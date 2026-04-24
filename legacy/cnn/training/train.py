import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from legacy.cnn.models.resnet import create_model
from legacy.cnn.training.common import apply_class_multipliers, class_weight_tensor, create_loader
from legacy.cnn.training.evaluate import evaluate_checkpoint
from legacy.cnn.training.losses import make_loss
from src.utils.config import load_config, save_config
from src.evaluation.metrics import compute_metrics
from src.utils.seed import set_seed


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Training requires PyTorch. Install the base dependencies with "
            "`pip install -r requirements.txt`, then install the PyTorch stack with "
            "`pip install -r requirements-cu124.txt`."
        ) from exc
    return torch


def train_one_epoch(torch, model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        running_loss += float(loss.item()) * batch_size
        total += batch_size

    return running_loss / max(1, total)


def train_one_epoch_amp(torch, model, loader, criterion, optimizer, scaler, device, amp_enabled: bool, grad_clip_norm: float | None) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if grad_clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        running_loss += float(loss.item()) * batch_size
        total += batch_size

    return running_loss / max(1, total)


def evaluate_loader(torch, model, loader, class_names, device) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="selection/eval", leave=False):
            logits = model(images.to(device, non_blocking=True))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return compute_metrics(y_true, y_pred, class_names)


def split_semantics(config: dict) -> dict:
    train_split = config.get("train_split", "train")
    official_test_split = config.get("official_test_split", config.get("test_split", config.get("val_split", "validation")))
    inner_selection_split = config.get("inner_selection_split", config.get("selection_split"))
    allow_official = bool(config.get("training", {}).get("allow_official_test_for_selection", False))
    if not inner_selection_split and not allow_official:
        raise ValueError(
            "Historical CNN training requires an explicit train-only inner_selection_split. "
            "Refusing to fall back to the official held-out test split."
        )

    used_official = False
    effective_selection_split = inner_selection_split
    selection_strategy = config.get("selection_strategy", "explicit_inner_split_required")
    if not effective_selection_split:
        effective_selection_split = official_test_split
        used_official = True
        selection_strategy = "explicit_opt_in_official_test"
    return {
        "train_split": train_split,
        "official_test_split": official_test_split,
        "inner_selection_split": inner_selection_split,
        "effective_selection_split": effective_selection_split,
        "used_official_test_for_selection": used_official,
        "selection_strategy": selection_strategy,
    }


def train(config: dict) -> Path:
    torch = _require_torch()
    start_time = time.time()
    set_seed(int(config["training"]["seed"]))
    requested_device = str(config["training"].get("device", "cuda"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested, but no CUDA device is available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    run_dir = Path(config["training"]["output_dir"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    semantics = split_semantics(config)

    class_names = config["classes"]
    train_loader = create_loader(
        manifest_path=config.get("processed_manifest", "processed_manifest.csv"),
        split=semantics["train_split"],
        class_names=class_names,
        train=True,
        config=config,
    )
    val_loader = create_loader(
        manifest_path=config.get("processed_manifest", "processed_manifest.csv"),
        split=semantics["effective_selection_split"],
        class_names=class_names,
        train=False,
        config=config,
    )

    model = create_model(
        name=config["model"]["name"],
        num_classes=len(class_names),
        pretrained=bool(config["model"]["pretrained"]),
    ).to(device)
    if bool(config["training"].get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    class_weights = None
    if bool(config["training"].get("use_class_weights", True)):
        class_weights = class_weight_tensor(train_loader.dataset.class_counts()).to(device)
        class_weights = apply_class_multipliers(
            class_weights,
            class_names,
            config["training"].get("class_weight_multipliers", {}),
        ).to(device)
    criterion = make_loss(config["training"].get("loss", {"name": "cross_entropy"}), class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    amp_enabled = device.type == "cuda" and bool(config["training"].get("mixed_precision", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_model.pt"

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch_amp(
            torch,
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            float(config["training"].get("grad_clip_norm", 0)) or None,
        )
        metrics = evaluate_loader(torch, model, val_loader, class_names, device)
        scheduler.step(metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "selection_split": semantics["effective_selection_split"],
            "selection_split_role": (
                "official_held_out_test" if semantics["used_official_test_for_selection"] else "inner_validation"
            ),
            "official_test_split": semantics["official_test_split"],
            "used_official_test_for_selection": semantics["used_official_test_for_selection"],
            "selection_strategy": semantics["selection_strategy"],
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_macro_f1_present_classes": metrics["macro_f1_present_classes"],
            "val_weighted_f1": metrics["weighted_f1"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            epochs_without_improvement = 0
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "class_names": class_names,
                    "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
                    "args": config,
                    "best_metrics": metrics,
                },
                best_path,
            )
            with (run_dir / "best_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "model_state_dict": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                "class_names": class_names,
                "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
                "args": config,
                "last_metrics": metrics,
            },
            last_path,
        )

        print(
            f"epoch={epoch} loss={train_loss:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} val_acc={metrics['accuracy']:.4f}"
        )

        patience = int(config["training"].get("early_stopping_patience", 0))
        if patience > 0 and epochs_without_improvement >= patience:
            break

    evaluate_checkpoint(best_path, config, run_dir)
    write_run_metadata(torch, run_dir, config, device, start_time)
    return run_dir


def write_run_metadata(torch, run_dir: Path, config: dict, device, start_time: float) -> None:
    semantics = split_semantics(config)
    metadata = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "seed": int(config["training"]["seed"]),
        "duration_seconds": round(time.time() - start_time, 3),
        "train_split": semantics["train_split"],
        "official_test_split": semantics["official_test_split"],
        "inner_selection_split": semantics["inner_selection_split"],
        "effective_selection_split": semantics["effective_selection_split"],
        "selection_strategy": semantics["selection_strategy"],
        "used_official_test_for_selection": semantics["used_official_test_for_selection"],
    }
    try:
        import torchaudio
        import torchvision

        metadata["torchaudio"] = torchaudio.__version__
        metadata["torchvision"] = torchvision.__version__
    except ImportError:
        pass

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        metadata["gpu_name"] = props.name
        metadata["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
        metadata["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)

    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet classifier from the event manifest.")
    parser.add_argument("--config", default="legacy/cnn/configs/nitro4060.yaml")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--processed-manifest", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["processed_manifest"] = args.processed_manifest or args.manifest
    run_dir = train(config)
    print(f"Run written to {run_dir}")


if __name__ == "__main__":
    main()
