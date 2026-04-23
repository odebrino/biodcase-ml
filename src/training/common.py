from pathlib import Path

import pandas as pd

from src.data.dataset import BioacousticDataset


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Training and evaluation require PyTorch. Install the base dependencies with "
            "`pip install -r requirements.txt`, then install the PyTorch stack with "
            "`pip install -r requirements-cu124.txt`."
        ) from exc
    return torch, DataLoader


def create_loader(
    manifest_path: str | Path,
    split: str,
    class_names: list[str],
    train: bool,
    config: dict,
):
    torch, DataLoader = _require_torch()
    from torch.utils.data import WeightedRandomSampler

    data_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    cache_cfg = config.get("cache", {})
    dataset = BioacousticDataset(
        manifest_path=manifest_path,
        split=split,
        class_names=class_names,
        img_size=int(config["model"]["img_size"]),
        audio_cfg=config["audio"],
        train=train,
        mode=data_cfg.get("mode", "spectrogram"),
        cache_cfg=cache_cfg if cache_cfg.get("enabled", True) else None,
        augmentation=config.get("augmentation", {}) if train else {},
    )
    num_workers = int(training_cfg.get("num_workers", 0))
    sampler = None
    shuffle = train
    if train and training_cfg.get("sampler") == "weighted":
        sampler_class_weights = class_weight_tensor(dataset.class_counts())
        sampler_class_weights = apply_class_multipliers(
            sampler_class_weights,
            class_names,
            training_cfg.get("class_weight_multipliers", {}),
        )
        sample_weights = dataset.sample_weights(sampler_class_weights)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(training_cfg.get("pin_memory", torch.cuda.is_available())) and torch.cuda.is_available(),
        persistent_workers=bool(training_cfg.get("persistent_workers", False)) and num_workers > 0,
    )


def class_weight_tensor(class_counts: list[int]):
    torch, _ = _require_torch()
    total = sum(class_counts)
    weights = []
    for count in class_counts:
        weights.append(total / max(1, count))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor / weights_tensor.mean()


def apply_class_multipliers(weights, class_names: list[str], multipliers: dict[str, float]):
    if not multipliers:
        return weights
    adjusted = weights.clone()
    for label, factor in multipliers.items():
        if label in class_names:
            adjusted[class_names.index(label)] *= float(factor)
    return adjusted / adjusted.mean()


def save_predictions(rows: list[dict], out_path: str | Path) -> None:
    pd.DataFrame(rows).to_csv(out_path, index=False)
