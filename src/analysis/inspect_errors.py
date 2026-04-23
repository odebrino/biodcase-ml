import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.spectrogram import event_tensor
from src.utils.config import load_config


def infer_split_from_audio_path(audio_path: str, default_split: str) -> str:
    if "/train/" in audio_path:
        return "train"
    if "/validation/" in audio_path:
        return "validation"
    return default_split


def discover_default_report(outputs_root: Path) -> Path:
    candidates = sorted(outputs_root.glob("*/bpd_error_report.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No bpd_error_report.csv found under outputs/runs/. "
            "Pass --report explicitly."
        )
    return candidates[-1]


def tensor_to_image(tensor) -> Image.Image:
    array = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def export_errors(report_path: Path, config: dict, out_dir: Path, limit: int, split: str | None = None) -> int:
    report = pd.read_csv(report_path)
    if report.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = report.head(limit)
    count = 0
    for idx, row in rows.iterrows():
        row_dict = row.to_dict()
        default_split = split or config.get("official_test_split", config.get("test_split", config.get("val_split", "validation")))
        row_dict["split"] = infer_split_from_audio_path(str(row_dict.get("audio_path", "")), default_split)
        row_dict["label"] = row_dict["y_true_label"]
        tensor = event_tensor(row_dict, config["audio"], int(config["model"]["img_size"]))
        name = (
            f"{idx:03d}_{row_dict['dataset']}_{Path(row_dict['filename']).stem}_"
            f"{row_dict['y_true_label']}_as_{row_dict['y_pred_label']}.png"
        )
        tensor_to_image(tensor).save(out_dir / name)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PNGs for the main bpd/bmd errors.")
    parser.add_argument("--report", default=None)
    parser.add_argument("--config", default="configs/nitro4060.yaml")
    parser.add_argument("--out", default="outputs/error_samples")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--split", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report) if args.report else discover_default_report(Path("outputs/runs"))
    count = export_errors(report_path, load_config(args.config), Path(args.out), args.limit, args.split)
    print(f"exported {count} images")


if __name__ == "__main__":
    main()
