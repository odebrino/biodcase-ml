import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.spectrogram import event_tensor
from src.utils.config import load_config


def tensor_to_image(tensor) -> Image.Image:
    array = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def export_errors(report_path: Path, config: dict, out_dir: Path, limit: int) -> int:
    report = pd.read_csv(report_path)
    if report.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = report.head(limit)
    count = 0
    for idx, row in rows.iterrows():
        row_dict = row.to_dict()
        row_dict["split"] = "validation"
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
    parser.add_argument("--report", default="outputs/runs/20260421-213619/bpd_error_report.csv")
    parser.add_argument("--config", default="configs/nitro4060.yaml")
    parser.add_argument("--out", default="outputs/error_samples")
    parser.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = export_errors(Path(args.report), load_config(args.config), Path(args.out), args.limit)
    print(f"exported {count} images")


if __name__ == "__main__":
    main()

