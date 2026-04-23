from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.data.representations import valid_manifest_rows
from src.data.spectrogram import event_tensor, literal_time_frequency_crop, resize_spectrogram
from src.utils.config import load_config


def grayscale_image(tensor, size: int) -> Image.Image:
    resized = resize_spectrogram(tensor, size)
    array = (resized.clamp(0, 1).detach().cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(array, mode="L").convert("RGB")


def rgb_tensor_image(tensor, size: int) -> Image.Image:
    import numpy as np

    resized = tensor
    if tensor.shape[-1] != size or tensor.shape[-2] != size:
        channels = [resize_spectrogram(tensor[channel], size) for channel in range(tensor.shape[0])]
        resized = tensor.new_empty((tensor.shape[0], size, size))
        for channel, value in enumerate(channels):
            resized[channel] = value
    array = (resized.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def make_panel(literal_crop: Image.Image, mask_highlight: Image.Image, title: str) -> Image.Image:
    width, height = literal_crop.size
    panel = Image.new("RGB", (width * 2, height + 56), "white")
    panel.paste(literal_crop, (0, 28))
    panel.paste(mask_highlight, (width, 28))
    draw = ImageDraw.Draw(panel)
    draw.text((4, 6), "literal crop", fill="black")
    draw.text((width + 4, 6), "time crop + band mask", fill="black")
    draw.text((4, height + 34), title[:120], fill="black")
    return panel


def export_crop_verification(
    manifest_path: Path,
    config_path: Path,
    out_dir: Path,
    img_size: int = 224,
    splits: list[str] | None = None,
    per_class: int = 2,
    limit: int | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    manifest = valid_manifest_rows(pd.read_csv(manifest_path), splits=splits)
    if per_class:
        manifest = manifest.groupby(["split", "label"], group_keys=False).head(per_class).reset_index(drop=True)
    if limit is not None:
        manifest = manifest.head(limit).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="crop verification"):
        row_dict = row.to_dict()
        literal = literal_time_frequency_crop(row_dict, config["audio"])
        mask_tensor = event_tensor(row_dict, config["audio"], img_size)
        literal_image = grayscale_image(literal.values, img_size)
        mask_image = rgb_tensor_image(mask_tensor, img_size)
        title = f"{row_dict.get('dataset')} {row_dict.get('filename')} row={row_dict.get('source_row')} label={row_dict.get('label')}"
        panel = make_panel(literal_image, mask_image, title)

        image_name = f"{idx:04d}_{row_dict.get('split')}_{row_dict.get('label')}_{row_dict.get('dataset')}_row{row_dict.get('source_row')}.png"
        image_path = out_dir / image_name
        panel.save(image_path)
        rows.append(
            {
                "image_path": str(image_path),
                "split": row_dict.get("split", ""),
                "dataset": row_dict.get("dataset", ""),
                "filename": row_dict.get("filename", ""),
                "source_row": row_dict.get("source_row", ""),
                "label": row_dict.get("label", ""),
                "label_raw": row_dict.get("label_raw", ""),
                "annotated_start_seconds": row_dict.get("clip_start_seconds", row_dict.get("start_seconds", "")),
                "annotated_end_seconds": row_dict.get("clip_end_seconds", row_dict.get("end_seconds", "")),
                "annotated_low_frequency": row_dict.get("low_frequency", ""),
                "annotated_high_frequency": row_dict.get("high_frequency", ""),
                "crop_time_min_seconds": float(literal.times_seconds.min().item()) if literal.times_seconds.numel() else "",
                "crop_time_max_seconds": float(literal.times_seconds.max().item()) if literal.times_seconds.numel() else "",
                "crop_frequency_min_hz": float(literal.frequencies_hz.min().item()) if literal.frequencies_hz.numel() else "",
                "crop_frequency_max_hz": float(literal.frequencies_hz.max().item()) if literal.frequencies_hz.numel() else "",
                "crop_bins_frequency": int(literal.values.shape[0]),
                "crop_bins_time": int(literal.values.shape[1]),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "crop_verification_manifest.csv", index=False)
    with (out_dir / "crop_verification_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "rows": len(summary),
                "img_size": img_size,
                "config": str(config_path),
                "spectrogram_config": config["audio"],
                "purpose": "Manual verification of annotation-to-spectrogram coordinate mapping.",
            },
            handle,
            indent=2,
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export small crop panels for manual coordinate verification.")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--out", default="outputs/crop_verification")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_crop_verification(
        manifest_path=Path(args.manifest),
        config_path=Path(args.config),
        out_dir=Path(args.out),
        img_size=args.img_size,
        splits=args.splits,
        per_class=args.per_class,
        limit=args.limit,
    )
    print(f"Wrote {len(summary)} crop verification rows")


if __name__ == "__main__":
    main()
