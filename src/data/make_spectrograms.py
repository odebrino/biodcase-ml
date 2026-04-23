import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.spectrogram import event_tensor


def tensor_to_rgb_image(tensor) -> Image.Image:
    array = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def make_spectrograms(
    manifest_path: Path,
    out_root: Path,
    processed_manifest_path: Path,
    audio_cfg: dict,
    img_size: int,
    splits: list[str] | None = None,
    limit: int | None = None,
    max_per_class_per_split: int | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    if "valid_event" in manifest.columns:
        manifest = manifest[manifest["valid_event"].map(lambda value: str(value).lower() == "true")]
    if splits:
        manifest = manifest[manifest["split"].isin(splits)]
    if max_per_class_per_split is not None:
        manifest = (
            manifest.groupby(["split", "label"], group_keys=False)
            .head(max_per_class_per_split)
            .reset_index(drop=True)
        )
    if limit is not None:
        manifest = manifest.head(limit).reset_index(drop=True)

    rows = []
    for manifest_index, row in tqdm(manifest.iterrows(), total=len(manifest), desc="spectrograms"):
        row_dict = row.to_dict()
        image = tensor_to_rgb_image(event_tensor(row_dict, audio_cfg, img_size))

        output_dir = out_root / str(row["split"]) / str(row["label"])
        output_dir.mkdir(parents=True, exist_ok=True)
        image_name = f"{row['dataset']}__{Path(row['filename']).stem}__row{manifest_index}.png"
        image_path = output_dir / image_name
        image.save(image_path)

        enriched = row_dict
        enriched["image_path"] = str(image_path)
        rows.append(enriched)

    processed = pd.DataFrame(rows)
    processed_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_manifest_path, index=False)
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sample spectrogram PNGs from the clean manifest.")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--out", default="processed_images")
    parser.add_argument("--processed-manifest", default="processed_manifest.csv")
    parser.add_argument("--sample-rate", type=int, default=250)
    parser.add_argument("--margin-seconds", type=float, default=1.0)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=16)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=125.0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-per-class-per-split", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_cfg = {
        "sample_rate": args.sample_rate,
        "margin_seconds": args.margin_seconds,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "f_min": args.f_min,
        "f_max": args.f_max,
    }
    processed = make_spectrograms(
        manifest_path=Path(args.manifest),
        out_root=Path(args.out),
        processed_manifest_path=Path(args.processed_manifest),
        audio_cfg=audio_cfg,
        img_size=args.img_size,
        splits=args.splits,
        limit=args.limit,
        max_per_class_per_split=args.max_per_class_per_split,
    )
    print(f"Wrote {len(processed)} spectrogram PNGs")


if __name__ == "__main__":
    main()
