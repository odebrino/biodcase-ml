from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.spectrogram import (
    literal_time_frequency_crop,
    resize_spectrogram,
)
from src.utils.config import load_config


HANDCRAFTED_FEATURE_NAMES = [
    "duration_seconds",
    "real_duration_seconds",
    "low_frequency",
    "high_frequency",
    "frequency_span_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_85_hz",
    "mean_intensity",
    "std_intensity",
    "max_intensity",
    "time_centroid_seconds",
    "frequency_contrast",
    "temporal_contrast",
    "low_half_energy_ratio",
    "high_half_energy_ratio",
]


def valid_manifest_rows(manifest: pd.DataFrame, splits: list[str] | None = None) -> pd.DataFrame:
    frame = manifest.copy()
    if "valid_event" in frame.columns:
        frame = frame[frame["valid_event"].map(lambda value: str(value).lower() == "true")]
    if splits:
        frame = frame[frame["split"].isin(splits)]
    return frame.reset_index(drop=True)


def literal_patch_vector(row: dict, audio_cfg: dict, img_size: int) -> np.ndarray:
    crop = literal_time_frequency_crop(row, audio_cfg)
    patch = resize_spectrogram(crop.values, img_size)
    return patch.detach().cpu().numpy().astype(np.float32).reshape(-1)


def handcrafted_descriptor_vector(row: dict, audio_cfg: dict) -> np.ndarray:
    crop = literal_time_frequency_crop(row, audio_cfg)
    values = crop.values.detach().cpu().numpy().astype(np.float64)
    freqs = crop.frequencies_hz.detach().cpu().numpy().astype(np.float64)
    times = crop.times_seconds.detach().cpu().numpy().astype(np.float64)
    weights = np.maximum(values, 0.0)
    total = float(weights.sum())

    if total > 0:
        freq_weights = weights.sum(axis=1)
        time_weights = weights.sum(axis=0)
        spectral_centroid = float((freqs * freq_weights).sum() / max(1e-12, freq_weights.sum()))
        spectral_bandwidth = float(
            np.sqrt((((freqs - spectral_centroid) ** 2) * freq_weights).sum() / max(1e-12, freq_weights.sum()))
        )
        cumsum = np.cumsum(freq_weights)
        rolloff_idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1], side="left"))
        spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        time_centroid = float((times * time_weights).sum() / max(1e-12, time_weights.sum()))
    else:
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0
        spectral_rolloff = 0.0
        time_centroid = 0.0

    half = max(1, values.shape[0] // 2)
    low_energy = float(weights[:half].sum())
    high_energy = float(weights[half:].sum())
    if values.shape[0] > 1:
        freq_profile = values.mean(axis=1)
        frequency_contrast = float(np.percentile(freq_profile, 90) - np.percentile(freq_profile, 10))
    else:
        frequency_contrast = 0.0
    if values.shape[1] > 1:
        time_profile = values.mean(axis=0)
        temporal_contrast = float(np.percentile(time_profile, 90) - np.percentile(time_profile, 10))
    else:
        temporal_contrast = 0.0

    duration = float(row.get("duration_seconds", row.get("real_duration_seconds", 0.0)))
    real_duration = float(row.get("real_duration_seconds", duration))
    low_frequency = float(row["low_frequency"])
    high_frequency = float(row["high_frequency"])
    return np.asarray(
        [
            duration,
            real_duration,
            low_frequency,
            high_frequency,
            max(0.0, high_frequency - low_frequency),
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            float(values.mean()),
            float(values.std()),
            float(values.max()) if values.size else 0.0,
            time_centroid,
            frequency_contrast,
            temporal_contrast,
            low_energy / max(1e-12, total),
            high_energy / max(1e-12, total),
        ],
        dtype=np.float32,
    )


def hybrid_vector(row: dict, audio_cfg: dict, img_size: int) -> np.ndarray:
    return np.concatenate(
        [literal_patch_vector(row, audio_cfg, img_size), handcrafted_descriptor_vector(row, audio_cfg)],
    ).astype(np.float32)


def representation_vector(row: dict, audio_cfg: dict, family: str, img_size: int) -> np.ndarray:
    if family == "patch":
        return literal_patch_vector(row, audio_cfg, img_size)
    if family == "handcrafted":
        return handcrafted_descriptor_vector(row, audio_cfg)
    if family == "hybrid":
        return hybrid_vector(row, audio_cfg, img_size)
    raise ValueError(f"Unknown representation family: {family}")


def feature_names(family: str, img_size: int) -> list[str]:
    patch_names = [f"patch_{idx}" for idx in range(img_size * img_size)]
    if family == "patch":
        return patch_names
    if family == "handcrafted":
        return HANDCRAFTED_FEATURE_NAMES
    if family == "hybrid":
        return patch_names + HANDCRAFTED_FEATURE_NAMES
    raise ValueError(f"Unknown representation family: {family}")


def _apply_pca_if_requested(matrix: np.ndarray, family: str, img_size: int, pca_components: int | None):
    if not pca_components:
        return matrix, feature_names(family, img_size), None

    from sklearn.decomposition import PCA

    patch_width = img_size * img_size
    if family == "handcrafted":
        pca = PCA(n_components=pca_components, random_state=0)
        transformed = pca.fit_transform(matrix)
        names = [f"pca_{idx}" for idx in range(transformed.shape[1])]
        return transformed.astype(np.float32), names, pca
    if family == "patch":
        pca = PCA(n_components=pca_components, random_state=0)
        transformed = pca.fit_transform(matrix)
        names = [f"patch_pca_{idx}" for idx in range(transformed.shape[1])]
        return transformed.astype(np.float32), names, pca
    if family == "hybrid":
        pca = PCA(n_components=pca_components, random_state=0)
        patch_part = matrix[:, :patch_width]
        descriptor_part = matrix[:, patch_width:]
        patch_pca = pca.fit_transform(patch_part)
        transformed = np.concatenate([patch_pca, descriptor_part], axis=1)
        names = [f"patch_pca_{idx}" for idx in range(patch_pca.shape[1])] + HANDCRAFTED_FEATURE_NAMES
        return transformed.astype(np.float32), names, pca
    raise ValueError(f"Unknown representation family: {family}")


def export_representations(
    manifest_path: Path,
    config_path: Path,
    out_path: Path,
    family: str,
    img_size: int,
    splits: list[str] | None = None,
    limit: int | None = None,
    pca_components: int | None = None,
) -> dict:
    config = load_config(config_path)
    manifest = valid_manifest_rows(pd.read_csv(manifest_path), splits=splits)
    if limit is not None:
        manifest = manifest.head(limit).reset_index(drop=True)

    vectors = []
    labels = []
    metadata = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"{family} representation"):
        row_dict = row.to_dict()
        vectors.append(representation_vector(row_dict, config["audio"], family, img_size))
        labels.append(str(row_dict["label"]))
        metadata.append(
            {
                "split": row_dict.get("split", ""),
                "dataset": row_dict.get("dataset", ""),
                "filename": row_dict.get("filename", ""),
                "source_row": row_dict.get("source_row", ""),
                "label": row_dict.get("label", ""),
                "label_raw": row_dict.get("label_raw", ""),
                "low_frequency": row_dict.get("low_frequency", ""),
                "high_frequency": row_dict.get("high_frequency", ""),
                "clip_start_seconds": row_dict.get("clip_start_seconds", row_dict.get("start_seconds", "")),
                "clip_end_seconds": row_dict.get("clip_end_seconds", row_dict.get("end_seconds", "")),
            }
        )

    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32)
    matrix, names, pca = _apply_pca_if_requested(matrix, family, img_size, pca_components)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=matrix,
        y=np.asarray(labels),
        feature_names=np.asarray(names),
        metadata_json=np.asarray([json.dumps(item, sort_keys=True) for item in metadata]),
    )
    manifest_out = out_path.with_suffix(".manifest.csv")
    pd.DataFrame(metadata).to_csv(manifest_out, index=False)
    summary = {
        "out": str(out_path),
        "manifest": str(manifest_out),
        "family": family,
        "rows": int(matrix.shape[0]),
        "features": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "pca_components": pca_components,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist() if pca is not None else None,
        "spectrogram_config": config["audio"],
        "crop_semantics": "literal time-frequency crop from annotation coordinates",
    }
    with out_path.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export classical-ML-ready representations from the manifest.")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--out", default="outputs/representations/patch_train.npz")
    parser.add_argument("--family", choices=["patch", "handcrafted", "hybrid"], default="patch")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pca-components", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_representations(
        manifest_path=Path(args.manifest),
        config_path=Path(args.config),
        out_path=Path(args.out),
        family=args.family,
        img_size=args.img_size,
        splits=args.splits,
        limit=args.limit,
        pca_components=args.pca_components,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
