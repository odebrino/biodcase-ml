from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

from src.data.labels import normalize_label
from src.data.spectrogram import augment_spectrogram, cached_event_tensor


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install the PyTorch stack described in the README.") from exc
    return torch


class BioacousticDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        class_names: list[str],
        img_size: int,
        audio_cfg: dict,
        train: bool = False,
        mode: str = "spectrogram",
        cache_cfg: dict | str | Path | None = None,
        augmentation: dict | None = None,
    ) -> None:
        self.frame = pd.read_csv(manifest_path)
        self.frame = self.frame[self.frame["split"] == split].reset_index(drop=True)
        if "valid_event" in self.frame.columns:
            self.frame = self.frame[self.frame["valid_event"].map(lambda value: str(value).lower() == "true")].reset_index(drop=True)
        if "label_raw" not in self.frame.columns:
            self.frame["label_raw"] = self.frame["label"]
        self.frame["label"] = self.frame["label"].map(normalize_label)

        if self.frame.empty:
            raise ValueError(f"No rows found for split '{split}' in {manifest_path}")

        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.img_size = img_size
        self.audio_cfg = audio_cfg
        self.train = train
        self.mode = mode
        self.cache_cfg = cache_cfg
        self.augmentation = augmentation or {}

        unknown = sorted(set(self.frame["label"]) - set(class_names))
        if unknown:
            raise ValueError(f"Labels not declared in config: {unknown}")
        if mode == "image" and "image_path" not in self.frame.columns:
            raise ValueError("Image mode requires an image_path column.")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index].to_dict()
        label = self.class_to_idx[row["label"]]

        if self.mode == "image":
            tensor = self._image_tensor(row["image_path"])
        elif self.mode == "spectrogram":
            tensor = cached_event_tensor(row, self.audio_cfg, self.img_size, self.cache_cfg)
        else:
            raise ValueError(f"Unknown dataset mode: {self.mode}")

        if self.train and self.mode == "spectrogram":
            tensor = augment_spectrogram(tensor, self.augmentation)

        metadata = {
            "audio_path": row.get("audio_path", ""),
            "dataset": row.get("dataset", ""),
            "filename": row.get("filename", ""),
            "label": row["label"],
            "label_raw": row.get("label_raw", row["label"]),
            "label_display": row.get("label_display", ""),
            "source_row": row.get("source_row", ""),
            "low_frequency": row.get("low_frequency", ""),
            "high_frequency": row.get("high_frequency", ""),
            "duration_seconds": row.get("duration_seconds", ""),
            "real_duration_seconds": row.get("real_duration_seconds", ""),
            "clip_start_seconds": row.get("clip_start_seconds", row.get("start_seconds", "")),
            "clip_end_seconds": row.get("clip_end_seconds", row.get("end_seconds", "")),
        }
        if "image_path" in row:
            metadata["image_path"] = row["image_path"]
        return tensor, label, metadata

    def _image_tensor(self, image_path: str | Path):
        torch = require_torch()
        image = Image.open(image_path).convert("RGB").resize((self.img_size, self.img_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def class_counts(self) -> list[int]:
        counts = self.frame["label"].value_counts()
        return [int(counts.get(label, 0)) for label in self.class_names]

    def sample_weights(self, class_weights):
        torch = require_torch()
        values = []
        for label in self.frame["label"]:
            values.append(float(class_weights[self.class_to_idx[label]]))
        return torch.tensor(values, dtype=torch.double)
