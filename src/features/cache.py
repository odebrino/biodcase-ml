from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.utils.reproducibility import file_signature, stable_hash


FEATURE_CACHE_VERSION = 3


def feature_cache_key(
    manifest_path: str | Path,
    split_name: str,
    feature_spec: dict[str, Any],
    audio_cfg: dict[str, Any],
    row_identity: list[dict[str, Any]],
) -> str:
    payload = {
        "version": FEATURE_CACHE_VERSION,
        "manifest": file_signature(manifest_path),
        "split_name": split_name,
        "feature_spec": feature_spec,
        "audio_cfg": audio_cfg,
        "row_identity": row_identity,
    }
    return stable_hash(payload)


def feature_cache_path(cache_root: str | Path, cache_key: str) -> Path:
    root = Path(cache_root)
    return root / cache_key[:2] / f"{cache_key}.joblib"


def load_feature_cache(cache_root: str | Path, cache_key: str):
    path = feature_cache_path(cache_root, cache_key)
    if not path.exists():
        return None
    return joblib.load(path)


def save_feature_cache(cache_root: str | Path, cache_key: str, payload) -> Path:
    path = feature_cache_path(cache_root, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path
