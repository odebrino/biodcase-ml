from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.data.spectrogram_presets import resolve_spectrogram_preset


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    parent = config.pop("extends", None)
    if not parent:
        if "audio" in config:
            config["audio"] = resolve_spectrogram_preset(config["audio"])
        return config

    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
        if not parent_path.exists():
            parent_path = Path(parent)
    base = load_config(parent_path)
    if isinstance(config.get("audio"), dict) and config["audio"].get("preset"):
        inherited_audio = base.get("audio", {})
        base = deepcopy(base)
        base["audio"] = {
            key: value
            for key, value in inherited_audio.items()
            if key in {"margin_seconds", "normalization", "db_min", "db_max"}
        }
    merged = deep_update(base, config)
    if "audio" in merged:
        merged["audio"] = resolve_spectrogram_preset(merged["audio"])
    return merged


def save_config(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
