from __future__ import annotations

from copy import deepcopy


SPECTROGRAM_PRESETS = {
    "aplose_512_98": {
        "sample_rate": 250,
        "n_fft": 512,
        "win_length": 256,
        "hop_length": 5,
        "n_mels": 256,
        "f_min": 0.0,
        "f_max": 125.0,
        "overlap_percent": 98,
        "scale": "Default",
        "crop_frequency_semantics": "time_crop_with_frequency_band_mask",
        "preset_note": (
            "Guideline-inspired APLOSE setting nfft=512, winsize=256, overlap=98. "
            "This repository still uses its current mel-tensor representation; this is not a literal APLOSE display export."
        ),
    },
    "aplose_256_90": {
        "sample_rate": 250,
        "n_fft": 256,
        "win_length": 256,
        "hop_length": 26,
        "n_mels": 128,
        "f_min": 0.0,
        "f_max": 125.0,
        "overlap_percent": 90,
        "scale": "Default",
        "crop_frequency_semantics": "time_crop_with_frequency_band_mask",
        "preset_note": (
            "Guideline-inspired APLOSE setting nfft=256, winsize=256, overlap=90. "
            "This repository still uses its current mel-tensor representation; this is not a literal APLOSE display export."
        ),
    },
}


def resolve_spectrogram_preset(audio_cfg: dict) -> dict:
    resolved = deepcopy(audio_cfg)
    preset_name = resolved.get("preset")
    if not preset_name:
        return resolved
    if preset_name not in SPECTROGRAM_PRESETS:
        known = ", ".join(sorted(SPECTROGRAM_PRESETS))
        raise ValueError(f"Unknown spectrogram preset '{preset_name}'. Known presets: {known}")

    preset = deepcopy(SPECTROGRAM_PRESETS[preset_name])
    preset.update(resolved)
    preset["preset"] = preset_name
    return preset
