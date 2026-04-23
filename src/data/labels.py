from __future__ import annotations


CANONICAL_LABELS = ("bma", "bmb", "bmz", "bmd", "bp20", "bp20plus", "bpd")

LABEL_DISPLAY_NAMES = {
    "bma": "Bm-A",
    "bmb": "Bm-B",
    "bmz": "Bm-Z",
    "bmd": "Bm-D",
    "bp20": "Bp-20",
    "bp20plus": "Bp-20Plus",
    "bpd": "Bp-40Down",
}

LABEL_ALIASES = {
    "bma": "bma",
    "bmb": "bmb",
    "bmz": "bmz",
    "bmd": "bmd",
    "bp20": "bp20",
    "bp20plus": "bp20plus",
    "bpd": "bpd",
    "bp40down": "bpd",
}


def label_alias_key(label: object) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def normalize_label(label: object) -> str:
    key = label_alias_key(label)
    if key not in LABEL_ALIASES:
        raise ValueError(f"unknown label alias: {label!r}")
    return LABEL_ALIASES[key]


def label_display_name(label: object) -> str:
    return LABEL_DISPLAY_NAMES[normalize_label(label)]
