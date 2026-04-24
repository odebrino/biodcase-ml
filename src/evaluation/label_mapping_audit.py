from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


NOTEBOOK_3CLASS_MAPPING = {
    "bma": "ABZ",
    "bmb": "ABZ",
    "bmz": "ABZ",
    "bmd": "DDswp",
    "bpd": "DDswp",
    "bp20": "20Hz20Plus",
    "bp20plus": "20Hz20Plus",
}


def _counts(frame: pd.DataFrame, group_columns: list[str]) -> list[dict]:
    columns = group_columns + ["label"]
    present = [column for column in columns if column in frame.columns]
    if "label" not in present:
        return []
    return (
        frame.groupby(present, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(present)
        .to_dict("records")
    )


def _aggregated_counts(frame: pd.DataFrame, group_columns: list[str]) -> list[dict]:
    if "label" not in frame.columns:
        return []
    work = frame.copy()
    work["notebook_3class_label"] = work["label"].astype(str).map(NOTEBOOK_3CLASS_MAPPING).fillna(work["label"].astype(str))
    present = [column for column in group_columns if column in work.columns] + ["notebook_3class_label"]
    return (
        work.groupby(present, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(present)
        .to_dict("records")
    )


def write_label_mapping_audit(manifest: pd.DataFrame, config: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_labels = list(config.get("classes", sorted(manifest["label"].dropna().astype(str).unique())))
    aggregation_cfg = config.get("label_aggregation", {}) or {}
    aggregation_enabled = bool(aggregation_cfg.get("enabled", False))
    current_uses_notebook_mapping = aggregation_enabled and dict(aggregation_cfg.get("mapping", {})) == NOTEBOOK_3CLASS_MAPPING
    manifest_labels = sorted(manifest["label"].dropna().astype(str).unique()) if "label" in manifest.columns else []
    notebook_labels = sorted(set(NOTEBOOK_3CLASS_MAPPING))
    unknown_for_notebook = sorted(set(manifest_labels) - set(notebook_labels))
    split_counts = _counts(manifest, ["split"])
    domain_counts = _counts(manifest, ["split", "dataset"])
    aggregated_split_counts = _aggregated_counts(manifest, ["split"])
    aggregated_domain_counts = _aggregated_counts(manifest, ["split", "dataset"])
    conclusion = (
        "The maintained strict KNN config uses the original 7 labels, not the notebook 3-class aggregation. "
        "That label granularity mismatch can make the notebook's reported KNN accuracy non-comparable, especially "
        "because ABZ, DDswp, and 20Hz20Plus collapse known ambiguous subclasses. The larger issue remains evaluation "
        "protocol: same-domain/random splits are optimistic while dataset/domain-aware CV collapses."
    )
    if current_uses_notebook_mapping:
        conclusion = (
            "The config explicitly enables the notebook 3-class aggregation. This must only be used if the grading "
            "specification confirms that aggregated labels, not the original 7 labels, are the official target."
        )

    report = {
        "current_config_labels": current_labels,
        "manifest_labels": manifest_labels,
        "notebook_3class_mapping": NOTEBOOK_3CLASS_MAPPING,
        "label_aggregation_config": aggregation_cfg,
        "current_uses_notebook_3class_mapping": current_uses_notebook_mapping,
        "unknown_labels_for_notebook_mapping": unknown_for_notebook,
        "class_counts_by_split": split_counts,
        "class_counts_by_domain": domain_counts,
        "notebook_3class_counts_by_split": aggregated_split_counts,
        "notebook_3class_counts_by_domain": aggregated_domain_counts,
        "could_mapping_explain_notebook_gap": not current_uses_notebook_mapping,
        "conclusion": conclusion,
    }
    json_path = output_dir / "label_mapping_audit.json"
    md_path = output_dir / "label_mapping_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Label Mapping Audit",
        "",
        "## Current Strict KNN Labels",
        "",
        "- " + ", ".join(f"`{label}`" for label in current_labels),
        "",
        "## Notebook 3-Class Mapping",
        "",
    ]
    for label, mapped in NOTEBOOK_3CLASS_MAPPING.items():
        lines.append(f"- `{label}` -> `{mapped}`")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
            "## Class Counts By Split",
            "",
        ]
    )
    for row in split_counts:
        lines.append(f"- `{row.get('split')}` / `{row.get('label')}`: {int(row.get('count', 0))}")
    lines.extend(["", "## Notebook 3-Class Counts By Split", ""])
    for row in aggregated_split_counts:
        lines.append(f"- `{row.get('split')}` / `{row.get('notebook_3class_label')}`: {int(row.get('count', 0))}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
