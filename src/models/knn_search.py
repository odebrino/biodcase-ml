from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    ParameterGrid,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None

from src.data.representations import valid_manifest_rows
from src.evaluation.label_mapping_audit import write_label_mapping_audit
from src.evaluation.metrics import compute_metrics
from src.evaluation.split_checks import assert_split_integrity, build_split_integrity_report, write_split_integrity_report
from src.features.feature_sets import build_feature_matrix
from src.models.knn_pipeline import (
    apply_label_aggregation,
    build_model_pipeline,
    build_submission_config_from_candidate,
    encode_labels,
    feature_cache_config,
    load_manifest_and_validate,
)
from src.utils.config import load_config, save_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train-only KNN feature and hyperparameter search.")
    parser.add_argument("--config", default="configs/knn_search.yaml")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/reports/knn_search")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--stage", default="full", choices=["1", "2", "3", "full"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cv-focused", action="store_true", help="Rank by StratifiedKFold CV first, with domain CV as diagnostics.")
    return parser.parse_args()


def _candidate_group_column(train_frame: pd.DataFrame, requested_group_column: str | None = None) -> str | None:
    if requested_group_column in train_frame.columns:
        return requested_group_column
    for column in ["domain", "site", "recording_id", "session_id", "dataset", "audio_path"]:
        if column in train_frame.columns and train_frame[column].nunique(dropna=False) >= 2:
            return column
    return None


def _safe_stratified_splits(train_frame: pd.DataFrame, y: np.ndarray, n_splits: int, seed: int):
    min_class_count = int(np.bincount(y).min())
    effective_splits = max(2, min(n_splits, min_class_count))
    splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    return list(splitter.split(train_frame, y))


def _cv_scenarios(train_frame: pd.DataFrame, y: np.ndarray, cv_cfg: dict, seed: int) -> list[dict]:
    requested_group_column = cv_cfg.get("group_column")
    group_column = _candidate_group_column(train_frame, requested_group_column)
    groups = train_frame[group_column].astype(str).to_numpy() if group_column else None
    n_splits = int(cv_cfg.get("n_splits", 4))
    scenarios = [
        {
            "name": "stratified_kfold",
            "strategy": "StratifiedKFold",
            "group_column": None,
            "splits": _safe_stratified_splits(train_frame, y, n_splits=n_splits, seed=seed),
            "is_domain_aware": False,
        }
    ]

    if groups is not None and len(np.unique(groups)) >= max(2, n_splits):
        if StratifiedGroupKFold is not None:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            scenario_name = "stratified_group_kfold"
        else:
            splitter = GroupKFold(n_splits=n_splits)
            scenario_name = "group_kfold"
        scenarios.append(
            {
                "name": scenario_name,
                "strategy": splitter.__class__.__name__,
                "group_column": group_column,
                "splits": list(splitter.split(train_frame, y, groups)),
                "is_domain_aware": True,
            }
        )

    max_logo_groups = int(cv_cfg.get("max_leave_one_group_splits", 12))
    if groups is not None and 2 <= len(np.unique(groups)) <= max_logo_groups:
        splitter = LeaveOneGroupOut()
        scenarios.append(
            {
                "name": "leave_one_domain_out",
                "strategy": "LeaveOneGroupOut",
                "group_column": group_column,
                "splits": list(splitter.split(train_frame, y, groups)),
                "is_domain_aware": True,
            }
        )
    return scenarios


def _primary_scenario(scenarios: list[dict], cv_cfg: dict) -> dict:
    requested = str(cv_cfg.get("primary", "domain_aware")).lower()
    if requested in {"stratified", "stratified_kfold"}:
        return scenarios[0]
    for preferred in ["leave_one_domain_out", "stratified_group_kfold", "group_kfold"]:
        for scenario in scenarios:
            if scenario["name"] == preferred:
                return scenario
    return scenarios[0]


def _expand_stage(stage_cfg: dict, feature_sets: list[str], seed: int, max_candidates: int | None) -> list[dict]:
    grid = {
        "feature_set": feature_sets,
        "imputer": stage_cfg.get("imputers", ["median"]),
        "scaler": stage_cfg.get("scalers", ["standard"]),
        "reducer": stage_cfg.get("reducers", [{"type": "none"}]),
        "n_neighbors": stage_cfg.get("n_neighbors", [5]),
        "weights": stage_cfg.get("weights", ["distance"]),
        "metric": stage_cfg.get("metrics", ["euclidean"]),
        "algorithm": stage_cfg.get("algorithms", ["auto"]),
        "leaf_size": stage_cfg.get("leaf_sizes", [30]),
        "p": stage_cfg.get("p_values", [2]),
    }
    candidates = []
    for candidate in ParameterGrid(grid):
        reducer = deepcopy(candidate["reducer"])
        metric = str(candidate["metric"]).lower()
        algorithm = str(candidate["algorithm"]).lower()
        if metric == "cosine" and algorithm in {"ball_tree", "kd_tree"}:
            continue
        if metric != "minkowski":
            candidate["p"] = 2
        candidate["reducer"] = reducer
        candidates.append(candidate)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    limit = stage_cfg.get("max_candidates")
    if max_candidates is not None:
        limit = min(limit, max_candidates) if limit is not None else max_candidates
    if limit is not None:
        limit = int(limit)
        if bool(stage_cfg.get("ensure_feature_coverage", False)):
            covered = []
            seen_features = set()
            for candidate in candidates:
                if candidate["feature_set"] not in seen_features:
                    covered.append(candidate)
                    seen_features.add(candidate["feature_set"])
                if len(covered) >= limit:
                    break
            remaining = [candidate for candidate in candidates if candidate not in covered]
            candidates = (covered + remaining)[:limit]
        else:
            candidates = candidates[:limit]
    return candidates


def _evaluate_candidate(
    candidate: dict,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    class_names: list[str],
    seed: int,
    n_jobs: int,
) -> dict:
    try:
        min_fit_rows = min(len(train_idx) for train_idx, _ in splits)
        pipeline = build_model_pipeline(
            {"imputer": candidate["imputer"], "scaler": candidate["scaler"], "reducer": candidate["reducer"]},
            candidate,
            seed=seed,
            fit_rows=min_fit_rows,
        )
        scoring = {
            "accuracy": "accuracy",
            "macro_precision": make_scorer(precision_score, average="macro", zero_division=0),
            "macro_f1": make_scorer(f1_score, average="macro"),
            "weighted_f1": make_scorer(f1_score, average="weighted"),
        }
        cv = cross_validate(
            pipeline,
            X,
            y,
            cv=splits,
            scoring=scoring,
            n_jobs=n_jobs,
            error_score="raise",
            return_train_score=False,
        )
        return {
            **candidate,
            "status": "ok",
            "cv_accuracy_mean": float(np.mean(cv["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv["test_accuracy"])),
            "cv_macro_precision_mean": float(np.mean(cv["test_macro_precision"])),
            "cv_macro_precision_std": float(np.std(cv["test_macro_precision"])),
            "cv_macro_f1_mean": float(np.mean(cv["test_macro_f1"])),
            "cv_macro_f1_std": float(np.std(cv["test_macro_f1"])),
            "cv_weighted_f1_mean": float(np.mean(cv["test_weighted_f1"])),
            "cv_weighted_f1_std": float(np.std(cv["test_weighted_f1"])),
            "worst_cv_accuracy": float(np.min(cv["test_accuracy"])),
            "worst_cv_macro_precision": float(np.min(cv["test_macro_precision"])),
            "worst_cv_macro_f1": float(np.min(cv["test_macro_f1"])),
            "worst_cv_weighted_f1": float(np.min(cv["test_weighted_f1"])),
        }
    except Exception as exc:
        return {
            **candidate,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "cv_accuracy_mean": float("nan"),
            "cv_accuracy_std": float("nan"),
            "cv_macro_precision_mean": float("nan"),
            "cv_macro_precision_std": float("nan"),
            "cv_macro_f1_mean": float("nan"),
            "cv_macro_f1_std": float("nan"),
            "cv_weighted_f1_mean": float("nan"),
            "cv_weighted_f1_std": float("nan"),
            "worst_cv_accuracy": float("nan"),
            "worst_cv_macro_precision": float("nan"),
            "worst_cv_macro_f1": float("nan"),
            "worst_cv_weighted_f1": float("nan"),
        }


def _scenario_for_name(scenarios: list[dict], name: str) -> dict | None:
    for scenario in scenarios:
        if scenario["name"] == name:
            return scenario
    return None


def _domain_scenario(scenarios: list[dict], primary: dict) -> dict:
    if primary["is_domain_aware"]:
        return primary
    for preferred in ["leave_one_domain_out", "stratified_group_kfold", "group_kfold"]:
        scenario = _scenario_for_name(scenarios, preferred)
        if scenario is not None:
            return scenario
    return primary


def _flatten_cv_scores(prefix: str, result: dict) -> dict:
    return {
        f"{prefix}_accuracy_mean": result.get("cv_accuracy_mean", float("nan")),
        f"{prefix}_accuracy_std": result.get("cv_accuracy_std", float("nan")),
        f"{prefix}_macro_precision_mean": result.get("cv_macro_precision_mean", float("nan")),
        f"{prefix}_macro_precision_std": result.get("cv_macro_precision_std", float("nan")),
        f"{prefix}_macro_f1_mean": result.get("cv_macro_f1_mean", float("nan")),
        f"{prefix}_macro_f1_std": result.get("cv_macro_f1_std", float("nan")),
        f"{prefix}_weighted_f1_mean": result.get("cv_weighted_f1_mean", float("nan")),
        f"{prefix}_weighted_f1_std": result.get("cv_weighted_f1_std", float("nan")),
    }


def _candidate_report_fields(candidate: dict) -> dict:
    preprocessing = {
        "imputer": candidate.get("imputer"),
        "scaler": candidate.get("scaler"),
        "reducer": candidate.get("reducer"),
    }
    knn_params = {
        "n_neighbors": int(candidate.get("n_neighbors", 5)),
        "weights": candidate.get("weights"),
        "metric": candidate.get("metric"),
        "algorithm": candidate.get("algorithm"),
        "leaf_size": int(candidate.get("leaf_size", 30)),
        "p": int(candidate.get("p", 2)),
    }
    return {
        "feature_family": candidate.get("feature_set"),
        "preprocessing": json.dumps(preprocessing, sort_keys=True),
        "knn_params": json.dumps(knn_params, sort_keys=True),
    }


def _evaluate_candidate_for_search(
    candidate: dict,
    X: np.ndarray,
    y: np.ndarray,
    random_scenario: dict,
    domain_scenario: dict,
    class_names: list[str],
    seed: int,
    n_jobs: int,
) -> dict:
    random_result = _evaluate_candidate(candidate, X, y, random_scenario["splits"], class_names, seed=seed, n_jobs=n_jobs)
    domain_result = (
        random_result
        if domain_scenario["name"] == random_scenario["name"]
        else _evaluate_candidate(candidate, X, y, domain_scenario["splits"], class_names, seed=seed, n_jobs=n_jobs)
    )
    status = "ok" if random_result["status"] == "ok" and domain_result["status"] == "ok" else "error"
    error_parts = [result.get("error", "") for result in [random_result, domain_result] if result.get("status") != "ok"]
    return {
        **candidate,
        "status": status,
        "error": " | ".join(error_parts),
        "cv_accuracy_mean": domain_result["cv_accuracy_mean"],
        "cv_accuracy_std": domain_result["cv_accuracy_std"],
        "cv_macro_precision_mean": domain_result["cv_macro_precision_mean"],
        "cv_macro_precision_std": domain_result["cv_macro_precision_std"],
        "cv_macro_f1_mean": domain_result["cv_macro_f1_mean"],
        "cv_macro_f1_std": domain_result["cv_macro_f1_std"],
        "cv_weighted_f1_mean": domain_result["cv_weighted_f1_mean"],
        "cv_weighted_f1_std": domain_result["cv_weighted_f1_std"],
        "random_cv_accuracy_mean": random_result["cv_accuracy_mean"],
        "random_cv_accuracy_std": random_result["cv_accuracy_std"],
        "random_cv_macro_precision_mean": random_result["cv_macro_precision_mean"],
        "random_cv_macro_precision_std": random_result["cv_macro_precision_std"],
        "random_cv_macro_f1_mean": random_result["cv_macro_f1_mean"],
        "random_cv_macro_f1_std": random_result["cv_macro_f1_std"],
        "random_cv_weighted_f1_mean": random_result["cv_weighted_f1_mean"],
        "random_cv_weighted_f1_std": random_result["cv_weighted_f1_std"],
        "stratified_cv_accuracy_mean": random_result["cv_accuracy_mean"],
        "stratified_cv_accuracy_std": random_result["cv_accuracy_std"],
        "stratified_cv_macro_precision_mean": random_result["cv_macro_precision_mean"],
        "stratified_cv_macro_precision_std": random_result["cv_macro_precision_std"],
        "stratified_cv_macro_f1_mean": random_result["cv_macro_f1_mean"],
        "stratified_cv_macro_f1_std": random_result["cv_macro_f1_std"],
        "stratified_cv_weighted_f1_mean": random_result["cv_weighted_f1_mean"],
        "stratified_cv_weighted_f1_std": random_result["cv_weighted_f1_std"],
        "domain_cv_accuracy_mean": domain_result["cv_accuracy_mean"],
        "domain_cv_macro_precision_mean": domain_result["cv_macro_precision_mean"],
        "domain_cv_macro_f1_mean": domain_result["cv_macro_f1_mean"],
        "domain_cv_weighted_f1_mean": domain_result["cv_weighted_f1_mean"],
        "domain_cv_accuracy_std": domain_result["cv_accuracy_std"],
        "domain_cv_macro_precision_std": domain_result["cv_macro_precision_std"],
        "domain_cv_macro_f1_std": domain_result["cv_macro_f1_std"],
        "domain_cv_weighted_f1_std": domain_result["cv_weighted_f1_std"],
        "worst_domain_accuracy": domain_result["worst_cv_accuracy"],
        "worst_domain_macro_precision": domain_result["worst_cv_macro_precision"],
        "worst_domain_macro_f1": domain_result["worst_cv_macro_f1"],
        "worst_domain_weighted_f1": domain_result["worst_cv_weighted_f1"],
        "random_cv_scenario": random_scenario["name"],
        "domain_cv_scenario": domain_scenario["name"],
        **_candidate_report_fields(candidate),
    }


def _best_candidate_metrics(best_candidate: dict, X: np.ndarray, y: np.ndarray, splits, class_names: list[str], seed: int, n_jobs: int) -> dict:
    pipeline = build_model_pipeline(
        {"imputer": best_candidate["imputer"], "scaler": best_candidate["scaler"], "reducer": best_candidate["reducer"]},
        best_candidate,
        seed=seed,
        fit_rows=min(len(train_idx) for train_idx, _ in splits),
    )
    y_pred = cross_val_predict(pipeline, X, y, cv=splits, n_jobs=n_jobs, method="predict")
    metrics = compute_metrics(y.tolist(), y_pred.tolist(), class_names)
    return {
        "candidate": best_candidate,
        "cv_metrics": metrics,
    }


def _ranking_columns(cv_focused: bool) -> tuple[list[str], list[bool]]:
    if cv_focused:
        return (
            [
                "random_cv_weighted_f1_mean",
                "random_cv_accuracy_mean",
                "random_cv_macro_f1_mean",
                "random_cv_macro_precision_mean",
                "random_cv_weighted_f1_std",
            ],
            [False, False, False, False, True],
        )
    return (
        [
            "domain_cv_macro_f1_mean",
            "domain_cv_accuracy_mean",
            "random_cv_macro_f1_mean",
            "domain_cv_macro_f1_std",
        ],
        [False, False, False, True],
    )


def _stage_feature_sets(stage_cfg: dict, prior_results: pd.DataFrame | None, cv_focused: bool = False) -> list[str]:
    if stage_cfg.get("feature_sets_from_previous_top") and prior_results is not None and not prior_results.empty:
        top_n = int(stage_cfg["feature_sets_from_previous_top"])
        sort_columns, ascending = _ranking_columns(cv_focused)
        available = [column for column in sort_columns if column in prior_results.columns]
        if len(available) < len(sort_columns):
            sort_columns = ["cv_macro_f1_mean", "cv_accuracy_mean"]
            ascending = [False, False]
        return (
            prior_results.sort_values(sort_columns, ascending=ascending[: len(sort_columns)], na_position="last")["feature_set"]
            .drop_duplicates()
            .head(top_n)
            .tolist()
        )
    return list(stage_cfg.get("feature_sets", []))


def _apply_quick_mode(search_cfg: dict) -> dict:
    updated = deepcopy(search_cfg)
    updated["limit_per_class"] = int(updated.get("limit_per_class", 1000))
    preferred_feature_sets = [
        "handcrafted_stats",
        "waveform_spectral_stats",
        "notebook_lowfreq_band_features",
        "relative_lowfreq_shape_features",
        "temporal_lowfreq_shape_features",
    ]
    for stage in updated.get("stages", []):
        if "feature_sets" in stage:
            stage["feature_sets"] = [name for name in preferred_feature_sets if name in stage["feature_sets"]] or list(stage["feature_sets"])[:3]
        stage["n_neighbors"] = list(stage.get("n_neighbors", [5]))[:4]
        stage["metrics"] = list(stage.get("metrics", ["euclidean"]))[:2]
        stage["reducers"] = list(stage.get("reducers", [{"type": "none"}]))[:3]
        stage["scalers"] = list(stage.get("scalers", ["standard"]))[:3]
        stage["max_candidates"] = min(int(stage.get("max_candidates", 30)), 12)
    return updated


def _configured_stages(search_cfg: dict, stage: str) -> list[dict]:
    stages = list(search_cfg.get("stages", []))
    if stage == "1":
        return stages[:1]
    if stage == "2":
        return stages[:2]
    return stages


def _scenario_metadata(scenario: dict) -> dict:
    return {
        "name": scenario["name"],
        "strategy": scenario["strategy"],
        "group_column": scenario["group_column"],
        "n_splits": len(scenario["splits"]),
        "is_domain_aware": bool(scenario["is_domain_aware"]),
    }


def _write_feature_family_comparison(valid_results: pd.DataFrame, output_dir: Path, cv_focused: bool = False) -> pd.DataFrame:
    rows = []
    sort_columns, ascending = _ranking_columns(cv_focused)
    for feature_set, group in valid_results.groupby("feature_set"):
        best = group.sort_values(
            sort_columns,
            ascending=ascending,
            na_position="last",
        ).iloc[0]
        rows.append(
            {
                "feature_set": feature_set,
                "best_random_cv_weighted_f1_mean": best.get("random_cv_weighted_f1_mean", np.nan),
                "best_random_cv_accuracy_mean": best.get("random_cv_accuracy_mean", np.nan),
                "best_random_cv_macro_precision_mean": best.get("random_cv_macro_precision_mean", np.nan),
                "best_random_cv_macro_f1_mean": best.get("random_cv_macro_f1_mean", np.nan),
                "best_domain_cv_accuracy_mean": best.get("domain_cv_accuracy_mean", best["cv_accuracy_mean"]),
                "best_domain_cv_macro_f1_mean": best.get("domain_cv_macro_f1_mean", best["cv_macro_f1_mean"]),
                "best_worst_domain_accuracy": best.get("worst_domain_accuracy", np.nan),
                "best_worst_domain_macro_f1": best.get("worst_domain_macro_f1", np.nan),
                "candidate_count": int(len(group)),
                "best_scaler": best["scaler"],
                "best_reducer": best["reducer"],
                "best_metric": best["metric"],
                "best_n_neighbors": int(best["n_neighbors"]),
            }
        )
    if cv_focused:
        frame = pd.DataFrame(rows).sort_values(
            [
                "best_random_cv_weighted_f1_mean",
                "best_random_cv_accuracy_mean",
                "best_random_cv_macro_f1_mean",
            ],
            ascending=[False, False, False],
        )
    else:
        frame = pd.DataFrame(rows).sort_values(
            ["best_domain_cv_macro_f1_mean", "best_domain_cv_accuracy_mean", "best_random_cv_macro_f1_mean"],
            ascending=[False, False, False],
        )
    frame.to_csv(output_dir / "feature_family_comparison.csv", index=False)
    return frame


def _write_domain_cv_results(
    valid_results: pd.DataFrame,
    scenarios: list[dict],
    feature_cache: dict[str, tuple[np.ndarray, pd.DataFrame, list[str], Path | None]],
    y: np.ndarray,
    class_names: list[str],
    seed: int,
    n_jobs: int,
    output_dir: Path,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    top_candidates = valid_results.head(top_n).to_dict("records")
    for candidate_rank, candidate in enumerate(top_candidates, start=1):
        X_train, _, _, _ = feature_cache[candidate["feature_set"]]
        for scenario in scenarios:
            result = _evaluate_candidate(
                candidate,
                X_train,
                y,
                scenario["splits"],
                class_names,
                seed=seed,
                n_jobs=n_jobs,
            )
            rows.append(
                {
                    "candidate_rank": candidate_rank,
                    "scenario": scenario["name"],
                    "strategy": scenario["strategy"],
                    "group_column": scenario["group_column"],
                    "n_splits": len(scenario["splits"]),
                    "is_domain_aware": bool(scenario["is_domain_aware"]),
                    "feature_set": candidate["feature_set"],
                    "scaler": candidate["scaler"],
                    "reducer": candidate["reducer"],
                    "n_neighbors": int(candidate["n_neighbors"]),
                    "weights": candidate["weights"],
                    "metric": candidate["metric"],
                    "algorithm": candidate["algorithm"],
                    "cv_accuracy_mean": result["cv_accuracy_mean"],
                    "cv_accuracy_std": result["cv_accuracy_std"],
                    "cv_macro_precision_mean": result["cv_macro_precision_mean"],
                    "cv_macro_precision_std": result["cv_macro_precision_std"],
                    "cv_macro_f1_mean": result["cv_macro_f1_mean"],
                    "cv_macro_f1_std": result["cv_macro_f1_std"],
                    "cv_weighted_f1_mean": result["cv_weighted_f1_mean"],
                    "cv_weighted_f1_std": result["cv_weighted_f1_std"],
                    "worst_fold_accuracy": result["worst_cv_accuracy"],
                    "worst_fold_macro_precision": result["worst_cv_macro_precision"],
                    "worst_fold_macro_f1": result["worst_cv_macro_f1"],
                    "worst_fold_weighted_f1": result["worst_cv_weighted_f1"],
                    "status": result["status"],
                    "error": result.get("error", ""),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "domain_cv_results.csv", index=False)
    return frame


def _write_notebook_feature_ablation(
    train_frame: pd.DataFrame,
    manifest_path: Path,
    train_split: str,
    audio_cfg: dict,
    cache_cfg: dict,
    feature_cache: dict[str, tuple[np.ndarray, pd.DataFrame, list[str], Path | None]],
    y: np.ndarray,
    class_names: list[str],
    random_scenario: dict,
    domain_scenario: dict,
    seed: int,
    n_jobs: int,
    output_dir: Path,
    ablation_cfg: dict,
    cv_focused: bool = False,
) -> pd.DataFrame:
    feature_sets = list(
        ablation_cfg.get(
            "feature_sets",
            [
                "waveform_spectral_stats",
                "handcrafted_stats",
                "notebook_lowfreq_band_features",
                "relative_lowfreq_shape_features",
                "temporal_lowfreq_shape_features",
                "lowfreq_relative_temporal",
                "lowfreq_all",
                "lowfreq_all_plus_mfcc",
                "lowfreq_all_plus_logmel",
                "lowfreq_all_plus_waveform_spectral",
            ],
        )
    )
    candidate_template = {
        "imputer": ablation_cfg.get("imputer", "median"),
        "scaler": ablation_cfg.get("scaler", "robust_l2"),
        "reducer": deepcopy(ablation_cfg.get("reducer", {"type": "none"})),
        "n_neighbors": int(ablation_cfg.get("n_neighbors", 1)),
        "weights": ablation_cfg.get("weights", "uniform"),
        "metric": ablation_cfg.get("metric", "cosine"),
        "algorithm": ablation_cfg.get("algorithm", "brute"),
        "leaf_size": int(ablation_cfg.get("leaf_size", 30)),
        "p": int(ablation_cfg.get("p", 2)),
    }
    rows = []
    for feature_set in feature_sets:
        candidate = {**deepcopy(candidate_template), "feature_set": feature_set}
        try:
            if feature_set not in feature_cache:
                feature_cache[feature_set] = build_feature_matrix(
                    train_frame,
                    manifest_path=manifest_path,
                    split_name=train_split,
                    audio_cfg=audio_cfg,
                    feature_name=feature_set,
                    cache_cfg=cache_cfg,
                    show_progress=True,
                    desc=f"{feature_set} ablation features",
                )
            X_train, _, _, cache_path = feature_cache[feature_set]
            result = _evaluate_candidate_for_search(
                candidate,
                X_train,
                y,
                random_scenario=random_scenario,
                domain_scenario=domain_scenario,
                class_names=class_names,
                seed=seed,
                n_jobs=n_jobs,
            )
            rows.append(
                {
                    **result,
                    "stage": "notebook_feature_ablation",
                    "feature_dimension": int(X_train.shape[1]),
                    "feature_count": int(X_train.shape[1]),
                    "cache_path": str(cache_path) if cache_path else None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **candidate,
                    "stage": "notebook_feature_ablation",
                    "feature_dimension": np.nan,
                    "feature_count": np.nan,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "random_cv_accuracy_mean": np.nan,
                    "random_cv_accuracy_std": np.nan,
                    "random_cv_macro_precision_mean": np.nan,
                    "random_cv_macro_precision_std": np.nan,
                    "random_cv_macro_f1_mean": np.nan,
                    "random_cv_macro_f1_std": np.nan,
                    "random_cv_weighted_f1_mean": np.nan,
                    "random_cv_weighted_f1_std": np.nan,
                    "stratified_cv_accuracy_mean": np.nan,
                    "stratified_cv_accuracy_std": np.nan,
                    "stratified_cv_macro_precision_mean": np.nan,
                    "stratified_cv_macro_precision_std": np.nan,
                    "stratified_cv_macro_f1_mean": np.nan,
                    "stratified_cv_macro_f1_std": np.nan,
                    "stratified_cv_weighted_f1_mean": np.nan,
                    "stratified_cv_weighted_f1_std": np.nan,
                    "domain_cv_accuracy_mean": np.nan,
                    "domain_cv_macro_precision_mean": np.nan,
                    "domain_cv_macro_f1_mean": np.nan,
                    "domain_cv_weighted_f1_mean": np.nan,
                    "domain_cv_accuracy_std": np.nan,
                    "domain_cv_macro_precision_std": np.nan,
                    "domain_cv_macro_f1_std": np.nan,
                    "domain_cv_weighted_f1_std": np.nan,
                    "worst_domain_accuracy": np.nan,
                    "worst_domain_macro_precision": np.nan,
                    "worst_domain_macro_f1": np.nan,
                    "worst_domain_weighted_f1": np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_columns, ascending = _ranking_columns(cv_focused)
        frame = frame.sort_values(sort_columns, ascending=ascending, na_position="last")
    frame.to_csv(output_dir / "notebook_feature_ablation.csv", index=False)
    lines = [
        "# Notebook Feature Ablation",
        "",
        "This compares Projet.ipynb-inspired low-frequency classical features using train-only CV.",
        "The official held-out validation split is not used for this ranking.",
        f"Ranking mode: `{'StratifiedKFold CV first' if cv_focused else 'domain-aware CV first'}`.",
        "",
        "| feature_set | dim | random_acc | random_weighted_f1 | random_macro_f1 | domain_acc | domain_macro_f1 | worst_domain_acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        if row.get("status") != "ok":
            continue
        lines.append(
            f"| `{row['feature_set']}` | {int(row['feature_dimension'])} | "
            f"{float(row['random_cv_accuracy_mean']):.4f} | {float(row.get('random_cv_weighted_f1_mean', float('nan'))):.4f} | "
            f"{float(row['random_cv_macro_f1_mean']):.4f} | "
            f"{float(row['domain_cv_accuracy_mean']):.4f} | {float(row['domain_cv_macro_f1_mean']):.4f} | "
            f"{float(row['worst_domain_accuracy']):.4f} |"
        )
    (output_dir / "notebook_feature_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def run_knn_search(
    config: dict,
    manifest_path: Path,
    output_dir: Path,
    quick: bool = False,
    max_candidates: int | None = None,
    n_jobs: int = 1,
    cache_features_override: bool = False,
    stage: str = "full",
    resume: bool = False,
    cv_focused: bool = False,
) -> Path:
    seed = int(config.get("seed", 42))
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume and (output_dir / "best_knn_config.yaml").exists() and (output_dir / "search_results.csv").exists():
        return output_dir
    search_cfg = deepcopy(config.get("search", {}))
    if quick:
        search_cfg = _apply_quick_mode(search_cfg)
    if cache_features_override:
        config.setdefault("feature_cache", {})
        config["feature_cache"]["enabled"] = True

    manifest = load_manifest_and_validate(manifest_path, config)
    write_label_mapping_audit(manifest, config, Path("outputs/reports/quality"))
    report = build_split_integrity_report(
        manifest,
        train_split=config.get("train_split", "train"),
        test_split=config.get("test_split", config.get("val_split", "validation")),
    )
    write_split_integrity_report(
        report,
        Path("outputs/reports/quality/split_integrity.json"),
        Path("outputs/reports/quality/split_integrity.md"),
    )
    assert_split_integrity(report)

    train_split = config.get("train_split", "train")
    train_frame = valid_manifest_rows(manifest, splits=[train_split])
    train_frame = apply_label_aggregation(train_frame, config)
    limit_per_class = search_cfg.get("limit_per_class")
    task_mode = str(search_cfg.get("task_mode") or config.get("task_mode") or ("3class_notebook_cv" if config.get("label_aggregation", {}).get("enabled") else "7class_strict_cv"))
    dataset_mode = str(search_cfg.get("dataset_mode") or ("full_train_cv" if limit_per_class is None else "balanced_train_cv"))
    if limit_per_class is not None:
        train_frame = (
            train_frame.sort_values(["label", "dataset", "filename", "source_row"])
            .groupby("label", group_keys=False)
            .head(int(limit_per_class))
            .reset_index(drop=True)
        )
    class_names = list(config["classes"])
    y = encode_labels(train_frame["label"].astype(str).tolist(), class_names)
    scenarios = _cv_scenarios(train_frame, y, search_cfg.get("cv", {}), seed=seed)
    primary = _primary_scenario(scenarios, search_cfg.get("cv", {}))
    random_scenario = _scenario_for_name(scenarios, "stratified_kfold") or scenarios[0]
    domain_scenario = _domain_scenario(scenarios, primary)
    ranking_scenario = random_scenario if cv_focused else domain_scenario
    split_strategy = _scenario_metadata(primary)
    split_strategy["all_scenarios"] = [_scenario_metadata(scenario) for scenario in scenarios]
    split_strategy["random_scenario"] = _scenario_metadata(random_scenario)
    split_strategy["domain_ranking_scenario"] = _scenario_metadata(domain_scenario)
    split_strategy["final_ranking_scenario"] = _scenario_metadata(ranking_scenario)
    split_strategy["ranking_mode"] = "cv_focused_stratified_first" if cv_focused else "domain_aware_first"
    split_strategy["task_mode"] = task_mode
    split_strategy["dataset_mode"] = dataset_mode
    split_strategy["note"] = "Search uses the official training split only. The official validation split is excluded from CV and hyperparameter selection."

    cache_cfg = feature_cache_config(config)
    all_stage_results = []
    prior_results = None
    feature_cache: dict[str, tuple[np.ndarray, pd.DataFrame, list[str], Path | None]] = {}
    evaluated_count = 0

    for stage_index, stage_cfg in enumerate(_configured_stages(search_cfg, stage), start=1):
        stage_name = stage_cfg.get("name", f"stage_{stage_index}")
        feature_sets = _stage_feature_sets(stage_cfg, prior_results, cv_focused=cv_focused)
        candidates = _expand_stage(stage_cfg, feature_sets, seed=seed + stage_index, max_candidates=max_candidates)
        stage_rows = []
        for candidate in candidates:
            if max_candidates is not None and evaluated_count >= max_candidates:
                break
            feature_set = candidate["feature_set"]
            if feature_set not in feature_cache:
                feature_cache[feature_set] = build_feature_matrix(
                    train_frame,
                    manifest_path=manifest_path,
                    split_name=train_split,
                    audio_cfg=config["audio"],
                    feature_name=feature_set,
                    cache_cfg=cache_cfg,
                    show_progress=True,
                    desc=f"{feature_set} search features",
                )
            X_train, metadata, feature_names, cache_path = feature_cache[feature_set]
            result = _evaluate_candidate_for_search(
                candidate,
                X_train,
                y,
                random_scenario=random_scenario,
                domain_scenario=domain_scenario,
                class_names=class_names,
                seed=seed,
                n_jobs=n_jobs,
            )
            result.update(
                {
                    "stage": stage_name,
                    "cv_scenario": domain_scenario["name"],
                    "cv_group_column": domain_scenario["group_column"],
                    "cv_is_domain_aware": bool(domain_scenario["is_domain_aware"]),
                    "task_mode": task_mode,
                    "dataset_mode": dataset_mode,
                    "feature_dimension": int(X_train.shape[1]),
                    "feature_count": int(X_train.shape[1]),
                    "cache_path": str(cache_path) if cache_path else None,
                }
            )
            stage_rows.append(result)
            evaluated_count += 1
        stage_frame = pd.DataFrame(stage_rows)
        all_stage_results.append(stage_frame)
        valid_stage_results = stage_frame[stage_frame["status"] == "ok"].copy()
        prior_results = valid_stage_results
        if max_candidates is not None and evaluated_count >= max_candidates:
            break

    if not all_stage_results:
        raise ValueError("No search stages were configured.")

    results = pd.concat(all_stage_results, ignore_index=True)
    valid_results = results[results["status"] == "ok"].copy()
    if valid_results.empty:
        results.to_csv(output_dir / "search_results.csv", index=False)
        raise ValueError("All KNN search candidates failed. See search_results.csv for error messages.")

    ablation = _write_notebook_feature_ablation(
        train_frame=train_frame,
        manifest_path=manifest_path,
        train_split=train_split,
        audio_cfg=config["audio"],
        cache_cfg=cache_cfg,
        feature_cache=feature_cache,
        y=y,
        class_names=class_names,
        random_scenario=random_scenario,
        domain_scenario=domain_scenario,
        seed=seed,
        n_jobs=n_jobs,
        output_dir=output_dir,
        ablation_cfg=search_cfg.get("notebook_ablation", {}),
        cv_focused=cv_focused,
    )
    if not ablation.empty:
        results = pd.concat([results, ablation], ignore_index=True)
        valid_results = results[results["status"] == "ok"].copy()
    results["task_mode"] = results.get("task_mode", task_mode)
    results["dataset_mode"] = results.get("dataset_mode", dataset_mode)
    if "feature_dimension" not in results.columns and "feature_count" in results.columns:
        results["feature_dimension"] = results["feature_count"]
    valid_results["task_mode"] = valid_results.get("task_mode", task_mode)
    valid_results["dataset_mode"] = valid_results.get("dataset_mode", dataset_mode)
    if "feature_dimension" not in valid_results.columns and "feature_count" in valid_results.columns:
        valid_results["feature_dimension"] = valid_results["feature_count"]

    ranking_columns, ranking_ascending = _ranking_columns(cv_focused)
    valid_results = valid_results.sort_values(ranking_columns, ascending=ranking_ascending, na_position="last").reset_index(drop=True)
    valid_results.to_csv(output_dir / "search_results.csv", index=False)
    if cv_focused:
        valid_results.to_csv(output_dir / "cv_focused_search_results.csv", index=False)
    _write_feature_family_comparison(valid_results, output_dir, cv_focused=cv_focused)
    domain_cv = _write_domain_cv_results(
        valid_results,
        scenarios,
        feature_cache,
        y,
        class_names,
        seed=seed,
        n_jobs=n_jobs,
        output_dir=output_dir,
        top_n=int(search_cfg.get("domain_cv_top_n", 5)),
    )
    best_candidate = valid_results.iloc[0].to_dict()
    best_feature_set = best_candidate["feature_set"]
    X_best, _, _, _ = feature_cache[best_feature_set]
    best_cv = _best_candidate_metrics(best_candidate, X_best, y, ranking_scenario["splits"], class_names, seed=seed, n_jobs=n_jobs)

    best_config = build_submission_config_from_candidate(
        config,
        {
            "feature_set": best_candidate["feature_set"],
            "imputer": best_candidate["imputer"],
            "scaler": best_candidate["scaler"],
            "reducer": best_candidate["reducer"],
            "n_neighbors": int(best_candidate["n_neighbors"]),
            "weights": best_candidate["weights"],
            "metric": best_candidate["metric"],
            "algorithm": best_candidate["algorithm"],
            "leaf_size": int(best_candidate["leaf_size"]),
            "p": int(best_candidate.get("p", 2)),
        },
        cv_summary={
            "best_cv_accuracy": best_candidate["random_cv_accuracy_mean"] if cv_focused else best_candidate["domain_cv_accuracy_mean"],
            "best_cv_macro_f1": best_candidate["random_cv_macro_f1_mean"] if cv_focused else best_candidate["domain_cv_macro_f1_mean"],
            "best_cv_weighted_f1": best_candidate.get("random_cv_weighted_f1_mean") if cv_focused else best_candidate.get("domain_cv_weighted_f1_mean"),
            "task_mode": task_mode,
            "dataset_mode": dataset_mode,
            "split_strategy": split_strategy,
        },
    )
    save_config(best_config, output_dir / "best_knn_config.yaml")
    save_config(config, output_dir / "search_config_used.yaml")

    best_metrics = {
        "best_candidate": best_candidate,
        "cv_metrics": best_cv["cv_metrics"],
    }
    with (output_dir / "best_cv_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2)
    with (output_dir / "split_strategy.json").open("w", encoding="utf-8") as handle:
        json.dump(split_strategy, handle, indent=2)

    title = "CV-Focused KNN Search Summary" if cv_focused else "KNN Search Summary"
    lines = [
        f"# {title}",
        "",
        f"- evaluated candidates: `{len(valid_results)}`",
        f"- best feature_set: `{best_candidate['feature_set']}`",
        f"- task mode: `{task_mode}`",
        f"- dataset mode: `{dataset_mode}`",
        f"- ranking mode: `{split_strategy['ranking_mode']}`",
        f"- best random CV weighted-F1: `{best_candidate.get('random_cv_weighted_f1_mean', float('nan')):.4f}`",
        f"- best random CV weighted-F1 std: `{best_candidate.get('random_cv_weighted_f1_std', float('nan')):.4f}`",
        f"- best random CV accuracy: `{best_candidate.get('random_cv_accuracy_mean', float('nan')):.4f}`",
        f"- best random CV accuracy std: `{best_candidate.get('random_cv_accuracy_std', float('nan')):.4f}`",
        f"- best random CV macro precision: `{best_candidate.get('random_cv_macro_precision_mean', float('nan')):.4f}`",
        f"- best random CV macro-F1: `{best_candidate.get('random_cv_macro_f1_mean', float('nan')):.4f}`",
        f"- best random CV macro-F1 std: `{best_candidate.get('random_cv_macro_f1_std', float('nan')):.4f}`",
        f"- best domain-aware CV accuracy: `{best_candidate.get('domain_cv_accuracy_mean', best_candidate['cv_accuracy_mean']):.4f}` ± `{best_candidate.get('domain_cv_accuracy_std', best_candidate['cv_accuracy_std']):.4f}`",
        f"- best domain-aware CV macro-F1: `{best_candidate.get('domain_cv_macro_f1_mean', best_candidate['cv_macro_f1_mean']):.4f}` ± `{best_candidate.get('domain_cv_macro_f1_std', best_candidate['cv_macro_f1_std']):.4f}`",
        f"- worst-domain accuracy: `{best_candidate.get('worst_domain_accuracy', float('nan')):.4f}`",
        f"- worst-domain macro-F1: `{best_candidate.get('worst_domain_macro_f1', float('nan')):.4f}`",
        f"- primary CV scenario: `{split_strategy['name']}`",
        f"- final ranking scenario: `{split_strategy['final_ranking_scenario']['name']}`",
        f"- split strategy: `{split_strategy['strategy']}`",
        f"- group column: `{split_strategy['group_column']}`",
        f"- domain-aware primary: `{split_strategy['is_domain_aware']}`",
        "",
        "Top 10 candidates:",
        "",
    ]
    for _, row in valid_results.head(10).iterrows():
        lines.append(
            f"- `{row['feature_set']}` | scaler=`{row['scaler']}` | reducer=`{row['reducer']}` | "
            f"k=`{int(row['n_neighbors'])}` | weights=`{row['weights']}` | metric=`{row['metric']}` | "
            f"random_acc=`{row.get('random_cv_accuracy_mean', float('nan')):.4f}` | "
            f"random_weighted_f1=`{row.get('random_cv_weighted_f1_mean', float('nan')):.4f}` | "
            f"random_macro_f1=`{row.get('random_cv_macro_f1_mean', float('nan')):.4f}` | "
            f"domain_acc=`{row.get('domain_cv_accuracy_mean', row['cv_accuracy_mean']):.4f}` | "
            f"domain_macro_f1=`{row.get('domain_cv_macro_f1_mean', row['cv_macro_f1_mean']):.4f}`"
        )
    (output_dir / "search_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if cv_focused:
        (output_dir / "cv_focused_search_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = run_knn_search(
        config,
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        quick=bool(args.quick),
        max_candidates=args.max_candidates,
        n_jobs=int(args.n_jobs),
        cache_features_override=bool(args.cache_features),
        stage=str(args.stage),
        resume=bool(args.resume),
        cv_focused=bool(args.cv_focused),
    )
    print(f"KNN search outputs written to {output_dir}")


if __name__ == "__main__":
    main()
