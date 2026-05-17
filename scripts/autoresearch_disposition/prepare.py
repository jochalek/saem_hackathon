from __future__ import annotations

import ast
import json
import platform
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "data" / "raw").exists():
            return parent
    raise RuntimeError("Could not find repo root containing data/raw")


ROOT = _find_repo_root()
RAW_XLSX_PATH = ROOT / "data" / "raw" / "Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = ROOT / "outputs" / "disposition_autoresearch"
CACHE_DIR = OUT_DIR / "cache"
RUNS_DIR = OUT_DIR / "runs"

TRIAGE_SHEET = "Triage_Data"
FOUR_HOUR_SHEET = "Four_Hour_Data"
DISPOSITION_SHEET = "Disposition"

ID_COL = "encounter_id"
TARGET_COL = "encounter_disposition_label"
ARRIVAL_DATE_COL = "encounter_arrival_date"

CLASS_NAMES = ["Discharge", "Floor", "ICU"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_NAMES)}

RANDOM_SEED = 42
TRAIN_TIME_BUDGET_SECONDS = 300

# Cross-validation defaults.
# - "stratified" (default): standard class-stratified KFold
# - "group": GroupKFold using CV_GROUP_COLUMN
CV_N_SPLITS = 5
CV_STRATEGY = "stratified"
CV_GROUP_COLUMN: str | None = None

NESTED_COLUMNS = [
    "triage.labs",
    "ed_course.vitals_timeseries",
    "ed_course.labs_timeseries",
    "ed_course.interventions",
]

LEAKY_OR_EXCLUDED_COLUMNS = {
    "narrative_notes_structured_brief_hpi",
    "narrative_notes_structured_hpi",
    "narrative_notes_structured_physical_exam_pertinent_positives",
    "narrative_notes_structured_mdm",
    "narrative_notes_structured_clinical_course",
    "narrative.notes_structured_ed_meds_procedures",
}


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    class_weights: np.ndarray
    encounter_ids_train: np.ndarray
    encounter_ids_val: np.ndarray


@dataclass
class CVMetadata:
    seed: int
    cv_strategy: str
    cv_group_column: str | None
    n_rows_total: int
    n_features: int
    n_splits_requested: int
    n_splits_used: int
    class_counts_total: dict[str, int]
    fold_sizes: dict[str, int]
    fold_class_counts: dict[str, dict[str, int]]
    cache_npz: str
    cache_json: str


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _cache_paths(seed: int, cv_strategy: str, n_splits: int) -> tuple[Path, Path]:
    stem = f"dataset_seed{seed}_{slugify(cv_strategy)}_{int(n_splits)}fold"
    return CACHE_DIR / f"{stem}.npz", CACHE_DIR / f"{stem}.json"


def parse_nested_cell(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    parsed: Any = None
    for candidate in (text, text.replace("null", "None")):
        try:
            parsed = ast.literal_eval(candidate)
            break
        except Exception:
            parsed = None

    if parsed is None:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    out: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            out.append(item)
    return out


def summarize_nested_column(df: pd.DataFrame, col: str, top_k_categories: int = 6) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame({ID_COL: df[ID_COL].astype(str)})

    prefix = slugify(col)
    parsed = df[col].map(parse_nested_cell)

    numeric_keys: set[str] = set()
    categorical_keys: set[str] = set()
    cat_counts: dict[str, dict[str, int]] = {}

    for items in parsed:
        for item in items:
            for key, val in item.items():
                if key in {"encounter_id", "minute"}:
                    continue
                if isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val):
                    numeric_keys.add(str(key))
                else:
                    key_s = str(key)
                    categorical_keys.add(key_s)
                    cat_counts.setdefault(key_s, {})
                    val_s = str(val).strip()
                    if val is not None and val_s:
                        cat_counts[key_s][val_s] = cat_counts[key_s].get(val_s, 0) + 1

    top_categories: dict[str, list[str]] = {}
    for key, counts in cat_counts.items():
        ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top_categories[key] = [k for k, _ in ordered[:top_k_categories]]

    rows: list[dict[str, Any]] = []
    for encounter_id, items in zip(df[ID_COL].astype(str), parsed):
        rec: dict[str, Any] = {ID_COL: encounter_id, f"{prefix}__n_events": float(len(items))}

        minute_values = [
            float(item.get("minute"))
            for item in items
            if isinstance(item.get("minute"), (int, float, np.integer, np.floating))
        ]
        if minute_values:
            mins = np.asarray(minute_values, dtype=float)
            rec[f"{prefix}__minute_min"] = float(np.min(mins))
            rec[f"{prefix}__minute_max"] = float(np.max(mins))
            rec[f"{prefix}__minute_span"] = float(np.max(mins) - np.min(mins))
        else:
            rec[f"{prefix}__minute_min"] = np.nan
            rec[f"{prefix}__minute_max"] = np.nan
            rec[f"{prefix}__minute_span"] = np.nan

        for key in sorted(numeric_keys):
            vals = [
                float(item.get(key))
                for item in items
                if isinstance(item.get(key), (int, float, np.integer, np.floating))
                and not pd.isna(item.get(key))
            ]
            feat_prefix = f"{prefix}__{slugify(key)}"
            if vals:
                arr = np.asarray(vals, dtype=float)
                rec[f"{feat_prefix}__mean"] = float(arr.mean())
                rec[f"{feat_prefix}__min"] = float(arr.min())
                rec[f"{feat_prefix}__max"] = float(arr.max())
                rec[f"{feat_prefix}__last"] = float(arr[-1])
                rec[f"{feat_prefix}__delta"] = float(arr[-1] - arr[0])
            else:
                rec[f"{feat_prefix}__mean"] = np.nan
                rec[f"{feat_prefix}__min"] = np.nan
                rec[f"{feat_prefix}__max"] = np.nan
                rec[f"{feat_prefix}__last"] = np.nan
                rec[f"{feat_prefix}__delta"] = np.nan

        for key in sorted(categorical_keys):
            values = [
                str(item.get(key)).strip()
                for item in items
                if item.get(key) is not None and str(item.get(key)).strip()
            ]
            base = f"{prefix}__{slugify(key)}"
            rec[f"{base}__nunique"] = float(len(set(values)))
            for cat in top_categories.get(key, []):
                rec[f"{base}__count__{slugify(cat)}"] = float(sum(v == cat for v in values))

        rows.append(rec)

    return pd.DataFrame(rows)


def _build_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if ARRIVAL_DATE_COL not in df.columns:
        return out

    dt = pd.to_datetime(df[ARRIVAL_DATE_COL], errors="coerce")
    out["arrival_month"] = dt.dt.month
    out["arrival_day"] = dt.dt.day
    out["arrival_dayofweek"] = dt.dt.dayofweek
    out["arrival_is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
    return out


def _validate_tables(triage_df: pd.DataFrame, fourh_df: pd.DataFrame, dispo_df: pd.DataFrame) -> None:
    required = {
        "Triage_Data": [ID_COL],
        "Four_Hour_Data": [ID_COL],
        "Disposition": [ID_COL, TARGET_COL],
    }

    tables = {
        "Triage_Data": triage_df,
        "Four_Hour_Data": fourh_df,
        "Disposition": dispo_df,
    }

    for table_name, req in required.items():
        missing = [col for col in req if col not in tables[table_name].columns]
        if missing:
            raise ValueError(f"Missing columns in {table_name}: {missing}")

    for table_name, table in tables.items():
        dupes = int(table[ID_COL].duplicated().sum())
        if dupes:
            raise ValueError(f"{table_name} has duplicated encounter_id rows: {dupes}")


def _load_raw_merged_table() -> pd.DataFrame:
    if not RAW_XLSX_PATH.exists():
        raise FileNotFoundError(f"Expected workbook at {RAW_XLSX_PATH}")

    triage_df = pd.read_excel(RAW_XLSX_PATH, sheet_name=TRIAGE_SHEET)
    fourh_df = pd.read_excel(RAW_XLSX_PATH, sheet_name=FOUR_HOUR_SHEET)
    dispo_df = pd.read_excel(RAW_XLSX_PATH, sheet_name=DISPOSITION_SHEET)
    _validate_tables(triage_df, fourh_df, dispo_df)

    merged = triage_df.merge(fourh_df, on=ID_COL, how="inner", validate="one_to_one")
    merged = merged.merge(dispo_df[[ID_COL, TARGET_COL]], on=ID_COL, how="inner", validate="one_to_one")
    return merged


def _build_feature_frame(merged: pd.DataFrame) -> pd.DataFrame:
    nested_blocks: list[pd.DataFrame] = []
    for col in NESTED_COLUMNS:
        if col in merged.columns:
            nested_blocks.append(summarize_nested_column(merged, col))

    if nested_blocks:
        nested_df = nested_blocks[0]
        for block in nested_blocks[1:]:
            nested_df = nested_df.merge(block, on=ID_COL, how="left", validate="one_to_one")
    else:
        nested_df = pd.DataFrame({ID_COL: merged[ID_COL].astype(str)})

    datetime_df = _build_datetime_features(merged)

    drop_cols = set(LEAKY_OR_EXCLUDED_COLUMNS)
    drop_cols.update(NESTED_COLUMNS)
    drop_cols.update({ID_COL, TARGET_COL, ARRIVAL_DATE_COL})
    for col in merged.columns:
        if col.startswith("narrative_notes_structured_") or col.startswith("narrative.notes_structured_"):
            drop_cols.add(col)

    base = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore").copy()

    bool_cols = [c for c in base.columns if str(base[c].dtype) == "bool"]
    for col in bool_cols:
        base[col] = base[col].astype(float)

    numeric_base = base.select_dtypes(include=[np.number]).copy()
    cat_cols = [c for c in base.columns if c not in numeric_base.columns]
    categorical_base = (
        pd.get_dummies(base[cat_cols], dummy_na=True, dtype=float)
        if cat_cols
        else pd.DataFrame(index=base.index)
    )

    nested_no_id = nested_df.drop(columns=[ID_COL], errors="ignore")
    feature_df = pd.concat(
        [numeric_base, categorical_base, datetime_df.reset_index(drop=True), nested_no_id.reset_index(drop=True)],
        axis=1,
    )

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    medians = feature_df.median(numeric_only=True)
    feature_df = feature_df.fillna(medians)
    feature_df = feature_df.fillna(0.0)

    # enforce deterministic column order for stable cache
    feature_df = feature_df.reindex(sorted(feature_df.columns), axis=1)
    return feature_df.astype(np.float32)


def _class_count_dict(y_idx: np.ndarray) -> dict[str, int]:
    counts = np.bincount(y_idx, minlength=len(CLASS_NAMES))
    return {CLASS_NAMES[i]: int(v) for i, v in enumerate(counts)}


def _compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    weights = np.ones(len(CLASS_NAMES), dtype=np.float32)
    present_classes = np.unique(y_train)
    if len(present_classes) == 0:
        return weights

    raw = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train,
    ).astype(np.float32)
    for cls_idx, w in zip(present_classes.tolist(), raw.tolist()):
        weights[int(cls_idx)] = float(w)
    return weights


def _build_fold_assignments(
    merged: pd.DataFrame,
    y_idx: np.ndarray,
    strategy: str,
    n_splits: int,
    seed: int,
    group_col: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_rows = len(merged)
    fold_ids = np.full(n_rows, -1, dtype=np.int64)

    strategy_norm = strategy.strip().lower()
    effective_splits = int(n_splits)

    if strategy_norm == "stratified":
        min_class = int(np.bincount(y_idx, minlength=len(CLASS_NAMES)).min())
        effective_splits = min(max(2, n_splits), max(2, min_class))

        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
        for fold_idx, (_, val_idx) in enumerate(splitter.split(np.zeros(n_rows), y_idx)):
            fold_ids[val_idx] = fold_idx

    elif strategy_norm == "group":
        if group_col is None:
            raise ValueError("group strategy requires a non-null group_col")
        if group_col not in merged.columns:
            raise ValueError(f"group_col '{group_col}' not found in merged table")

        groups = merged[group_col].fillna("__NA__").astype(str).to_numpy()
        unique_groups = np.unique(groups)
        effective_splits = min(max(2, n_splits), len(unique_groups))

        splitter = GroupKFold(n_splits=effective_splits)
        for fold_idx, (_, val_idx) in enumerate(splitter.split(np.zeros(n_rows), y_idx, groups=groups)):
            fold_ids[val_idx] = fold_idx
    else:
        raise ValueError(f"Unsupported CV strategy: {strategy}")

    if (fold_ids < 0).any():
        raise RuntimeError("Some rows were not assigned to any CV fold")

    info = {
        "strategy_requested": strategy,
        "strategy_used": strategy_norm,
        "n_splits_requested": int(n_splits),
        "n_splits_used": int(effective_splits),
    }
    return fold_ids, info


def _build_and_cache_cv_dataset(
    seed: int,
    cv_strategy: str,
    n_splits: int,
    group_col: str | None,
) -> CVMetadata:
    merged = _load_raw_merged_table()

    labels = merged[TARGET_COL].astype(str)
    unknown_labels = sorted(set(labels.unique()) - set(CLASS_NAMES))
    if unknown_labels:
        raise ValueError(f"Unknown target labels found: {unknown_labels}")

    y_idx = labels.map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    encounter_ids = merged[ID_COL].astype(str).to_numpy()

    feature_df = _build_feature_frame(merged)
    fold_ids, split_info = _build_fold_assignments(
        merged=merged,
        y_idx=y_idx,
        strategy=cv_strategy,
        n_splits=n_splits,
        seed=seed,
        group_col=group_col,
    )

    cache_npz, cache_json = _cache_paths(seed=seed, cv_strategy=cv_strategy, n_splits=split_info["n_splits_used"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_npz,
        x=feature_df.to_numpy(dtype=np.float32),
        y=y_idx.astype(np.int64),
        fold_ids=fold_ids.astype(np.int64),
        encounter_ids=encounter_ids.astype(str),
        feature_names=np.asarray(feature_df.columns.astype(str).tolist(), dtype=str),
        class_names=np.asarray(CLASS_NAMES, dtype=str),
    )

    fold_sizes: dict[str, int] = {}
    fold_class_counts: dict[str, dict[str, int]] = {}
    for fold_idx in sorted(np.unique(fold_ids).tolist()):
        mask = fold_ids == fold_idx
        fold_sizes[str(fold_idx)] = int(mask.sum())
        fold_class_counts[str(fold_idx)] = _class_count_dict(y_idx[mask])

    metadata = CVMetadata(
        seed=seed,
        cv_strategy=split_info["strategy_used"],
        cv_group_column=group_col,
        n_rows_total=int(len(y_idx)),
        n_features=int(feature_df.shape[1]),
        n_splits_requested=int(split_info["n_splits_requested"]),
        n_splits_used=int(split_info["n_splits_used"]),
        class_counts_total=_class_count_dict(y_idx),
        fold_sizes=fold_sizes,
        fold_class_counts=fold_class_counts,
        cache_npz=str(cache_npz),
        cache_json=str(cache_json),
    )

    cache_json.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return metadata


def _load_cached_cv_metadata(seed: int, cv_strategy: str, n_splits: int) -> CVMetadata | None:
    _, cache_json = _cache_paths(seed=seed, cv_strategy=cv_strategy, n_splits=n_splits)
    if not cache_json.exists():
        return None

    payload = json.loads(cache_json.read_text(encoding="utf-8"))
    return CVMetadata(**payload)


def load_or_create_cv_dataset_bundles(
    force_rebuild: bool = False,
    seed: int = RANDOM_SEED,
    cv_strategy: str = CV_STRATEGY,
    n_splits: int = CV_N_SPLITS,
    group_col: str | None = CV_GROUP_COLUMN,
) -> list[DatasetBundle]:
    if force_rebuild:
        metadata = _build_and_cache_cv_dataset(
            seed=seed,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            group_col=group_col,
        )
    else:
        metadata = _load_cached_cv_metadata(seed=seed, cv_strategy=cv_strategy, n_splits=n_splits)
        if metadata is None:
            metadata = _build_and_cache_cv_dataset(
                seed=seed,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                group_col=group_col,
            )

    cache_npz = Path(metadata.cache_npz)
    if not cache_npz.exists():
        metadata = _build_and_cache_cv_dataset(
            seed=seed,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            group_col=group_col,
        )
        cache_npz = Path(metadata.cache_npz)

    with np.load(cache_npz, allow_pickle=False) as data:
        x_all = data["x"].astype(np.float32)
        y_all = data["y"].astype(np.int64)
        fold_ids = data["fold_ids"].astype(np.int64)
        encounter_ids = data["encounter_ids"].astype(str)
        feature_names = data["feature_names"].astype(str).tolist()
        class_names = data["class_names"].astype(str).tolist()

    bundles: list[DatasetBundle] = []
    for fold_idx in sorted(np.unique(fold_ids).tolist()):
        val_mask = fold_ids == fold_idx
        train_mask = ~val_mask

        y_train = y_all[train_mask]
        class_weights = _compute_class_weights(y_train)

        bundle = DatasetBundle(
            x_train=x_all[train_mask],
            y_train=y_train,
            x_val=x_all[val_mask],
            y_val=y_all[val_mask],
            feature_names=feature_names,
            class_names=class_names,
            class_weights=class_weights,
            encounter_ids_train=encounter_ids[train_mask],
            encounter_ids_val=encounter_ids[val_mask],
        )
        bundles.append(bundle)

    return bundles


def load_or_create_dataset_bundle(
    force_rebuild: bool = False,
    seed: int = RANDOM_SEED,
) -> DatasetBundle:
    # Backward-compatible helper: return the first CV fold.
    bundles = load_or_create_cv_dataset_bundles(
        force_rebuild=force_rebuild,
        seed=seed,
        cv_strategy=CV_STRATEGY,
        n_splits=CV_N_SPLITS,
        group_col=CV_GROUP_COLUMN,
    )
    return bundles[0]


def compute_classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)

    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = np.argmax(probs, axis=1)

    metrics: dict[str, float] = {}
    metrics["accuracy"] = float((y_pred == y_true).mean())
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    try:
        metrics["log_loss"] = float(log_loss(y_true, probs, labels=np.arange(n_classes)))
    except ValueError:
        metrics["log_loss"] = float("nan")

    one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
    try:
        metrics["macro_auroc"] = float(roc_auc_score(one_hot, probs, average="macro", multi_class="ovr"))
    except ValueError:
        metrics["macro_auroc"] = float("nan")

    icu_idx = class_names.index("ICU") if "ICU" in class_names else n_classes - 1
    floor_idx = class_names.index("Floor") if "Floor" in class_names else 1

    y_true_icu = (y_true == icu_idx).astype(int)
    y_pred_icu = (y_pred == icu_idx).astype(int)
    metrics["icu_recall"] = float(recall_score(y_true_icu, y_pred_icu, zero_division=0))
    metrics["icu_precision"] = float(precision_score(y_true_icu, y_pred_icu, zero_division=0))
    try:
        metrics["icu_pr_auc"] = float(average_precision_score(y_true_icu, probs[:, icu_idx]))
    except ValueError:
        metrics["icu_pr_auc"] = float("nan")

    y_true_floor = (y_true == floor_idx).astype(int)
    y_pred_floor = (y_pred == floor_idx).astype(int)
    metrics["floor_recall"] = float(recall_score(y_true_floor, y_pred_floor, zero_division=0))
    metrics["floor_precision"] = float(precision_score(y_true_floor, y_pred_floor, zero_division=0))

    return metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(get_device()),
        "cuda_available": bool(torch.cuda.is_available()),
        "raw_xlsx": str(RAW_XLSX_PATH),
        "output_dir": str(OUT_DIR),
        "cv_strategy": CV_STRATEGY,
        "cv_splits": CV_N_SPLITS,
    }

    if torch.cuda.is_available():
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
        except Exception:
            pass

    return info


def main() -> None:
    set_seed(RANDOM_SEED)

    metadata = _build_and_cache_cv_dataset(
        seed=RANDOM_SEED,
        cv_strategy=CV_STRATEGY,
        n_splits=CV_N_SPLITS,
        group_col=CV_GROUP_COLUMN,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    prepare_summary = {
        "environment": describe_environment(),
        "cv_strategy": metadata.cv_strategy,
        "cv_group_column": metadata.cv_group_column,
        "n_rows_total": metadata.n_rows_total,
        "n_features": metadata.n_features,
        "n_splits_requested": metadata.n_splits_requested,
        "n_splits_used": metadata.n_splits_used,
        "class_counts_total": metadata.class_counts_total,
        "fold_sizes": metadata.fold_sizes,
        "fold_class_counts": metadata.fold_class_counts,
        "cache_npz": metadata.cache_npz,
        "cache_json": metadata.cache_json,
    }
    (OUT_DIR / "prepare_summary.json").write_text(json.dumps(prepare_summary, indent=2), encoding="utf-8")

    print("Prepared disposition CV dataset cache")
    print(f"- rows:         {metadata.n_rows_total}")
    print(f"- features:     {metadata.n_features}")
    print(f"- cv_strategy:  {metadata.cv_strategy}")
    print(f"- folds:        {metadata.n_splits_used}")
    print(f"- cache:        {metadata.cache_npz}")


if __name__ == "__main__":
    main()
