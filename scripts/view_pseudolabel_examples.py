from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None
    CUPY_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX_PATH = ROOT / "data" / "raw" / "Hackathon_Data_Release_1_SHARE.xlsx"
PSEUDO_PATH = ROOT / "outputs" / "pseudo_labels" / "drug_target_pseudo_labels.csv"
OUT_DIR = ROOT / "outputs" / "pseudo_labels"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


REQUIRED_PSEUDO_COLS = {
    "encounter_id",
    "drug_target_pseudo",
    "drug_target_confidence",
}
REQUIRED_FOURH_COLS = {
    "encounter_id",
    "narrative_notes_structured_brief_hpi",
}
REQUIRED_DISPO_COLS = {
    "encounter_id",
    "encounter_disposition_label",
}
REQUIRED_TRIAGE_COLS = {
    "encounter_id",
    "triage_heart_rate",
    "triage_respiratory_rate",
    "triage_snapshot.systolic_bp",
    "triage_snapshot.diastolic_bp",
    "triage_snapshot.oxygen_saturation",
    "triage_temperature_c",
    "triage_gcs",
    "triage_supplemental_oxygen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample pseudo-labeled encounters by drug category and print "
            "narrative_notes_structured_brief_hpi examples."
        )
    )
    parser.add_argument(
        "--raw-xlsx-path",
        type=Path,
        default=RAW_XLSX_PATH,
        help="Path to Hackathon_Data_Release_1_SHARE.xlsx.",
    )
    parser.add_argument(
        "--pseudo-path",
        type=Path,
        default=PSEUDO_PATH,
        help="Path to pseudo-label CSV (default: outputs/pseudo_labels/drug_target_pseudo_labels.csv).",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=3,
        help="Number of examples to print for each category (default: 3).",
    )
    parser.add_argument(
        "--include-no-festival",
        action="store_true",
        help="Include No_Festival in category output (default: only Drug_* categories).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help=(
            "Optional explicit category list (example: Drug_1 Drug_2 Drug_3). "
            "If omitted, categories are inferred from pseudo labels."
        ),
    )
    parser.add_argument(
        "--sampling",
        choices=["top", "random"],
        default="top",
        help="Sampling strategy within each category (default: top confidence).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed (used when --sampling random).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "pseudolabel_brief_hpi_examples.md",
        help="Path for markdown output report.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=OUT_DIR / "pseudolabel_brief_hpi_examples_metadata.json",
        help="Path for run metadata json.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns in {name}: {missing}")


def load_joined_examples(raw_xlsx_path: Path, pseudo_path: Path) -> pd.DataFrame:
    if not raw_xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {raw_xlsx_path}")
    if not pseudo_path.exists():
        raise FileNotFoundError(f"Pseudo-label file not found: {pseudo_path}")

    pseudo_df = pd.read_csv(pseudo_path)
    validate_columns(pseudo_df, REQUIRED_PSEUDO_COLS, "pseudo labels")

    fourh_df = pd.read_excel(
        raw_xlsx_path,
        sheet_name="Four_Hour_Data",
        usecols=["encounter_id", "narrative_notes_structured_brief_hpi"],
    )
    validate_columns(fourh_df, REQUIRED_FOURH_COLS, "Four_Hour_Data")

    dispo_df = pd.read_excel(
        raw_xlsx_path,
        sheet_name="Disposition",
        usecols=["encounter_id", "encounter_disposition_label"],
    )
    validate_columns(dispo_df, REQUIRED_DISPO_COLS, "Disposition")

    triage_vital_cols = [
        "encounter_id",
        "triage_heart_rate",
        "triage_respiratory_rate",
        "triage_snapshot.systolic_bp",
        "triage_snapshot.diastolic_bp",
        "triage_snapshot.oxygen_saturation",
        "triage_temperature_c",
        "triage_gcs",
        "triage_supplemental_oxygen",
    ]
    triage_df = pd.read_excel(
        raw_xlsx_path,
        sheet_name="Triage_Data",
        usecols=triage_vital_cols,
    )
    validate_columns(triage_df, REQUIRED_TRIAGE_COLS, "Triage_Data")

    merged = (
        pseudo_df[["encounter_id", "drug_target_pseudo", "drug_target_confidence"]]
        .merge(
            fourh_df[["encounter_id", "narrative_notes_structured_brief_hpi"]],
            on="encounter_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            dispo_df[["encounter_id", "encounter_disposition_label"]],
            on="encounter_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            triage_df[triage_vital_cols],
            on="encounter_id",
            how="left",
            validate="one_to_one",
        )
    )

    merged["narrative_notes_structured_brief_hpi"] = (
        merged["narrative_notes_structured_brief_hpi"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return merged


def category_sort_key(label: str) -> tuple[int, Any]:
    m = re.fullmatch(r"Drug_(\d+)", label)
    if m:
        return (0, int(m.group(1)))
    if label == "No_Festival":
        return (1, 0)
    return (2, label)


def choose_categories(df: pd.DataFrame, include_no_festival: bool, categories: list[str] | None) -> list[str]:
    if categories is not None:
        missing = [c for c in categories if c not in set(df["drug_target_pseudo"].unique())]
        if missing:
            raise ValueError(f"Requested category not found in pseudo labels: {missing}")
        return categories

    inferred = sorted(df["drug_target_pseudo"].dropna().unique().tolist(), key=category_sort_key)
    if include_no_festival:
        return inferred
    return [c for c in inferred if str(c).startswith("Drug_")]


def sample_examples(df: pd.DataFrame, category: str, n: int, sampling: str, seed: int) -> pd.DataFrame:
    subset = df[df["drug_target_pseudo"] == category].copy()
    if subset.empty:
        return subset

    n_take = min(n, len(subset))
    if sampling == "random":
        return subset.sample(n=n_take, random_state=seed).sort_values(
            "drug_target_confidence", ascending=False
        )

    return subset.sort_values("drug_target_confidence", ascending=False).head(n_take)


def _fmt_int_like(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return str(int(round(float(value))))


def _fmt_float(value: Any, decimals: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{decimals}f}"


def format_vitals_shorthand(row: dict[str, Any]) -> str:
    hr = _fmt_int_like(row.get("triage_heart_rate"))
    rr = _fmt_int_like(row.get("triage_respiratory_rate"))
    sbp = _fmt_int_like(row.get("triage_snapshot.systolic_bp"))
    dbp = _fmt_int_like(row.get("triage_snapshot.diastolic_bp"))
    spo2 = _fmt_int_like(row.get("triage_snapshot.oxygen_saturation"))
    temp_c = _fmt_float(row.get("triage_temperature_c"), decimals=1)
    gcs = _fmt_int_like(row.get("triage_gcs"))
    supp_o2 = _fmt_int_like(row.get("triage_supplemental_oxygen"))

    return (
        f"HR {hr} | RR {rr} | BP {sbp}/{dbp} | SpO2 {spo2}% | "
        f"Temp {temp_c}C | GCS {gcs} | SuppO2 {supp_o2}"
    )


def format_report(df: pd.DataFrame, categories: list[str], per_category: int, sampling: str, seed: int) -> str:
    lines: list[str] = []
    lines.append("# Pseudo-labeled Brief HPI Examples")
    lines.append("")
    lines.append(f"- Total pseudo-labeled rows: `{len(df)}`")
    lines.append(f"- Categories shown: `{', '.join(categories)}`")
    lines.append(f"- Examples per category: `{per_category}`")
    lines.append(f"- Sampling: `{sampling}`")
    lines.append(f"- Seed: `{seed}`")
    lines.append("")

    for category in categories:
        cat_df = df[df["drug_target_pseudo"] == category].copy()
        lines.append(f"## {category}")
        lines.append("")

        if cat_df.empty:
            lines.append("No rows found for this category.")
            lines.append("")
            continue

        dispo_counts = (
            cat_df["encounter_disposition_label"]
            .fillna("<missing>")
            .value_counts()
            .rename_axis("encounter_disposition_label")
            .reset_index(name="count")
        )
        lines.append("Disposition counts in category:")
        lines.append("")
        lines.append(dispo_counts.to_markdown(index=False))
        lines.append("")

        sampled = sample_examples(cat_df, category=category, n=per_category, sampling=sampling, seed=seed)
        if sampled.empty:
            lines.append("No sampled rows for this category.")
            lines.append("")
            continue

        for i, rec in enumerate(sampled.to_dict(orient="records"), start=1):
            hpi_text = str(rec.get("narrative_notes_structured_brief_hpi", "")).strip() or "<missing>"
            vitals = format_vitals_shorthand(rec)
            lines.append(f"### Example {i}")
            lines.append(f"- encounter_id: `{rec.get('encounter_id')}`")
            lines.append(f"- drug_target_confidence: `{float(rec.get('drug_target_confidence', 0.0)):.4f}`")
            lines.append(
                f"- encounter_disposition_label: `{str(rec.get('encounter_disposition_label')) if pd.notna(rec.get('encounter_disposition_label')) else '<missing>'}`"
            )
            lines.append(f"- vitals_shorthand: `{vitals}`")
            lines.append(f"- narrative_notes_structured_brief_hpi: {hpi_text}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_metadata(args: argparse.Namespace, categories: list[str], df: pd.DataFrame) -> dict[str, Any]:
    gpu_info = {
        "torch_cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cupy_available": bool(CUPY_AVAILABLE),
        "gpu_path_used": False,
        "note": "This script is IO/reporting only; GPU acceleration not required.",
    }

    counts = (
        df.groupby(["drug_target_pseudo", "encounter_disposition_label"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )

    return {
        "raw_xlsx_path": str(args.raw_xlsx_path),
        "pseudo_path": str(args.pseudo_path),
        "output_path": str(args.output),
        "metadata_output_path": str(args.metadata_output),
        "per_category": int(args.per_category),
        "sampling": args.sampling,
        "seed": int(args.seed),
        "categories": categories,
        "n_rows_joined": int(len(df)),
        "gpu": gpu_info,
        "category_disposition_counts": counts.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    joined_df = load_joined_examples(
        raw_xlsx_path=args.raw_xlsx_path,
        pseudo_path=args.pseudo_path,
    )
    categories = choose_categories(
        joined_df,
        include_no_festival=bool(args.include_no_festival),
        categories=args.categories,
    )

    if not categories:
        raise ValueError(
            "No categories selected. Use --include-no-festival or --categories to choose categories."
        )

    report = format_report(
        joined_df,
        categories=categories,
        per_category=int(args.per_category),
        sampling=str(args.sampling),
        seed=int(args.seed),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)

    metadata = build_metadata(args=args, categories=categories, df=joined_df)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
