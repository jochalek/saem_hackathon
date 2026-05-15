from __future__ import annotations

import ast
from pathlib import Path
import re

import numpy as np
import pandas as pd
from docx import Document


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
XLSX_PATH = RAW_DIR / "Hackathon_Data_Release_1_SHARE.xlsx"
CODEBOOK_PATH = RAW_DIR / "Hackathon_Codebook_Release_1_SHARE.docx"
OUT_DIR = ROOT / "outputs" / "eda"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def parse_list_cell(v: object) -> list:
    if pd.isna(v):
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def save_codebook_extract() -> pd.DataFrame:
    doc = Document(CODEBOOK_PATH)
    records: list[dict[str, str]] = []

    for t_idx, table in enumerate(doc.tables, start=1):
        for r_idx, row in enumerate(table.rows, start=1):
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                rec = {f"col_{i+1}": v for i, v in enumerate(cells)}
                rec["table_index"] = str(t_idx)
                rec["row_index"] = str(r_idx)
                records.append(rec)

    codebook_df = pd.DataFrame(records)
    codebook_df.to_csv(OUT_DIR / "codebook_tables_raw.csv", index=False)

    var_rows = []
    for _, row in codebook_df.iterrows():
        first = str(row.get("col_1", "")).strip()
        second = str(row.get("col_2", "")).strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_\.]{1,80}", first):
            var_rows.append({"variable": first, "description": second})

    var_df = pd.DataFrame(var_rows).drop_duplicates()
    var_df.to_csv(OUT_DIR / "codebook_variable_descriptions.csv", index=False)
    return var_df


def summarize_sheet(sheet_name: str, df: pd.DataFrame) -> dict[str, object]:
    safe = slugify(sheet_name)

    n_rows, _ = df.shape
    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda x: x["missing_count"] / max(n_rows, 1))
        .sort_values(["missing_pct", "missing_count"], ascending=False)
    )
    missing.to_csv(OUT_DIR / f"{safe}__missingness.csv")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    if num_cols:
        num_summary = df[num_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
        num_summary.to_csv(OUT_DIR / f"{safe}__numeric_summary.csv")

    cat_summary_records = []
    for col in cat_cols:
        s = df[col]
        top = s.value_counts(dropna=True).head(5)
        cat_summary_records.append(
            {
                "column": col,
                "nunique_non_null": int(s.nunique(dropna=True)),
                "missing_count": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean()),
                "top_values": " | ".join([f"{k}: {v}" for k, v in top.items()]),
            }
        )
    cat_summary = pd.DataFrame(cat_summary_records).sort_values("missing_pct", ascending=False)
    cat_summary.to_csv(OUT_DIR / f"{safe}__categorical_summary.csv", index=False)

    dtypes = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
    dtypes.to_csv(OUT_DIR / f"{safe}__dtypes.csv", index=False)

    id_candidates = [
        c for c in df.columns if re.search(r"(^id$|_id$|encounter|patient)", c.lower())
    ]

    return {
        "sheet": sheet_name,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "numeric_cols": len(num_cols),
        "categorical_cols": len(cat_cols),
        "id_candidates": id_candidates,
        "top_missing_cols": missing.head(10).reset_index().rename(columns={"index": "column"}),
    }


def nested_feature_eda(triage_df: pd.DataFrame, fourh_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    triage_lens = triage_df["triage.labs"].map(parse_list_cell).map(len)
    out["nested_triage_labs_len"] = triage_lens.value_counts().sort_index().rename_axis("len").reset_index(name="count")

    for col in ["ed_course.vitals_timeseries", "ed_course.labs_timeseries", "ed_course.interventions"]:
        lens = fourh_df[col].map(parse_list_cell).map(len)
        out[f"nested_{slugify(col)}_len"] = lens.value_counts().sort_index().rename_axis("len").reset_index(name="count")

    intervention_rows = []
    for items in fourh_df["ed_course.interventions"].map(parse_list_cell):
        for it in items:
            if isinstance(it, dict):
                intervention_rows.append(str(it.get("event_name", "")).strip())
    intervention_counts = (
        pd.Series(intervention_rows, name="event_name").replace("", np.nan).dropna().value_counts().rename_axis("event_name").reset_index(name="count")
    )
    out["intervention_event_counts"] = intervention_counts

    return out


def sex_hpi_mismatch_eda(triage_df: pd.DataFrame, fourh_df: pd.DataFrame) -> pd.DataFrame:
    merged = triage_df[["encounter_id", "triage_sex_gender"]].merge(
        fourh_df[["encounter_id", "narrative_notes_structured_hpi"]], on="encounter_id", how="inner"
    )

    def extract_hpi_gender(hpi: object) -> str:
        t = str(hpi).lower()
        if "non-binary" in t:
            return "Non-binary"
        if re.search(r"\bmale\b", t):
            return "Male"
        if re.search(r"\bfemale\b", t):
            return "Female"
        return "Unknown"

    merged["hpi_gender"] = merged["narrative_notes_structured_hpi"].map(extract_hpi_gender)
    merged["sex_matches_hpi"] = merged["triage_sex_gender"] == merged["hpi_gender"]

    summary = (
        merged.groupby(["triage_sex_gender", "hpi_gender", "sex_matches_hpi"]).size().rename("count").reset_index()
    )
    return summary


def leakage_scan_eda(fourh_df: pd.DataFrame) -> pd.DataFrame:
    col = "narrative_notes_structured_clinical_course"
    s = fourh_df[col].fillna("").str.lower()
    patterns = {
        "mentions_discharge": r"discharge",
        "mentions_floor": r"floor",
        "mentions_icu": r"icu",
        "mentions_final_plan": r"final plan",
    }
    rows = []
    for name, pat in patterns.items():
        rows.append({"pattern": name, "count": int(s.str.contains(pat, regex=True).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    codebook_vars = save_codebook_extract()

    xls = pd.ExcelFile(XLSX_PATH)
    summaries = []
    sheets = {}

    for sheet in xls.sheet_names:
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
        sheets[sheet] = df
        summaries.append(summarize_sheet(sheet, df))

    summary_df = pd.DataFrame(
        [
            {
                "sheet": s["sheet"],
                "n_rows": s["n_rows"],
                "n_cols": s["n_cols"],
                "numeric_cols": s["numeric_cols"],
                "categorical_cols": s["categorical_cols"],
                "id_candidates": ", ".join(s["id_candidates"]),
            }
            for s in summaries
        ]
    )
    summary_df.to_csv(OUT_DIR / "sheet_overview.csv", index=False)

    overlap_rows = []
    sheet_names = list(sheets.keys())
    for i, s1 in enumerate(sheet_names):
        for s2 in sheet_names[i + 1 :]:
            shared = [c for c in sheets[s1].columns if c in set(sheets[s2].columns)]
            id_shared = [c for c in shared if re.search(r"(^id$|_id$|encounter|patient)", c.lower())]
            for c in id_shared:
                a = set(sheets[s1][c].dropna().unique())
                b = set(sheets[s2][c].dropna().unique())
                inter = len(a & b)
                overlap_rows.append(
                    {
                        "sheet_a": s1,
                        "sheet_b": s2,
                        "id_col": c,
                        "unique_a": len(a),
                        "unique_b": len(b),
                        "intersection": inter,
                        "coverage_of_a": inter / max(len(a), 1),
                        "coverage_of_b": inter / max(len(b), 1),
                    }
                )
    overlap_df = pd.DataFrame(overlap_rows)
    if not overlap_df.empty:
        overlap_df.to_csv(OUT_DIR / "id_overlap_checks.csv", index=False)

    triage_df = sheets["Triage_Data"]
    fourh_df = sheets["Four_Hour_Data"]
    dispo_df = sheets["Disposition"]

    nested = nested_feature_eda(triage_df, fourh_df)
    for name, df in nested.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    sex_mismatch = sex_hpi_mismatch_eda(triage_df, fourh_df)
    sex_mismatch.to_csv(OUT_DIR / "sex_hpi_mismatch_summary.csv", index=False)

    leakage = leakage_scan_eda(fourh_df)
    leakage.to_csv(OUT_DIR / "clinical_course_leakage_scan.csv", index=False)

    dispo_dist = dispo_df["encounter_disposition_label"].value_counts().rename_axis("label").reset_index(name="count")
    dispo_dist["pct"] = dispo_dist["count"] / dispo_dist["count"].sum()
    dispo_dist.to_csv(OUT_DIR / "disposition_distribution.csv", index=False)

    lines = []
    lines.append("# EDA Report: Hackathon Data Release 1")
    lines.append("")
    lines.append(f"- Workbook: `{XLSX_PATH}`")
    lines.append(f"- Codebook: `{CODEBOOK_PATH}`")
    lines.append("")
    lines.append("## Sheet Overview")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Disposition Label Distribution")
    lines.append(dispo_dist.to_markdown(index=False))
    lines.append("")

    lines.append("## Notes")
    lines.append("- Use `Triage_Data` only for drug-identification model features.")
    lines.append("- Use `Four_Hour_Data` (+ triage data) for disposition model features.")
    lines.append("- Exclude `narrative_notes_structured_clinical_course` in final hackathon-compatible model.")
    lines.append("")

    if not overlap_df.empty:
        lines.append("## ID Overlap Checks")
        lines.append(overlap_df.to_markdown(index=False))
        lines.append("")

    lines.append("## Nested Column Length Checks")
    for name, df in nested.items():
        lines.append(f"### {name}")
        lines.append(df.to_markdown(index=False))
        lines.append("")

    lines.append("## Potential Label Leakage in `narrative_notes_structured_clinical_course`")
    lines.append(leakage.to_markdown(index=False))
    lines.append("")

    lines.append("## Sex vs HPI Text Consistency")
    lines.append(sex_mismatch.to_markdown(index=False))
    lines.append("")

    lines.append("## Top Missingness (First 10 Columns Per Sheet)")
    for s in summaries:
        lines.append(f"### {s['sheet']}")
        lines.append(s["top_missing_cols"].to_markdown(index=False))
        lines.append("")

    if not codebook_vars.empty:
        lines.append("## Codebook Variable Rows (Heuristic Extract)")
        lines.append(codebook_vars.head(50).to_markdown(index=False))
        lines.append("")

    (OUT_DIR / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA complete. Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
