from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from discover_pseudo_labels import (
    MODEL_NAME,
    RAPIDS_AVAILABLE,
    RANDOM_SEED,
    build_outputs,
    build_tabular_features,
    build_text_embeddings,
    discover_nested_columns,
    expand_nested_features,
    stage1_split,
    stage2_consensus,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = ROOT / "outputs" / "pseudo_labels_no_disposition"
EMB_CACHE_DIR = OUT_DIR / "embedding_cache"

np.random.seed(RANDOM_SEED)

TRIAGE_SHEET = "Triage_Data"
FOUR_HOUR_SHEET = "Four_Hour_Data"
DISPOSITION_SHEET = "Disposition"

SAFE_FOUR_HOUR_TEXT_COLS = [
    "encounter_id",
    "narrative_notes_structured_brief_hpi",
    "narrative_notes_structured_hpi",
]

LEAKAGE_PATTERNS = [
    r"disposition",
    r"clinical_course",
    r"structured_mdm",
    r"ed_meds_procedures",
    r"ed_course\.interventions",
    r"ed_course_reassessment_4h",
    r"ed_course\.reassessment_4h",
    r"intubat",
    r"positive_pressure",
    r"\bcvc\b",
]


def validate_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {missing}")


def infer_leakage_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        col_l = col.lower()
        if any(re.search(pat, col_l) for pat in LEAKAGE_PATTERNS):
            out.append(col)
    return sorted(set(out))


def load_modeling_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")

    triage_df = pd.read_excel(path, sheet_name=TRIAGE_SHEET)
    fourh_df = pd.read_excel(path, sheet_name=FOUR_HOUR_SHEET)
    dispo_df = pd.read_excel(path, sheet_name=DISPOSITION_SHEET)

    validate_columns(triage_df, ["encounter_id"], TRIAGE_SHEET)
    validate_columns(fourh_df, SAFE_FOUR_HOUR_TEXT_COLS, FOUR_HOUR_SHEET)
    validate_columns(dispo_df, ["encounter_id", "encounter_disposition_label"], DISPOSITION_SHEET)

    excluded_fourh_cols = sorted([c for c in fourh_df.columns if c not in SAFE_FOUR_HOUR_TEXT_COLS])

    joined = triage_df.merge(
        fourh_df[SAFE_FOUR_HOUR_TEXT_COLS],
        on="encounter_id",
        how="inner",
        validate="one_to_one",
    )

    dropped_leakage_cols = infer_leakage_columns(joined.columns.tolist())
    if dropped_leakage_cols:
        joined = joined.drop(columns=dropped_leakage_cols, errors="ignore")

    remaining_leakage_cols = infer_leakage_columns(joined.columns.tolist())
    if remaining_leakage_cols:
        raise RuntimeError(
            "Leakage columns remain in modeling table after filtering: "
            f"{remaining_leakage_cols}"
        )

    return joined, dispo_df, excluded_fourh_cols, dropped_leakage_cols


def build_safe_text_corpus(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    text_cols = [
        "triage_chief_complaint",
        "triage_brief_note",
        "narrative_notes_structured_brief_hpi",
        "narrative_notes_structured_hpi",
    ]
    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        raise ValueError("No text columns available to build corpus")

    pieces = []
    for col in cols:
        part = (
            f"[{col}] "
            + df[col]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        pieces.append(part)

    corpus = pd.Series([""] * len(df), index=df.index, dtype="object")
    for part in pieces:
        corpus = corpus + " " + part
    return corpus.str.strip(), cols


def text_leakage_scan(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    terms = {
        "mentions_discharge": r"\bdischarge\b",
        "mentions_floor": r"\bfloor\b",
        "mentions_icu": r"\bicu\b",
        "mentions_admit": r"\badmit(?:ted|ting)?\b",
    }
    rows: list[dict[str, Any]] = []
    for col in text_cols:
        s = df[col].fillna("").astype(str).str.lower()
        for term_name, pat in terms.items():
            rows.append(
                {
                    "column": col,
                    "pattern": term_name,
                    "count": int(s.str.contains(pat, regex=True).sum()),
                }
            )
    return pd.DataFrame(rows)


def disposition_balance_table(pseudo_df: pd.DataFrame, dispo_df: pd.DataFrame) -> pd.DataFrame:
    tmp = pseudo_df[["encounter_id", "drug_target_pseudo"]].merge(
        dispo_df[["encounter_id", "encounter_disposition_label"]],
        on="encounter_id",
        how="left",
        validate="one_to_one",
    )
    counts = pd.crosstab(tmp["drug_target_pseudo"], tmp["encounter_disposition_label"])
    if counts.empty:
        return pd.DataFrame(columns=["drug_target_pseudo", "encounter_disposition_label", "count", "row_pct"])

    pct = counts.div(counts.sum(axis=1), axis=0)
    return counts.stack().rename("count").reset_index().merge(
        pct.stack().rename("row_pct").reset_index(),
        on=["drug_target_pseudo", "encounter_disposition_label"],
        how="left",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    joined, dispo_df, excluded_fourh_cols, dropped_leakage_cols = load_modeling_tables(RAW_PATH)

    nested_cols = discover_nested_columns(joined)
    nested_features = expand_nested_features(joined, nested_cols)

    tabular = build_tabular_features(joined, nested_features)

    corpus, used_text_cols = build_safe_text_corpus(joined)
    embeddings, emb_meta = build_text_embeddings(
        corpus,
        EMB_CACHE_DIR / "text_embeddings_safe_hpi_triage.npy",
        model_name=MODEL_NAME,
    )

    tabular_matrix = tabular.drop(columns=["encounter_id"]).to_numpy(dtype=np.float32)
    X_full = np.hstack([tabular_matrix, embeddings]).astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    festival_like, prob_festival, stage1_detail, stage1_threshold = stage1_split(joined, X_scaled)

    festival_idx = np.where(festival_like)[0]
    if len(festival_idx) < 12:
        raise RuntimeError("Festival-like subset too small for stage-2 clustering")

    X_stage2_fest = X_scaled[festival_idx]
    stage2_labels, vote_probs_fest, vote_df, stage2_stats = stage2_consensus(X_stage2_fest, n_clusters=3)

    pseudo_df, cluster_map = build_outputs(
        joined=joined,
        festival_like=festival_like,
        prob_festival=prob_festival,
        consensus_labels_fest=stage2_labels,
        vote_probs_fest=vote_probs_fest,
        X_cluster_all=X_scaled,
        X_cluster_fest=X_stage2_fest,
    )

    stage1_bal_acc = np.nan
    pos = stage1_detail["stage1_anchor_pos"].to_numpy().astype(bool)
    neg = stage1_detail["stage1_anchor_neg"].to_numpy().astype(bool)
    if pos.sum() >= 5 and neg.sum() >= 5:
        tpr = float((festival_like[pos]).mean())
        tnr = float((~festival_like[neg]).mean())
        stage1_bal_acc = 0.5 * (tpr + tnr)

    stability_summary = pd.DataFrame(
        [
            {"metric": "n_total", "value": len(joined)},
            {"metric": "n_festival_like", "value": int(festival_like.sum())},
            {"metric": "n_no_festival", "value": int((~festival_like).sum())},
            {"metric": "stage1_threshold", "value": float(stage1_threshold)},
            {"metric": "stage1_anchor_balanced_accuracy", "value": stage1_bal_acc},
            {"metric": "stage2_pairwise_ari_mean", "value": stage2_stats["stage2_pairwise_ari_mean"]},
            {"metric": "stage2_pairwise_ari_min", "value": stage2_stats["stage2_pairwise_ari_min"]},
            {"metric": "stage2_pairwise_ari_max", "value": stage2_stats["stage2_pairwise_ari_max"]},
            {"metric": "stage2_consensus_conf_mean", "value": stage2_stats["stage2_consensus_conf_mean"]},
        ]
    )

    dispo_balance = disposition_balance_table(pseudo_df, dispo_df)
    text_leakage = text_leakage_scan(joined, used_text_cols)

    pseudo_df.to_csv(OUT_DIR / "drug_target_pseudo_labels.csv", index=False)

    stage1_out = pd.concat(
        [joined[["encounter_id"]], stage1_detail.reset_index(drop=True)],
        axis=1,
    )
    stage1_out.to_csv(OUT_DIR / "stage1_split_diagnostics.csv", index=False)

    vote_out = vote_df.copy()
    vote_out.insert(0, "encounter_id", joined.loc[festival_idx, "encounter_id"].to_numpy())
    vote_out.to_csv(OUT_DIR / "stage2_consensus_votes.csv", index=False)

    stability_summary.to_csv(OUT_DIR / "diagnostics_stability_summary.csv", index=False)
    dispo_balance.to_csv(OUT_DIR / "diagnostics_disposition_balance.csv", index=False)
    text_leakage.to_csv(OUT_DIR / "diagnostics_text_leakage_scan.csv", index=False)

    metadata = {
        "input_workbook": str(RAW_PATH),
        "pipeline_variant": "no_disposition_safe_hpi",
        "rapids_available": RAPIDS_AVAILABLE,
        "random_seed": RANDOM_SEED,
        "embedding_meta": emb_meta,
        "nested_columns_expanded": nested_cols,
        "text_columns_used": used_text_cols,
        "excluded_four_hour_columns": excluded_fourh_cols,
        "explicitly_dropped_leakage_columns": dropped_leakage_cols,
        "cluster_to_drug_mapping": cluster_map,
        "output_rows": int(len(pseudo_df)),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "# Pseudo-Label Discovery Diagnostics (No Disposition / Leakage-Reduced)",
        "",
        f"- Input workbook: `{RAW_PATH}`",
        f"- RAPIDS enabled: `{RAPIDS_AVAILABLE}`",
        f"- Embedding model: `{emb_meta.get('model')}` on `{emb_meta.get('device')}`",
        "- Modeling inputs: `Triage_Data` + restricted HPI text from `Four_Hour_Data`",
        f"- Excluded Four_Hour_Data columns (non-HPI): `{len(excluded_fourh_cols)}`",
        f"- Explicitly dropped leakage columns from modeling frame: `{', '.join(dropped_leakage_cols) if dropped_leakage_cols else 'None'}`",
        f"- Expanded nested columns: `{', '.join(nested_cols) if nested_cols else 'None'}`",
        f"- Stage-1 festival-like count: `{int(festival_like.sum())}` / `{len(joined)}`",
        f"- Stage-1 threshold: `{stage1_threshold:.4f}`",
        "",
        "## Stability Summary",
        stability_summary.to_markdown(index=False),
        "",
        "## Text Leakage Scan",
        text_leakage.to_markdown(index=False),
        "",
        "## Disposition Balance (diagnostic only; not used for modeling)",
        dispo_balance.to_markdown(index=False),
    ]

    (OUT_DIR / "diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"No-disposition pseudo-label discovery complete. Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
