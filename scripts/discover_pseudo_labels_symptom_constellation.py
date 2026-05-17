from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = ROOT / "outputs" / "pseudo_labels_symptom_constellation"
RANDOM_SEED = 42
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


@dataclass(frozen=True)
class SymptomDef:
    name: str
    patterns: tuple[str, ...]
    weight: float


SYMPTOMS: list[SymptomDef] = [
    SymptomDef("tachycardia", (r"tachycard", r"heart rate\s*(?:>|of\s*)?\s*1[01]\d", r"hr\s*(?:>|of\s*)?\s*1[01]\d"), 1.0),
    SymptomDef("diaphoresis", (r"diaphoret", r"sweat(?:ing)?"), 1.0),
    SymptomDef("mydriasis", (r"mydriasis", r"dilated pupils?"), 1.2),
    SymptomDef("agitation", (r"agitat", r"restless", r"combative"), 1.0),
    SymptomDef("hyperthermia", (r"hypertherm", r"temp(?:erature)?\s*(?:>|of\s*)?\s*(?:10[01]|39)"), 1.4),
    SymptomDef("clonus", (r"clonus", r"hyperreflex"), 1.5),
    SymptomDef("hallucination", (r"hallucin", r"psychosis", r"paranoid"), 1.0),
    SymptomDef("seizure", (r"seizure", r"convuls"), 1.3),
    SymptomDef("hypertension", (r"hypertens", r"bp\s*(?:>|of\s*)?\s*1[6-9]\d"), 0.9),
    SymptomDef("chest_pain", (r"chest pain", r"palpitation"), 0.8),
]

PROTOTYPES = {
    "Drug_1": {"tachycardia": 1.0, "diaphoresis": 1.0, "mydriasis": 1.0, "agitation": 0.9, "hypertension": 0.8},
    "Drug_2": {"hyperthermia": 1.0, "clonus": 1.0, "mydriasis": 0.8, "agitation": 0.8, "seizure": 0.6},
    "Drug_3": {"hallucination": 1.0, "agitation": 0.9, "tachycardia": 0.6, "hypertension": 0.5, "mydriasis": 0.5},
}

SEVERITY_PATTERNS = {
    "mild": r"\bmild\b",
    "moderate": r"\bmoderate\b",
    "severe": r"\bsevere\b|\bextreme\b|\bmarked\b",
}
ONSET_PATTERNS = {
    "sudden": r"\bsudden\b|\bacute\b",
    "hours": r"\b\d+\s*hours?\b|\bthis evening\b|\btoday\b",
    "days": r"\b\d+\s*days?\b|\bfor days\b",
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_joined(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    if "Triage_Data" not in xls.sheet_names:
        raise ValueError("Expected Triage_Data sheet")
    base = pd.read_excel(path, sheet_name="Triage_Data")
    for s in xls.sheet_names:
        if s == "Triage_Data":
            continue
        df = pd.read_excel(path, sheet_name=s)
        if "encounter_id" in df.columns:
            keep = ["encounter_id"] + [c for c in df.columns if c in {
                "narrative_notes_structured_brief_hpi",
                "narrative_notes_structured_hpi",
                "triage_brief_note",
                "triage_chief_complaint",
            }]
            df = df[keep].drop_duplicates("encounter_id")
            base = base.merge(df, on="encounter_id", how="left")
    return base


def build_hpi_text(df: pd.DataFrame) -> pd.Series:
    cols = [
        "triage_chief_complaint",
        "triage_brief_note",
        "narrative_notes_structured_brief_hpi",
        "narrative_notes_structured_hpi",
    ]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("No HPI-like text columns found")
    text = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def extract_triples(text: str) -> list[dict[str, str]]:
    t = text.lower()
    triples: list[dict[str, str]] = []

    severity = "unknown"
    for k, pat in SEVERITY_PATTERNS.items():
        if re.search(pat, t):
            severity = k
            break

    onset = "unknown"
    for k, pat in ONSET_PATTERNS.items():
        if re.search(pat, t):
            onset = k
            break

    for s in SYMPTOMS:
        if any(re.search(p, t) for p in s.patterns):
            triples.append({"symptom": s.name, "severity": severity, "onset_timing": onset})
    return triples


def symptom_vector(triples: list[dict[str, str]]) -> np.ndarray:
    sev_scale = {"unknown": 0.8, "mild": 0.8, "moderate": 1.0, "severe": 1.2}
    names = [s.name for s in SYMPTOMS]
    base = {n: 0.0 for n in names}
    for tr in triples:
        sym = tr["symptom"]
        if sym in base:
            w = next(s.weight for s in SYMPTOMS if s.name == sym)
            base[sym] = max(base[sym], w * sev_scale.get(tr["severity"], 1.0))
    return np.array([base[n] for n in names], dtype=np.float32)


def load_clinicalbert() -> tuple[AutoTokenizer, AutoModel, str]:
    token = os.getenv("HF_TOKEN", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token or None)
    model = AutoModel.from_pretrained(MODEL_NAME, token=token or None).to(device)
    model.eval()
    return tokenizer, model, device


def bert_embedding(texts: list[str], tokenizer: AutoTokenizer, model: AutoModel, device: str) -> np.ndarray:
    outs: list[np.ndarray] = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            tok = tokenizer(batch, return_tensors="pt", truncation=True, max_length=256, padding=True)
            tok = {k: v.to(device) for k, v in tok.items()}
            h = model(**tok).last_hidden_state[:, 0, :]
            outs.append(h.detach().cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def proto_matrix() -> tuple[list[str], np.ndarray]:
    names = [s.name for s in SYMPTOMS]
    rows = []
    drug_names = []
    for d, spec in PROTOTYPES.items():
        drug_names.append(d)
        rows.append([float(spec.get(n, 0.0)) for n in names])
    return drug_names, np.array(rows, dtype=np.float32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    load_env_file(ROOT / ".env")

    df = load_joined(RAW_PATH)
    text = build_hpi_text(df)

    tokenizer, model, device = load_clinicalbert()
    text_emb = bert_embedding(text.tolist(), tokenizer, model, device)

    triples = [extract_triples(t) for t in text.tolist()]
    sym_vec = np.vstack([symptom_vector(t) for t in triples])

    drug_names, proto = proto_matrix()
    sim = cosine_similarity(sym_vec, proto)

    evidence = (sym_vec > 0).sum(axis=1)
    max_sim = sim.max(axis=1)
    best_idx = sim.argmax(axis=1)

    # confidence: similarity + symptom specificity
    conf = 0.7 * max_sim + 0.3 * np.clip(evidence / 5.0, 0, 1)

    labels = np.array([drug_names[i] for i in best_idx], dtype=object)

    no_party_mask = (max_sim < 0.35) | (evidence < 2)

    # residual clustering for low-confidence potential party-drug cases
    residual_mask = (~no_party_mask) & (conf < 0.55)
    if residual_mask.sum() >= 9:
        X_resid = np.hstack([sym_vec[residual_mask], text_emb[residual_mask]])
        km = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=20)
        resid_cluster = km.fit_predict(X_resid)
        # map by nearest prototype in symptom subspace
        centers_sym = km.cluster_centers_[:, : sym_vec.shape[1]]
        c2p = cosine_similarity(centers_sym, proto).argmax(axis=1)
        labels[residual_mask] = [drug_names[c2p[c]] for c in resid_cluster]

    labels[no_party_mask] = "No_Party_Drug"

    out = pd.DataFrame({
        "encounter_id": df["encounter_id"],
        "drug_target_pseudo": labels,
        "drug_target_confidence": conf,
        "symptom_evidence_count": evidence,
        "prototype_max_similarity": max_sim,
        "P_No_Party_Drug": no_party_mask.astype(float),
        "P_Drug_1": sim[:, 0] if sim.shape[1] > 0 else 0.0,
        "P_Drug_2": sim[:, 1] if sim.shape[1] > 1 else 0.0,
        "P_Drug_3": sim[:, 2] if sim.shape[1] > 2 else 0.0,
    })

    triples_json = pd.DataFrame({
        "encounter_id": df["encounter_id"],
        "symptom_triples_json": [json.dumps(t) for t in triples],
    })

    # diagnostics similar to other pseudo-label pipelines
    summary = pd.DataFrame([
        {"metric": "n_total", "value": int(len(out))},
        {"metric": "n_no_party_drug", "value": int((out["drug_target_pseudo"] == "No_Party_Drug").sum())},
        {"metric": "n_drug_1", "value": int((out["drug_target_pseudo"] == "Drug_1").sum())},
        {"metric": "n_drug_2", "value": int((out["drug_target_pseudo"] == "Drug_2").sum())},
        {"metric": "n_drug_3", "value": int((out["drug_target_pseudo"] == "Drug_3").sum())},
        {"metric": "mean_confidence", "value": float(out["drug_target_confidence"].mean())},
        {"metric": "mean_symptom_evidence", "value": float(out["symptom_evidence_count"].mean())},
        {"metric": "mean_prototype_similarity", "value": float(out["prototype_max_similarity"].mean())},
        {"metric": "n_residual_clustered", "value": int(residual_mask.sum())},
    ])

    dispo_df = pd.read_excel(RAW_PATH, sheet_name="Disposition", usecols=["encounter_id", "encounter_disposition_label"])
    tmp = out[["encounter_id", "drug_target_pseudo"]].merge(dispo_df, on="encounter_id", how="left")
    counts = pd.crosstab(tmp["drug_target_pseudo"], tmp["encounter_disposition_label"])
    if counts.empty:
        dispo_balance = pd.DataFrame(columns=["drug_target_pseudo", "encounter_disposition_label", "count", "row_pct"])
    else:
        pct = counts.div(counts.sum(axis=1), axis=0)
        dispo_balance = counts.stack().rename("count").reset_index().merge(
            pct.stack().rename("row_pct").reset_index(),
            on=["drug_target_pseudo", "encounter_disposition_label"],
            how="left",
        )

    out.to_csv(OUT_DIR / "drug_target_pseudo_labels.csv", index=False)
    triples_json.to_csv(OUT_DIR / "symptom_triples_extracted.csv", index=False)
    summary.to_csv(OUT_DIR / "diagnostics_stability_summary.csv", index=False)
    dispo_balance.to_csv(OUT_DIR / "diagnostics_disposition_balance.csv", index=False)

    meta = {
        "model": MODEL_NAME,
        "device": device,
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
        "random_seed": RANDOM_SEED,
        "n_rows": int(len(df)),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = [
        "# Symptom Constellation Pseudo-Label Diagnostics",
        "",
        f"- Input workbook: `{RAW_PATH}`",
        f"- Model: `{MODEL_NAME}` on `{device}`",
        f"- Rows: `{len(out)}`",
        "",
        "## Stability Summary",
        summary.to_markdown(index=False),
        "",
        "## Disposition Balance",
        dispo_balance.to_markdown(index=False),
    ]
    (OUT_DIR / "diagnostics_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Done. Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
