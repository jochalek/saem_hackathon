from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = ROOT / "outputs" / "pseudo_labels_llm_structured"
RANDOM_SEED = 42

TRIAGE_SHEET = "Triage_Data"
FOUR_HOUR_SHEET = "Four_Hour_Data"

TEXT_COLS = [
    "triage_chief_complaint",
    "triage_brief_note",
    "narrative_notes_structured_brief_hpi",
    "narrative_notes_structured_hpi",
    "narrative_notes_structured_physical_exam_pertinent_positives",
]

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover pseudo labels via LLM structured extraction + clustering")
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH, help="Path to source workbook")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-5.5"),
        help="OpenAI model name",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows processed (head N)")
    parser.add_argument("--n-clusters", type=int, default=4, help="Requested KMeans cluster count")
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="Number of extracted-entity examples to print and include in diagnostics report",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def validate_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {missing}")


def load_joined(path: Path) -> pd.DataFrame:
    triage_df = pd.read_excel(path, sheet_name=TRIAGE_SHEET)
    fourh_df = pd.read_excel(path, sheet_name=FOUR_HOUR_SHEET)

    validate_columns(triage_df, ["encounter_id", "triage_chief_complaint", "triage_brief_note"], TRIAGE_SHEET)
    validate_columns(
        fourh_df,
        [
            "encounter_id",
            "narrative_notes_structured_brief_hpi",
            "narrative_notes_structured_hpi",
            "narrative_notes_structured_physical_exam_pertinent_positives",
        ],
        FOUR_HOUR_SHEET,
    )

    keep_fourh = [
        "encounter_id",
        "narrative_notes_structured_brief_hpi",
        "narrative_notes_structured_hpi",
        "narrative_notes_structured_physical_exam_pertinent_positives",
    ]

    joined = triage_df.merge(
        fourh_df[keep_fourh],
        on="encounter_id",
        how="inner",
        validate="one_to_one",
    )
    return joined


def normalize_text(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return " ".join(str(v).split())


def build_prompt_record(row: pd.Series) -> dict[str, str]:
    return {c: normalize_text(row.get(c, "")) for c in TEXT_COLS}


def extraction_schema() -> dict[str, Any]:
    return {
        "name": "hpi_structured_extraction",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "symptoms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "polarity": {"type": "string", "enum": ["present", "absent", "uncertain"]},
                            "severity": {"type": "string", "enum": ["mild", "moderate", "severe", "unknown"]},
                            "duration_minutes": {"type": ["integer", "null"]},
                            "onset": {"type": "string", "enum": ["sudden", "gradual", "unknown"]},
                        },
                        "required": ["name", "polarity", "severity", "duration_minutes", "onset"],
                    },
                },
                "arrival_mode": {
                    "type": "string",
                    "enum": ["ems", "festival_tent_transfer", "walk_in", "private_vehicle", "police", "other", "unknown"],
                },
                "patient_origin": {
                    "type": "string",
                    "enum": ["main_stage", "campground", "vip_tent", "beach_shuttle_stop", "medical_tent", "offsite", "unknown"],
                },
                "time_prior_to_arrival_minutes": {"type": ["integer", "null"]},
                "suspected_substance_exposure": {"type": "string", "enum": ["yes", "no", "unclear"]},
                "mental_status": {
                    "type": "string",
                    "enum": ["normal", "anxious", "agitated", "confused", "somnolent", "psychotic", "unknown"],
                },
                "confidence": {"type": "number"},
            },
            "required": [
                "symptoms",
                "arrival_mode",
                "patient_origin",
                "time_prior_to_arrival_minutes",
                "suspected_substance_exposure",
                "mental_status",
                "confidence",
            ],
        },
        "strict": True,
    }


def parse_json_text(s: str) -> dict[str, Any]:
    s = s.strip()
    if not s:
        raise ValueError("Empty model response")
    return json.loads(s)


def llm_extract_one(client: OpenAI, model: str, rec: dict[str, str], max_retries: int = 4) -> dict[str, Any]:
    schema = extraction_schema()
    system = (
        "You are a clinical information extraction engine. "
        "Extract only facts supported by the provided text. "
        "Return JSON that exactly matches the schema."
    )
    user_payload = {
        "task": "extract_structured_features",
        "fields": rec,
        "notes": (
            "If a fact is not present, use unknown/unclear/null as appropriate. "
            "When extracting time_prior_to_arrival_minutes or symptom duration_minutes, "
            "return integer minutes only (no decimals, no units string)."
        ),
    }

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
                response_format={"type": "json_schema", "json_schema": schema},
            )
            content = resp.choices[0].message.content or "{}"
            return parse_json_text(content)
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            non_retryable = ("invalid_request_error" in msg) or ("unsupported value" in msg) or ("error code: 400" in msg)
            if non_retryable or attempt == max_retries - 1:
                break
            time.sleep(1.25 * (attempt + 1))

    raise RuntimeError(f"LLM extraction failed after retries: {last_err}")


def coerce_extraction(obj: dict[str, Any]) -> dict[str, Any]:
    out = dict(obj)
    out.setdefault("symptoms", [])
    out.setdefault("arrival_mode", "unknown")
    out.setdefault("patient_origin", "unknown")
    out.setdefault("time_prior_to_arrival_minutes", None)
    out.setdefault("suspected_substance_exposure", "unclear")
    out.setdefault("mental_status", "unknown")
    out.setdefault("confidence", 0.0)

    # normalize for featurization
    out["symptoms"] = [s for s in out["symptoms"] if isinstance(s, dict) and s.get("name")]
    out["arrival_mode"] = str(out["arrival_mode"])
    out["patient_origin"] = str(out["patient_origin"])
    out["suspected_substance_exposure"] = str(out["suspected_substance_exposure"])
    out["mental_status"] = str(out["mental_status"])

    try:
        out["confidence"] = float(out["confidence"])
    except Exception:  # noqa: BLE001
        out["confidence"] = 0.0

    tpa = out["time_prior_to_arrival_minutes"]
    if tpa is None:
        out["time_prior_to_arrival_minutes"] = None
    else:
        try:
            tpa_i = int(round(float(tpa)))
            out["time_prior_to_arrival_minutes"] = tpa_i if tpa_i >= 0 else None
        except Exception:  # noqa: BLE001
            out["time_prior_to_arrival_minutes"] = None

    normalized_symptoms: list[dict[str, Any]] = []
    for s in out["symptoms"]:
        s2 = dict(s)
        dur = s2.get("duration_minutes")
        if dur is None:
            s2["duration_minutes"] = None
        else:
            try:
                dur_i = int(round(float(dur)))
                s2["duration_minutes"] = dur_i if dur_i >= 0 else None
            except Exception:  # noqa: BLE001
                s2["duration_minutes"] = None
        normalized_symptoms.append(s2)
    out["symptoms"] = normalized_symptoms

    return out


def summarize_named_entities(extraction: dict[str, Any]) -> dict[str, Any]:
    symptoms = sorted(
        {
            str(s.get("name")).strip().lower()
            for s in extraction.get("symptoms", [])
            if isinstance(s, dict) and s.get("name")
        }
    )
    return {
        "symptoms": symptoms,
        "time_prior_to_arrival_minutes": extraction.get("time_prior_to_arrival_minutes"),
        "arrival_mode": extraction.get("arrival_mode", "unknown"),
        "patient_origin": extraction.get("patient_origin", "unknown"),
    }


def featurize_extractions(extractions: list[dict[str, Any]]) -> np.ndarray:
    symptom_sets = [[f"symptom:{str(s.get('name')).strip().lower()}" for s in ex["symptoms"]] for ex in extractions]

    mlb_sym = MultiLabelBinarizer()
    X_sym = mlb_sym.fit_transform(symptom_sets)

    cat_rows = [
        [
            ex["arrival_mode"],
            ex["patient_origin"],
            ex["suspected_substance_exposure"],
            ex["mental_status"],
        ]
        for ex in extractions
    ]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = ohe.fit_transform(cat_rows)

    num = np.array(
        [
            [
                float(
                    -1.0
                    if ex["time_prior_to_arrival_minutes"] is None
                    else ex["time_prior_to_arrival_minutes"]
                ),
                float(ex["confidence"]),
                float(len(ex["symptoms"])),
            ]
            for ex in extractions
        ],
        dtype=np.float32,
    )

    scaler = StandardScaler()
    X_num = scaler.fit_transform(num)

    return np.hstack([X_sym, X_cat, X_num]).astype(np.float32)


def cluster_and_label(X: np.ndarray, n_clusters: int = 4) -> tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=30)
    labels = km.fit_predict(X)

    dists = km.transform(X)
    min_dist = dists.min(axis=1)
    conf = 1.0 / (1.0 + min_dist)
    return labels, conf


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    load_env_file(ROOT / ".env")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env.")

    model = str(args.model).strip() or "gpt-5.5"
    client = OpenAI(api_key=api_key)

    df = load_joined(args.raw_path)
    if args.max_rows is not None:
        if int(args.max_rows) <= 0:
            raise ValueError("--max-rows must be > 0 when provided")
        df = df.head(int(args.max_rows)).copy()

    if df.empty:
        raise ValueError("No rows available after loading/filtering input data")

    extracted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for i, row in df.iterrows():
        rec = build_prompt_record(row)
        try:
            obj = llm_extract_one(client=client, model=model, rec=rec)
            obj = coerce_extraction(obj)
        except Exception as e:  # noqa: BLE001
            obj = coerce_extraction({})
            errors.append({"encounter_id": row["encounter_id"], "error": str(e)})
        extracted.append(obj)

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            print(f"Processed {i + 1}/{len(df)}")

    X = featurize_extractions(extracted)
    n_clusters_requested = int(args.n_clusters)
    if n_clusters_requested <= 0:
        raise ValueError("--n-clusters must be > 0")
    n_clusters_used = min(n_clusters_requested, len(df))

    cluster_idx, cluster_conf = cluster_and_label(X, n_clusters=n_clusters_used)

    cluster_name_map = {c: f"Drug_{c + 1}" for c in sorted(set(cluster_idx.tolist()))}
    cluster_names = np.array([cluster_name_map[c] for c in cluster_idx], dtype=object)

    out_df = pd.DataFrame(
        {
            "encounter_id": df["encounter_id"],
            "drug_target_pseudo": cluster_names,
            "drug_target_confidence": cluster_conf,
            "cluster_index": cluster_idx,
        }
    )

    struct_df = pd.DataFrame(
        {
            "encounter_id": df["encounter_id"],
            "structured_extraction_json": [json.dumps(x, ensure_ascii=False) for x in extracted],
        }
    )

    summary = pd.DataFrame(
        [
            {"metric": "n_total", "value": int(len(df))},
            {"metric": "n_extraction_errors", "value": int(len(errors))},
            {"metric": "mean_cluster_confidence", "value": float(np.mean(cluster_conf))},
            {"metric": "n_clusters_requested", "value": n_clusters_requested},
            {"metric": "n_clusters_used", "value": n_clusters_used},
            {"metric": "openai_model", "value": model},
        ]
    )

    out_df.to_csv(args.out_dir / "drug_target_pseudo_labels.csv", index=False)
    struct_df.to_csv(args.out_dir / "structured_extractions.csv", index=False)
    summary.to_csv(args.out_dir / "diagnostics_stability_summary.csv", index=False)
    pd.DataFrame(errors).to_csv(args.out_dir / "extraction_errors.csv", index=False)

    n_show = max(0, min(int(args.show_examples), len(df)))
    example_lines: list[str] = []
    if n_show > 0:
        example_lines.extend(["", "## Sample extracted named entities", ""])
        for encounter_id, obj in zip(df["encounter_id"].head(n_show), extracted[:n_show], strict=False):
            ent = summarize_named_entities(obj)
            line_block = [
                f"- encounter_id={encounter_id}",
                f"  symptoms={ent['symptoms']}",
                f"  time_prior_to_arrival_minutes={ent['time_prior_to_arrival_minutes']}",
                f"  context=arrival_mode={ent['arrival_mode']}, origin={ent['patient_origin']}",
            ]
            example_lines.extend(line_block)

    metadata = {
        "input_workbook": str(args.raw_path),
        "pipeline_variant": "llm_structured_extraction",
        "openai_model": model,
        "openai_api_key_present": bool(api_key),
        "text_columns_used": TEXT_COLS,
        "n_rows": int(len(df)),
        "max_rows_requested": None if args.max_rows is None else int(args.max_rows),
        "n_features_after_featurization": int(X.shape[1]),
        "n_clusters_requested": n_clusters_requested,
        "n_clusters_used": n_clusters_used,
        "random_seed": RANDOM_SEED,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_lines = [
        "# LLM Structured Extraction Pseudo-Label Diagnostics",
        "",
        f"- Input workbook: `{args.raw_path}`",
        f"- OpenAI model: `{model}`",
        f"- Rows processed: `{len(df)}`",
        f"- Extraction errors: `{len(errors)}`",
        f"- Clusters requested: `{n_clusters_requested}`",
        f"- Clusters used: `{n_clusters_used}`",
        "",
        "## Stability Summary",
        summary.to_markdown(index=False),
    ]
    report_lines.extend(example_lines)
    (args.out_dir / "diagnostics_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    if n_show > 0:
        print("\nSample extracted named entities")
        for encounter_id, obj in zip(df["encounter_id"].head(n_show), extracted[:n_show], strict=False):
            ent = summarize_named_entities(obj)
            print(f"- encounter_id={encounter_id}")
            print(f"  symptoms={ent['symptoms']}")
            print(f"  time_prior_to_arrival_minutes={ent['time_prior_to_arrival_minutes']}")
            print(f"  context=arrival_mode={ent['arrival_mode']}, origin={ent['patient_origin']}")

    print(f"Done. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
