from __future__ import annotations

import ast
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    import cupy as cp

    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

    RAPIDS_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None
    cuKMeans = None
    cuPCA = None
    cuNearestNeighbors = None
    RAPIDS_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Hackathon_Data_Release_1_SHARE.xlsx"
OUT_DIR = ROOT / "outputs" / "pseudo_labels"
EMB_CACHE_DIR = OUT_DIR / "embedding_cache"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax_rows(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def parse_nested_cell(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if not s:
        return []

    parsed: Any = None
    for candidate in (s, s.replace("null", "None")):
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


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        try:
            return x.to_numpy()
        except Exception:
            pass
    if RAPIDS_AVAILABLE and cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def load_and_join_sheets(path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    xls = pd.ExcelFile(path)
    sheets = {name: pd.read_excel(path, sheet_name=name) for name in xls.sheet_names}

    if "Triage_Data" not in sheets:
        raise ValueError("Expected sheet 'Triage_Data' to exist")

    merged = sheets["Triage_Data"].copy()
    for sheet_name, df in sheets.items():
        if sheet_name == "Triage_Data":
            continue
        if "encounter_id" not in df.columns:
            continue

        rename_map: dict[str, str] = {}
        for col in df.columns:
            if col == "encounter_id":
                continue
            if col in merged.columns:
                rename_map[col] = f"{slugify(sheet_name)}__{slugify(col)}"
        if rename_map:
            df = df.rename(columns=rename_map)

        merged = merged.merge(df, on="encounter_id", how="inner", validate="one_to_one")

    return sheets, merged


def discover_nested_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "triage.labs",
        "ed_course.vitals_timeseries",
        "ed_course.labs_timeseries",
        "ed_course.interventions",
    ]
    out = [c for c in preferred if c in df.columns]

    for col in df.select_dtypes(include=["object"]).columns:
        if col in out:
            continue
        sample = df[col].dropna().astype(str).head(25)
        if sample.empty:
            continue
        starts_like = sample.str.strip().str.startswith("[").mean()
        if starts_like < 0.6:
            continue
        parsed_ratio = sample.map(lambda x: len(parse_nested_cell(x)) >= 0).mean()
        if parsed_ratio >= 0.8:
            out.append(col)
    return out


def summarize_nested_column(df: pd.DataFrame, col: str, top_k_categories: int = 8) -> pd.DataFrame:
    prefix = slugify(col)
    parsed = df[col].map(parse_nested_cell)

    numeric_keys: set[str] = set()
    categorical_keys: set[str] = set()
    cat_values: dict[str, Counter[str]] = {}

    for items in parsed:
        for item in items:
            for k, v in item.items():
                if k in {"encounter_id", "minute"}:
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                    numeric_keys.add(k)
                else:
                    categorical_keys.add(k)
                    cat_values.setdefault(k, Counter())
                    if v is not None and str(v).strip() != "":
                        cat_values[k].update([str(v).strip()])

    top_categories = {
        k: [val for val, _ in ctr.most_common(top_k_categories)] for k, ctr in cat_values.items()
    }

    rows: list[dict[str, Any]] = []
    for encounter_id, items in zip(df["encounter_id"], parsed):
        rec: dict[str, Any] = {"encounter_id": encounter_id, f"{prefix}__n_events": len(items)}

        minutes = [float(it.get("minute")) for it in items if isinstance(it.get("minute"), (int, float))]
        if minutes:
            m = np.asarray(minutes, dtype=float)
            rec[f"{prefix}__minute__min"] = float(np.min(m))
            rec[f"{prefix}__minute__max"] = float(np.max(m))
            rec[f"{prefix}__minute__mean"] = float(np.mean(m))
            rec[f"{prefix}__minute__std"] = float(np.std(m))
            rec[f"{prefix}__minute__span"] = float(np.max(m) - np.min(m))
        else:
            rec[f"{prefix}__minute__min"] = np.nan
            rec[f"{prefix}__minute__max"] = np.nan
            rec[f"{prefix}__minute__mean"] = np.nan
            rec[f"{prefix}__minute__std"] = np.nan
            rec[f"{prefix}__minute__span"] = np.nan

        for key in sorted(numeric_keys):
            vals = [
                float(it.get(key))
                for it in items
                if isinstance(it.get(key), (int, float, np.integer, np.floating)) and not pd.isna(it.get(key))
            ]
            if vals:
                a = np.asarray(vals, dtype=float)
                rec[f"{prefix}__{slugify(key)}__mean"] = float(a.mean())
                rec[f"{prefix}__{slugify(key)}__std"] = float(a.std())
                rec[f"{prefix}__{slugify(key)}__min"] = float(a.min())
                rec[f"{prefix}__{slugify(key)}__max"] = float(a.max())
                rec[f"{prefix}__{slugify(key)}__first"] = float(a[0])
                rec[f"{prefix}__{slugify(key)}__last"] = float(a[-1])
                rec[f"{prefix}__{slugify(key)}__delta"] = float(a[-1] - a[0])
            else:
                rec[f"{prefix}__{slugify(key)}__mean"] = np.nan
                rec[f"{prefix}__{slugify(key)}__std"] = np.nan
                rec[f"{prefix}__{slugify(key)}__min"] = np.nan
                rec[f"{prefix}__{slugify(key)}__max"] = np.nan
                rec[f"{prefix}__{slugify(key)}__first"] = np.nan
                rec[f"{prefix}__{slugify(key)}__last"] = np.nan
                rec[f"{prefix}__{slugify(key)}__delta"] = np.nan

        for key in sorted(categorical_keys):
            vals = [str(it.get(key)).strip() for it in items if it.get(key) is not None and str(it.get(key)).strip()]
            rec[f"{prefix}__{slugify(key)}__nunique"] = len(set(vals))
            rec[f"{prefix}__{slugify(key)}__non_null"] = len(vals)
            for cat in top_categories.get(key, []):
                rec[f"{prefix}__{slugify(key)}__count__{slugify(cat)}"] = int(sum(v == cat for v in vals))

        rows.append(rec)

    return pd.DataFrame(rows)


def expand_nested_features(df: pd.DataFrame, nested_cols: list[str]) -> pd.DataFrame:
    if not nested_cols:
        return pd.DataFrame({"encounter_id": df["encounter_id"]})

    out = pd.DataFrame({"encounter_id": df["encounter_id"]})
    for col in nested_cols:
        block = summarize_nested_column(df, col)
        out = out.merge(block, on="encounter_id", how="left", validate="one_to_one")
    return out


def build_text_corpus(df: pd.DataFrame, include_clinical_course: bool = False) -> tuple[pd.Series, list[str]]:
    text_cols = [
        "triage_chief_complaint",
        "triage_brief_note",
        "narrative_notes_structured_brief_hpi",
        "narrative_notes_structured_hpi",
        "narrative_notes_structured_physical_exam_pertinent_positives",
        "narrative_notes_structured_mdm",
    ]
    if include_clinical_course:
        text_cols.append("narrative_notes_structured_clinical_course")

    cols = [c for c in text_cols if c in df.columns]
    pieces = []
    for col in cols:
        pieces.append((f"[{col}] " + df[col].fillna("").astype(str).str.strip()).str.replace(r"\s+", " ", regex=True))

    corpus = pd.Series([""] * len(df), index=df.index, dtype="object")
    for p in pieces:
        corpus = corpus + " " + p
    corpus = corpus.str.strip()
    return corpus, cols


def build_text_embeddings(corpus: pd.Series, cache_path: Path, model_name: str = MODEL_NAME) -> tuple[np.ndarray, dict[str, Any]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix(".meta.json")

    if cache_path.exists() and meta_path.exists():
        emb = np.load(cache_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("n_rows", -1)) == len(corpus):
            return emb, meta

    if SentenceTransformer is None:
        tfidf = TruncatedSVD(n_components=128, random_state=RANDOM_SEED)
        from sklearn.feature_extraction.text import TfidfVectorizer

        vect = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
        X = vect.fit_transform(corpus.tolist())
        emb = tfidf.fit_transform(X).astype(np.float32)
        meta = {
            "model": "tfidf_svd_fallback",
            "device": "cpu",
            "n_rows": int(len(corpus)),
            "dim": int(emb.shape[1]),
        }
    else:
        device = "cpu"
        if torch is not None and torch.cuda.is_available():
            device = "cuda"

        model = SentenceTransformer(model_name, device=device)
        emb = model.encode(
            corpus.tolist(),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        meta = {
            "model": model_name,
            "device": device,
            "n_rows": int(len(corpus)),
            "dim": int(emb.shape[1]),
        }

    np.save(cache_path, emb)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return emb, meta


def build_tabular_features(joined: pd.DataFrame, nested_features: pd.DataFrame) -> pd.DataFrame:
    keep_id = joined[["encounter_id"]].copy()

    dt_df = pd.DataFrame(index=joined.index)
    for col in joined.columns:
        if str(joined[col].dtype).startswith("datetime64"):
            dt = pd.to_datetime(joined[col], errors="coerce")
            dt_df[f"{slugify(col)}__hour"] = dt.dt.hour
            dt_df[f"{slugify(col)}__dow"] = dt.dt.dayofweek
            dt_df[f"{slugify(col)}__month"] = dt.dt.month

    numeric_df = joined.select_dtypes(include=[np.number]).copy()

    drop_obj_cols = {
        "encounter_id",
        "triage.labs",
        "ed_course.vitals_timeseries",
        "ed_course.labs_timeseries",
        "ed_course.interventions",
        "narrative_notes_structured_brief_hpi",
        "narrative_notes_structured_hpi",
        "narrative_notes_structured_physical_exam_pertinent_positives",
        "narrative_notes_structured_mdm",
        "narrative_notes_structured_clinical_course",
        "triage_chief_complaint",
        "triage_brief_note",
    }

    cat_cols = [
        c
        for c in joined.select_dtypes(include=["object"]).columns
        if c not in drop_obj_cols and c != "encounter_id"
    ]
    cat_df = pd.get_dummies(joined[cat_cols], dummy_na=True, dtype=float) if cat_cols else pd.DataFrame(index=joined.index)

    nested_df = nested_features.drop(columns=["encounter_id"], errors="ignore")
    feat = pd.concat([keep_id, numeric_df, dt_df, cat_df, nested_df], axis=1)

    non_id_cols = [c for c in feat.columns if c != "encounter_id"]
    for col in non_id_cols:
        if feat[col].dtype == bool:
            feat[col] = feat[col].astype(float)

    feat[non_id_cols] = feat[non_id_cols].replace([np.inf, -np.inf], np.nan)
    medians = feat[non_id_cols].median(numeric_only=True)
    feat[non_id_cols] = feat[non_id_cols].fillna(medians)
    feat[non_id_cols] = feat[non_id_cols].fillna(0.0)

    return feat


def compute_stage1_heuristics(df: pd.DataFrame) -> pd.DataFrame:
    festival_terms = [
        "festival",
        "main stage",
        "campground",
        "vip tent",
        "beach shuttle",
        "medical tent",
        "rave",
        "dance",
        "dj",
    ]
    community_terms = [
        "community patient",
        "home",
        "workplace",
        "shopping area",
        "street",
        "restaurant",
        "gym",
    ]

    text_cols = [
        c
        for c in [
            "triage_chief_complaint",
            "triage_brief_note",
            "narrative_notes_structured_brief_hpi",
            "narrative_notes_structured_hpi",
        ]
        if c in df.columns
    ]
    text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

    def term_hits(terms: list[str]) -> pd.Series:
        out = pd.Series(0, index=df.index, dtype=float)
        for t in terms:
            out += text.str.contains(re.escape(t), regex=True).astype(float)
        return out

    festival_hits = term_hits(festival_terms)
    community_hits = term_hits(community_terms)

    arrival = df.get("triage_mode_of_arrival", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    arrival_festival = arrival.str.contains("festival").astype(float)

    hpi = df.get("narrative_notes_structured_hpi", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    from_loc = hpi.str.extract(r"from ([^,]+) with", expand=False).fillna("")
    loc_festival = from_loc.str.contains("main stage|campground|vip tent|festival|beach shuttle", regex=True).astype(float)
    loc_community = from_loc.str.contains("home|workplace|shopping area|street|restaurant|gym", regex=True).astype(float)

    heuristic_score = (
        1.5 * arrival_festival
        + 1.1 * festival_hits
        + 1.0 * loc_festival
        - 0.9 * community_hits
        - 1.0 * loc_community
    )

    out = pd.DataFrame(
        {
            "festival_term_hits": festival_hits,
            "community_term_hits": community_hits,
            "arrival_festival": arrival_festival,
            "location_festival": loc_festival,
            "location_community": loc_community,
            "heuristic_score": heuristic_score,
        },
        index=df.index,
    )
    return out


def knn_mean_distance(X_fit: np.ndarray, X_query: np.ndarray, k: int = 15) -> np.ndarray:
    k = int(max(2, min(k, len(X_fit))))

    if RAPIDS_AVAILABLE and cuNearestNeighbors is not None:
        try:
            nn = cuNearestNeighbors(n_neighbors=k)
            nn.fit(X_fit.astype(np.float32))
            dists, _ = nn.kneighbors(X_query.astype(np.float32))
            d = to_numpy(dists)
            return d.mean(axis=1)
        except Exception:
            pass

    nn_cpu = NearestNeighbors(n_neighbors=k)
    nn_cpu.fit(X_fit)
    d_cpu, _ = nn_cpu.kneighbors(X_query)
    return d_cpu.mean(axis=1)


def select_stage1_threshold(prob_festival: np.ndarray, pos_anchor: np.ndarray, neg_anchor: np.ndarray) -> float:
    if pos_anchor.sum() >= 5 and neg_anchor.sum() >= 5:
        best_t = 0.5
        best_score = -1.0
        for t in np.linspace(0.2, 0.8, 61):
            tpr = float((prob_festival[pos_anchor] >= t).mean())
            tnr = float((prob_festival[neg_anchor] < t).mean())
            score = 0.5 * (tpr + tnr)
            if score > best_score:
                best_score = score
                best_t = float(t)
        return float(np.clip(best_t, 0.3, 0.75))

    return 0.5


def stage1_split(df: pd.DataFrame, X_stage1: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    heur = compute_stage1_heuristics(df)

    pos_anchor = (
        (heur["arrival_festival"].to_numpy() > 0)
        | (heur["festival_term_hits"].to_numpy() >= 2)
        | (heur["location_festival"].to_numpy() > 0)
    )
    neg_anchor = (
        (heur["arrival_festival"].to_numpy() == 0)
        & ((heur["community_term_hits"].to_numpy() >= 2) | (heur["location_community"].to_numpy() > 0))
    )

    if pos_anchor.sum() < 10:
        q = np.quantile(heur["heuristic_score"].to_numpy(), 0.7)
        pos_anchor = heur["heuristic_score"].to_numpy() >= q
    if neg_anchor.sum() < 10:
        q = np.quantile(heur["heuristic_score"].to_numpy(), 0.3)
        neg_anchor = heur["heuristic_score"].to_numpy() <= q

    dist_to_pos = knn_mean_distance(X_stage1[pos_anchor], X_stage1, k=12)
    dist_to_neg = knn_mean_distance(X_stage1[neg_anchor], X_stage1, k=12)
    global_knn = knn_mean_distance(X_stage1, X_stage1, k=18)

    relative = (dist_to_neg - dist_to_pos) / np.clip(np.abs(dist_to_neg) + np.abs(dist_to_pos), 1e-6, None)
    density = -zscore(global_knn)

    score_raw = 0.55 * zscore(heur["heuristic_score"].to_numpy()) + 0.30 * zscore(relative) + 0.15 * density
    prob_festival = sigmoid(score_raw)

    threshold = select_stage1_threshold(prob_festival, pos_anchor, neg_anchor)
    festival_like = prob_festival >= threshold

    # Guardrail for degenerate splits.
    if festival_like.sum() < 30 or festival_like.sum() > len(festival_like) - 15:
        threshold = float(np.quantile(prob_festival, 0.3))
        festival_like = prob_festival >= threshold

    detail = heur.copy()
    detail["stage1_prob_festival"] = prob_festival
    detail["stage1_anchor_pos"] = pos_anchor.astype(int)
    detail["stage1_anchor_neg"] = neg_anchor.astype(int)
    detail["stage1_distance_to_festival_anchor"] = dist_to_pos
    detail["stage1_distance_to_nonfestival_anchor"] = dist_to_neg
    detail["stage1_global_knn_distance"] = global_knn
    detail["stage1_score_raw"] = score_raw
    detail["stage1_label_festival_like"] = festival_like.astype(int)

    return festival_like, prob_festival, detail, threshold


def fit_kmeans_labels(X: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if RAPIDS_AVAILABLE and cuKMeans is not None:
        try:
            model = cuKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                max_iter=300,
                n_init=20,
                init="k-means||",
            )
            labels = model.fit_predict(X.astype(np.float32))
            return to_numpy(labels).astype(int)
        except Exception:
            pass

    model_cpu = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    return model_cpu.fit_predict(X)


def reduce_for_clustering(X: np.ndarray, max_components: int = 32) -> np.ndarray:
    n_comp = int(min(max_components, X.shape[1], max(2, X.shape[0] - 1)))

    if RAPIDS_AVAILABLE and cuPCA is not None:
        try:
            pca = cuPCA(n_components=n_comp, random_state=RANDOM_SEED)
            out = pca.fit_transform(X.astype(np.float32))
            return to_numpy(out).astype(np.float32)
        except Exception:
            pass

    pca_cpu = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    return pca_cpu.fit_transform(X).astype(np.float32)


def best_permutation_map(reference: np.ndarray, candidate: np.ndarray, n_clusters: int = 3) -> dict[int, int]:
    best_map = {i: i for i in range(n_clusters)}
    best_score = -1
    for perm in itertools.permutations(range(n_clusters)):
        mapping = {i: perm[i] for i in range(n_clusters)}
        mapped = np.array([mapping.get(int(v), 0) for v in candidate], dtype=int)
        score = int((mapped == reference).sum())
        if score > best_score:
            best_score = score
            best_map = mapping
    return best_map


def remap_labels(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.array([mapping.get(int(v), 0) for v in labels], dtype=int)


def stage2_consensus(X_festival: np.ndarray, n_clusters: int = 3) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float]]:
    X_red = reduce_for_clustering(X_festival, max_components=28)

    model_labels: dict[str, np.ndarray] = {
        "kmeans_seed_13": fit_kmeans_labels(X_red, n_clusters=n_clusters, seed=13),
        "kmeans_seed_47": fit_kmeans_labels(X_red, n_clusters=n_clusters, seed=47),
        "kmeans_seed_97": fit_kmeans_labels(X_red, n_clusters=n_clusters, seed=97),
        "gmm_full": GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=RANDOM_SEED).fit_predict(X_red),
        "agglo_ward": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X_red),
    }

    ref_name = "kmeans_seed_13"
    ref = model_labels[ref_name]

    aligned: dict[str, np.ndarray] = {ref_name: ref}
    for name, labels in model_labels.items():
        if name == ref_name:
            continue
        mapping = best_permutation_map(ref, labels, n_clusters=n_clusters)
        aligned[name] = remap_labels(labels, mapping)

    model_names = list(aligned.keys())
    stacked = np.vstack([aligned[m] for m in model_names]).T  # n x m

    vote_counts = np.zeros((stacked.shape[0], n_clusters), dtype=float)
    for k in range(n_clusters):
        vote_counts[:, k] = (stacked == k).sum(axis=1)
    vote_probs = vote_counts / vote_counts.sum(axis=1, keepdims=True)
    consensus = vote_probs.argmax(axis=1)

    # Deterministic cluster -> Drug mapping by cluster size (largest -> Drug_1).
    counts = np.bincount(consensus, minlength=n_clusters)
    ordered = list(np.argsort(-counts))
    cluster_to_drug = {cluster: i + 1 for i, cluster in enumerate(ordered)}

    vote_df = pd.DataFrame({"row_idx": np.arange(len(consensus))})
    for m in model_names:
        vote_df[f"model__{m}"] = aligned[m]
    vote_df["cluster_consensus"] = consensus
    vote_df["cluster_consensus_confidence"] = vote_probs.max(axis=1)
    vote_df["drug_index_consensus"] = [cluster_to_drug[int(c)] for c in consensus]

    ari_scores = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a = aligned[model_names[i]]
            b = aligned[model_names[j]]
            ari_scores.append(float(adjusted_rand_score(a, b)))

    stats = {
        "stage2_pairwise_ari_mean": float(np.mean(ari_scores)) if ari_scores else np.nan,
        "stage2_pairwise_ari_min": float(np.min(ari_scores)) if ari_scores else np.nan,
        "stage2_pairwise_ari_max": float(np.max(ari_scores)) if ari_scores else np.nan,
        "stage2_consensus_conf_mean": float(vote_probs.max(axis=1).mean()),
    }

    return consensus, vote_probs, vote_df, stats


def cluster_affinities(X_all: np.ndarray, X_fest: np.ndarray, fest_labels: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    centroids = []
    global_mean = X_fest.mean(axis=0)
    for k in range(n_clusters):
        members = X_fest[fest_labels == k]
        if len(members) == 0:
            centroids.append(global_mean)
        else:
            centroids.append(members.mean(axis=0))
    centroids = np.vstack(centroids)

    dists = np.sqrt(np.sum((X_all[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    temp = float(np.median(dists)) if np.median(dists) > 0 else 1.0
    probs = softmax_rows(-dists, temperature=temp)
    return probs


def build_outputs(
    joined: pd.DataFrame,
    festival_like: np.ndarray,
    prob_festival: np.ndarray,
    consensus_labels_fest: np.ndarray,
    vote_probs_fest: np.ndarray,
    X_cluster_all: np.ndarray,
    X_cluster_fest: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    n = len(joined)
    n_clusters = 3

    # Map cluster index -> Drug index by festival subset prevalence.
    counts = np.bincount(consensus_labels_fest, minlength=n_clusters)
    ordered = list(np.argsort(-counts))
    cluster_to_drug = {cluster: i + 1 for i, cluster in enumerate(ordered)}

    affinity_all = cluster_affinities(X_cluster_all, X_cluster_fest, consensus_labels_fest, n_clusters=n_clusters)

    conditional_drug_probs = affinity_all.copy()
    festival_indices = np.where(festival_like)[0]
    if len(festival_indices) > 0:
        blended = 0.7 * vote_probs_fest + 0.3 * affinity_all[festival_indices]
        blended = blended / np.clip(blended.sum(axis=1, keepdims=True), 1e-9, None)
        conditional_drug_probs[festival_indices] = blended

    # Reorder from cluster-space to Drug_1..Drug_3 order.
    ordered_conditional = np.zeros_like(conditional_drug_probs)
    for cluster_idx in range(n_clusters):
        drug_idx = cluster_to_drug[cluster_idx] - 1
        ordered_conditional[:, drug_idx] = conditional_drug_probs[:, cluster_idx]

    p_no = np.clip(1.0 - prob_festival, 1e-6, 1.0)
    p_drug = np.clip(prob_festival[:, None] * ordered_conditional, 1e-6, None)

    probs = np.column_stack([p_no, p_drug])
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

    pseudo = np.array(["No_Festival"] * n, dtype=object)
    fest_drug_indices = [cluster_to_drug[int(c)] for c in consensus_labels_fest]
    pseudo[festival_like] = [f"Drug_{i}" for i in fest_drug_indices]

    assigned_conf = np.zeros(n, dtype=float)
    assigned_conf[pseudo == "No_Festival"] = probs[pseudo == "No_Festival", 0]
    for drug_i in (1, 2, 3):
        mask = pseudo == f"Drug_{drug_i}"
        assigned_conf[mask] = probs[mask, drug_i]

    out = pd.DataFrame(
        {
            "encounter_id": joined["encounter_id"],
            "drug_target_pseudo": pseudo,
            "drug_target_confidence": assigned_conf,
            "P_No_Festival": probs[:, 0],
            "P_Drug_1": probs[:, 1],
            "P_Drug_2": probs[:, 2],
            "P_Drug_3": probs[:, 3],
            "stage1_prob_festival_like": prob_festival,
        }
    )

    return out, {f"cluster_{k}": v for k, v in cluster_to_drug.items()}


def leakage_sensitivity_check(
    joined: pd.DataFrame,
    tabular_matrix: np.ndarray,
    base_stage1_festival: np.ndarray,
    base_stage1_prob: np.ndarray,
    base_stage2_labels: np.ndarray,
    base_festival_idx: np.ndarray,
) -> pd.DataFrame:
    corpus_with_leak, _ = build_text_corpus(joined, include_clinical_course=True)
    emb_leak, _ = build_text_embeddings(
        corpus_with_leak,
        EMB_CACHE_DIR / "text_embeddings_with_clinical_course.npy",
        model_name=MODEL_NAME,
    )

    X_alt = np.hstack([tabular_matrix, emb_leak]).astype(np.float32)
    scaler = StandardScaler()
    X_alt_scaled = scaler.fit_transform(X_alt)

    alt_festival_like, alt_prob_festival, _, _ = stage1_split(joined, X_alt_scaled)

    # Compare stage-2 on the baseline festival subset only.
    if len(base_festival_idx) >= 12:
        X_alt_fest = X_alt_scaled[base_festival_idx]
        alt_stage2_labels, _, _, _ = stage2_consensus(X_alt_fest, n_clusters=3)
        remap = best_permutation_map(base_stage2_labels, alt_stage2_labels, n_clusters=3)
        alt_remapped = remap_labels(alt_stage2_labels, remap)
        stage2_flip_rate = float((alt_remapped != base_stage2_labels).mean())
        stage2_ari = float(adjusted_rand_score(base_stage2_labels, alt_remapped))
    else:
        stage2_flip_rate = np.nan
        stage2_ari = np.nan

    checks = pd.DataFrame(
        [
            {
                "check": "stage1_label_flip_rate_with_clinical_course",
                "value": float((alt_festival_like != base_stage1_festival).mean()),
            },
            {
                "check": "stage1_mean_abs_prob_shift_with_clinical_course",
                "value": float(np.mean(np.abs(alt_prob_festival - base_stage1_prob))),
            },
            {
                "check": "stage2_flip_rate_on_baseline_festival_subset",
                "value": stage2_flip_rate,
            },
            {
                "check": "stage2_ari_on_baseline_festival_subset",
                "value": stage2_ari,
            },
        ]
    )
    return checks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sheets, joined = load_and_join_sheets(RAW_PATH)

    nested_cols = discover_nested_columns(joined)
    nested_features = expand_nested_features(joined, nested_cols)

    tabular = build_tabular_features(joined, nested_features)

    corpus, used_text_cols = build_text_corpus(joined, include_clinical_course=False)
    embeddings, emb_meta = build_text_embeddings(
        corpus,
        EMB_CACHE_DIR / "text_embeddings_no_clinical_course.npy",
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

    # Diagnostics: disposition balance.
    if "encounter_disposition_label" in joined.columns:
        tmp = pseudo_df[["encounter_id", "drug_target_pseudo"]].merge(
            joined[["encounter_id", "encounter_disposition_label"]], on="encounter_id", how="left"
        )
        counts = pd.crosstab(tmp["drug_target_pseudo"], tmp["encounter_disposition_label"])
        pct = counts.div(counts.sum(axis=1), axis=0)
        disposition_balance = counts.stack().rename("count").reset_index().merge(
            pct.stack().rename("row_pct").reset_index(),
            on=["drug_target_pseudo", "encounter_disposition_label"],
            how="left",
        )
    else:
        disposition_balance = pd.DataFrame()

    # Stability + sensitivity checks.
    stage1_bal_acc = np.nan
    pos = stage1_detail["stage1_anchor_pos"].to_numpy().astype(bool)
    neg = stage1_detail["stage1_anchor_neg"].to_numpy().astype(bool)
    if pos.sum() >= 5 and neg.sum() >= 5:
        tpr = float((festival_like[pos]).mean())
        tnr = float((~festival_like[neg]).mean())
        stage1_bal_acc = 0.5 * (tpr + tnr)

    sensitivity = leakage_sensitivity_check(
        joined=joined,
        tabular_matrix=tabular_matrix,
        base_stage1_festival=festival_like,
        base_stage1_prob=prob_festival,
        base_stage2_labels=stage2_labels,
        base_festival_idx=festival_idx,
    )

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

    # Save files.
    pseudo_df.to_csv(OUT_DIR / "drug_target_pseudo_labels.csv", index=False)

    stage1_out = pd.concat(
        [joined[["encounter_id"]], stage1_detail.reset_index(drop=True)],
        axis=1,
    )
    stage1_out.to_csv(OUT_DIR / "stage1_split_diagnostics.csv", index=False)

    vote_out = vote_df.copy()
    vote_out.insert(0, "encounter_id", joined.loc[festival_idx, "encounter_id"].to_numpy())
    vote_out.to_csv(OUT_DIR / "stage2_consensus_votes.csv", index=False)

    disposition_balance.to_csv(OUT_DIR / "diagnostics_disposition_balance.csv", index=False)
    stability_summary.to_csv(OUT_DIR / "diagnostics_stability_summary.csv", index=False)
    sensitivity.to_csv(OUT_DIR / "diagnostics_leakage_sensitivity.csv", index=False)

    metadata = {
        "rapids_available": RAPIDS_AVAILABLE,
        "embedding_meta": emb_meta,
        "nested_columns_expanded": nested_cols,
        "text_columns_used": used_text_cols,
        "cluster_to_drug_mapping": cluster_map,
        "output_rows": int(len(pseudo_df)),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_lines = [
        "# Pseudo-Label Discovery Diagnostics",
        "",
        f"- Input workbook: `{RAW_PATH}`",
        f"- RAPIDS enabled: `{RAPIDS_AVAILABLE}`",
        f"- Embedding model: `{emb_meta.get('model')}` on `{emb_meta.get('device')}`",
        f"- Expanded nested columns: {', '.join(nested_cols) if nested_cols else 'None'}",
        f"- Stage-1 festival-like count: `{int(festival_like.sum())}` / `{len(joined)}`",
        f"- Stage-1 threshold: `{stage1_threshold:.4f}`",
        "",
        "## Stability Summary",
        stability_summary.to_markdown(index=False),
        "",
        "## Leakage / Sensitivity Checks",
        sensitivity.to_markdown(index=False),
    ]

    if not disposition_balance.empty:
        report_lines.extend(
            [
                "",
                "## Disposition Balance",
                disposition_balance.to_markdown(index=False),
            ]
        )

    (OUT_DIR / "diagnostics_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Pseudo-label discovery complete. Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
