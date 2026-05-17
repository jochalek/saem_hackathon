# autoresearch_disposition

Autoresearch-style training scaffold for **SAEM disposition prediction** (`Discharge` / `Floor` / `ICU`) on tabular ED data.

This adaptation uses **k-fold cross-validation** (default: stratified folds), rotating held-out partitions so every sample is evaluated.

## Files

- `prepare.py` — fixed data prep + feature cache + evaluation utilities (do not modify in loop)
- `train.py` — model and training loop (the file to iterate)
- `program.md` — autonomous experiment instructions
- `pyproject.toml` — local GPU dependencies (`torch` + RAPIDS extras)

## Quick start

```bash
cd scripts/autoresearch_disposition
uv sync
uv run python prepare.py
uv run python train.py
```

Optional CV overrides:

```bash
DISPO_CV_STRATEGY=stratified uv run python train.py
DISPO_CV_FOLDS=5 uv run python train.py
DISPO_TRAIN_SECONDS=60 uv run python train.py
```

Outputs are written under:

- `outputs/disposition_autoresearch/cache/`
- `outputs/disposition_autoresearch/runs/<timestamp>/`

## Tabular Data Guides
https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
