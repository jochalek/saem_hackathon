# disposition-autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may16`). The branch `autoresearch/disposition-<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/disposition-<tag>` from current master/main.
3. **Read the in-scope files**: this subproject is intentionally small. Read these files for full context:
   - `README.md` — local subproject context.
   - `prepare.py` — fixed constants, data prep, feature cache, and evaluation utilities. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify cached data exists**: check that `outputs/disposition_autoresearch/cache/` contains cached dataset files. If not, tell the human to run `uv run python prepare.py` from this directory.
5. **Initialize results.tsv**: create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed 5-minute total budget** by default (`TRAIN_TIME_BUDGET_SECONDS` in `prepare.py`), distributed across CV folds. You launch it as:

`uv run python train.py`

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, regularization, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only for the loop and contains the fixed data prep and evaluation utilities.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness in `prepare.py`.

**The goal is simple: maximize `cv_macro_f1_mean`.**

Guardrails (track every run):
- keep `cv_icu_recall_mean` strong,
- improve `cv_icu_pr_auc_mean` when possible,
- avoid major degradation in `cv_log_loss_mean`.

Secondary metric:
- `cv_macro_auroc_mean`.

**CV strategy note**: by default this setup uses stratified k-fold (`CV_STRATEGY = stratified` in `prepare.py`).

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but avoid major blowups. There is 24GB VRAM.

**Simplicity criterion**: all else equal, simpler is better. Tiny gains with lots of ugly complexity are usually not worth it. If a simplification keeps performance equal or better, that's a win.

**The first run**: your first run should always be baseline (`train.py` unchanged).

## Output format

Once the script finishes, it prints a summary like this:

```
---
primary_metric:         cv_macro_f1_mean
cv_macro_f1_mean:      0.752902
cv_macro_f1_std:       0.040000
cv_macro_auroc_mean:   0.891839
cv_icu_pr_auc_mean:    0.590192
cv_icu_recall_mean:    0.750000
cv_icu_recall_min:     0.500000
cv_log_loss_mean:      1.375221
training_seconds:      300.0
peak_vram_mb:          1234.5
num_steps_total:       123
epochs_completed_total:45
num_params_m:          0.331
feature_dim:           768
cv_folds:              5
run_dir:               /.../outputs/disposition_autoresearch/runs/<timestamp>
```

Extract key metrics from the run log using:

```
grep "^cv_macro_f1_mean:\|^cv_macro_auroc_mean:\|^cv_icu_pr_auc_mean:\|^cv_icu_recall_mean:\|^cv_log_loss_mean:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break descriptions).

TSV header:

```
commit	cv_macro_f1_mean	cv_macro_auroc_mean	cv_icu_pr_auc_mean	cv_icu_recall_mean	cv_log_loss_mean	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `cv_macro_f1_mean` (higher is better) — use `0.000000` for crashes
3. `cv_macro_auroc_mean`
4. `cv_icu_pr_auc_mean`
5. `cv_icu_recall_mean`
6. `cv_log_loss_mean`
7. peak memory in GB, round to .1f (`peak_vram_mb / 1024`) — use `0.0` for crashes
8. status: `keep`, `discard`, or `crash`
9. short text description of what this experiment tried

Example:

```
commit	cv_macro_f1_mean	cv_macro_auroc_mean	cv_icu_pr_auc_mean	cv_icu_recall_mean	cv_log_loss_mean	memory_gb	status	description
a1b2c3d	0.640000	0.860000	0.520000	0.700000	0.890000	2.1	keep	baseline
b2c3d4e	0.655000	0.868000	0.541000	0.750000	0.870000	2.2	keep	increase hidden dim
c3d4e5f	0.631000	0.851000	0.500000	0.625000	0.940000	2.1	discard	switch activation
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/disposition-may16` or `autoresearch/disposition-may16-gpu0`).

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Tune `train.py` with one experimental idea.
3. git commit.
4. Run the experiment: `uv run python train.py > run.log 2>&1` (redirect everything — do NOT use tee or flood context).
5. Read out results with grep command above.
6. If grep output is empty, run crashed. Use `tail -n 50 run.log` to inspect stack trace and attempt a fix. If it fails repeatedly, abandon that idea.
7. Record results in `results.tsv` (do not commit `results.tsv`; keep it untracked).
8. If `cv_macro_f1_mean` improved (higher), advance the branch and keep commit.
9. If `cv_macro_f1_mean` is equal or worse, reset back to previous best.

The idea is autonomous iteration: keep improvements, discard regressions, keep advancing the best branch state.

**Timeout**: each experiment should take ~5 minutes (+ startup/eval overhead). If a run exceeds 10 minutes, kill it and treat as failure (discard and revert).

**Crashes**: if a run crashes (OOM/bug/etc.), use judgment. If easy fix, fix and rerun. If the idea is fundamentally broken, log `crash` and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
