from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prepare import (
    CLASS_NAMES,
    CV_N_SPLITS,
    CV_STRATEGY,
    OUT_DIR,
    RANDOM_SEED,
    RUNS_DIR,
    TRAIN_TIME_BUDGET_SECONDS,
    DatasetBundle,
    compute_classification_metrics,
    describe_environment,
    get_device,
    load_or_create_cv_dataset_bundles,
    set_seed,
)


# Agent-tunable hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 2.0e-3
WEIGHT_DECAY = 2.0e-4
HIDDEN_DIM = 256
DEPTH = 3
DROPOUT = 0.20
GRAD_CLIP_NORM = 1.0
EVAL_EVERY_STEPS = 20
MAX_EPOCHS = 10_000

# Primary optimization target for autonomous loop: maximize this metric.
PRIMARY_METRIC = "macro_f1"


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class FoldRunResult:
    fold_index: int
    metrics: dict[str, float]
    peak_vram_mb: float
    num_steps: int
    epochs_completed: int
    train_seconds: float
    num_params_m: float


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        dim_in = input_dim
        for _ in range(DEPTH):
            blocks.extend(
                [
                    nn.Linear(dim_in, HIDDEN_DIM),
                    nn.BatchNorm1d(HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                ]
            )
            dim_in = HIDDEN_DIM

        self.backbone = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Linear(dim_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


def _normalize_features(bundle: DatasetBundle) -> tuple[DatasetBundle, NormalizationStats]:
    x_train = bundle.x_train.astype(np.float32)
    x_val = bundle.x_val.astype(np.float32)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    x_train_norm = (x_train - mean) / std
    x_val_norm = (x_val - mean) / std

    normalized = DatasetBundle(
        x_train=x_train_norm.astype(np.float32),
        y_train=bundle.y_train,
        x_val=x_val_norm.astype(np.float32),
        y_val=bundle.y_val,
        feature_names=bundle.feature_names,
        class_names=bundle.class_names,
        class_weights=bundle.class_weights,
        encounter_ids_train=bundle.encounter_ids_train,
        encounter_ids_val=bundle.encounter_ids_val,
    )
    return normalized, NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def _make_dataloaders(bundle: DatasetBundle) -> tuple[DataLoader, DataLoader]:
    x_train = torch.tensor(bundle.x_train, dtype=torch.float32)
    y_train = torch.tensor(bundle.y_train, dtype=torch.long)
    x_val = torch.tensor(bundle.x_val, dtype=torch.float32)
    y_val = torch.tensor(bundle.y_val, dtype=torch.long)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(256, BATCH_SIZE), shuffle=False, drop_last=False)
    return train_loader, val_loader


def _evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    logits_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            logits_chunks.append(logits.detach().cpu().numpy())
            y_chunks.append(y_batch.detach().cpu().numpy())

    logits_all = np.concatenate(logits_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)

    probs = torch.softmax(torch.tensor(logits_all), dim=1).numpy()
    return compute_classification_metrics(y_all, probs, class_names=CLASS_NAMES)


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _train_single_fold(
    fold_index: int,
    bundle_raw: DatasetBundle,
    device: torch.device,
    train_seconds: int,
    run_dir: Any,
) -> tuple[FoldRunResult, NormalizationStats]:
    set_seed(RANDOM_SEED + fold_index)

    bundle, norm_stats = _normalize_features(bundle_raw)
    train_loader, val_loader = _make_dataloaders(bundle)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = MLPClassifier(input_dim=bundle.x_train.shape[1], num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    class_weights = torch.tensor(bundle.class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    start = time.perf_counter()
    best_metrics: dict[str, float] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")

    global_step = 0
    epochs_completed = 0

    while (time.perf_counter() - start) < train_seconds and epochs_completed < MAX_EPOCHS:
        epochs_completed += 1
        model.train()

        for x_batch, y_batch in train_loader:
            if (time.perf_counter() - start) >= train_seconds:
                break

            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            if GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            optimizer.step()
            global_step += 1

            if global_step % EVAL_EVERY_STEPS == 0:
                val_metrics = _evaluate(model, val_loader, device)
                score = float(val_metrics.get(PRIMARY_METRIC, float("nan")))

                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_metrics = val_metrics
                    best_model_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best_model_state is None:
        final_metrics = _evaluate(model, val_loader, device)
        best_metrics = final_metrics
        best_model_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    else:
        model.load_state_dict(best_model_state)
        final_metrics = _evaluate(model, val_loader, device)
        current_score = float(final_metrics.get(PRIMARY_METRIC, float("nan")))
        if np.isfinite(current_score) and current_score >= best_score:
            best_metrics = final_metrics
            best_score = current_score

    elapsed = float(time.perf_counter() - start)

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    params_count = _count_parameters(model)

    fold_artifact = {
        "state_dict": model.state_dict(),
        "feature_names": bundle.feature_names,
        "class_names": CLASS_NAMES,
        "normalization_mean": norm_stats.mean,
        "normalization_std": norm_stats.std,
        "metrics": best_metrics,
        "fold_index": int(fold_index),
    }
    torch.save(fold_artifact, run_dir / f"model_fold{fold_index}.pt")

    np.savez_compressed(
        run_dir / f"normalization_stats_fold{fold_index}.npz",
        mean=norm_stats.mean,
        std=norm_stats.std,
        feature_names=np.asarray(bundle.feature_names, dtype=str),
    )

    result = FoldRunResult(
        fold_index=fold_index,
        metrics=best_metrics,
        peak_vram_mb=peak_vram_mb,
        num_steps=int(global_step),
        epochs_completed=int(epochs_completed),
        train_seconds=elapsed,
        num_params_m=float(params_count / 1_000_000),
    )
    return result, norm_stats


def _aggregate_fold_results(results: list[FoldRunResult]) -> dict[str, float]:
    if not results:
        raise ValueError("Cannot aggregate empty fold results")

    metric_keys = sorted(results[0].metrics.keys())
    out: dict[str, float] = {}

    for key in metric_keys:
        vals = np.asarray([float(r.metrics.get(key, np.nan)) for r in results], dtype=float)
        out[f"cv_{key}_mean"] = float(np.nanmean(vals))
        out[f"cv_{key}_std"] = float(np.nanstd(vals))

    out["cv_macro_f1_min"] = float(np.nanmin([r.metrics.get("macro_f1", np.nan) for r in results]))
    out["cv_icu_recall_min"] = float(np.nanmin([r.metrics.get("icu_recall", np.nan) for r in results]))
    return out


def main() -> None:
    set_seed(RANDOM_SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    total_train_seconds = int(os.getenv("DISPO_TRAIN_SECONDS", str(TRAIN_TIME_BUDGET_SECONDS)))
    cv_strategy = os.getenv("DISPO_CV_STRATEGY", CV_STRATEGY)
    cv_folds_requested = int(os.getenv("DISPO_CV_FOLDS", str(CV_N_SPLITS)))

    bundles = load_or_create_cv_dataset_bundles(
        force_rebuild=False,
        seed=RANDOM_SEED,
        cv_strategy=cv_strategy,
        n_splits=cv_folds_requested,
    )
    if len(bundles) < 2:
        raise RuntimeError("Need at least 2 folds for cross-validation")

    now_tag = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / now_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Allocate fixed total budget across folds.
    base = max(1, total_train_seconds // len(bundles))
    budgets = [base for _ in bundles]
    remainder = max(0, total_train_seconds - base * len(bundles))
    for i in range(min(remainder, len(budgets))):
        budgets[i] += 1

    device = get_device()

    start_all = time.perf_counter()
    fold_results: list[FoldRunResult] = []

    for fold_idx, (bundle, fold_budget) in enumerate(zip(bundles, budgets)):
        result, _ = _train_single_fold(
            fold_index=fold_idx,
            bundle_raw=bundle,
            device=device,
            train_seconds=int(max(1, fold_budget)),
            run_dir=run_dir,
        )
        fold_results.append(result)

    elapsed_all = float(time.perf_counter() - start_all)

    aggregate = _aggregate_fold_results(fold_results)
    primary_metric_name = "cv_macro_f1_mean"

    peak_vram_mb = float(max(r.peak_vram_mb for r in fold_results))
    num_steps_total = int(sum(r.num_steps for r in fold_results))
    epochs_total = int(sum(r.epochs_completed for r in fold_results))

    fold_metrics_payload = [
        {
            "fold_index": int(r.fold_index),
            **{f"val_{k}": float(v) for k, v in r.metrics.items()},
            "training_seconds": float(r.train_seconds),
            "peak_vram_mb": float(r.peak_vram_mb),
            "num_steps": int(r.num_steps),
            "epochs_completed": int(r.epochs_completed),
            "num_params_m": float(r.num_params_m),
            "train_rows": int(len(bundles[r.fold_index].y_train)),
            "val_rows": int(len(bundles[r.fold_index].y_val)),
        }
        for r in fold_results
    ]

    (run_dir / "fold_metrics.json").write_text(
        json.dumps(_to_serializable(fold_metrics_payload), indent=2),
        encoding="utf-8",
    )

    summary = {
        "primary_metric": primary_metric_name,
        "cv_strategy": cv_strategy,
        "cv_folds": int(len(bundles)),
        "cv_fold_budget_seconds": [int(x) for x in budgets],
        **aggregate,
        "training_seconds": elapsed_all,
        "peak_vram_mb": peak_vram_mb,
        "num_steps_total": num_steps_total,
        "epochs_completed_total": epochs_total,
        "num_params_m": float(fold_results[0].num_params_m),
        "feature_dim": int(bundles[0].x_train.shape[1]),
        "class_names": CLASS_NAMES,
        "device": str(device),
        "run_dir": str(run_dir),
    }

    (run_dir / "metrics.json").write_text(json.dumps(_to_serializable(summary), indent=2), encoding="utf-8")

    run_meta = {
        "env": describe_environment(),
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "hidden_dim": HIDDEN_DIM,
            "depth": DEPTH,
            "dropout": DROPOUT,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "max_epochs": MAX_EPOCHS,
            "total_train_seconds": total_train_seconds,
            "primary_metric": PRIMARY_METRIC,
            "cv_strategy": cv_strategy,
            "cv_folds_requested": cv_folds_requested,
            "cv_folds_used": len(bundles),
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(_to_serializable(run_meta), indent=2), encoding="utf-8")

    print("---")
    print(f"primary_metric:         {summary['primary_metric']}")
    print(f"cv_macro_f1_mean:      {summary['cv_macro_f1_mean']:.6f}")
    print(f"cv_macro_f1_std:       {summary['cv_macro_f1_std']:.6f}")
    print(f"cv_macro_auroc_mean:   {summary['cv_macro_auroc_mean']:.6f}")
    print(f"cv_icu_pr_auc_mean:    {summary['cv_icu_pr_auc_mean']:.6f}")
    print(f"cv_icu_recall_mean:    {summary['cv_icu_recall_mean']:.6f}")
    print(f"cv_icu_recall_min:     {summary['cv_icu_recall_min']:.6f}")
    print(f"cv_log_loss_mean:      {summary['cv_log_loss_mean']:.6f}")
    print(f"training_seconds:      {summary['training_seconds']:.1f}")
    print(f"peak_vram_mb:          {summary['peak_vram_mb']:.1f}")
    print(f"num_steps_total:       {summary['num_steps_total']}")
    print(f"epochs_completed_total:{summary['epochs_completed_total']}")
    print(f"num_params_m:          {summary['num_params_m']:.3f}")
    print(f"feature_dim:           {summary['feature_dim']}")
    print(f"cv_folds:              {summary['cv_folds']}")
    print(f"run_dir:               {summary['run_dir']}")


if __name__ == "__main__":
    main()
