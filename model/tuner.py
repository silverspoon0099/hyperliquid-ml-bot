"""Optuna hyperparameter tuning (spec §10.2).

Objective: minimize mean val log-loss across a cheap subset of walk-forward
folds (tuning.tuning_fold_indices — default [1, 4, 7] = early/mid/late regime).
MedianPruner kills trials whose first-fold score is worse than the median of
completed first-fold scores, so bad trials cost ~1 fold instead of 3.

Search space + bounds live in config.yaml under `tuning.search_space` so you
can edit without touching code. Fixed base params (objective, num_class, metric,
boosting_type, verbose, n_jobs, seed) are inherited from model.params.

Best params are written to model/models/{SYMBOL}_wf/best_params.json, ready
for a final 8-fold walk-forward run (spec 2.3 → 2.4 SHAP → 2.5 trim).

Usage:
    python -m model.tuner --symbol BTC
    python -m model.tuner --symbol BTC --n-trials 100   # override config
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from model.trainer import feature_columns, make_folds
from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("tuner")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _suggest(trial: optuna.Trial, space: dict) -> dict:
    """Build a LightGBM param dict from an Optuna trial, driven by config ranges."""
    return {
        "num_leaves":        trial.suggest_int("num_leaves",
                                               *space["num_leaves"]),
        "learning_rate":     trial.suggest_float("learning_rate",
                                                 *space["learning_rate"], log=True),
        "feature_fraction":  trial.suggest_float("feature_fraction",
                                                 *space["feature_fraction"]),
        "bagging_fraction":  trial.suggest_float("bagging_fraction",
                                                 *space["bagging_fraction"]),
        "min_child_samples": trial.suggest_int("min_child_samples",
                                               *space["min_child_samples"]),
        "lambda_l1":         trial.suggest_float("lambda_l1",
                                                 *space["lambda_l1"], log=True),
        "lambda_l2":         trial.suggest_float("lambda_l2",
                                                 *space["lambda_l2"], log=True),
    }


def run(symbol: str, cfg: dict, n_trials: int | None = None, timeout: int | None = None) -> dict:
    tcfg = cfg["tuning"]
    n_trials = n_trials or tcfg["n_trials"]
    timeout = timeout or tcfg["timeout_sec"]
    fold_ids = tcfg["tuning_fold_indices"]  # 1-based

    # ── Load features + precompute fold slices once (shared across trials) ──
    features_dir = PROJECT_ROOT / cfg["features"]["output_dir"]
    parquet = features_dir / f"{symbol}_features.parquet"
    log.info(f"loading {parquet}")
    df = pd.read_parquet(parquet)
    df = df[df["label"] >= 0].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    feat_cols = feature_columns(df)
    n_classes = len(cfg["labeling"]["classes"])

    wf = cfg["walk_forward"]
    purge = cfg["model"]["walk_forward_purge_bars"]
    all_folds = make_folds(
        df["_ts"],
        train_months=wf["train_months"],
        val_months=wf["val_months"],
        step_months=wf["step_months"],
        purge_bars=purge,
    )
    log.info(f"{len(all_folds)} total walk-forward folds; tuning on {fold_ids}")

    # Pre-materialize train/val numpy arrays for the fold subset — avoids
    # reslicing the DataFrame inside every trial.
    cached: list[dict] = []
    for i in fold_ids:
        if i < 1 or i > len(all_folds):
            raise ValueError(f"tuning_fold_indices contains {i}, out of range 1..{len(all_folds)}")
        t0, t1, v0, v1 = all_folds[i - 1]
        tr = df[(df["_ts"] >= t0) & (df["_ts"] < t1)]
        va = df[(df["_ts"] >= v0) & (df["_ts"] < v1)]
        cached.append({
            "fold": i,
            "X_tr":  tr[feat_cols].to_numpy(dtype=np.float32),
            "y_tr":  tr["label"].to_numpy(dtype=np.int32),
            "X_val": va[feat_cols].to_numpy(dtype=np.float32),
            "y_val": va["label"].to_numpy(dtype=np.int32),
            "val_from": v0.date(),
            "val_to":   v1.date(),
        })
        log.info(f"  fold {i}: train={t0.date()}->{t1.date()} ({len(tr):,}) "
                 f"val={v0.date()}->{v1.date()} ({len(va):,})")

    base_params = dict(cfg["model"]["params"])
    num_boost_round = cfg["model"]["num_boost_round"]
    early_stopping = cfg["model"]["early_stopping_rounds"]
    search_space = tcfg["search_space"]

    # ── Optuna objective ────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        tuned = _suggest(trial, search_space)
        params = {**base_params, **tuned}

        fold_losses: list[float] = []
        for step, fold in enumerate(cached):
            train_data = lgb.Dataset(fold["X_tr"], label=fold["y_tr"], feature_name=feat_cols)
            val_data = lgb.Dataset(fold["X_val"], label=fold["y_val"],
                                   reference=train_data, feature_name=feat_cols)
            booster = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[val_data],
                valid_names=["val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping, first_metric_only=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            proba = booster.predict(fold["X_val"], num_iteration=booster.best_iteration)
            ll = float(log_loss(fold["y_val"], proba, labels=list(range(n_classes))))
            fold_losses.append(ll)

            # Report to Optuna after each fold; allow pruner to kill early.
            running_mean = float(np.mean(fold_losses))
            trial.report(running_mean, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_losses))

    # ── Run study ───────────────────────────────────────────────────────────
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=tcfg["pruner_n_warmup_steps"])
    sampler = optuna.samplers.TPESampler(seed=tcfg["seed"])
    study = optuna.create_study(direction=tcfg["direction"], sampler=sampler, pruner=pruner)

    log.info(f"starting study: n_trials={n_trials} timeout={timeout}s")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    # ── Summary + persist ───────────────────────────────────────────────────
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    log.info(f"done — {len(completed)} completed, {len(pruned)} pruned")
    log.info(f"best trial #{study.best_trial.number}: mean_val_logloss = {study.best_value:.4f}")
    log.info("best params:")
    for k, v in study.best_params.items():
        log.info(f"  {k:20s} {v!r}")

    out_dir = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "symbol": symbol,
        "n_trials_requested": n_trials,
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "best_trial_number": study.best_trial.number,
        "best_mean_val_logloss": float(study.best_value),
        "best_params": study.best_params,
        "tuning_fold_indices": fold_ids,
        "base_params_frozen": {
            k: v for k, v in base_params.items()
            if k not in study.best_params  # what we did NOT tune
        },
    }
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"saved -> {out_dir / 'best_params.json'}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--n-trials", type=int, default=None, help="override tuning.n_trials")
    ap.add_argument("--timeout", type=int, default=None, help="override tuning.timeout_sec")
    args = ap.parse_args()
    cfg = load_config()
    run(args.symbol, cfg, n_trials=args.n_trials, timeout=args.timeout)


if __name__ == "__main__":
    main()
