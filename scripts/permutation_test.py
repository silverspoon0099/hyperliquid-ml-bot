"""Label permutation sanity test.

Hypothesis: if features are not leaking label information, then training on
permuted (shuffled) labels should collapse val metrics to the class-prior
baseline. Any accuracy/log-loss BETTER than prior on permuted training implies
a feature leaks the label (e.g. look-ahead from a forward-looking indicator,
or a label-derived column that slipped into the feature matrix).

Method:
  1. Slice BTC fold-1 train / val (same boundaries as walk-forward fold 1).
  2. Random-shuffle ONLY the training labels. Val labels stay real.
  3. Train LightGBM with identical params.
  4. Compare val metrics to:
       - max class prior (accuracy ceiling if model only learns priors)
       - H(val prior) in nats (log-loss floor from predicting priors exactly)
       - real-label fold-1 from walk-forward (0.962 logloss, 49.4% acc)

Usage:
    python -m scripts.permutation_test
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("permutation_test")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NON_FEATURE_COLS = {
    "timestamp", "_ts",  # _ts is a synthetic datetime used only for fold slicing
    "open", "high", "low", "close", "volume",
    "label", "holding_bars", "exit_price", "pnl_pct",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def main(seed: int = 42) -> None:
    cfg = load_config()
    classes = cfg["labeling"]["classes"]
    n_classes = len(classes)

    parquet = PROJECT_ROOT / cfg["features"]["output_dir"] / "BTC_features.parquet"
    log.info(f"loading {parquet}")
    df = pd.read_parquet(parquet)
    df = df[df["label"] >= 0].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Match fold 1 of the walk-forward exactly for apples-to-apples.
    t0 = pd.Timestamp("2024-10-27", tz="UTC")
    t1 = t0 + pd.DateOffset(months=9)
    purge = pd.Timedelta(minutes=5 * cfg["model"]["walk_forward_purge_bars"])
    v0 = t1 + purge
    v1 = v0 + pd.DateOffset(months=1)

    train_df = df[(df["_ts"] >= t0) & (df["_ts"] < t1)]
    val_df = df[(df["_ts"] >= v0) & (df["_ts"] < v1)]
    feat_cols = feature_columns(df)

    X_tr = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_tr_real = train_df["label"].to_numpy(dtype=np.int32)
    X_val = val_df[feat_cols].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy(dtype=np.int32)

    log.info(
        f"fold-1 geometry: train {t0.date()}->{t1.date()} ({len(train_df):,}) "
        f"val {v0.date()}->{v1.date()} ({len(val_df):,})"
    )

    # Baselines from val label distribution.
    val_prior = np.bincount(y_val, minlength=n_classes) / len(y_val)
    max_prior = float(val_prior.max())
    prior_entropy = float(-(val_prior * np.log(np.maximum(val_prior, 1e-12))).sum())
    log.info(
        f"val label distribution: {dict(zip(['LONG','SHORT','NEUTRAL'], val_prior.round(4)))}"
    )
    log.info(f"  => accuracy ceiling for a prior-only predictor: {max_prior:.4f}")
    log.info(f"  => log-loss floor for a prior-only predictor:   {prior_entropy:.4f}")

    # Shuffle ONLY train labels. This kills feature->label correlation in training.
    rng = np.random.default_rng(seed)
    y_tr_perm = rng.permutation(y_tr_real)

    params = dict(cfg["model"]["params"])
    num_boost_round = cfg["model"]["num_boost_round"]
    early_stopping = cfg["model"]["early_stopping_rounds"]

    log.info(f"training on PERMUTED labels (seed={seed})...")
    train_data = lgb.Dataset(X_tr, label=y_tr_perm, feature_name=feat_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feat_cols)
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
    proba = booster.predict(X_val, num_iteration=booster.best_iteration)
    pred = proba.argmax(axis=1)
    ll = float(log_loss(y_val, proba, labels=list(range(n_classes))))
    acc = float((pred == y_val).mean())
    pred_dist = np.bincount(pred, minlength=n_classes) / len(pred)

    log.info("=== PERMUTED-LABEL RESULTS ===")
    log.info(f"  best_iteration: {booster.best_iteration}")
    log.info(f"  val log_loss:   {ll:.4f}   (prior floor: {prior_entropy:.4f})")
    log.info(f"  val accuracy:   {acc:.4f}   (prior ceiling: {max_prior:.4f})")
    log.info(f"  prediction distribution: LONG={pred_dist[0]:.3f} SHORT={pred_dist[1]:.3f} NEUTRAL={pred_dist[2]:.3f}")
    log.info(f"  (val real dist:         LONG={val_prior[0]:.3f} SHORT={val_prior[1]:.3f} NEUTRAL={val_prior[2]:.3f})")

    # Reference: real-label fold-1 from walk-forward was logloss=0.962, acc=0.494.
    log.info("=== INTERPRETATION ===")
    eps_ll = 0.02
    eps_acc = 0.02
    if ll >= prior_entropy - eps_ll and acc <= max_prior + eps_acc:
        log.info("  PASS: permuted-label metrics match prior baselines within noise.")
        log.info("        Pipeline is clean (no feature-to-label leakage).")
        log.info("        Real-fold gain of ~0.04 log-loss over prior is genuine, albeit weak.")
    else:
        log.warning("  FAIL: permuted-label metrics BEAT the prior baseline.")
        log.warning("        At least one feature leaks label information. Hunt for the source:")
        log.warning("        candidates: forward-looking rolling windows, label-derived cols,")
        log.warning("        off-by-one shifts in 1H merge, label/pnl_pct accidentally kept as feature.")


if __name__ == "__main__":
    main()
