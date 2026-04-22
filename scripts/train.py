"""Train LightGBM on a features parquet with inverse-frequency sample weights.

Implements spec §9.4 ("class_weight='balanced'" guidance) and §10.1 walk-forward
purge. This is a SMOKE trainer: 80/20 time-ordered split on a single features
parquet with a 48-bar purge gap. Walk-forward + Optuna (§10.2) come later.

Usage:
    python -m scripts.train --symbol BTC
    python -m scripts.train --symbol BTC --no-weights     # ablation run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Columns that are NOT features. Everything else in the parquet is a feature.
NON_FEATURE_COLS = {
    "timestamp",
    "_ts",  # synthetic datetime used only for slicing
    "open",
    "high",
    "low",
    "close",
    "volume",
    "label",
    "holding_bars",
    "exit_price",
    "pnl_pct",
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def time_split(
    df: pd.DataFrame, train_frac: float, purge_bars: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ordered split with a purge gap between train and val (spec §10.1)."""
    n = len(df)
    train_end = int(n * train_frac)
    val_start = train_end + purge_bars
    train = df.iloc[:train_end]
    val = df.iloc[val_start:]
    return train, val


def balanced_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-row inverse-frequency weight, matching sklearn's class_weight='balanced'.

        w[c] = N / (n_classes * count[c])
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    class_w = len(y) / (n_classes * np.maximum(counts, 1.0))
    return class_w[y]


def train(symbol: str, use_weights: bool = True) -> dict:
    cfg = load_config()
    classes = cfg["labeling"]["classes"]
    n_classes = len(classes)
    inv_classes = {v: k for k, v in classes.items()}
    class_names = [inv_classes[i] for i in range(n_classes)]
    purge = cfg["model"]["walk_forward_purge_bars"]
    threshold = cfg["model"]["signal_threshold"]

    features_dir = PROJECT_ROOT / cfg["features"]["output_dir"]
    parquet_path = features_dir / f"{symbol}_features.parquet"
    print(f"[train] loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[train] shape={df.shape}  label counts:\n{df['label'].value_counts().sort_index()}")

    # Labeler emits -1 for the trailing max_holding_bars rows with no forward
    # window. Drop them defensively even though builder already should.
    df = df[df["label"] >= 0].reset_index(drop=True)

    feat_cols = feature_columns(df)
    print(f"[train] {len(feat_cols)} feature columns")

    train_df, val_df = time_split(df, train_frac=0.8, purge_bars=purge)
    print(f"[train] train={len(train_df)} val={len(val_df)} purge_bars={purge}")

    X_tr = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_tr = train_df["label"].to_numpy(dtype=np.int32)
    X_val = val_df[feat_cols].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy(dtype=np.int32)

    if use_weights:
        w_tr = balanced_weights(y_tr, n_classes)
        w_val = balanced_weights(y_val, n_classes)
        # Report the class weight schedule used.
        counts_tr = np.bincount(y_tr, minlength=n_classes)
        class_w = len(y_tr) / (n_classes * np.maximum(counts_tr, 1.0))
        print("[train] class weights (train):", dict(zip(class_names, class_w.round(3))))
    else:
        w_tr = None
        w_val = None
        print("[train] no class weighting (ablation)")

    params = dict(cfg["model"]["params"])
    num_boost_round = cfg["model"]["num_boost_round"]
    early_stopping = cfg["model"]["early_stopping_rounds"]

    train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feat_cols)
    val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data, feature_name=feat_cols)

    print(f"[train] fitting LightGBM: num_boost_round={num_boost_round} early_stopping={early_stopping}")
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping, first_metric_only=False),
            lgb.log_evaluation(period=50),
        ],
    )

    # Predictions on val set — unweighted metrics for the REAL distribution.
    proba_val = booster.predict(X_val, num_iteration=booster.best_iteration)
    pred_val = proba_val.argmax(axis=1)

    ll = log_loss(y_val, proba_val, labels=list(range(n_classes)))
    acc = (pred_val == y_val).mean()
    cm = confusion_matrix(y_val, pred_val, labels=list(range(n_classes)))
    report = classification_report(
        y_val, pred_val, labels=list(range(n_classes)), target_names=class_names, digits=4, zero_division=0
    )

    # Confidence / threshold-firing diagnostics.
    max_proba = proba_val.max(axis=1)
    fire_mask = max_proba >= threshold
    fire_rate = fire_mask.mean()
    directional_fire_mask = fire_mask & (pred_val != classes["NEUTRAL"])
    directional_fire_rate = directional_fire_mask.mean()

    # Hit rate among fired directional signals: was the fired class the true class?
    if directional_fire_mask.any():
        directional_hit = (pred_val[directional_fire_mask] == y_val[directional_fire_mask]).mean()
    else:
        directional_hit = float("nan")

    print("\n=== VAL METRICS (unweighted, real distribution) ===")
    print(f"log_loss: {ll:.4f}")
    print(f"accuracy: {acc:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names]))
    print("\nclassification report:")
    print(report)

    print(f"confidence mean={max_proba.mean():.4f}  std={max_proba.std():.4f}")
    print(f"fire_rate (max_proba>={threshold}): {fire_rate:.4%} of val bars")
    print(f"directional fire rate (non-NEUTRAL, conf>={threshold}): {directional_fire_rate:.4%} of val bars")
    print(f"directional hit rate | fired: {directional_hit:.4%}")

    out_dir = PROJECT_ROOT / "model" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_weighted" if use_weights else "_unweighted"
    model_path = out_dir / f"{symbol}_lgbm{suffix}.txt"
    booster.save_model(str(model_path))

    metrics = {
        "symbol": symbol,
        "use_weights": use_weights,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "n_features": len(feat_cols),
        "best_iteration": booster.best_iteration,
        "log_loss": float(ll),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
        "fire_rate": float(fire_rate),
        "directional_fire_rate": float(directional_fire_rate),
        "directional_hit_rate": float(directional_hit) if not np.isnan(directional_hit) else None,
        "confidence_mean": float(max_proba.mean()),
        "confidence_std": float(max_proba.std()),
    }
    metrics_path = out_dir / f"{symbol}_lgbm{suffix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[train] saved model -> {model_path}")
    print(f"[train] saved metrics -> {metrics_path}")

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--no-weights", action="store_true", help="ablation: train without class weights")
    args = ap.parse_args()
    train(args.symbol, use_weights=not args.no_weights)


if __name__ == "__main__":
    main()
