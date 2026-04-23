"""Probability calibration on OOF walk-forward preds (spec §2.6 step 2).

LightGBM multiclass probs are softmaxed leaf scores — overconfident on
easy examples, underconfident on borderline. Since we threshold on
max_proba, miscalibration distorts the fire/hit trade directly.

Method:
  1. Concat each fold's val predictions into a true OOF stack (folds are
     time-sequential, non-overlapping — no leakage).
  2. Per-class 1-vs-rest IsotonicRegression (target = one-hot).
  3. At apply time, transform each class then row-renormalize.

Outputs:
  model/models/{SYMBOL}_wf_{tag}/calibrators.pkl
  model/models/{SYMBOL}_wf_{tag}/calibration_report.json

Usage:
    python -m model.calibration --symbol BTC --tag tuned_trim120
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from model.trainer import make_folds
from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("calibration")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fit(symbol: str, cfg: dict, tag: str, max_train_date: str | None = None) -> dict:
    suffix = f"_{tag}" if tag else ""
    model_dir = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf{suffix}"
    local_sel = model_dir / "selected_features.json"
    base_sel = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf" / "selected_features.json"
    sel_path = local_sel if local_sel.exists() else base_sel
    if not sel_path.exists():
        raise FileNotFoundError(
            f"missing selected_features.json in {model_dir} or base wf dir"
        )
    selected = json.loads(sel_path.read_text())["features"]
    log.info(f"using {len(selected)} features from {sel_path}")

    features_dir = PROJECT_ROOT / cfg["features"]["output_dir"]
    df = pd.read_parquet(features_dir / f"{symbol}_features.parquet")
    df = df[df["label"] >= 0].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    if max_train_date is not None:
        cutoff = pd.Timestamp(max_train_date, tz="UTC")
        df = df[df["_ts"] < cutoff].reset_index(drop=True)
        log.info(f"max_train_date={max_train_date}: truncated to {len(df):,} rows")

    classes = cfg["labeling"]["classes"]
    n_classes = len(classes)
    class_names = ["LONG", "SHORT", "NEUTRAL"]  # matches index 0/1/2

    wf = cfg["walk_forward"]
    purge = cfg["model"]["walk_forward_purge_bars"]
    folds = make_folds(df["_ts"], wf["train_months"], wf["val_months"],
                       wf["step_months"], purge)

    oof_proba_parts: list[np.ndarray] = []
    oof_y_parts: list[np.ndarray] = []
    for i, (_, _, v0, v1) in enumerate(folds, start=1):
        bp = model_dir / f"fold_{i}.txt"
        if not bp.exists():
            log.warning(f"[fold {i}] missing {bp}; skip")
            continue
        booster = lgb.Booster(model_file=str(bp))
        val_df = df[(df["_ts"] >= v0) & (df["_ts"] < v1)]
        X = val_df[selected].to_numpy(dtype=np.float32)
        y = val_df["label"].to_numpy(dtype=np.int32)
        p = booster.predict(X)
        oof_proba_parts.append(p)
        oof_y_parts.append(y)
        log.info(f"[fold {i}] OOF {len(y):,} rows")

    oof_proba = np.vstack(oof_proba_parts)
    oof_y = np.concatenate(oof_y_parts)
    log.info(f"stacked OOF: {oof_proba.shape}")

    raw_ll = float(log_loss(oof_y, oof_proba, labels=list(range(n_classes))))
    raw_brier = {
        class_names[c]: float(brier_score_loss(
            (oof_y == c).astype(int), oof_proba[:, c]
        )) for c in range(n_classes)
    }

    calibrators: list[IsotonicRegression] = []
    for c in range(n_classes):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_proba[:, c], (oof_y == c).astype(float))
        calibrators.append(iso)
        log.info(f"  fit isotonic for {class_names[c]}")

    calibrated = np.zeros_like(oof_proba)
    for c in range(n_classes):
        calibrated[:, c] = calibrators[c].transform(oof_proba[:, c])
    calibrated = calibrated / np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-12)

    cal_ll = float(log_loss(oof_y, calibrated, labels=list(range(n_classes))))
    cal_brier = {
        class_names[c]: float(brier_score_loss(
            (oof_y == c).astype(int), calibrated[:, c]
        )) for c in range(n_classes)
    }

    log.info("=== CALIBRATION REPORT ===")
    log.info(f"  OOF log_loss:  raw={raw_ll:.4f}  cal={cal_ll:.4f}  Δ={cal_ll - raw_ll:+.4f}")
    for c in class_names:
        log.info(
            f"  Brier[{c:>7s}]: raw={raw_brier[c]:.4f}  cal={cal_brier[c]:.4f}  "
            f"Δ={cal_brier[c] - raw_brier[c]:+.4f}"
        )

    with open(model_dir / "calibrators.pkl", "wb") as f:
        pickle.dump(calibrators, f)
    report = {
        "symbol": symbol,
        "tag": tag,
        "n_oof_rows": int(len(oof_y)),
        "raw_log_loss": raw_ll,
        "calibrated_log_loss": cal_ll,
        "raw_brier": raw_brier,
        "calibrated_brier": cal_brier,
    }
    with open(model_dir / "calibration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"saved -> {model_dir/'calibrators.pkl'}")
    log.info(f"saved -> {model_dir/'calibration_report.json'}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--tag", default="tuned_trim120")
    ap.add_argument(
        "--max-train-date", default=None,
        help="must match the same flag used in trainer; recomputes identical folds",
    )
    args = ap.parse_args()
    cfg = load_config()
    fit(args.symbol, cfg, tag=args.tag, max_train_date=args.max_train_date)


if __name__ == "__main__":
    main()
