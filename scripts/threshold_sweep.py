"""Threshold sweep (spec §2.6 step 1).

For each walk-forward fold × threshold, compute:
  fire_rate, directional_fire_rate, directional_hit_rate,
  per-side breakdown: LONG fire-count / hit|fire, SHORT fire-count / hit|fire.

Goal: find the elbow where hit-rate plateaus while fire-rate drops. The
config default `signal_threshold: 0.65` was never validated.

If --calibrators is given, sweep runs on calibrated probabilities instead.

Usage:
    python -m scripts.threshold_sweep --symbol BTC --tag tuned_trim120
    python -m scripts.threshold_sweep --symbol BTC --tag tuned_trim120 \\
        --calibrators model/models/BTC_wf_tuned_trim120/calibrators.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from model.trainer import feature_columns, make_folds
from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("threshold_sweep")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def apply_calibrators(proba: np.ndarray, calibrators: list) -> np.ndarray:
    out = np.zeros_like(proba)
    for c in range(proba.shape[1]):
        out[:, c] = calibrators[c].transform(proba[:, c])
    row = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row, 1e-12)


def sweep_fold(y_val: np.ndarray, proba: np.ndarray,
               long_idx: int, short_idx: int, neutral_idx: int) -> list[dict]:
    rows = []
    max_proba = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    for t in THRESHOLDS:
        fire = max_proba >= t
        dir_fire = fire & (pred != neutral_idx)

        def _side(side_idx: int):
            sf = fire & (pred == side_idx)
            n = int(sf.sum())
            hit = float((y_val[sf] == side_idx).mean()) if n > 0 else None
            return n, hit

        ln, lh = _side(long_idx)
        sn, sh = _side(short_idx)
        dh = (float((pred[dir_fire] == y_val[dir_fire]).mean())
              if dir_fire.any() else None)
        rows.append({
            "threshold": t,
            "fire_rate": float(fire.mean()),
            "directional_fire_rate": float(dir_fire.mean()),
            "directional_hit_rate": dh,
            "long_fire_count": ln,
            "long_hit_rate": lh,
            "short_fire_count": sn,
            "short_hit_rate": sh,
        })
    return rows


def _pooled_side_hit(df: pd.DataFrame, side: str) -> tuple[float | None, int]:
    n = int(df[f"{side}_fire_count"].sum())
    if n == 0:
        return None, 0
    # Per-fold hit_count = fire_count × hit_rate; pooled = sum/sum.
    hits = (df[f"{side}_fire_count"] * df[f"{side}_hit_rate"].fillna(0.0)).sum()
    return float(hits / n), n


def run(symbol: str, cfg: dict, tag: str,
        calibrators_path: Path | None = None) -> pd.DataFrame:
    suffix = f"_{tag}" if tag else ""
    model_dir = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf{suffix}"
    if not model_dir.exists():
        raise FileNotFoundError(f"missing {model_dir}")

    sel_path = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf" / "selected_features.json"
    selected = None
    if sel_path.exists():
        selected = json.loads(sel_path.read_text())["features"]
        log.info(f"using {len(selected)} selected features from {sel_path}")

    features_dir = PROJECT_ROOT / cfg["features"]["output_dir"]
    parquet = features_dir / f"{symbol}_features.parquet"
    df = pd.read_parquet(parquet)
    df = df[df["label"] >= 0].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    feat_cols = selected if selected else feature_columns(df)

    classes = cfg["labeling"]["classes"]
    long_idx, short_idx, neutral_idx = classes["LONG"], classes["SHORT"], classes["NEUTRAL"]
    wf = cfg["walk_forward"]
    purge = cfg["model"]["walk_forward_purge_bars"]
    folds = make_folds(df["_ts"], wf["train_months"], wf["val_months"],
                       wf["step_months"], purge)

    calibrators = None
    if calibrators_path is not None:
        with open(calibrators_path, "rb") as f:
            calibrators = pickle.load(f)
        log.info(f"loaded calibrators from {calibrators_path}")

    all_rows: list[dict] = []
    for i, (_, _, v0, v1) in enumerate(folds, start=1):
        bp = model_dir / f"fold_{i}.txt"
        if not bp.exists():
            log.warning(f"[fold {i}] missing {bp}; skip")
            continue
        booster = lgb.Booster(model_file=str(bp))
        val_df = df[(df["_ts"] >= v0) & (df["_ts"] < v1)]
        X = val_df[feat_cols].to_numpy(dtype=np.float32)
        y = val_df["label"].to_numpy(dtype=np.int32)
        proba = booster.predict(X)
        if calibrators is not None:
            proba = apply_calibrators(proba, calibrators)
        rows = sweep_fold(y, proba, long_idx, short_idx, neutral_idx)
        for r in rows:
            r["fold"] = i
            r["val_start"] = str(v0.date())
            r["val_end"] = str(v1.date())
        all_rows.extend(rows)
        log.info(f"[fold {i}] val={v0.date()}->{v1.date()} n={len(y):,}")

    df_sweep = pd.DataFrame(all_rows)

    # Per-threshold aggregate across folds.
    agg_rows = []
    for t in THRESHOLDS:
        sub = df_sweep[df_sweep["threshold"] == t]
        lh, ln = _pooled_side_hit(sub, "long")
        sh, sn = _pooled_side_hit(sub, "short")
        agg_rows.append({
            "threshold": t,
            "fire_rate_mean": float(sub["fire_rate"].mean()),
            "fire_rate_std": float(sub["fire_rate"].std(ddof=0)),
            "dir_hit_mean": float(sub["directional_hit_rate"].mean()),
            "dir_hit_std": float(sub["directional_hit_rate"].std(ddof=0)),
            "n_folds_with_fires": int(sub["directional_hit_rate"].notna().sum()),
            "long_fire_total": ln,
            "long_hit_pooled": lh,
            "short_fire_total": sn,
            "short_hit_pooled": sh,
        })
    agg = pd.DataFrame(agg_rows)

    suffix_cal = "_calibrated" if calibrators_path is not None else "_raw"
    out_csv = model_dir / f"threshold_sweep{suffix_cal}.csv"
    out_summary = model_dir / f"threshold_sweep_summary{suffix_cal}.csv"
    df_sweep.to_csv(out_csv, index=False)
    agg.to_csv(out_summary, index=False)
    log.info(f"saved -> {out_csv}")
    log.info(f"saved -> {out_summary}")

    label = "CALIBRATED" if calibrators_path is not None else "RAW"
    log.info(f"=== THRESHOLD SWEEP ({label}, mean±std across 8 folds) ===")
    log.info(f"  {'thr':>4} {'fire%':>7} {'hit%':>8} "
             f"{'L_n':>6} {'L_hit%':>8} {'S_n':>6} {'S_hit%':>8}  folds_firing")
    for _, r in agg.iterrows():
        lh = f"{r['long_hit_pooled']*100:6.2f}" if r["long_hit_pooled"] is not None else "   n/a"
        sh = f"{r['short_hit_pooled']*100:6.2f}" if r["short_hit_pooled"] is not None else "   n/a"
        dh = f"{r['dir_hit_mean']*100:5.2f}±{r['dir_hit_std']*100:4.1f}" if not np.isnan(r["dir_hit_mean"]) else "    n/a"
        log.info(
            f"  {r['threshold']:>4.2f} "
            f"{r['fire_rate_mean']*100:>5.2f}±{r['fire_rate_std']*100:<4.1f} "
            f"{dh:>8} "
            f"{int(r['long_fire_total']):>6} {lh:>8} "
            f"{int(r['short_fire_total']):>6} {sh:>8}  "
            f"{int(r['n_folds_with_fires'])}/8"
        )
    return df_sweep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--tag", default="tuned_trim120")
    ap.add_argument("--calibrators", default=None,
                    help="path to calibrators.pkl (enables calibrated sweep)")
    args = ap.parse_args()
    cfg = load_config()
    calib_path = Path(args.calibrators) if args.calibrators else None
    run(args.symbol, cfg, tag=args.tag, calibrators_path=calib_path)


if __name__ == "__main__":
    main()
