"""Out-of-time holdout evaluation (spec §2.6 step 3).

The tail 2026-03-29 → 2026-04-21 is untouched by:
  - Optuna tuning (folds 1/4/7 val windows: 2025-07, 2025-10, 2026-01)
  - SHAP computation (used 8 fold val sets, ending 2026-03-28)
  - Feature trim decision (derived from SHAP aggregate)
  - Walk-forward training (last fold val ends 2026-03-28)

Apply fold-8 booster (trained on most-recent regime) to the tail. Report
raw + calibrated log_loss, directional fire/hit at chosen thresholds, and
LONG vs SHORT precision. This is the clean "would-have-traded" reference.

Usage:
    python -m scripts.oot_eval --symbol BTC --tag tuned_trim120 \\
        --thresholds 0.55 0.60 0.65 0.70
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("oot_eval")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def apply_calibrators(proba: np.ndarray, calibrators: list) -> np.ndarray:
    out = np.zeros_like(proba)
    for c in range(proba.shape[1]):
        out[:, c] = calibrators[c].transform(proba[:, c])
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def eval_at_threshold(y: np.ndarray, proba: np.ndarray, t: float,
                      classes: dict) -> dict:
    long_idx, short_idx, neutral_idx = classes["LONG"], classes["SHORT"], classes["NEUTRAL"]
    max_p = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    fire = max_p >= t
    dir_fire = fire & (pred != neutral_idx)
    long_fire = fire & (pred == long_idx)
    short_fire = fire & (pred == short_idx)
    return {
        "threshold": float(t),
        "fire_count": int(fire.sum()),
        "fire_rate": float(fire.mean()),
        "dir_fire_count": int(dir_fire.sum()),
        "dir_fire_rate": float(dir_fire.mean()),
        "dir_hit_rate": (float((pred[dir_fire] == y[dir_fire]).mean())
                         if dir_fire.any() else None),
        "long_fire_count": int(long_fire.sum()),
        "long_hit_rate": (float((y[long_fire] == long_idx).mean())
                          if long_fire.any() else None),
        "short_fire_count": int(short_fire.sum()),
        "short_hit_rate": (float((y[short_fire] == short_idx).mean())
                           if short_fire.any() else None),
    }


def _fmt_hit(v: float | None) -> str:
    return f"{v*100:6.2f}" if v is not None else "   n/a"


def _log_threshold_table(name: str, rows: list[dict]) -> None:
    log.info(f"=== OOT @ THRESHOLDS ({name}) ===")
    log.info(
        f"  {'thr':>4} {'fire_n':>7} {'fire%':>7} {'dir_hit%':>9} "
        f"{'L_n':>5} {'L_hit%':>8} {'S_n':>5} {'S_hit%':>8}"
    )
    for d in rows:
        log.info(
            f"  {d['threshold']:>4.2f} {d['fire_count']:>7} "
            f"{d['fire_rate']*100:>6.2f} "
            f"{_fmt_hit(d['dir_hit_rate']):>9} "
            f"{d['long_fire_count']:>5} {_fmt_hit(d['long_hit_rate']):>8} "
            f"{d['short_fire_count']:>5} {_fmt_hit(d['short_hit_rate']):>8}"
        )


def run(symbol: str, cfg: dict, tag: str, thresholds: list[float],
        oot_start: str = "2026-03-29") -> dict:
    suffix = f"_{tag}" if tag else ""
    model_dir = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf{suffix}"
    local_sel = model_dir / "selected_features.json"
    base_sel = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf" / "selected_features.json"
    sel_path = local_sel if local_sel.exists() else base_sel
    selected = json.loads(sel_path.read_text())["features"]
    log.info(f"using {len(selected)} features from {sel_path}")

    features_dir = PROJECT_ROOT / cfg["features"]["output_dir"]
    df = pd.read_parquet(features_dir / f"{symbol}_features.parquet")
    df = df[df["label"] >= 0].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    oot_ts = pd.Timestamp(oot_start, tz="UTC")
    oot = df[df["_ts"] >= oot_ts]
    if len(oot) == 0:
        raise RuntimeError(f"no rows at or after {oot_start}")
    log.info(
        f"OOT slice: {len(oot):,} rows  "
        f"{oot['_ts'].iloc[0]} -> {oot['_ts'].iloc[-1]}"
    )

    classes = cfg["labeling"]["classes"]
    n_classes = len(classes)
    class_names = ["LONG", "SHORT", "NEUTRAL"]

    y = oot["label"].to_numpy(dtype=np.int32)
    X = oot[selected].to_numpy(dtype=np.float32)

    prior = np.bincount(y, minlength=n_classes) / len(y)
    prior_ent = float(-(prior * np.log(np.maximum(prior, 1e-12))).sum())
    log.info(
        f"OOT label dist: "
        f"LONG={prior[0]*100:.1f}% SHORT={prior[1]*100:.1f}% NEUTRAL={prior[2]*100:.1f}%"
    )
    log.info(f"OOT prior-entropy floor: {prior_ent:.4f}")

    fold_files = sorted(
        model_dir.glob("fold_*.txt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not fold_files:
        raise RuntimeError(f"no fold_*.txt found in {model_dir}")
    last_fold = fold_files[-1]
    log.info(f"loading latest fold booster: {last_fold.name}")
    booster = lgb.Booster(model_file=str(last_fold))
    raw = booster.predict(X)

    cal = None
    calibrators_path = model_dir / "calibrators.pkl"
    if calibrators_path.exists():
        with open(calibrators_path, "rb") as f:
            calibrators = pickle.load(f)
        cal = apply_calibrators(raw, calibrators)

    raw_ll = float(log_loss(y, raw, labels=list(range(n_classes))))
    cal_ll = (float(log_loss(y, cal, labels=list(range(n_classes))))
              if cal is not None else None)
    log.info(
        f"OOT log_loss:  raw={raw_ll:.4f}  "
        f"calibrated={'-' if cal_ll is None else f'{cal_ll:.4f}'}  "
        f"prior={prior_ent:.4f}"
    )

    report = {
        "symbol": symbol,
        "tag": tag,
        "oot_start": oot_start,
        "oot_end": str(oot["_ts"].iloc[-1]),
        "n_samples": int(len(oot)),
        "label_distribution": {class_names[c]: float(prior[c]) for c in range(n_classes)},
        "prior_entropy": prior_ent,
        "raw_log_loss": raw_ll,
        "calibrated_log_loss": cal_ll,
        "thresholds_raw": [eval_at_threshold(y, raw, t, classes) for t in thresholds],
        "thresholds_calibrated": (
            [eval_at_threshold(y, cal, t, classes) for t in thresholds]
            if cal is not None else None
        ),
    }

    _log_threshold_table("raw", report["thresholds_raw"])
    if report["thresholds_calibrated"] is not None:
        _log_threshold_table("calibrated", report["thresholds_calibrated"])

    out_path = model_dir / "oot_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"saved -> {out_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--tag", default="tuned_trim120")
    ap.add_argument("--oot-start", default="2026-03-29")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.55, 0.60, 0.65, 0.70])
    args = ap.parse_args()
    cfg = load_config()
    run(args.symbol, cfg, tag=args.tag, thresholds=args.thresholds,
        oot_start=args.oot_start)


if __name__ == "__main__":
    main()
