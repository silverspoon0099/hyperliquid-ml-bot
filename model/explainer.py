"""SHAP feature importance (spec §2.4, §7.3).

TreeExplainer on each tuned walk-forward fold's val set. Importance per
feature = mean |shap| over (samples, classes). We aggregate across folds
so the ranking is stable, not overfit to a single regime.

Val rows are subsampled per fold to keep memory reasonable — shap output is
(n_samples, n_features, n_classes) so full val × 269 × 3 × 8 folds = >50M
floats. 2k rows/fold is enough for stable top-N rankings.

Outputs (under model/models/{SYMBOL}_wf_tuned/):
    shap_importance.csv   — full table (rank, feature, mean_abs_shap, share, cum_share)
    shap_importance.json  — same + meta (n_folds_used, sample_per_fold)

Used by spec §2.5 (feature trim — target 120-150 features).

Usage:
    python -m model.explainer --symbol BTC
    python -m model.explainer --symbol BTC --sample-per-fold 3000
    python -m model.explainer --symbol BTC --tag tuned  # defaults
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from model.trainer import feature_columns, make_folds
from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("explainer")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(symbol: str, cfg: dict, sample_per_fold: int = 2000,
        tag: str = "tuned", seed: int = 42) -> pd.DataFrame:
    suffix = f"_{tag}" if tag else ""
    model_dir = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf{suffix}"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"missing {model_dir} — run trainer with --tag {tag!r} first"
        )

    features_dir = PROJECT_ROOT / cfg["features"]["output_dir"]
    parquet = features_dir / f"{symbol}_features.parquet"
    log.info(f"loading {parquet}")
    df = pd.read_parquet(parquet)
    df = df[df["label"] >= 0].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    feat_cols = feature_columns(df)
    log.info(f"{len(df):,} rows, {len(feat_cols)} features")

    wf = cfg["walk_forward"]
    purge = cfg["model"]["walk_forward_purge_bars"]
    folds = make_folds(df["_ts"], wf["train_months"], wf["val_months"],
                       wf["step_months"], purge)
    log.info(f"{len(folds)} walk-forward folds; sample_per_fold={sample_per_fold}")

    rng = np.random.default_rng(seed)
    per_fold_imp: list[np.ndarray] = []
    used_folds: list[int] = []

    for i, (_, _, v0, v1) in enumerate(folds, start=1):
        booster_path = model_dir / f"fold_{i}.txt"
        if not booster_path.exists():
            log.warning(f"[fold {i}] no model at {booster_path}; skipping")
            continue
        booster = lgb.Booster(model_file=str(booster_path))

        val_df = df[(df["_ts"] >= v0) & (df["_ts"] < v1)]
        X_val = val_df[feat_cols].to_numpy(dtype=np.float32)
        if len(X_val) == 0:
            log.warning(f"[fold {i}] empty val set; skipping")
            continue

        if len(X_val) > sample_per_fold:
            idx = rng.choice(len(X_val), size=sample_per_fold, replace=False)
            X_sample = X_val[idx]
        else:
            X_sample = X_val

        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(X_sample)
        # shap>=0.42 returns ndarray (n_samples, n_features, n_classes) for
        # LightGBM multiclass; older versions return a list of per-class arrays.
        arr = np.stack(shap_vals, axis=-1) if isinstance(shap_vals, list) else shap_vals
        imp = np.abs(arr).mean(axis=(0, 2))  # -> (n_features,)
        per_fold_imp.append(imp)
        used_folds.append(i)
        top_idx = int(imp.argmax())
        log.info(
            f"[fold {i}] val={v0.date()}->{v1.date()} "
            f"sampled={len(X_sample):,}/{len(X_val):,}  "
            f"top feature: {feat_cols[top_idx]} ({imp[top_idx]:.4f})"
        )

    if not per_fold_imp:
        raise RuntimeError("no folds produced SHAP values")

    stacked = np.vstack(per_fold_imp)  # (n_folds, n_features)
    mean_imp = stacked.mean(axis=0)
    std_imp = stacked.std(axis=0)

    out = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": mean_imp,
        "std_abs_shap": std_imp,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = float(out["mean_abs_shap"].sum())
    out["share"] = out["mean_abs_shap"] / max(total, 1e-12)
    out["cum_share"] = out["share"].cumsum()
    out.insert(0, "rank", range(1, len(out) + 1))

    csv_path = model_dir / "shap_importance.csv"
    json_path = model_dir / "shap_importance.json"
    out.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({
            "symbol": symbol,
            "tag": tag,
            "n_folds_used": len(per_fold_imp),
            "used_folds": used_folds,
            "sample_per_fold": sample_per_fold,
            "features": out.to_dict(orient="records"),
        }, f, indent=2)
    log.info(f"saved -> {csv_path}")
    log.info(f"saved -> {json_path}")

    log.info("=== TOP 20 features by mean |SHAP| (avg across folds) ===")
    for _, row in out.head(20).iterrows():
        log.info(
            f"  {int(row['rank']):3d}  {row['feature']:40s}  "
            f"mean|shap|={row['mean_abs_shap']:.5f}  "
            f"share={row['share']*100:5.2f}%  "
            f"cum={row['cum_share']*100:5.2f}%"
        )

    log.info("=== CUMULATIVE COVERAGE (for spec §2.5 trim) ===")
    for k in [50, 100, 120, 150, 200]:
        if k <= len(out):
            cum = float(out.loc[k - 1, "cum_share"])
            log.info(f"  top {k:3d} features cover {cum*100:5.2f}% of total |SHAP|")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--tag", default="tuned",
                    help="model subdir suffix (empty = baseline)")
    ap.add_argument("--sample-per-fold", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config()
    run(args.symbol, cfg,
        sample_per_fold=args.sample_per_fold, tag=args.tag, seed=args.seed)


if __name__ == "__main__":
    main()
