"""Binary walk-forward trainer (LONG-vs-rest or SHORT-vs-rest).

Separate model per direction: positive class = predicted side, negative class =
everything else (including NEUTRAL + the opposite side). This removes the
multiclass capacity wasted on modelling NEUTRAL as its own class; each model
answers one clean binary question and can produce calibrated P(positive)
directly.

Usage:
    python -m model.binary_trainer --symbol BTC --side LONG --tag sym_v2_bin_long \\
        --max-train-date 2025-12-31 --best-params model/models/BTC_wf/best_params.json \\
        --feature-list model/models/BTC_wf_sym_v2/selected_features.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_recall_fscore_support

from model.trainer import make_folds, feature_columns
from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("binary_trainer")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BinaryFoldMetrics:
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_rows: int
    val_rows: int
    pos_rate_train: float
    pos_rate_val: float
    best_iteration: int
    log_loss: float
    auc: float | None
    threshold: float
    fire_rate: float
    precision_at_thr: float | None
    recall_at_thr: float | None


def _side_to_label_idx(side: str, classes: dict) -> int:
    side = side.upper()
    if side not in classes:
        raise ValueError(f"--side must be LONG or SHORT; got {side}")
    if side == "NEUTRAL":
        raise ValueError("binary classifier is for LONG or SHORT, not NEUTRAL")
    return classes[side]


def binary_walk_forward(symbol: str, cfg: dict, side: str, tag: str,
                        best_params_path: Path | None,
                        feature_list_path: Path | None,
                        max_train_date: str | None) -> dict:
    classes = cfg["labeling"]["classes"]
    pos_idx = _side_to_label_idx(side, classes)
    threshold = cfg["model"]["signal_threshold"]
    purge_bars = cfg["model"]["walk_forward_purge_bars"]
    wf = cfg["walk_forward"]

    features_dir = PROJECT_ROOT / cfg["features"]["output_dir"]
    parquet_path = features_dir / f"{symbol}_features.parquet"
    log.info(f"loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df[df["label"] >= 0].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    if max_train_date is not None:
        cutoff = pd.Timestamp(max_train_date, tz="UTC")
        n_before = len(df)
        df = df[df["_ts"] < cutoff].reset_index(drop=True)
        log.info(f"max_train_date={max_train_date}: truncated {n_before:,} -> {len(df):,}")

    feat_cols = feature_columns(df)
    if feature_list_path is not None:
        selected = json.loads(Path(feature_list_path).read_text())["features"]
        feat_cols = selected
        log.info(f"restricted to {len(feat_cols)} features from {feature_list_path}")

    log.info(f"BINARY classifier: side={side} (pos_idx={pos_idx}) "
             f"{len(df):,} rows, span {df['_ts'].iloc[0]} -> {df['_ts'].iloc[-1]}")
    df["_y_bin"] = (df["label"].to_numpy(dtype=np.int32) == pos_idx).astype(np.int32)
    log.info(f"overall positive rate: {df['_y_bin'].mean()*100:.2f}%")

    folds = make_folds(
        df["_ts"],
        train_months=wf["train_months"],
        val_months=wf["val_months"],
        step_months=wf["step_months"],
        purge_bars=purge_bars,
    )
    log.info(f"walk-forward: {len(folds)} folds")

    # Start from multiclass defaults but override objective to binary.
    params = dict(cfg["model"]["params"])
    params["objective"] = "binary"
    params["metric"] = "binary_logloss"
    params.pop("num_class", None)

    if best_params_path is not None:
        tuned = json.loads(Path(best_params_path).read_text()).get("best_params", {})
        # Tuned params from multiclass run are still useful (num_leaves, lr, reg, …).
        # Skip anything that only makes sense for multiclass (none right now).
        log.info(f"overriding with {len(tuned)} tuned params from {best_params_path}")
        for k, v in tuned.items():
            params[k] = v

    num_boost_round = cfg["model"]["num_boost_round"]
    early_stopping = cfg["model"]["early_stopping_rounds"]

    suffix = f"_{tag}" if tag else ""
    out_dir = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics: list[BinaryFoldMetrics] = []
    for i, (t0, t1, v0, v1) in enumerate(folds, start=1):
        train_df = df[(df["_ts"] >= t0) & (df["_ts"] < t1)]
        val_df = df[(df["_ts"] >= v0) & (df["_ts"] < v1)]

        X_tr = train_df[feat_cols].to_numpy(dtype=np.float32)
        y_tr = train_df["_y_bin"].to_numpy(dtype=np.int32)
        X_val = val_df[feat_cols].to_numpy(dtype=np.float32)
        y_val = val_df["_y_bin"].to_numpy(dtype=np.int32)

        pos_rate_train = float(y_tr.mean())
        pos_rate_val = float(y_val.mean())
        # Use scale_pos_weight to balance — binary version of the class_weight knob.
        neg_over_pos = (1 - pos_rate_train) / max(pos_rate_train, 1e-9)
        params_fold = dict(params)
        params_fold["scale_pos_weight"] = neg_over_pos

        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feat_cols)

        log.info(f"[fold {i}/{len(folds)}] train={t0.date()}->{t1.date()} ({len(train_df):,}) "
                 f"val={v0.date()}->{v1.date()} ({len(val_df):,})  "
                 f"pos_rate tr/val = {pos_rate_train*100:.1f}%/{pos_rate_val*100:.1f}%")

        booster = lgb.train(
            params_fold,
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
        ll = float(log_loss(y_val, proba, labels=[0, 1]))
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_val, proba))
        except Exception:
            auc = None

        fire_mask = proba >= threshold
        p_at_thr = None
        r_at_thr = None
        if fire_mask.any():
            p_at_thr = float(y_val[fire_mask].mean())  # precision = P(pos | fire)
        if y_val.sum() > 0:
            r_at_thr = float(fire_mask[y_val == 1].mean())  # recall

        fm = BinaryFoldMetrics(
            fold=i,
            train_start=str(t0.date()),
            train_end=str(t1.date()),
            val_start=str(v0.date()),
            val_end=str(v1.date()),
            train_rows=int(len(train_df)),
            val_rows=int(len(val_df)),
            pos_rate_train=pos_rate_train,
            pos_rate_val=pos_rate_val,
            best_iteration=int(booster.best_iteration),
            log_loss=ll,
            auc=auc,
            threshold=threshold,
            fire_rate=float(fire_mask.mean()),
            precision_at_thr=p_at_thr,
            recall_at_thr=r_at_thr,
        )
        fold_metrics.append(fm)
        booster.save_model(str(out_dir / f"fold_{i}.txt"))

        p_str = f"{p_at_thr*100:.2f}%" if p_at_thr is not None else "n/a"
        r_str = f"{r_at_thr*100:.2f}%" if r_at_thr is not None else "n/a"
        auc_str = f"{auc:.4f}" if auc is not None else "n/a"
        log.info(f"[fold {i}] best_iter={fm.best_iteration} logloss={ll:.4f} auc={auc_str} "
                 f"fire@{threshold:.2f}={fm.fire_rate:.3%} precision={p_str} recall={r_str}")

    summary = pd.DataFrame([asdict(f) for f in fold_metrics])[
        ["fold", "val_start", "val_end", "pos_rate_val",
         "log_loss", "auc", "fire_rate", "precision_at_thr", "recall_at_thr"]
    ]
    log.info("\n=== BINARY WF SUMMARY ===\n" + summary.to_string(index=False))

    def _mean(key: str) -> float:
        vals = [getattr(f, key) for f in fold_metrics if getattr(f, key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    log.info("=== AGGREGATES ===")
    for k in ("log_loss", "auc", "fire_rate", "precision_at_thr", "recall_at_thr"):
        log.info(f"  {k:24s} {_mean(k):.4f}")

    result = {
        "symbol": symbol,
        "side": side,
        "n_folds": len(fold_metrics),
        "threshold": threshold,
        "aggregate": {k: _mean(k) for k in
                      ("log_loss", "auc", "fire_rate", "precision_at_thr", "recall_at_thr")},
        "folds": [asdict(f) for f in fold_metrics],
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    # Copy the feature list into the model dir so downstream scripts find it.
    if feature_list_path is not None:
        (out_dir / "selected_features.json").write_text(Path(feature_list_path).read_text())
    log.info(f"saved -> {out_dir}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--side", required=True, choices=["LONG", "SHORT"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--best-params", default=None)
    ap.add_argument("--feature-list", default=None)
    ap.add_argument("--max-train-date", default=None)
    args = ap.parse_args()
    cfg = load_config()
    binary_walk_forward(
        args.symbol, cfg, side=args.side, tag=args.tag,
        best_params_path=Path(args.best_params) if args.best_params else None,
        feature_list_path=Path(args.feature_list) if args.feature_list else None,
        max_train_date=args.max_train_date,
    )


if __name__ == "__main__":
    main()
