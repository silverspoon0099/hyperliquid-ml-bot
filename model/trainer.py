"""Walk-forward training (spec §10.1) — rolling fixed-width windows.

Each fold:
    train window (train_months) │ purge (48 bars = max_holding_bars) │ val window (val_months)
Window slides by step_months. Rolling (not anchored/expanding) — each fold
mirrors the production retraining cadence (Phase 4 step 4.9).

Metrics per fold:
    val log_loss, accuracy, per-class P/R/F1, confusion matrix,
    fire-rate / directional-hit-rate at signal_threshold.
Aggregate: mean ± std across folds.

Usage:
    python -m model.trainer --symbol BTC
    python -m model.trainer --symbol BTC --class-weights     # override config
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, log_loss

from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("trainer")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NON_FEATURE_COLS = {
    "timestamp", "_ts",  # _ts is a synthetic datetime used only for fold slicing
    "open", "high", "low", "close", "volume",
    "label", "holding_bars", "exit_price", "pnl_pct",
}


@dataclass
class FoldMetrics:
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_rows: int
    val_rows: int
    best_iteration: int
    log_loss: float
    accuracy: float
    # Per-class arrays are [LONG, SHORT, NEUTRAL].
    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]
    confusion: list[list[int]]
    confidence_mean: float
    confidence_std: float
    fire_rate: float
    directional_fire_rate: float
    directional_hit_rate: float | None


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def balanced_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """sklearn-style inverse-frequency: w[c] = N / (n_classes * count[c])."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    class_w = len(y) / (n_classes * np.maximum(counts, 1.0))
    return class_w[y]


def make_folds(
    ts: pd.Series,
    train_months: int,
    val_months: int,
    step_months: int,
    purge_bars: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return list of (train_start, train_end, val_start, val_end) timestamps.

    Walks from the first timestamp forward in step_months increments.
    val_start sits purge_bars AFTER train_end; represented in time as
    purge_bars * inferred_bar_interval.
    """
    data_start = ts.iloc[0]
    data_end = ts.iloc[-1]
    # Infer bar interval from median delta (5min = 300s, robust to gaps).
    bar_interval = pd.Timedelta(seconds=int(ts.diff().dt.total_seconds().median()))
    purge_delta = bar_interval * purge_bars

    folds = []
    fold_idx = 0
    cursor = data_start
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months)
        val_start = train_end + purge_delta
        val_end = val_start + pd.DateOffset(months=val_months)
        if val_end > data_end:
            break
        folds.append((train_start, train_end, val_start, val_end))
        cursor = cursor + pd.DateOffset(months=step_months)
        fold_idx += 1
    return folds


def evaluate_fold(
    booster: lgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: list[str],
    neutral_idx: int,
    threshold: float,
) -> dict:
    n_classes = len(class_names)
    proba = booster.predict(X_val, num_iteration=booster.best_iteration)
    pred = proba.argmax(axis=1)

    ll = log_loss(y_val, proba, labels=list(range(n_classes)))
    acc = float((pred == y_val).mean())

    report = classification_report(
        y_val, pred,
        labels=list(range(n_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    precision = [report[c]["precision"] for c in class_names]
    recall = [report[c]["recall"] for c in class_names]
    f1 = [report[c]["f1-score"] for c in class_names]
    support = [int(report[c]["support"]) for c in class_names]
    cm = confusion_matrix(y_val, pred, labels=list(range(n_classes))).tolist()

    max_proba = proba.max(axis=1)
    fire_mask = max_proba >= threshold
    directional_fire_mask = fire_mask & (pred != neutral_idx)
    directional_hit = (
        float((pred[directional_fire_mask] == y_val[directional_fire_mask]).mean())
        if directional_fire_mask.any() else None
    )

    return dict(
        log_loss=float(ll),
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        confusion=cm,
        confidence_mean=float(max_proba.mean()),
        confidence_std=float(max_proba.std()),
        fire_rate=float(fire_mask.mean()),
        directional_fire_rate=float(directional_fire_mask.mean()),
        directional_hit_rate=directional_hit,
    )


def walk_forward(symbol: str, cfg: dict, use_weights: bool, tag: str = "",
                 best_params_path: Path | None = None,
                 feature_list_path: Path | None = None) -> dict:
    classes = cfg["labeling"]["classes"]
    n_classes = len(classes)
    inv_classes = {v: k for k, v in classes.items()}
    class_names = [inv_classes[i] for i in range(n_classes)]
    neutral_idx = classes["NEUTRAL"]
    threshold = cfg["model"]["signal_threshold"]
    purge_bars = cfg["model"]["walk_forward_purge_bars"]
    wf = cfg["walk_forward"]

    features_dir = PROJECT_ROOT / cfg["features"]["output_dir"]
    parquet_path = features_dir / f"{symbol}_features.parquet"
    log.info(f"loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df[df["label"] >= 0].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    feat_cols = feature_columns(df)
    if feature_list_path is not None:
        with open(feature_list_path) as f:
            selection = json.load(f)
        selected = selection["features"]
        missing = [c for c in selected if c not in feat_cols]
        if missing:
            raise ValueError(
                f"feature-list references {len(missing)} columns absent from parquet: {missing[:5]}..."
            )
        feat_cols = selected
        log.info(
            f"restricted to {len(feat_cols)} features from {feature_list_path} "
            f"(coverage={selection.get('shap_coverage', float('nan'))*100:.2f}% of SHAP)"
        )
    log.info(f"{len(df):,} rows, {len(feat_cols)} feature columns, span "
             f"{df['_ts'].iloc[0]} -> {df['_ts'].iloc[-1]}")

    folds = make_folds(
        df["_ts"],
        train_months=wf["train_months"],
        val_months=wf["val_months"],
        step_months=wf["step_months"],
        purge_bars=purge_bars,
    )
    log.info(
        f"walk-forward: {len(folds)} folds "
        f"(train={wf['train_months']}mo purge={purge_bars}bars val={wf['val_months']}mo step={wf['step_months']}mo) "
        f"class_weights={use_weights}"
    )

    params = dict(cfg["model"]["params"])
    if best_params_path is not None:
        with open(best_params_path) as f:
            best_params_blob = json.load(f)
        tuned = best_params_blob.get("best_params", {})
        log.info(f"overriding with {len(tuned)} tuned params from {best_params_path}")
        for k, v in tuned.items():
            log.info(f"  {k:20s} {params.get(k)!r} -> {v!r}")
        params.update(tuned)
    num_boost_round = cfg["model"]["num_boost_round"]
    early_stopping = cfg["model"]["early_stopping_rounds"]

    suffix = f"_{tag}" if tag else ""
    out_dir = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics: list[FoldMetrics] = []
    for i, (t0, t1, v0, v1) in enumerate(folds, start=1):
        train_mask = (df["_ts"] >= t0) & (df["_ts"] < t1)
        val_mask = (df["_ts"] >= v0) & (df["_ts"] < v1)
        train_df = df[train_mask]
        val_df = df[val_mask]

        X_tr = train_df[feat_cols].to_numpy(dtype=np.float32)
        y_tr = train_df["label"].to_numpy(dtype=np.int32)
        X_val = val_df[feat_cols].to_numpy(dtype=np.float32)
        y_val = val_df["label"].to_numpy(dtype=np.int32)

        w_tr = balanced_weights(y_tr, n_classes) if use_weights else None

        train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feat_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feat_cols)

        log.info(
            f"[fold {i}/{len(folds)}] train={t0.date()}->{t1.date()} ({len(train_df):,}) "
            f"val={v0.date()}->{v1.date()} ({len(val_df):,})"
        )
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping, first_metric_only=False),
                lgb.log_evaluation(period=0),  # silent per-iteration
            ],
        )
        metrics = evaluate_fold(booster, X_val, y_val, class_names, neutral_idx, threshold)
        fm = FoldMetrics(
            fold=i,
            train_start=str(t0.date()),
            train_end=str(t1.date()),
            val_start=str(v0.date()),
            val_end=str(v1.date()),
            train_rows=int(len(train_df)),
            val_rows=int(len(val_df)),
            best_iteration=int(booster.best_iteration),
            **metrics,
        )
        fold_metrics.append(fm)
        booster.save_model(str(out_dir / f"fold_{i}.txt"))
        log.info(
            f"[fold {i}] best_iter={fm.best_iteration} logloss={fm.log_loss:.4f} "
            f"acc={fm.accuracy:.4f} fire={fm.fire_rate:.3%} "
            f"hit|fire={fm.directional_hit_rate if fm.directional_hit_rate is None else f'{fm.directional_hit_rate:.3%}'}"
        )

    # Aggregate summary.
    summary_table = pd.DataFrame([asdict(f) for f in fold_metrics])[
        ["fold", "val_start", "val_end", "best_iteration",
         "log_loss", "accuracy", "fire_rate", "directional_fire_rate", "directional_hit_rate",
         "confidence_mean"]
    ]
    log.info("\n=== WALK-FORWARD SUMMARY (per fold) ===\n" + summary_table.to_string(index=False))

    def _mean_std(key: str) -> tuple[float, float]:
        vals = [getattr(f, key) for f in fold_metrics if getattr(f, key) is not None]
        return float(np.mean(vals)), float(np.std(vals))

    agg = {
        "log_loss":              _mean_std("log_loss"),
        "accuracy":              _mean_std("accuracy"),
        "fire_rate":             _mean_std("fire_rate"),
        "directional_fire_rate": _mean_std("directional_fire_rate"),
        "directional_hit_rate":  _mean_std("directional_hit_rate"),
        "confidence_mean":       _mean_std("confidence_mean"),
    }
    log.info("=== AGGREGATE (mean ± std across folds) ===")
    for k, (m, s) in agg.items():
        log.info(f"  {k:24s} {m:.4f} ± {s:.4f}")

    # Per-class aggregate (mean of per-fold precision/recall).
    prec_mat = np.array([f.precision for f in fold_metrics])
    rec_mat = np.array([f.recall for f in fold_metrics])
    log.info("=== PER-CLASS (mean across folds) ===")
    log.info(f"  {'':8s}  {'precision':>10s}  {'recall':>10s}")
    for i, name in enumerate(class_names):
        log.info(f"  {name:8s}  {prec_mat[:, i].mean():>10.4f}  {rec_mat[:, i].mean():>10.4f}")

    result = {
        "symbol": symbol,
        "use_weights": use_weights,
        "n_folds": len(fold_metrics),
        "config": {
            "train_months": wf["train_months"],
            "val_months": wf["val_months"],
            "step_months": wf["step_months"],
            "purge_bars": purge_bars,
            "signal_threshold": threshold,
        },
        "aggregate": {k: {"mean": m, "std": s} for k, (m, s) in agg.items()},
        "per_class_mean": {
            class_names[i]: {
                "precision": float(prec_mat[:, i].mean()),
                "recall": float(rec_mat[:, i].mean()),
            } for i in range(n_classes)
        },
        "folds": [asdict(f) for f in fold_metrics],
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"saved models -> {out_dir}/fold_*.txt, metrics -> {metrics_path}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument(
        "--class-weights", dest="class_weights", action="store_true", default=None,
        help="override config: use inverse-frequency weights",
    )
    ap.add_argument(
        "--no-class-weights", dest="class_weights", action="store_false",
        help="override config: disable class weights",
    )
    ap.add_argument(
        "--best-params", default=None,
        help="path to best_params.json (e.g. model/models/BTC_wf/best_params.json); "
             "merged over model.params",
    )
    ap.add_argument(
        "--tag", default="",
        help="suffix for the output dir, e.g. --tag tuned -> model/models/BTC_wf_tuned/",
    )
    ap.add_argument(
        "--feature-list", default=None,
        help="path to selected_features.json (spec §2.5 trim)",
    )
    args = ap.parse_args()
    cfg = load_config()
    use_weights = cfg["walk_forward"]["use_class_weights"] if args.class_weights is None else args.class_weights
    best_params_path = Path(args.best_params) if args.best_params else None
    feature_list_path = Path(args.feature_list) if args.feature_list else None
    walk_forward(args.symbol, cfg, use_weights=use_weights, tag=args.tag,
                 best_params_path=best_params_path,
                 feature_list_path=feature_list_path)


if __name__ == "__main__":
    main()
