"""Re-run triple-barrier labeling on an existing features parquet.

The builder is expensive (Cat 17 Hurst dominates ~4 min/symbol). When only the
labeler params change, we can reuse the features and just rewrite the label
columns in-place. The OHLCV + atr_14 columns are already in the parquet, which
is everything the labeler needs.

Usage:
    python -m scripts.relabel --symbol BTC
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from model.labeler import triple_barrier_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

LABEL_COLS = ["label", "holding_bars", "exit_price", "pnl_pct"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def relabel(symbol: str) -> None:
    cfg = load_config()
    lab = cfg["labeling"]
    features_dir = PROJECT_ROOT / cfg["features"]["output_dir"]
    path = features_dir / f"{symbol}_features.parquet"

    print(f"[relabel] loading {path}")
    df = pd.read_parquet(path)
    print(f"[relabel] shape={df.shape}")

    if "atr_14" not in df.columns:
        raise RuntimeError(f"atr_14 missing from {path} — rebuild features first")

    min_profit_pct = lab["min_profit_pct"].get(symbol)
    if min_profit_pct is None:
        raise KeyError(f"labeling.min_profit_pct.{symbol} not set in config")

    print(
        f"[relabel] params: tp_atr_mult={lab['tp_atr_mult']} sl_atr_mult={lab['sl_atr_mult']} "
        f"max_holding_bars={lab['max_holding_bars']} min_profit_pct[{symbol}]={min_profit_pct}"
    )
    print(f"[relabel] OLD label counts:\n{df['label'].value_counts().sort_index()}")

    new_labels = triple_barrier_labels(
        df,
        atr_col="atr_14",
        tp_atr_mult=lab["tp_atr_mult"],
        sl_atr_mult=lab["sl_atr_mult"],
        max_holding_bars=lab["max_holding_bars"],
        min_profit_pct=min_profit_pct,
        classes=lab["classes"],
    )
    # Overwrite existing label columns, preserve index alignment.
    for c in LABEL_COLS:
        df[c] = new_labels[c].to_numpy()

    # Drop sentinel -1 rows (last max_holding_bars rows have no forward window).
    before = len(df)
    df = df[df["label"] >= 0].reset_index(drop=True)
    print(f"[relabel] dropped {before - len(df)} rows with sentinel label=-1")

    print(f"[relabel] NEW label counts:\n{df['label'].value_counts().sort_index()}")
    dist = df["label"].value_counts(normalize=True).sort_index().mul(100).round(2)
    print(f"[relabel] NEW label %:\n{dist}")

    df.to_parquet(path, index=False)
    print(f"[relabel] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    args = ap.parse_args()
    relabel(args.symbol)


if __name__ == "__main__":
    main()
