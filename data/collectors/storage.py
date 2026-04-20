"""Parquet I/O helpers.

Append-only writes are idempotent on (symbol, timeframe, timestamp): if a row
with the same timestamp already exists it is overwritten. Partitioning is done
at the file level (one file per symbol+timeframe) to keep load simple; for very
large tables we can switch to pyarrow.dataset partitioning later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def symbol_slug(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def ohlcv_path(storage_dir: Path, symbol: str, timeframe: str) -> Path:
    return Path(storage_dir) / f"{symbol_slug(symbol)}_{timeframe}.parquet"


def write_ohlcv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df.to_parquet(path, index=False)


def read_ohlcv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=OHLCV_COLS)
    return pd.read_parquet(path)


def upsert_ohlcv(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    existing = read_ohlcv(path)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = (
        combined.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    write_ohlcv(combined, path)
    return combined


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
