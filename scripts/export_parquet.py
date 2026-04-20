"""Snapshot Postgres OHLCV → parquet for feature building / model training.

Why parquet when Postgres already has the data:
  * `features/builder.py` reads parquet (unchanged from the pre-Postgres era).
  * Training runs must be reproducible: Postgres keeps growing with new bars,
    but a parquet snapshot at `train_end` is immutable — rerunning training
    six months from now gives the exact same result.

Writes to `data.binance.storage_dir` using the filename convention defined in
`data/collectors/storage.py::ohlcv_path` so builder.py works with zero edits.

Usage:
    python -m scripts.export_parquet                         # all configured symbols × TFs, up to now
    python -m scripts.export_parquet --symbol BTC/USDT       # single symbol
    python -m scripts.export_parquet --until 2026-02-28      # cap at train_end (reproducible training snapshot)
    python -m scripts.export_parquet --train-snapshot        # shortcut: cap at splits.train_end from config
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data import db
from data.collectors.fetcher import EXCHANGE
from data.collectors.storage import OHLCV_COLS, ohlcv_path, write_ohlcv
from utils.config import load_config, resolve_path
from utils.logging_setup import get_logger

log = get_logger("export_parquet")


def _parse_until(s: str) -> datetime:
    """Accept `YYYY-MM-DD` or full ISO. Naive dates are treated as UTC midnight."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def export_one(
    symbol: str,
    timeframe: str,
    storage_dir: Path,
    start: datetime | None,
    end: datetime | None,
) -> int:
    """Export one (symbol, timeframe) slice. Returns row count written."""
    df = db.fetch_ohlcv(EXCHANGE, symbol, timeframe, start=start, end=end)
    if df.empty:
        log.warning(f"{symbol} {timeframe}: no rows in Postgres — skipping")
        return 0

    # Postgres → builder.py schema:
    #   * `ts` (timestamptz) → `timestamp` (int ms) to match storage.OHLCV_COLS
    #   * drop quote_volume / trades_count (builder doesn't read them, and
    #     rewriting the parquet with extra cols is wasted I/O)
    out = pd.DataFrame({
        "timestamp": (df["ts"].astype("int64") // 1_000_000),  # ns → ms
        "open":   df["open"].astype(float),
        "high":   df["high"].astype(float),
        "low":    df["low"].astype(float),
        "close":  df["close"].astype(float),
        "volume": df["volume"].astype(float),
    })[OHLCV_COLS]

    path = ohlcv_path(storage_dir, symbol, timeframe)
    write_ohlcv(out, path)
    first = pd.to_datetime(out["timestamp"].iloc[0], unit="ms", utc=True)
    last = pd.to_datetime(out["timestamp"].iloc[-1], unit="ms", utc=True)
    log.info(
        f"{symbol} {timeframe}: wrote {len(out):,} rows "
        f"({first.isoformat()} → {last.isoformat()}) → {path}"
    )
    return len(out)


def main() -> None:
    cfg = load_config()
    bcfg = cfg["data"]["binance"]
    storage_dir = resolve_path(bcfg["storage_dir"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="single symbol e.g. BTC/USDT (default: all configured)")
    parser.add_argument("--timeframe", help="single timeframe e.g. 5m (default: all configured)")
    parser.add_argument("--since", help="start date/time, ISO (default: beginning of data)")
    parser.add_argument("--until", help="end date/time, ISO (default: now). Inclusive.")
    parser.add_argument(
        "--train-snapshot", action="store_true",
        help="shortcut for --until=splits.train_end (reproducible training slice)",
    )
    args = parser.parse_args()

    if args.train_snapshot and args.until:
        parser.error("--train-snapshot and --until are mutually exclusive")

    end: datetime | None = None
    if args.train_snapshot:
        end = _parse_until(cfg["splits"]["train_end"])
        log.info(f"--train-snapshot: capping at splits.train_end = {end.isoformat()}")
    elif args.until:
        end = _parse_until(args.until)
    start: datetime | None = _parse_until(args.since) if args.since else None

    symbols = [args.symbol] if args.symbol else bcfg["symbols"]
    timeframes = [args.timeframe] if args.timeframe else bcfg["timeframes"]

    info = db.ping()
    log.info(f"DB ok: {info['user']}@{info['database']} (timescaledb {info['timescaledb']})")
    log.info(f"Exporting {symbols} × {timeframes} → {storage_dir}")

    total = 0
    for symbol in symbols:
        for tf in timeframes:
            total += export_one(symbol, tf, storage_dir, start, end)
    log.info(f"Done — {total:,} rows across {len(symbols) * len(timeframes)} files.")


if __name__ == "__main__":
    main()
