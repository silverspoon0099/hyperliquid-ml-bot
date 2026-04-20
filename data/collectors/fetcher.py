"""Binance Futures OHLCV fetcher with resumable pagination — Postgres backend.

Decision #23: Train on Binance OHLCV (deepest liquidity, cleanest candles),
execute on Hyperliquid. Decision #24: 18 months minimum history.

Resumability: queries `SELECT max(ts) FROM ohlcv WHERE ...` to find the cursor.
Writes per-chunk so an interrupted run loses at most one CCXT request worth.

Usage:
    python -m data.collectors.fetcher                        # all symbols, all tfs
    python -m data.collectors.fetcher --symbol BTC/USDT      # single symbol
    python -m data.collectors.fetcher --symbol BTC/USDT --timeframe 5m
    python -m data.collectors.fetcher --months 24            # override history depth
    python -m data.collectors.fetcher --no-resume            # rebuild from scratch
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
from tqdm import tqdm

from data import db
from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("fetcher")

EXCHANGE = "binance"
OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def make_exchange(market_type: str = "spot") -> ccxt.binance:
    """Build a Binance ccxt client.

    Uses `data-api.binance.vision` (Binance's official public market-data host)
    because the main api/fapi domains return HTTP 451 from many VPS regions.
    This endpoint:
      * serves SPOT OHLCV only (no futures, no auth, no orders)
      * is rate-limited identically to the main API
      * is a drop-in for /api/v3/klines, exchangeInfo, ping, etc.

    For ML training that's fine — BTC/USDT spot vs perp is >99% correlated and
    all our features are on OHLCV. Hyperliquid funding is collected separately.
    """
    ex = ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {
                "defaultType": market_type,
                # Without this, ccxt loadMarkets() fans out to spot+fapi+dapi+
                # options in parallel; the fapi call hits the geo-blocked host.
                "fetchMarkets": [market_type],
            },
        }
    )
    if market_type == "spot":
        ex.urls["api"]["public"] = "https://data-api.binance.vision/api/v3"
    return ex


def _flush_chunk(chunk: list[list], symbol: str, timeframe: str) -> int:
    """Upsert a CCXT chunk into Postgres. Returns rows written."""
    if not chunk:
        return 0
    df = pd.DataFrame(chunk, columns=OHLCV_COLS)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return db.upsert_ohlcv(df, exchange=EXCHANGE, symbol=symbol, timeframe=timeframe)


def fetch_range(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    request_limit: int,
    rate_limit_ms: int,
) -> int:
    """Paginate from start_ms to end_ms, writing each chunk to Postgres.

    Returns total rows written (sum of per-chunk writes; may overcount on
    duplicate upserts, which is harmless for progress logging).
    """
    tf_ms = TIMEFRAME_MS[timeframe]
    expected = max(1, (end_ms - start_ms) // tf_ms)
    cursor = start_ms
    total = 0
    consecutive_empty = 0

    pbar = tqdm(total=expected, desc=f"{symbol} {timeframe}", unit="bar")
    last_progress = 0

    while cursor < end_ms:
        try:
            chunk = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor, limit=request_limit
            )
        except ccxt.RateLimitExceeded:
            log.warning("Rate limited — sleeping 5s")
            time.sleep(5)
            continue
        except ccxt.NetworkError as exc:
            log.warning(f"Network error: {exc} — retrying in 3s")
            time.sleep(3)
            continue
        except ccxt.ExchangeError as exc:
            log.error(f"Exchange error for {symbol} {timeframe}: {exc}")
            break

        if not chunk:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                log.info(f"{symbol} {timeframe}: no more data at cursor={cursor}")
                break
            cursor += tf_ms * request_limit
            continue
        consecutive_empty = 0

        # Trim anything past end_ms before write
        chunk = [r for r in chunk if r[0] < end_ms]
        if chunk:
            total += _flush_chunk(chunk, symbol, timeframe)
            last_ts = chunk[-1][0]
            next_cursor = last_ts + tf_ms
        else:
            next_cursor = end_ms

        progress = (next_cursor - start_ms) // tf_ms
        pbar.update(max(0, progress - last_progress))
        last_progress = progress

        if next_cursor <= cursor:
            log.warning(f"No cursor advance ({next_cursor} <= {cursor}); stopping")
            break
        cursor = next_cursor

        time.sleep(rate_limit_ms / 1000.0)

    pbar.close()
    return total


def fetch_symbol_timeframe(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    request_limit: int,
    rate_limit_ms: int,
    resume: bool = True,
) -> int:
    tf_ms = TIMEFRAME_MS[timeframe]
    effective_start = start_ms

    if resume:
        last_ts = db.latest_ohlcv_ts(EXCHANGE, symbol, timeframe)
        if last_ts is not None:
            last_ms = int(last_ts.timestamp() * 1000)
            new_start = last_ms + tf_ms
            if new_start >= end_ms:
                log.info(
                    f"{symbol} {timeframe}: already up-to-date "
                    f"(last bar @ {last_ts.isoformat()})"
                )
                db.mark_collector_state(
                    "binance_ohlcv", f"{symbol}:{timeframe}", last_ts, "up_to_date"
                )
                return 0
            log.info(
                f"{symbol} {timeframe}: resuming from "
                f"{datetime.fromtimestamp(new_start / 1000, tz=timezone.utc).isoformat()}"
            )
            effective_start = new_start
        else:
            log.info(f"{symbol} {timeframe}: empty table, full backfill")

    written = fetch_range(
        exchange, symbol, timeframe,
        effective_start, end_ms, request_limit, rate_limit_ms,
    )
    last_ts_after = db.latest_ohlcv_ts(EXCHANGE, symbol, timeframe)
    db.mark_collector_state(
        "binance_ohlcv", f"{symbol}:{timeframe}", last_ts_after, "ok"
    )
    log.info(
        f"{symbol} {timeframe}: wrote {written} rows "
        f"(latest now {last_ts_after.isoformat() if last_ts_after else 'none'})"
    )
    return written


def main() -> None:
    cfg = load_config()
    bcfg = cfg["data"]["binance"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="single symbol e.g. BTC/USDT")
    parser.add_argument("--timeframe", help="single timeframe e.g. 5m")
    parser.add_argument("--months", type=int, default=bcfg["history_months"])
    parser.add_argument("--no-resume", action="store_true", help="rebuild from scratch (does NOT delete existing rows)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else bcfg["symbols"]
    timeframes = [args.timeframe] if args.timeframe else bcfg["timeframes"]

    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=args.months * 30)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    log.info(
        f"Fetching {symbols} x {timeframes} "
        f"from {start_dt.isoformat()} to {end_dt.isoformat()} → Postgres"
    )

    # Verify DB is reachable before we hit Binance.
    info = db.ping()
    log.info(
        f"DB ok: {info['user']}@{info['database']} "
        f"(timescaledb {info['timescaledb']})"
    )

    exchange = make_exchange(bcfg["market_type"])

    for symbol in symbols:
        for tf in timeframes:
            try:
                fetch_symbol_timeframe(
                    exchange, symbol, tf,
                    start_ms, end_ms,
                    request_limit=bcfg["request_limit"],
                    rate_limit_ms=bcfg["rate_limit_ms"],
                    resume=not args.no_resume,
                )
            except Exception as exc:
                log.exception(f"Failed {symbol} {tf}: {exc}")
                db.mark_collector_state(
                    "binance_ohlcv", f"{symbol}:{tf}", None, "error", str(exc)[:500]
                )


if __name__ == "__main__":
    main()
