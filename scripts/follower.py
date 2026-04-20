"""Live OHLCV follower — keeps Postgres current with newly-closed candles.

Polls data-api.binance.vision REST every `--interval` seconds. For each
(symbol, timeframe) pair:
  1. Reads `max(ts)` from Postgres (the resume point).
  2. Fetches everything since that point up to the most recent CLOSED candle
     boundary for that timeframe.
  3. Upserts via the same idempotent path as the backfill (`db.upsert_ohlcv`).

Gap detection is implicit: if we miss a tick (network blip, restart), the
next tick's `since` is older and the same code paginates forward to catch up.
No separate "gap detector" needed — the resume cursor IS the gap detector.

Closed-candle invariant: end_ms is floored to the timeframe boundary
(e.g. for 5m at 02:13:47, end_ms = 02:10:00). Any candle with open_time < end_ms
has fully elapsed and is safe to store.

Usage standalone:
    python -m scripts.follower                              # all configured symbols, every 30s
    python -m scripts.follower --symbol BTC/USDT            # single symbol
    python -m scripts.follower --interval 60                # poll every 60s
    python -m scripts.follower --once                       # one tick then exit (for cron/testing)
Usage under bot.py:
    bot.py imports `run(stop_event)` — graceful shutdown via threading.Event.
"""
from __future__ import annotations

import argparse
import signal
import threading
import time
from datetime import datetime, timezone

from data import db
from data.collectors.fetcher import (
    EXCHANGE,
    TIMEFRAME_MS,
    fetch_symbol_timeframe,
    make_exchange,
)
from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("follower")


def candle_floor_ms(now_ms: int, timeframe: str) -> int:
    """Most recent candle-open boundary that is fully ELAPSED (closed)."""
    tf_ms = TIMEFRAME_MS[timeframe]
    return (now_ms // tf_ms) * tf_ms


def tick(exchange, pairs: list[tuple[str, str]], request_limit: int, rate_limit_ms: int) -> None:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    for symbol, tf in pairs:
        last_ts = db.latest_ohlcv_ts(EXCHANGE, symbol, tf)
        if last_ts is None:
            log.warning(
                f"{symbol} {tf}: no rows in DB — run `python -m data.collectors.fetcher "
                f"--symbol {symbol}` first; skipping"
            )
            continue

        end_ms = candle_floor_ms(now_ms, tf)
        last_ms = int(last_ts.timestamp() * 1000)
        if last_ms + TIMEFRAME_MS[tf] >= end_ms:
            # Already current.
            continue

        try:
            written = fetch_symbol_timeframe(
                exchange, symbol, tf,
                start_ms=last_ms + TIMEFRAME_MS[tf],   # resume() will recompute, this is just a floor
                end_ms=end_ms,
                request_limit=request_limit,
                rate_limit_ms=rate_limit_ms,
                resume=True,
            )
            if written:
                log.info(f"{symbol} {tf}: +{written} bars (now @ {db.latest_ohlcv_ts(EXCHANGE, symbol, tf).isoformat()})")
        except Exception as exc:
            log.exception(f"{symbol} {tf}: tick failed: {exc}")
            db.mark_collector_state(
                "binance_ohlcv_follower", f"{symbol}:{tf}", None, "error", str(exc)[:500]
            )


def _interruptible_sleep(seconds: float, stop_event: threading.Event) -> None:
    """Sleep in 1s slices so a stop_event is honoured promptly."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if stop_event.is_set():
            return
        time.sleep(min(1.0, deadline - time.time()))


def run_follower(
    pairs: list[tuple[str, str]],
    interval_sec: int,
    request_limit: int,
    rate_limit_ms: int,
    market_type: str,
    stop_event: threading.Event,
    once: bool = False,
) -> None:
    """Core loop — shared by both standalone main() and bot.py supervisor."""
    info = db.ping()
    log.info(
        f"DB ok: {info['user']}@{info['database']} (timescaledb {info['timescaledb']})"
    )
    log.info(f"Following {len(pairs)} pair(s): {pairs}, interval={interval_sec}s")

    exchange = make_exchange(market_type)

    while not stop_event.is_set():
        t0 = time.time()
        try:
            tick(exchange, pairs, request_limit, rate_limit_ms)
        except Exception as exc:
            log.exception(f"follower tick crashed: {exc}")

        if once:
            break

        elapsed = time.time() - t0
        _interruptible_sleep(max(1.0, interval_sec - elapsed), stop_event)

    log.info("Follower stopped cleanly.")


# ─────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────
def run(stop_event: threading.Event | None = None) -> None:
    """Entry point usable from bot.py.

    If `stop_event` is provided, the loop watches it and shuts down gracefully.
    If None, installs its own SIGINT/SIGTERM handlers (standalone mode).
    """
    cfg = load_config()
    bcfg = cfg["data"]["binance"]
    scfg = cfg.get("services", {}).get("follower", {})
    interval_sec = int(scfg.get("interval_sec", 30))

    pairs = [(s, tf) for s in bcfg["symbols"] for tf in bcfg["timeframes"]]

    if stop_event is None:
        stop_event = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    run_follower(
        pairs=pairs,
        interval_sec=interval_sec,
        request_limit=bcfg["request_limit"],
        rate_limit_ms=bcfg["rate_limit_ms"],
        market_type=bcfg["market_type"],
        stop_event=stop_event,
    )


def main() -> None:
    cfg = load_config()
    bcfg = cfg["data"]["binance"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="single symbol e.g. BTC/USDT")
    parser.add_argument("--timeframe", help="single timeframe e.g. 5m")
    parser.add_argument("--interval", type=int, default=30, help="poll interval seconds")
    parser.add_argument("--once", action="store_true", help="run one tick and exit")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else bcfg["symbols"]
    timeframes = [args.timeframe] if args.timeframe else bcfg["timeframes"]
    pairs = [(s, tf) for s in symbols for tf in timeframes]

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    run_follower(
        pairs=pairs,
        interval_sec=args.interval,
        request_limit=bcfg["request_limit"],
        rate_limit_ms=bcfg["rate_limit_ms"],
        market_type=bcfg["market_type"],
        stop_event=stop_event,
        once=args.once,
    )


if __name__ == "__main__":
    main()
