"""Hyperliquid live data collector — 24/7 trades WS + funding REST poll.

Writes to Postgres hypertables `hl_trades` and `hl_funding`.

L2 order book (Phase 3 feature territory, #279-289) is NOT collected yet —
the schema lives in `hl_l2_snapshots` dormant; flip the flag on in config
when we're ready to eat the storage + compute.

CRITICAL (Decision #25): Trade tape has NO historical API. Start this
collector on Phase 1 Day 1 — the accumulated trade stream is what feeds
CVD / taker-imbalance features in Phase 2+.

Run standalone:
    python -m data.collectors.hyperliquid_ws                   # all coins
    python -m data.collectors.hyperliquid_ws --coins BTC TAO   # subset
Run under bot.py:
    bot.py imports `run(coins, stop_event)` — graceful shutdown via event.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import threading
import time
from datetime import datetime, timedelta, timezone

import aiohttp
import websockets

from data import db
from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("hl_collector")


# ─────────────────────────────────────────────────────────────────────────
# Trade batching — accumulate in memory, flush periodically
# ─────────────────────────────────────────────────────────────────────────
class TradesBatcher:
    """Collects trades across bursts and flushes to Postgres in batches.

    Flush triggers: either `batch_max` rows accumulated, or `flush_interval_s`
    elapsed since last flush — whichever first.
    """

    def __init__(self, coin: str, flush_interval_s: float = 2.0, batch_max: int = 1000):
        self.coin = coin
        self.flush_interval_s = flush_interval_s
        self.batch_max = batch_max
        self._buf: list[dict] = []
        self._lock = asyncio.Lock()

    async def add(self, rows: list[dict]) -> None:
        async with self._lock:
            self._buf.extend(rows)
            if len(self._buf) >= self.batch_max:
                await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._buf:
            return
        to_write, self._buf = self._buf, []
        try:
            written = await asyncio.to_thread(db.insert_trades_batch, to_write)
            log.info(f"[{self.coin}] trades flushed: {written} rows")
        except Exception as exc:
            log.exception(f"[{self.coin}] trade flush failed: {exc}")
            # Don't re-queue — losing a batch is preferable to unbounded growth.

    async def flusher_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.flush_interval_s)
            except asyncio.TimeoutError:
                pass
            async with self._lock:
                await self._flush_locked()


# ─────────────────────────────────────────────────────────────────────────
# Trades subscriber
# ─────────────────────────────────────────────────────────────────────────
async def trades_subscriber(
    ws_url: str,
    coin: str,
    batcher: TradesBatcher,
    reconnect_delay: int,
    max_reconnect_delay: int,
    stop_event: asyncio.Event,
) -> None:
    backoff = reconnect_delay
    while not stop_event.is_set():
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(
                    json.dumps(
                        {"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}
                    )
                )
                log.info(f"[{coin}] subscribed trades")
                backoff = reconnect_delay

                async for raw in ws:
                    if stop_event.is_set():
                        break
                    msg = json.loads(raw)
                    if msg.get("channel") != "trades":
                        continue
                    trades = msg.get("data", []) or []
                    if not trades:
                        continue
                    rows = []
                    for t in trades:
                        try:
                            ts_ms = int(t.get("time", time.time() * 1000))
                            rows.append(
                                {
                                    "ts": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                                    "coin": t.get("coin", coin),
                                    "trade_id": int(t["tid"]) if t.get("tid") is not None else None,
                                    "side": t.get("side"),  # "B" or "A"
                                    "price": float(t["px"]),
                                    "size": float(t["sz"]),
                                }
                            )
                        except (KeyError, ValueError, TypeError) as e:
                            log.warning(f"[{coin}] malformed trade {t}: {e}")
                    if rows:
                        await batcher.add(rows)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if stop_event.is_set():
                break
            log.warning(f"[{coin}] trades disconnect: {exc} — reconnect in {backoff}s")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                break  # stop signalled
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, max_reconnect_delay)


# ─────────────────────────────────────────────────────────────────────────
# Funding poller (REST, aligned to hour boundaries)
# ─────────────────────────────────────────────────────────────────────────
def _seconds_until_next_hour_plus_offset(offset_s: int = 5) -> float:
    """Seconds until (next UTC hour boundary + offset_s). +offset so funding is
    definitely applied server-side before we poll."""
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return (next_hour - now).total_seconds() + offset_s


async def funding_poller(
    rest_url: str,
    coins: list[str],
    stop_event: asyncio.Event,
    offset_s: int = 5,
) -> None:
    """Poll `metaAndAssetCtxs` once at startup, then every hour at :00:05.

    Writes one row per coin per hour into `hl_funding`. ON CONFLICT DO UPDATE
    handles duplicate polls (safe to restart).
    """
    url_meta = f"{rest_url.rstrip('/')}/info"

    async def poll_once() -> None:
        async with aiohttp.ClientSession() as session:
            payload = {"type": "metaAndAssetCtxs"}
            async with session.post(url_meta, json=payload, timeout=15) as resp:
                body = await resp.json()
        if not (isinstance(body, list) and len(body) == 2):
            log.warning(f"funding: unexpected shape {type(body)}")
            return
        meta, ctxs = body
        universe = meta.get("universe", [])
        name_to_idx = {u["name"]: i for i, u in enumerate(universe)}
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        rows = []
        for coin in coins:
            idx = name_to_idx.get(coin)
            if idx is None or idx >= len(ctxs):
                continue
            ctx = ctxs[idx]
            rows.append(
                {
                    "ts": now,
                    "coin": coin,
                    "funding_rate": float(ctx.get("funding", 0)),
                    "premium": float(ctx["premium"]) if ctx.get("premium") is not None else None,
                    "open_interest": float(ctx.get("openInterest", 0))
                    if ctx.get("openInterest") is not None
                    else None,
                    "mark_price": float(ctx.get("markPx", 0))
                    if ctx.get("markPx") is not None
                    else None,
                    "oracle_price": float(ctx.get("oraclePx", 0))
                    if ctx.get("oraclePx") is not None
                    else None,
                }
            )
        if rows:
            written = await asyncio.to_thread(db.insert_funding, rows)
            log.info(f"funding snapshot @ {now.isoformat()}: wrote {written} rows for {len(rows)} coins")

    # Initial poll immediately so we have a row before the first hour wraps.
    try:
        await poll_once()
    except Exception as exc:
        log.warning(f"funding: initial poll failed: {exc}")

    while not stop_event.is_set():
        wait_s = _seconds_until_next_hour_plus_offset(offset_s)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=wait_s)
            break  # stop signalled
        except asyncio.TimeoutError:
            pass
        try:
            await poll_once()
        except Exception as exc:
            log.warning(f"funding: poll failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────
async def run_collector(coins: list[str], stop_event: asyncio.Event) -> None:
    cfg = load_config()
    hcfg = cfg["data"]["hyperliquid"]

    # Verify DB is reachable before any WS traffic.
    info = await asyncio.to_thread(db.ping)
    log.info(f"DB ok: {info['user']}@{info['database']} (timescaledb {info['timescaledb']})")
    log.info(f"Starting Hyperliquid collector for {coins} (L2 disabled — Phase 3)")

    batchers = {c: TradesBatcher(c) for c in coins}
    tasks: list[asyncio.Task] = []

    for coin in coins:
        tasks.append(
            asyncio.create_task(
                trades_subscriber(
                    hcfg["ws_url"],
                    coin,
                    batchers[coin],
                    hcfg["reconnect_delay_sec"],
                    hcfg["max_reconnect_delay_sec"],
                    stop_event,
                ),
                name=f"trades_{coin}",
            )
        )
        tasks.append(
            asyncio.create_task(
                batchers[coin].flusher_loop(stop_event), name=f"flusher_{coin}"
            )
        )

    tasks.append(
        asyncio.create_task(
            funding_poller(hcfg["rest_url"], coins, stop_event), name="funding_poller"
        )
    )

    # Block until stop_event; then let tasks unwind cooperatively.
    await stop_event.wait()
    log.info("Stop signalled — draining trade buffers and cancelling tasks…")

    # Final flush for each batcher before cancelling.
    for b in batchers.values():
        async with b._lock:
            await b._flush_locked()

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    log.info("Hyperliquid collector stopped cleanly")


# ─────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────
def run(coins: list[str] | None = None, stop_event: threading.Event | None = None) -> None:
    """Entry point usable from bot.py.

    If `stop_event` is provided (threading.Event), the asyncio loop watches it
    and shuts down gracefully. If None, installs its own SIGINT/SIGTERM handler.
    """
    cfg = load_config()
    coins = coins or cfg["data"]["hyperliquid"]["coins"]

    async def _main():
        async_stop = asyncio.Event()

        if stop_event is None:
            # Standalone mode: install signal handlers.
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, async_stop.set)
        else:
            # Supervised mode: bridge threading.Event → asyncio.Event.
            async def _bridge():
                while not stop_event.is_set():
                    await asyncio.sleep(0.5)
                async_stop.set()
            asyncio.create_task(_bridge(), name="stop_bridge")

        await run_collector(coins, async_stop)

    asyncio.run(_main())


def main() -> None:
    cfg = load_config()
    default_coins = cfg["data"]["hyperliquid"]["coins"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=default_coins)
    args = parser.parse_args()

    try:
        run(args.coins)
    except KeyboardInterrupt:
        log.info("Collector stopped by user")


if __name__ == "__main__":
    main()
