"""PostgreSQL / TimescaleDB I/O layer.

Replaces the parquet-based `data/collectors/storage.py` for live data.
Parquet is still produced on demand by `scripts/export_parquet.py` for training.

Patterns mirror /nvme1/projects/dataverse/main/storage/miner/postgresql_miner_storage.py:
  * URL-form conninfo
  * Single global ConnectionPool with row_factory=dict_row at pool level
  * SQLAlchemy engine alongside for pandas.read_sql_query()
  * load_dotenv at module import
  * Idempotent writes via ON CONFLICT DO UPDATE / DO NOTHING
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Sequence

import pandas as pd
import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

_POOL: ConnectionPool | None = None
_ENGINE = None


# ─────────────────────────────────────────────────────────────────────────
# Connection pool + SQLAlchemy engine
# ─────────────────────────────────────────────────────────────────────────
def _build_url() -> str:
    return (
        f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@"
        f"{os.getenv('PG_HOST', '127.0.0.1')}:{os.getenv('PG_PORT', '5432')}/"
        f"{os.getenv('PG_DB')}"
    )


def get_pool() -> ConnectionPool:
    global _POOL
    if _POOL is None:
        max_size = max(int(os.getenv("PG_POOL_MAX", "10")), 2)
        _POOL = ConnectionPool(
            conninfo=_build_url(),
            min_size=int(os.getenv("PG_POOL_MIN", "2")),
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _POOL


def get_engine():
    """SQLAlchemy engine for pandas.read_sql_query (avoids pandas/psycopg warning)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(
            f"postgresql+psycopg://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@"
            f"{os.getenv('PG_HOST', '127.0.0.1')}:{os.getenv('PG_PORT', '5432')}/"
            f"{os.getenv('PG_DB')}"
        )
    return _ENGINE


def get_connection() -> Connection:
    """Pooled connection — use as `with get_connection() as conn: ...`.

    The pool's context manager auto-commits on clean exit and rolls back on
    exception, but we still call `conn.commit()` explicitly to match the
    miner-storage style.
    """
    return get_pool().connection()


def close_pool() -> None:
    global _POOL, _ENGINE
    if _POOL is not None:
        _POOL.close()
        _POOL = None
    if _ENGINE is not None:
        _ENGINE.dispose()
        _ENGINE = None


# ─────────────────────────────────────────────────────────────────────────
# OHLCV
# ─────────────────────────────────────────────────────────────────────────
_OHLCV_INSERT = """
INSERT INTO ohlcv (
    ts, exchange, symbol, timeframe,
    open, high, low, close, volume,
    quote_volume, trades_count
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (exchange, symbol, timeframe, ts)
DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low  = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    quote_volume = EXCLUDED.quote_volume,
    trades_count = EXCLUDED.trades_count;
"""


def upsert_ohlcv(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> int:
    """Upsert an OHLCV dataframe.

    `df` must have a `timestamp` column (ms since epoch or pd.Timestamp) plus
    open/high/low/close/volume. Optional: quote_volume, trades_count.
    Returns rowcount.
    """
    if df.empty:
        return 0

    df = df.copy()
    if pd.api.types.is_integer_dtype(df["timestamp"]):
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["ts"] = pd.to_datetime(df["timestamp"], utc=True)

    if "quote_volume" not in df:
        df["quote_volume"] = None
    if "trades_count" not in df:
        df["trades_count"] = None

    rows = [
        (
            r.ts, exchange, symbol, timeframe,
            r.open, r.high, r.low, r.close, r.volume,
            r.quote_volume, r.trades_count,
        )
        for r in df.itertuples(index=False)
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(_OHLCV_INSERT, rows)
            written = cur.rowcount
        conn.commit()
    return written or len(rows)


def latest_ohlcv_ts(
    exchange: str, symbol: str, timeframe: str
) -> datetime | None:
    """Most recent stored ts (for resumable backfills). None if empty."""
    sql = """
        SELECT max(ts) AS max_ts FROM ohlcv
        WHERE exchange = %s AND symbol = %s AND timeframe = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (exchange, symbol, timeframe))
            row = cur.fetchone()
    return row["max_ts"] if row else None


def fetch_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Read OHLCV back as a DataFrame for feature building / parquet export."""
    sql = """
        SELECT ts, open, high, low, close, volume, quote_volume, trades_count
        FROM ohlcv
        WHERE exchange = %(exchange)s
          AND symbol   = %(symbol)s
          AND timeframe = %(timeframe)s
          AND (%(start)s::timestamptz IS NULL OR ts >= %(start)s)
          AND (%(end)s::timestamptz   IS NULL OR ts <= %(end)s)
        ORDER BY ts
    """
    return pd.read_sql_query(
        sql,
        get_engine(),
        params={
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start,
            "end": end,
        },
    )


# ─────────────────────────────────────────────────────────────────────────
# Hyperliquid funding
# ─────────────────────────────────────────────────────────────────────────
_FUNDING_INSERT = """
INSERT INTO hl_funding
    (ts, coin, funding_rate, premium, open_interest, mark_price, oracle_price)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (coin, ts) DO UPDATE SET
    funding_rate  = EXCLUDED.funding_rate,
    premium       = EXCLUDED.premium,
    open_interest = EXCLUDED.open_interest,
    mark_price    = EXCLUDED.mark_price,
    oracle_price  = EXCLUDED.oracle_price;
"""


def insert_funding(rows: Sequence[dict]) -> int:
    """Each row: {ts, coin, funding_rate, premium?, open_interest?, mark_price?, oracle_price?}."""
    if not rows:
        return 0
    payload = [
        (
            r["ts"], r["coin"], r["funding_rate"],
            r.get("premium"), r.get("open_interest"),
            r.get("mark_price"), r.get("oracle_price"),
        )
        for r in rows
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(_FUNDING_INSERT, payload)
            written = cur.rowcount
        conn.commit()
    return written or len(payload)


# ─────────────────────────────────────────────────────────────────────────
# Hyperliquid trade tape
# ─────────────────────────────────────────────────────────────────────────
_TRADES_INSERT = """
INSERT INTO hl_trades (ts, coin, trade_id, side, price, size)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (coin, ts, trade_id, price, size) DO NOTHING;
"""


def insert_trades_batch(rows: Iterable[dict]) -> int:
    """Each row: {ts, coin, trade_id, side, price, size}.

    Currently uses executemany. If throughput becomes a bottleneck (>50k
    rows/sec sustained), swap in COPY via cursor.copy("COPY hl_trades ...").
    """
    payload = [
        (
            r["ts"], r["coin"], r.get("trade_id"),
            r["side"], r["price"], r["size"],
        )
        for r in rows
    ]
    if not payload:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(_TRADES_INSERT, payload)
            written = cur.rowcount
        conn.commit()
    return written or len(payload)


# ─────────────────────────────────────────────────────────────────────────
# Collector state — bookkeeping
# ─────────────────────────────────────────────────────────────────────────
_STATE_UPSERT = """
INSERT INTO collector_state (collector, key, last_ts, last_status, last_error, updated_at)
VALUES (%s, %s, %s, %s, %s, NOW())
ON CONFLICT (collector, key) DO UPDATE SET
    last_ts     = EXCLUDED.last_ts,
    last_status = EXCLUDED.last_status,
    last_error  = EXCLUDED.last_error,
    updated_at  = NOW();
"""


def mark_collector_state(
    collector: str,
    key: str,
    last_ts: datetime | None = None,
    last_status: str = "ok",
    last_error: str | None = None,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_STATE_UPSERT, (collector, key, last_ts, last_status, last_error))
        conn.commit()


def get_collector_state(collector: str, key: str) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM collector_state WHERE collector = %s AND key = %s",
                (collector, key),
            )
            return cur.fetchone()


# ─────────────────────────────────────────────────────────────────────────
# Health / smoke check
# ─────────────────────────────────────────────────────────────────────────
def ping() -> dict:
    """Quick verification that DB + extension are reachable."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database() AS db, current_user AS usr, version() AS ver;")
            info = cur.fetchone()
            cur.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"
            )
            ts_row = cur.fetchone()
    return {
        "database": info["db"],
        "user": info["usr"],
        "postgres": info["ver"],
        "timescaledb": ts_row["extversion"] if ts_row else None,
    }
