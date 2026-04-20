-- ML Trading Bot — initial schema (TimescaleDB)
-- Run as the database owner:
--   psql -h 127.0.0.1 -U lucasbogi -d ml_trading -f db/migrations/001_init.sql
--
-- Idempotent: safe to re-run (CREATE ... IF NOT EXISTS, ON CONFLICT DO NOTHING).
-- Designed for 5 years × 5 symbols, with two more symbols easily added later.

BEGIN;

-- TimescaleDB must already be installed (CREATE EXTENSION run by superuser).
-- We just verify it's loaded.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        RAISE EXCEPTION 'timescaledb extension is not installed in this database';
    END IF;
END$$;


-- ─────────────────────────────────────────────────────────────────────────
-- 1. OHLCV — Binance Futures, 5m and 1h
--    Source-of-truth for training. Never dropped.
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv (
    ts          TIMESTAMPTZ      NOT NULL,
    exchange    TEXT             NOT NULL,           -- 'binance'
    symbol      TEXT             NOT NULL,           -- 'BTC/USDT'
    timeframe   TEXT             NOT NULL,           -- '5m' | '1h'
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    quote_volume DOUBLE PRECISION,                   -- Binance returns this; nullable for safety
    trades_count INTEGER,
    inserted_at TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (exchange, symbol, timeframe, ts)
);

SELECT create_hypertable(
    'ohlcv', 'ts',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);
-- Space-partition on (exchange, symbol, timeframe) so each instrument lives
-- in its own chunk lineage. 8 partitions = headroom for 5 symbols × 2 tfs.
SELECT add_dimension(
    'ohlcv', 'symbol',
    number_partitions => 8,
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS ohlcv_symbol_tf_ts_desc
    ON ohlcv (symbol, timeframe, ts DESC);

-- Compression: columnar storage after 7 days. ~10x smaller, still queryable.
ALTER TABLE ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'exchange, symbol, timeframe',
    timescaledb.compress_orderby = 'ts DESC'
);

SELECT add_compression_policy('ohlcv', INTERVAL '7 days', if_not_exists => TRUE);
-- No retention policy — OHLCV is the source-of-truth.


-- ─────────────────────────────────────────────────────────────────────────
-- 2. hl_funding — Hyperliquid funding rates (HOURLY, not 8h)
--    Tiny table (~24 rows/day/coin). Cheap to keep forever.
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hl_funding (
    ts            TIMESTAMPTZ      NOT NULL,         -- funding apply time
    coin          TEXT             NOT NULL,         -- 'BTC', 'SOL', ...
    funding_rate  DOUBLE PRECISION NOT NULL,         -- e.g. 0.000125 = 0.0125%
    premium       DOUBLE PRECISION,                  -- mark - oracle, if available
    open_interest DOUBLE PRECISION,
    mark_price    DOUBLE PRECISION,
    oracle_price  DOUBLE PRECISION,
    inserted_at   TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (coin, ts)
);

SELECT create_hypertable(
    'hl_funding', 'ts',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS hl_funding_coin_ts_desc
    ON hl_funding (coin, ts DESC);

ALTER TABLE hl_funding SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'coin',
    timescaledb.compress_orderby = 'ts DESC'
);

SELECT add_compression_policy('hl_funding', INTERVAL '30 days', if_not_exists => TRUE);


-- ─────────────────────────────────────────────────────────────────────────
-- 3. hl_trades — Hyperliquid public trade tape
--    High-frequency: ~100k-1M rows/day/coin during active markets.
--    Compressed aggressively (after 2 days). Retention 365 days by default.
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hl_trades (
    ts            TIMESTAMPTZ      NOT NULL,         -- exchange timestamp
    coin          TEXT             NOT NULL,
    trade_id      BIGINT,                            -- Hyperliquid tid (nullable for safety)
    side          TEXT             NOT NULL,         -- 'B' (buy) | 'A' (sell, ask side)
    price         DOUBLE PRECISION NOT NULL,
    size          DOUBLE PRECISION NOT NULL,
    inserted_at   TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'hl_trades', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
SELECT add_dimension(
    'hl_trades', 'coin',
    number_partitions => 8,
    if_not_exists => TRUE
);

-- We don't have a real natural PK (trade_id can repeat across reconnects);
-- enforce uniqueness via index for ON CONFLICT upserts.
CREATE UNIQUE INDEX IF NOT EXISTS hl_trades_uniq
    ON hl_trades (coin, ts, trade_id, price, size);

CREATE INDEX IF NOT EXISTS hl_trades_coin_ts_desc
    ON hl_trades (coin, ts DESC);

ALTER TABLE hl_trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'coin',
    timescaledb.compress_orderby = 'ts DESC'
);

SELECT add_compression_policy('hl_trades', INTERVAL '2 days', if_not_exists => TRUE);
SELECT add_retention_policy('hl_trades', INTERVAL '365 days', if_not_exists => TRUE);


-- ─────────────────────────────────────────────────────────────────────────
-- 4. hl_l2_snapshots — DORMANT (Phase 3)
--    Schema kept so collectors can be enabled with no migration later.
--    Stored as JSONB to retain full top-N book without column explosion.
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hl_l2_snapshots (
    ts            TIMESTAMPTZ NOT NULL,
    coin          TEXT        NOT NULL,
    bids          JSONB       NOT NULL,   -- top-N: [[price, size], ...]
    asks          JSONB       NOT NULL,
    mid_price     DOUBLE PRECISION,
    spread_bps    DOUBLE PRECISION,
    inserted_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (coin, ts)
);

SELECT create_hypertable(
    'hl_l2_snapshots', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
SELECT add_dimension(
    'hl_l2_snapshots', 'coin',
    number_partitions => 8,
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS hl_l2_coin_ts_desc
    ON hl_l2_snapshots (coin, ts DESC);

ALTER TABLE hl_l2_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'coin',
    timescaledb.compress_orderby = 'ts DESC'
);

SELECT add_compression_policy('hl_l2_snapshots', INTERVAL '1 day', if_not_exists => TRUE);
SELECT add_retention_policy('hl_l2_snapshots', INTERVAL '180 days', if_not_exists => TRUE);


-- ─────────────────────────────────────────────────────────────────────────
-- 5. hl_book_features_5m — DORMANT (Phase 3)
--    Pre-aggregated L2 features at the same 5-minute grid as OHLCV.
--    Phase 3 will populate this from hl_l2_snapshots; included now so the
--    feature pipeline can JOIN against a stable schema.
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hl_book_features_5m (
    ts                  TIMESTAMPTZ NOT NULL,
    coin                TEXT NOT NULL,
    spread_bps_mean     DOUBLE PRECISION,
    spread_bps_max      DOUBLE PRECISION,
    bid_depth_top10     DOUBLE PRECISION,
    ask_depth_top10     DOUBLE PRECISION,
    book_imbalance      DOUBLE PRECISION,
    cvd                 DOUBLE PRECISION,            -- cumulative volume delta from trades
    taker_buy_ratio     DOUBLE PRECISION,
    inserted_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (coin, ts)
);

SELECT create_hypertable(
    'hl_book_features_5m', 'ts',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);


-- ─────────────────────────────────────────────────────────────────────────
-- 6. collector_state — bookkeeping for resumable backfills
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS collector_state (
    collector   TEXT NOT NULL,                       -- 'binance_ohlcv' | 'hl_trades' | ...
    key         TEXT NOT NULL,                       -- 'BTC/USDT:5m' | 'BTC' | ...
    last_ts     TIMESTAMPTZ,
    last_status TEXT,
    last_error  TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (collector, key)
);


-- ─────────────────────────────────────────────────────────────────────────
-- 7. schema_version — applied migrations
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO schema_version (version, name)
VALUES (1, '001_init')
ON CONFLICT (version) DO NOTHING;

COMMIT;
