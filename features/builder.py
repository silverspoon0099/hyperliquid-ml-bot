"""Master feature builder.

Loads raw 5min + 1H OHLCV from parquet, computes all 268 Phase 1 features,
merges 1H into 5min (using PREVIOUS 1H bar to avoid look-ahead), applies the
preprocessing pipeline, and writes the feature matrix.

Usage:
    python -m features.builder --symbol BTC/USDT
    python -m features.builder --symbol BTC/USDT --no-label  # skip triple-barrier
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.collectors.storage import ohlcv_path, read_ohlcv
from model.labeler import triple_barrier_labels
from utils.config import load_config, resolve_path
from utils.logging_setup import get_logger

from . import (
    adaptive_ma,
    candles,
    context,
    divergence,
    event_memory,
    extra_momentum,
    ichimoku,
    indicators,
    pivots,
    regime,
    sessions,
    stats,
    structure,
    volatility,
    volume,
    vwap,
)

log = get_logger("builder")


def coin_from_symbol(symbol: str) -> str:
    return symbol.split("/")[0].upper()


def load_ohlcv(symbol: str, timeframe: str, storage_dir: Path) -> pd.DataFrame:
    p = ohlcv_path(storage_dir, symbol, timeframe)
    df = read_ohlcv(p)
    if df.empty:
        raise FileNotFoundError(f"No data at {p} — run the fetcher first.")
    df["timestamp"] = pd.to_numeric(df["timestamp"], downcast="integer")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def merge_1h_into_5m(df_5m: pd.DataFrame, feats_1h: pd.DataFrame, ts_1h: pd.Series) -> pd.DataFrame:
    """Map 1H feature values to 5min bars using the PREVIOUS completed 1H bar.

    Per design: shift the 1H frame by 1 before mapping so a 5min bar at hour H
    sees features computed from the 1H bar that closed at H (not H+1).
    """
    # Build the lookup: 1H timestamp -> features (already shifted by 1 outside).
    lookup = feats_1h.copy()
    lookup.index = pd.to_datetime(ts_1h, unit="ms", utc=True)
    lookup = lookup.shift(1)  # use previous completed bar — Decision: avoid look-ahead.

    ts_5m = pd.to_datetime(df_5m["timestamp"], unit="ms", utc=True)
    hour_key = ts_5m.dt.floor("1h")
    mapped = lookup.reindex(hour_key).reset_index(drop=True)
    mapped.index = df_5m.index
    return mapped


def build_features(symbol: str, cfg: dict) -> pd.DataFrame:
    bcfg = cfg["data"]["binance"]
    fcfg = cfg["features"]
    storage_dir = resolve_path(bcfg["storage_dir"])

    log.info(f"Loading OHLCV for {symbol}")
    df_5m = load_ohlcv(symbol, "5m", storage_dir)
    df_1h = load_ohlcv(symbol, "1h", storage_dir)

    log.info(f"  5m bars: {len(df_5m):,}  1h bars: {len(df_1h):,}")

    ts_5m = pd.to_datetime(df_5m["timestamp"], unit="ms", utc=True)
    ts_1h = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)

    day_id = ts_5m.dt.floor("1D")
    session_id = day_id  # using UTC-day as session id for daily VWAP/range groupings

    parts: list[pd.DataFrame] = []

    # ── 5min indicator features ────────────────────────────────────────────
    log.info("Computing 5min indicators (Cat 1-2)")
    rsi_df = indicators.rsi_features(df_5m["close"], fcfg["rsi"])
    parts.append(rsi_df)

    wt5_df = indicators.wavetrend_features(
        df_5m["high"], df_5m["low"], df_5m["close"], fcfg["wavetrend"], suffix="5min"
    )
    parts.append(wt5_df)

    stoch_df = indicators.stoch_features(df_5m["high"], df_5m["low"], df_5m["close"], fcfg["stochastic"])
    parts.append(stoch_df)

    sq_df = indicators.squeeze_features(df_5m["high"], df_5m["low"], df_5m["close"], fcfg["squeeze"])
    parts.append(sq_df)

    macd5_df = indicators.macd_features_5m(df_5m["close"], fcfg["macd"])
    parts.append(macd5_df)

    adx5_df = indicators.adx_features_5m(df_5m["high"], df_5m["low"], df_5m["close"], fcfg["adx"])
    parts.append(adx5_df)

    ema5_df = indicators.ema_features_5m(df_5m["close"], fcfg["ema_periods"])
    parts.append(ema5_df)

    # ── Volatility (Cat 3) ────────────────────────────────────────────────
    log.info("Computing volatility (Cat 3)")
    vol_df = volatility.volatility_features(df_5m, fcfg, session_id)
    parts.append(vol_df)

    # ── Volume / VFI (Cat 4) ──────────────────────────────────────────────
    log.info("Computing volume + VFI (Cat 4)")
    vfi5_df = volume.vfi_features_5m(df_5m["close"], df_5m["high"], df_5m["low"], df_5m["volume"], fcfg["vfi"])
    parts.append(vfi5_df)
    vol_basic = volume.volume_features(df_5m)
    parts.append(vol_basic)

    # ── VWAP (Cat 5) ──────────────────────────────────────────────────────
    log.info("Computing VWAP (Cat 5)")
    vwap_df = vwap.vwap_features(df_5m, fcfg, day_id, session_id)
    parts.append(vwap_df)

    # ── Pivots (Cat 6) ────────────────────────────────────────────────────
    log.info("Computing Pivot Fibonacci (Cat 6)")
    pivots_df = pivots.pivot_features(df_5m, day_id, fcfg["pivots"]["test_tolerance_pct"])
    parts.append(pivots_df)

    # ── Sessions (Cat 7) ──────────────────────────────────────────────────
    log.info("Computing sessions (Cat 7)")
    sess_df = sessions.session_features(df_5m)
    parts.append(sess_df)

    # ── Candles (Cat 8) ───────────────────────────────────────────────────
    log.info("Computing candles (Cat 8)")
    candle_df = candles.candle_features(df_5m)
    parts.append(candle_df)

    # ── Mean reversion / stats (Cat 9) ────────────────────────────────────
    log.info("Computing mean-reversion (Cat 9)")
    bb_period = fcfg["bb"]["period"]
    bb_basis = df_5m["close"].rolling(bb_period, min_periods=bb_period).mean()
    bb_dev = df_5m["close"].rolling(bb_period, min_periods=bb_period).std(ddof=0)
    bb_up = bb_basis + fcfg["bb"]["std"] * bb_dev
    bb_lo = bb_basis - fcfg["bb"]["std"] * bb_dev
    bb_position = (df_5m["close"] - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)
    mr_df = stats.mean_reversion_features(df_5m, rsi_df["rsi_14"], bb_position)
    parts.append(mr_df)

    # ── Regime (Cat 10) ───────────────────────────────────────────────────
    log.info("Computing regime (Cat 10)")
    reg_df = regime.regime_features(
        df_5m,
        adx=adx5_df["adx"],
        di_plus=adx5_df["di_plus"],
        di_minus=adx5_df["di_minus"],
        bb_width_percentile=vol_df["bb_width_percentile"],
        atr_percentile=vol_df["atr_percentile"],
        atr_series=vol_df["atr_14"],
    )
    parts.append(reg_df)

    # ── Previous context (Cat 11) ─────────────────────────────────────────
    log.info("Computing previous context (Cat 11)")
    prev_ctx = context.previous_context_features(df_5m, day_id, pivots_df["pivot_P"])
    parts.append(prev_ctx)

    # ── Lagged dynamics (Cat 12) ──────────────────────────────────────────
    log.info("Computing lagged dynamics (Cat 12)")
    lag_df = context.lagged_features(
        df_5m,
        rsi_series=rsi_df["rsi_14"],
        wt1=wt5_df["wt1_5min"],
        adx=adx5_df["adx"],
        atr_series=vol_df["atr_14"],
        vwap_daily=vwap_df["vwap_daily"],
        di_plus=adx5_df["di_plus"],
        di_minus=adx5_df["di_minus"],
        squeeze_mom=sq_df["squeeze_momentum"],
    )
    parts.append(lag_df)

    # ── Divergences (Cat 13) ──────────────────────────────────────────────
    log.info("Computing divergences (Cat 13)")
    div_df = divergence.divergence_features(
        df_5m,
        rsi_series=rsi_df["rsi_14"],
        macd_hist=macd5_df["macd_hist_5min"],
        wt1=wt5_df["wt1_5min"],
        stoch_k=stoch_df["stoch_k"],
    )
    parts.append(div_df)

    # ── Money flow (Cat 14) ───────────────────────────────────────────────
    log.info("Computing money flow (Cat 14)")
    mf_df = volume.money_flow_features(df_5m, vfi5_df["vfi"], rsi_df["rsi_14"])
    parts.append(mf_df)

    # ── Extra momentum (Cat 15) ───────────────────────────────────────────
    log.info("Computing extra momentum (Cat 15)")
    em_df = extra_momentum.extra_momentum_features(df_5m, fcfg)
    parts.append(em_df)

    # ── Market structure (Cat 16) ─────────────────────────────────────────
    log.info("Computing market structure (Cat 16)")
    struct_df = structure.structure_features(df_5m, fcfg["swing"]["fractal_lookback"])
    parts.append(struct_df)

    # ── Statistical / Fractal (Cat 17) ────────────────────────────────────
    log.info("Computing statistical/fractal (Cat 17)")
    stats_df = stats.stats_features(df_5m, fcfg)
    parts.append(stats_df)

    # ── Adaptive MAs (Cat 18) ─────────────────────────────────────────────
    log.info("Computing adaptive MAs (Cat 18)")
    am_df = adaptive_ma.adaptive_ma_features(df_5m, fcfg)
    parts.append(am_df)

    # ── Ichimoku (Cat 19) ─────────────────────────────────────────────────
    log.info("Computing Ichimoku (Cat 19)")
    ich_df = ichimoku.ichimoku_features(df_5m, fcfg["ichimoku"])
    parts.append(ich_df)

    # ── Event memory (Cat 20) ─────────────────────────────────────────────
    log.info("Computing event memory (Cat 20)")
    em_cfg = fcfg["event_memory"]
    parts.append(event_memory.stoch_event_memory(stoch_df["stoch_k"], stoch_df["stoch_d"], em_cfg))
    parts.append(
        event_memory.wt_event_memory(
            wt5_df["wt1_5min"], wt5_df["wt2_5min"], fcfg["wavetrend"]["ob_level"], fcfg["wavetrend"]["os_level"]
        )
    )
    parts.append(event_memory.rsi_event_memory(rsi_df["rsi_14"], em_cfg))
    parts.append(event_memory.macd_event_memory(macd5_df["macd_hist_5min"], macd5_df["macd_line_5min"]))
    parts.append(event_memory.squeeze_event_memory(vol_df["squeeze_state"], sq_df["squeeze_momentum"]))
    parts.append(event_memory.adx_event_memory(adx5_df["adx"], adx5_df["di_plus"], adx5_df["di_minus"], em_cfg))
    parts.append(
        event_memory.price_event_memory(
            df_5m,
            pivots_df["dist_to_nearest_pivot_pct"],
            day_id,
            em_cfg["volume_spike_mult"],
            fcfg["pivots"]["test_tolerance_pct"],
        )
    )

    # ── 1H features ───────────────────────────────────────────────────────
    log.info("Computing 1H features")
    feats_1h = pd.concat(
        [
            indicators.wavetrend_features(
                df_1h["high"], df_1h["low"], df_1h["close"], fcfg["wavetrend"], suffix="1h"
            ),
            indicators.macd_features_1h(df_1h["close"], fcfg["macd"]),
            indicators.adx_features_1h(df_1h["high"], df_1h["low"], df_1h["close"], fcfg["adx"]),
            indicators.ema_features_1h(df_1h["close"], fcfg["ema_periods"]),
            volume.vfi_features_1h(df_1h["close"], df_1h["high"], df_1h["low"], df_1h["volume"], fcfg["vfi"]),
        ],
        axis=1,
    )
    feats_1h_mapped = merge_1h_into_5m(df_5m, feats_1h, df_1h["timestamp"])
    parts.append(feats_1h_mapped)

    # ── Assemble ──────────────────────────────────────────────────────────
    log.info("Assembling feature matrix")
    feature_matrix = pd.concat(parts, axis=1)
    feature_matrix.insert(0, "timestamp", df_5m["timestamp"].values)
    feature_matrix.insert(1, "open", df_5m["open"].values)
    feature_matrix.insert(2, "high", df_5m["high"].values)
    feature_matrix.insert(3, "low", df_5m["low"].values)
    feature_matrix.insert(4, "close", df_5m["close"].values)
    feature_matrix.insert(5, "volume", df_5m["volume"].values)

    log.info(
        f"Feature matrix shape: {feature_matrix.shape} "
        f"(features: {feature_matrix.shape[1] - 6})"
    )
    return feature_matrix


def add_labels(feature_matrix: pd.DataFrame, symbol: str, cfg: dict) -> pd.DataFrame:
    lcfg = cfg["labeling"]
    coin = coin_from_symbol(symbol)
    min_profit = lcfg["min_profit_pct"].get(coin, 0.5)
    log.info(
        f"Triple-barrier labels: tp={lcfg['tp_atr_mult']}x, sl={lcfg['sl_atr_mult']}x, "
        f"hold={lcfg['max_holding_bars']}, min_profit_pct={min_profit}"
    )
    labels = triple_barrier_labels(
        feature_matrix,
        atr_col="atr_14",
        tp_atr_mult=lcfg["tp_atr_mult"],
        sl_atr_mult=lcfg["sl_atr_mult"],
        max_holding_bars=lcfg["max_holding_bars"],
        min_profit_pct=min_profit,
        classes=lcfg["classes"],
    )
    out = pd.concat([feature_matrix, labels], axis=1)
    dist = out["label"].value_counts().to_dict()
    log.info(f"Label distribution: {dist}")
    return out


def trim_warmup(df: pd.DataFrame, warmup: int) -> pd.DataFrame:
    return df.iloc[warmup:].reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--no-label", action="store_true")
    parser.add_argument("--output-dir", default=cfg["features"]["output_dir"])
    args = parser.parse_args()

    matrix = build_features(args.symbol, cfg)
    matrix = trim_warmup(matrix, cfg["features"]["warmup_bars"])

    if not args.no_label:
        matrix = add_labels(matrix, args.symbol, cfg)
        # Drop unlabelable tail rows.
        matrix = matrix[matrix["label"] != -1].reset_index(drop=True)

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coin = coin_from_symbol(args.symbol)
    out_path = out_dir / f"{coin}_features.parquet"
    matrix.to_parquet(out_path, index=False)
    log.info(f"Saved {len(matrix):,} rows × {matrix.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    main()
