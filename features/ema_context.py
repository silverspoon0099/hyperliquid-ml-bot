"""Tier-1 setup-context features: EMA touches, confluence, patterns-at-level, MTF.

Layered on top of existing EMA + candle-pattern signals to encode the setups
visible on the user's chart (pullback-to-EMA bounce, pin/engulf at pivot,
confluence, MTF alignment). All features are emitted on the 5m frame; 1h
inputs are merged with the standard 1-bar shift to avoid look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


TOUCH_ATR_MULT = 0.3          # within 0.3·ATR counts as "at the level"
BOUNCE_LOOKAHEAD = 5          # bars after a touch to measure bounce strength
SLOPE_LOOKBACK_5M = 12        # 1h on 5m frame
SLOPE_LOOKBACK_1H = 4         # 4h on 1h frame
RANGE_WINDOW = 20


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _bars_since(mask: pd.Series) -> pd.Series:
    """Bars since the most recent True in `mask`. NaN before any True."""
    idx = np.arange(len(mask))
    last = pd.Series(np.where(mask.values, idx, np.nan), index=mask.index).ffill()
    return pd.Series(idx, index=mask.index) - last


def _touch_mask(close: pd.Series, level: pd.Series, atr: pd.Series) -> pd.Series:
    return (close - level).abs() <= (TOUCH_ATR_MULT * atr)


def ema_context_features(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    pivot_levels_5m: pd.DataFrame,
    atr_14: pd.Series,
    pin_bar: pd.Series,
    engulfing: pd.Series,
) -> pd.DataFrame:
    """Compute Tier-1 setup features.

    Parameters
    ----------
    df_5m : DataFrame with open/high/low/close/volume + ``timestamp`` ms col.
    df_1h : DataFrame same schema, aligned timestamps for merge-back.
    pivot_levels_5m : DataFrame with ``pivot_S3..pivot_R3`` already mapped on 5m.
    atr_14 : 5m atr_14 series (absolute, not pct).
    pin_bar, engulfing : signed {-1, 0, +1} candle-pattern series on 5m.
    """
    close = df_5m["close"].astype(float)
    high = df_5m["high"].astype(float)
    low = df_5m["low"].astype(float)
    out: dict[str, pd.Series] = {}

    # ── 5m EMA touch / bounce (ema21, ema50) ─────────────────────────────
    for span in (21, 50):
        ema = _ema(close, span)
        touch = _touch_mask(close, ema, atr_14)
        out[f"bars_since_ema{span}_touch_5m"] = _bars_since(touch)
        # Bounce strength: current (close - ema)/atr measured *at* the bar that
        # is exactly N bars after the most recent touch, forward-filled so every
        # bar carries the latest completed bounce reading. Causal — the value
        # surfaced at bar t uses only data up to t.
        position = (close - ema) / atr_14
        touch_done = touch.shift(BOUNCE_LOOKAHEAD).fillna(False).astype(bool)
        out[f"ema{span}_bounce_{BOUNCE_LOOKAHEAD}bar_5m"] = (
            position.where(touch_done).ffill()
        )

    # ── 1h EMA touch (ema50, ema200), merged to 5m ───────────────────────
    close_1h = df_1h["close"].astype(float)
    high_1h = df_1h["high"].astype(float)
    low_1h = df_1h["low"].astype(float)
    atr_1h = _atr(high_1h, low_1h, close_1h, 14)
    ts_1h = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    ts_5m = pd.to_datetime(df_5m["timestamp"], unit="ms", utc=True)

    hour_key = ts_5m.dt.floor("1h")
    for span in (50, 200):
        ema_1h_full = _ema(close_1h, span)
        touch_1h = _touch_mask(close_1h, ema_1h_full, atr_1h)
        bars_since_1h = _bars_since(touch_1h)
        # shift(1) before mapping — at hour H, only bars up to H-1 are known
        lookup = bars_since_1h.shift(1)
        lookup.index = ts_1h
        mapped = lookup.reindex(hour_key).reset_index(drop=True)
        out[f"bars_since_ema{span}_touch_1h"] = pd.Series(
            mapped.values, index=df_5m.index
        )

    # ── MTF trend alignment: sign(1h ema50 slope) × sign(5m ema50 slope) ─
    ema50_5m = _ema(close, 50)
    slope_5m = ema50_5m - ema50_5m.shift(SLOPE_LOOKBACK_5M)
    ema50_1h = _ema(close_1h, 50)
    slope_1h = (ema50_1h - ema50_1h.shift(SLOPE_LOOKBACK_1H)).shift(1)
    slope_1h.index = ts_1h
    slope_1h_mapped = slope_1h.reindex(hour_key).reset_index(drop=True)
    slope_1h_on_5m = pd.Series(slope_1h_mapped.values, index=df_5m.index)
    out["mtf_trend_alignment"] = (
        np.sign(slope_5m).fillna(0) * np.sign(slope_1h_on_5m).fillna(0)
    )

    # ── EMA × pivot confluence (min distance between ema_K and any pivot, /atr)
    pivot_cols = [c for c in pivot_levels_5m.columns if c.startswith("pivot_")]
    pivots_arr = pivot_levels_5m[pivot_cols].values  # (n, 7)
    for span in (21, 50):
        ema = _ema(close, span).values
        # broadcast |ema - pivot| over 7 levels, take min
        diff = np.abs(pivots_arr - ema[:, None])
        min_diff = np.nanmin(diff, axis=1)
        out[f"ema{span}_pivot_confluence"] = pd.Series(
            min_diff / atr_14.values, index=df_5m.index
        )

    # ── pin_bar / engulfing × level proximity (sign preserved) ───────────
    # Proximity to ANY of {ema21, ema50, pivot_S3..R3}.
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    near_ema = (_touch_mask(close, ema21, atr_14) | _touch_mask(close, ema50, atr_14))
    # near any pivot level
    close_arr = close.values[:, None]
    near_pivot = (np.abs(pivots_arr - close_arr) <= (TOUCH_ATR_MULT * atr_14.values[:, None]))
    near_pivot_any = pd.Series(np.nansum(near_pivot.astype(int), axis=1) > 0,
                               index=df_5m.index)

    pin = pin_bar.reindex(df_5m.index).astype(float).fillna(0.0)
    eng = engulfing.reindex(df_5m.index).astype(float).fillna(0.0)

    out["pin_bar_at_ema"] = pin.where(near_ema, 0.0)
    out["pin_bar_at_pivot"] = pin.where(near_pivot_any, 0.0)
    out["engulf_at_ema"] = eng.where(near_ema, 0.0)
    out["engulf_at_pivot"] = eng.where(near_pivot_any, 0.0)

    # ── pullback depth vs last 20-bar range (0=at high, 1=at low) ────────
    hi20 = close.rolling(RANGE_WINDOW).max()
    lo20 = close.rolling(RANGE_WINDOW).min()
    rng = (hi20 - lo20).replace(0, np.nan)
    out["pullback_depth_pct"] = ((hi20 - close) / rng).clip(0, 1)
    out["dist_from_range_high_20_pct"] = (hi20 - close) / close * 100.0
    out["dist_from_range_low_20_pct"] = (close - lo20) / close * 100.0

    return pd.DataFrame(out, index=df_5m.index)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()
