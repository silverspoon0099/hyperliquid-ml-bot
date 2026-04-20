"""Cat 11 — Previous Context (8) + Cat 12 — Lagged Dynamics (8)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import pct, safe_div


def previous_context_features(
    df: pd.DataFrame, day_id: pd.Series, pivot_p: pd.Series
) -> pd.DataFrame:
    grp = df.groupby(day_id)
    daily = grp.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    prev = daily.shift(1)

    # Map prev-day stats onto each 5min bar.
    def map_to_5m(s: pd.Series) -> pd.Series:
        return s.reindex(day_id.values).set_axis(df.index)

    prev_open = map_to_5m(prev["open"])
    prev_high = map_to_5m(prev["high"])
    prev_low = map_to_5m(prev["low"])
    prev_close = map_to_5m(prev["close"])
    prev_volume = map_to_5m(prev["volume"])

    today_open = map_to_5m(daily["open"])

    prev_session_dir = np.sign(prev_close - prev_open)

    # Volume rank of prev session vs last 20 sessions.
    rolling_vol_rank = prev["volume"].rolling(20, min_periods=20).rank(pct=True) * 100
    prev_vol_rank = map_to_5m(rolling_vol_rank)

    close = df["close"]
    return pd.DataFrame(
        {
            "prev_day_range_pct": pct(prev_high - prev_low, prev_close),
            "prev_day_close_vs_pivot": pct(prev_close - pivot_p, pivot_p),
            "gap_pct": pct(today_open - prev_close, prev_close),
            "dist_to_prev_day_high_pct": pct(close - prev_high, close),
            "dist_to_prev_day_low_pct": pct(close - prev_low, close),
            "prev_session_direction": prev_session_dir,
            "prev_session_volume_rank": prev_vol_rank,
            "daily_open_dist_pct": pct(close - today_open, today_open),
        }
    )


def lagged_features(
    df: pd.DataFrame,
    rsi_series: pd.Series,
    wt1: pd.Series,
    adx: pd.Series,
    atr_series: pd.Series,
    vwap_daily: pd.Series,
    di_plus: pd.Series,
    di_minus: pd.Series,
    squeeze_mom: pd.Series,
) -> pd.DataFrame:
    volume = df["volume"]
    sma_vol3 = volume.rolling(3, min_periods=3).mean()
    di_spread = di_plus - di_minus

    return pd.DataFrame(
        {
            "rsi_5bar_ago": rsi_series.shift(5),
            "wt1_slope_5bar": (wt1 - wt1.shift(5)) / 5,
            "adx_slope_5bar": (adx - adx.shift(5)) / 5,
            "atr_slope_5bar": pct(atr_series - atr_series.shift(5), atr_series.shift(5)),
            "volume_slope_5bar": pct(sma_vol3 - sma_vol3.shift(5), sma_vol3.shift(5)),
            "vwap_slope_5bar": pct(vwap_daily - vwap_daily.shift(5), vwap_daily.shift(5)),
            "di_spread_change_5bar": di_spread - di_spread.shift(5),
            "squeeze_mom_slope": (squeeze_mom - squeeze_mom.shift(3)) / 3,
        }
    )
