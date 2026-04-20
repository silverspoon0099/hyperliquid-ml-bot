"""Cat 5 — VWAP (8 features). Daily and session VWAP with bands."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import pct, safe_div


def vwap_group(typical: pd.Series, volume: pd.Series, group: pd.Series) -> pd.Series:
    """Group-aware cumulative VWAP — resets at each group boundary."""
    pv = typical * volume
    cum_pv = pv.groupby(group).cumsum()
    cum_v = volume.groupby(group).cumsum()
    return safe_div(cum_pv, cum_v)


def vwap_features(
    df: pd.DataFrame,
    cfg: dict,
    day_id: pd.Series,
    session_id: pd.Series,
) -> pd.DataFrame:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    typical = (high + low + close) / 3
    bands_window = cfg["vwap"]["bands_window"]

    daily_vwap = vwap_group(typical, volume, day_id)
    session_vwap = vwap_group(typical, volume, session_id)

    # Bands — daily-reset rolling stdev of close.
    def daily_stdev(x: pd.Series) -> pd.Series:
        return x.groupby(day_id).transform(
            lambda s: s.rolling(bands_window, min_periods=bands_window).std(ddof=0)
        )

    stdev_band = daily_stdev(close)
    upper = daily_vwap + stdev_band
    lower = daily_vwap - stdev_band
    band_pos = safe_div(close - lower, upper - lower)

    return pd.DataFrame(
        {
            "vwap_daily": daily_vwap,
            "vwap_dist_pct": pct(close - daily_vwap, close),
            "vwap_session": session_vwap,
            "vwap_session_dist_pct": pct(close - session_vwap, close),
            "vwap_slope": pct(daily_vwap - daily_vwap.shift(5), daily_vwap.shift(5)),
            "vwap_upper_band": upper,
            "vwap_lower_band": lower,
            "vwap_band_position": band_pos,
        }
    )
