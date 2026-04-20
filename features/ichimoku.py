"""Cat 19 — Ichimoku partial (5 features). Tenkan, Kijun, TK cross, cloud.

Canonical (Hosoda / TradingView ta.ichimoku):
    tenkan   = (maxH(9) + minL(9)) / 2
    kijun    = (maxH(26) + minL(26)) / 2
    senkou_a = (tenkan + kijun) / 2                  plotted 26 bars FORWARD
    senkou_b = (maxH(52) + minL(52)) / 2             plotted 26 bars FORWARD

"Cloud projected over bar t" therefore = the span values computed at t - kijun.
At current bar we read senkou_*.shift(kijun) — no lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import pct


def ichimoku_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    t = cfg["tenkan"]
    k = cfg["kijun"]
    sb = cfg["senkou_b"]

    tenkan = (high.rolling(t, min_periods=t).max() + low.rolling(t, min_periods=t).min()) / 2
    kijun = (high.rolling(k, min_periods=k).max() + low.rolling(k, min_periods=k).min()) / 2

    # Raw leading spans (computed at current bar from current data).
    span_a_raw = (tenkan + kijun) / 2
    span_b_raw = (high.rolling(sb, min_periods=sb).max() + low.rolling(sb, min_periods=sb).min()) / 2

    # Cloud over bar t = span values computed k bars ago.
    senkou_a = span_a_raw.shift(k)
    senkou_b = span_b_raw.shift(k)
    cloud_mid = (senkou_a + senkou_b) / 2

    return pd.DataFrame(
        {
            "tenkan_dist_pct": pct(close - tenkan, close),
            "kijun_dist_pct": pct(close - kijun, close),
            "tk_cross": np.sign(tenkan - kijun),
            "tk_spread": pct(tenkan - kijun, close),
            "cloud_dist_pct": pct(close - cloud_mid, close),
        }
    )
