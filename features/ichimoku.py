"""Cat 19 — Ichimoku partial (5 features). Tenkan, Kijun, TK cross, cloud."""
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
    senkou_a = ((tenkan + kijun) / 2)
    # Use the leading-span value at the same bar (no forward-shift; we want
    # current cloud projection's reference value vs current close).

    return pd.DataFrame(
        {
            "tenkan_dist_pct": pct(close - tenkan, close),
            "kijun_dist_pct": pct(close - kijun, close),
            "tk_cross": np.sign(tenkan - kijun),
            "tk_spread": pct(tenkan - kijun, close),
            "cloud_dist_pct": pct(close - senkou_a, close),
        }
    )
