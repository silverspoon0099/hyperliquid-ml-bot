"""Cat 10 — Market Regime (7 features)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import bars_since, safe_div


def regime_features(
    df: pd.DataFrame,
    adx: pd.Series,
    di_plus: pd.Series,
    di_minus: pd.Series,
    bb_width_percentile: pd.Series,
    atr_percentile: pd.Series,
    atr_series: pd.Series,
) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    trending = ((adx > 25) & ((di_plus - di_minus).abs() > 10)).astype(int)
    ranging = ((adx < 20) & (bb_width_percentile < 30)).astype(int)
    volatile = (atr_percentile > 80).astype(int)
    quiet = (atr_percentile < 20).astype(int)

    # Dominant regime label per bar (priority: trending > volatile > quiet > ranging > none).
    regime_label = pd.Series(0, index=close.index, dtype=int)
    regime_label[ranging == 1] = 4
    regime_label[quiet == 1] = 3
    regime_label[volatile == 1] = 2
    regime_label[trending == 1] = 1
    bars_in_regime = (regime_label.groupby((regime_label != regime_label.shift()).cumsum()).cumcount() + 1)

    # Efficiency ratio (UT Bot concept).
    delta14 = (close - close.shift(14)).abs()
    sum_abs = close.diff().abs().rolling(14, min_periods=14).sum()
    efficiency = safe_div(delta14, sum_abs)

    # Choppiness Index.
    atr_sum = atr_series.rolling(14, min_periods=14).sum()
    hh = high.rolling(14, min_periods=14).max()
    ll = low.rolling(14, min_periods=14).min()
    chop = 100 * np.log10(safe_div(atr_sum, hh - ll)) / np.log10(14)

    return pd.DataFrame(
        {
            "regime_trending": trending,
            "regime_ranging": ranging,
            "regime_volatile": volatile,
            "regime_quiet": quiet,
            "bars_in_current_regime": bars_in_regime,
            "efficiency_ratio": efficiency,
            "choppiness_index": chop,
        }
    )
