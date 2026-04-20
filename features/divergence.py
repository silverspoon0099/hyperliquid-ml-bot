"""Cat 13 — Divergence Detection (7 features).

Regular bullish: price LL + osc HL.
Regular bearish: price HH + osc LH.
Hidden bullish:  price HL + osc LL.
Hidden bearish:  price LH + osc HH.

Implementation uses fractal pivots (default 5-bar window: 2 left + center + 2 right)
and compares the most recent two confirmed pivots within `lookback` bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import bars_since


def fractal_pivots(series: pd.Series, lookback: int = 5) -> tuple[pd.Series, pd.Series]:
    """Return (pivot_high, pivot_low) — value at the pivot bar, NaN elsewhere.

    A bar is a pivot high if it's the max of `lookback` window centered on it.
    """
    half = lookback // 2
    rolling = series.rolling(lookback, center=True, min_periods=lookback)
    is_high = series == rolling.max()
    is_low = series == rolling.min()
    pivot_high = series.where(is_high)
    pivot_low = series.where(is_low)
    # Confirm pivots only after `half` bars passed (no future leak).
    pivot_high = pivot_high.shift(half)
    pivot_low = pivot_low.shift(half)
    return pivot_high, pivot_low


def detect_divergence(
    price: pd.Series, oscillator: pd.Series, lookback_bars: int = 14
) -> tuple[pd.Series, pd.Series]:
    """Return (regular_div, hidden_div) — both signed: +1 bullish, -1 bearish, 0 none.

    Project-wide convention: +1 = bullish/long-favoring, -1 = bearish/short-favoring.
    """
    p_high, p_low = fractal_pivots(price)
    o_high, o_low = fractal_pivots(oscillator)

    regular = pd.Series(0.0, index=price.index)
    hidden = pd.Series(0.0, index=price.index)

    # Walk through bars; for each row, look back lookback_bars to find a prior
    # confirmed pivot of the same type. Compare current vs prior.
    last_p_high_idx = -1
    last_p_high_val = np.nan
    last_o_high_val = np.nan
    last_p_low_idx = -1
    last_p_low_val = np.nan
    last_o_low_val = np.nan

    p_h_arr = p_high.to_numpy()
    p_l_arr = p_low.to_numpy()
    o_h_arr = o_high.to_numpy()
    o_l_arr = o_low.to_numpy()

    for i in range(len(price)):
        # Update most-recent pivots with the values confirmed at bar i.
        if not np.isnan(p_h_arr[i]):
            if last_p_high_idx >= 0 and (i - last_p_high_idx) <= lookback_bars:
                # Compare with previous high.
                if p_h_arr[i] > last_p_high_val and o_h_arr[i] < last_o_high_val:
                    regular.iloc[i] = -1  # regular bearish: price HH + osc LH
                elif p_h_arr[i] < last_p_high_val and o_h_arr[i] > last_o_high_val:
                    hidden.iloc[i] = -1  # hidden bearish: price LH + osc HH
            last_p_high_idx = i
            last_p_high_val = p_h_arr[i]
            last_o_high_val = o_h_arr[i]

        if not np.isnan(p_l_arr[i]):
            if last_p_low_idx >= 0 and (i - last_p_low_idx) <= lookback_bars:
                if p_l_arr[i] < last_p_low_val and o_l_arr[i] > last_o_low_val:
                    regular.iloc[i] = 1  # regular bullish: price LL + osc HL
                elif p_l_arr[i] > last_p_low_val and o_l_arr[i] < last_o_low_val:
                    hidden.iloc[i] = 1  # hidden bullish: price HL + osc LL
            last_p_low_idx = i
            last_p_low_val = p_l_arr[i]
            last_o_low_val = o_l_arr[i]

    return regular, hidden


def divergence_features(
    df: pd.DataFrame,
    rsi_series: pd.Series,
    macd_hist: pd.Series,
    wt1: pd.Series,
    stoch_k: pd.Series,
) -> pd.DataFrame:
    close = df["close"]
    rsi_div, rsi_hidden = detect_divergence(close, rsi_series)
    macd_div, _ = detect_divergence(close, macd_hist)
    wt_div, _ = detect_divergence(close, wt1)
    stoch_div, _ = detect_divergence(close, stoch_k)

    div_count = rsi_div.abs() + macd_div.abs() + wt_div.abs() + stoch_div.abs()
    any_div = (div_count > 0)
    freshness = bars_since(any_div)

    return pd.DataFrame(
        {
            "rsi_price_divergence": rsi_div,
            "macd_price_divergence": macd_div,
            "wt_price_divergence": wt_div,
            "stoch_price_divergence": stoch_div,
            "divergence_count": div_count,
            "hidden_divergence": rsi_hidden,
            "divergence_freshness": freshness,
        }
    )
