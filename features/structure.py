"""Cat 16 — Market Structure (10 features) — HH/HL/LH/LL via fractal pivots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import bars_since, pct, safe_div
from .divergence import fractal_pivots


def structure_features(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]

    p_high, p_low = fractal_pivots(high, lookback)
    _, p_low_low = fractal_pivots(low, lookback)
    p_low = p_low_low  # fractal_pivots(high) returned high-pivot of high; we need low pivots of low.

    # Recompute correctly — pivot highs from `high`, pivot lows from `low`.
    p_high, _ = fractal_pivots(high, lookback)
    _, p_low = fractal_pivots(low, lookback)

    # Forward-fill so each bar carries the most recent confirmed pivot value.
    last_high = p_high.ffill()
    last_low = p_low.ffill()

    swing_high_dist = pct(close - last_high, close)
    swing_low_dist = pct(close - last_low, close)

    # Structure type: HH+HL = uptrend (+1), LH+LL = downtrend (-1), else 0.
    high_pivots_only = p_high.dropna()
    low_pivots_only = p_low.dropna()

    def label_structure(idx_pos: int) -> int:
        # Look back through high pivots (last 2) and low pivots (last 2) up to bar idx_pos.
        hp = high_pivots_only[high_pivots_only.index <= close.index[idx_pos]].tail(2)
        lp = low_pivots_only[low_pivots_only.index <= close.index[idx_pos]].tail(2)
        if len(hp) < 2 or len(lp) < 2:
            return 0
        hh = hp.iloc[-1] > hp.iloc[-2]
        hl = lp.iloc[-1] > lp.iloc[-2]
        ll = lp.iloc[-1] < lp.iloc[-2]
        lh = hp.iloc[-1] < hp.iloc[-2]
        if hh and hl:
            return 1
        if ll and lh:
            return -1
        return 0

    structure_type = pd.Series([label_structure(i) for i in range(len(close))], index=close.index)

    # Bars since structure break = bars since structure_type changes.
    structure_changed = (structure_type != structure_type.shift(1)) & (structure_type != 0)
    bars_since_break = bars_since(structure_changed)

    swing_range_pct = pct(last_high - last_low, close)

    # Higher-highs count in last 20 high pivots.
    def hh_count(p: pd.Series, n: int = 20, direction: str = "up") -> pd.Series:
        out = []
        vals = p.dropna()
        positions = vals.index
        all_idx = p.index
        # Map each row to "count of pivots in last N where each is greater than its predecessor".
        rolling_count = []
        for idx in all_idx:
            window = vals[vals.index <= idx].tail(n)
            if len(window) < 2:
                rolling_count.append(0)
            elif direction == "up":
                rolling_count.append(int((window.diff().dropna() > 0).sum()))
            else:
                rolling_count.append(int((window.diff().dropna() < 0).sum()))
        return pd.Series(rolling_count, index=all_idx)

    higher_highs = hh_count(p_high, n=20, direction="up")
    lower_lows = hh_count(p_low, n=20, direction="down")

    # Swing ratio — most recent swing length / previous swing length (high-low alternation).
    pivots_combined = pd.concat(
        [p_high.rename("ph"), p_low.rename("pl")], axis=1
    )
    # Use absolute change between last_high and last_low.
    swing_size = (last_high - last_low).abs()
    swing_ratio = safe_div(swing_size, swing_size.shift(20))

    retrace_depth = safe_div(last_high - close, last_high - last_low)
    range_position = safe_div(
        close - low.rolling(20, min_periods=20).min(),
        high.rolling(20, min_periods=20).max() - low.rolling(20, min_periods=20).min(),
    )

    return pd.DataFrame(
        {
            "swing_high_dist_pct": swing_high_dist,
            "swing_low_dist_pct": swing_low_dist,
            "structure_type": structure_type,
            "bars_since_structure_break": bars_since_break,
            "swing_range_pct": swing_range_pct,
            "higher_highs_count_20": higher_highs,
            "lower_lows_count_20": lower_lows,
            "swing_ratio": swing_ratio,
            "retrace_depth": retrace_depth,
            "range_position": range_position,
        }
    )
