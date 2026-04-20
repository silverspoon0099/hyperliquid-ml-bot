"""Cat 16 — Market Structure (10 features) — HH/HL/LH/LL via fractal pivots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import bars_since, pct, safe_div
from .divergence import fractal_pivots


def _pivot_running_count(pivots: pd.Series, n: int, up: bool) -> pd.Series:
    """At each bar, count of direction-matching steps within the last n pivots. Forward-filled."""
    vals = pivots.dropna()
    if len(vals) < 2:
        return pd.Series(0.0, index=pivots.index)
    arr = vals.to_numpy()
    diffs = np.diff(arr)
    hits = (diffs > 0 if up else diffs < 0).astype(int)
    cs = np.concatenate([[0], np.cumsum(hits)])
    out = np.zeros(len(arr), dtype=float)
    for i in range(1, len(arr)):
        lo = max(0, i - (n - 1))
        out[i] = cs[i] - cs[lo]
    return pd.Series(out, index=vals.index).reindex(pivots.index).ffill().fillna(0)


def _swing_ratio(p_high: pd.Series, p_low: pd.Series) -> pd.Series:
    """Most recent swing length / previous swing length, from alternating H/L pivots.

    Walks pivots in order, enforces H/L alternation (consecutive same-kind pivots collapse
    to the more extreme one). Each bar carries the ratio of the last two completed swings.
    """
    n = len(p_high)
    ph = p_high.to_numpy()
    pl = p_low.to_numpy()
    out = np.full(n, np.nan)
    chain_prices: list[float] = []
    chain_kinds: list[str] = []
    for i in range(n):
        if not np.isnan(ph[i]):
            if chain_kinds and chain_kinds[-1] == "H":
                if ph[i] > chain_prices[-1]:
                    chain_prices[-1] = ph[i]
            else:
                chain_prices.append(float(ph[i]))
                chain_kinds.append("H")
        if not np.isnan(pl[i]):
            if chain_kinds and chain_kinds[-1] == "L":
                if pl[i] < chain_prices[-1]:
                    chain_prices[-1] = pl[i]
            else:
                chain_prices.append(float(pl[i]))
                chain_kinds.append("L")
        if len(chain_prices) >= 3:
            cur = abs(chain_prices[-1] - chain_prices[-2])
            prev = abs(chain_prices[-2] - chain_prices[-3])
            if prev > 0:
                out[i] = cur / prev
    return pd.Series(out, index=p_high.index).ffill()


def _structure_type_series(p_high: pd.Series, p_low: pd.Series) -> pd.Series:
    """+1 HH+HL uptrend, -1 LL+LH downtrend, 0 otherwise. Single pass, O(N)."""
    ph = p_high.to_numpy()
    pl = p_low.to_numpy()
    out = np.zeros(len(ph))
    last_hs: list[float] = []
    last_ls: list[float] = []
    for i in range(len(ph)):
        if not np.isnan(ph[i]):
            last_hs.append(float(ph[i]))
            if len(last_hs) > 2:
                last_hs.pop(0)
        if not np.isnan(pl[i]):
            last_ls.append(float(pl[i]))
            if len(last_ls) > 2:
                last_ls.pop(0)
        if len(last_hs) == 2 and len(last_ls) == 2:
            hh = last_hs[-1] > last_hs[-2]
            hl = last_ls[-1] > last_ls[-2]
            ll = last_ls[-1] < last_ls[-2]
            lh = last_hs[-1] < last_hs[-2]
            if hh and hl:
                out[i] = 1
            elif ll and lh:
                out[i] = -1
    return pd.Series(out, index=p_high.index)


def structure_features(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]

    # Pivot highs from `high`, pivot lows from `low`.
    p_high, _ = fractal_pivots(high, lookback)
    _, p_low = fractal_pivots(low, lookback)

    # Forward-fill so each bar carries the most recent confirmed pivot value.
    last_high = p_high.ffill()
    last_low = p_low.ffill()

    swing_high_dist = pct(close - last_high, close)
    swing_low_dist = pct(close - last_low, close)

    structure_type = _structure_type_series(p_high, p_low)

    structure_changed = (structure_type != structure_type.shift(1)) & (structure_type != 0)
    bars_since_break = bars_since(structure_changed)

    swing_range_pct = pct(last_high - last_low, close)

    higher_highs = _pivot_running_count(p_high, n=20, up=True)
    lower_lows = _pivot_running_count(p_low, n=20, up=False)

    swing_ratio = _swing_ratio(p_high, p_low)

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
