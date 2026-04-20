"""Cat 6 — Pivot Fibonacci S/R (13 features).

Daily Fibonacci pivots:
    P  = (prev_H + prev_L + prev_C) / 3
    R1 = P + 0.382 * (prev_H - prev_L)
    R2 = P + 0.618 * (prev_H - prev_L)
    R3 = P + 1.000 * (prev_H - prev_L)
    S1 = P - 0.382 * range
    S2 = P - 0.618 * range
    S3 = P - 1.000 * range

Context: which level is nearest, which zone, approach direction & speed,
how many times the level was tested today.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import pct


PIVOT_NAMES = ["S3", "S2", "S1", "P", "R1", "R2", "R3"]


def daily_pivots(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute next-day's pivot levels from each completed day. Returns one row per day."""
    prev = df_daily.shift(1)
    rng = prev["high"] - prev["low"]
    p = (prev["high"] + prev["low"] + prev["close"]) / 3.0
    out = pd.DataFrame(
        {
            "pivot_S3": p - 1.000 * rng,
            "pivot_S2": p - 0.618 * rng,
            "pivot_S1": p - 0.382 * rng,
            "pivot_P": p,
            "pivot_R1": p + 0.382 * rng,
            "pivot_R2": p + 0.618 * rng,
            "pivot_R3": p + 1.000 * rng,
        },
        index=df_daily.index,
    )
    return out


def pivot_features(df_5m: pd.DataFrame, day_id: pd.Series, tolerance_pct: float) -> pd.DataFrame:
    """Compute all pivot features on the 5min frame.

    Daily pivots are computed from prior-day OHLC then mapped to today's bars.
    """
    # Build per-day OHLC from 5min frame.
    daily = df_5m.groupby(day_id).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    pivots_daily = daily_pivots(daily)
    # Map onto 5min frame by day_id.
    mapped = pivots_daily.loc[day_id.values].set_index(df_5m.index)

    close = df_5m["close"]
    levels = mapped[[f"pivot_{n}" for n in PIVOT_NAMES]]

    # dist_to_nearest, nearest_pivot_type, pivot_zone
    abs_dist = (levels.sub(close, axis=0)).abs()
    nearest_idx = abs_dist.values.argmin(axis=1)
    nearest_dist = np.take_along_axis(abs_dist.values, nearest_idx[:, None], axis=1).ravel()
    dist_pct = (nearest_dist / close.replace(0, np.nan)) * 100.0
    nearest_type = nearest_idx  # 0..6 mapping to S3..R3

    # Zone — which interval the price falls into.
    arr = levels.values
    zone = np.full(len(close), -1, dtype=float)
    closes = close.values
    for i in range(len(close)):
        row = arr[i]
        if np.isnan(row).any():
            continue
        # Sorted by construction: S3 < S2 < S1 < P < R1 < R2 < R3.
        if closes[i] <= row[0]:
            zone[i] = 0
        elif closes[i] >= row[-1]:
            zone[i] = 5
        else:
            for z in range(6):
                if row[z] <= closes[i] < row[z + 1]:
                    zone[i] = z
                    break

    # Approach: direction & speed toward nearest level.
    nearest_level = np.take_along_axis(arr, nearest_idx[:, None], axis=1).ravel()
    nearest_level_s = pd.Series(nearest_level, index=close.index)
    diff = close - nearest_level_s
    diff_prev = diff.shift(1)
    # +1 if magnitude shrinking (approaching) and price falling toward (above level)
    approaching = (diff.abs() < diff_prev.abs())
    direction = np.where(diff > 0, -1, 1)  # price above level -> falling toward it = +1
    approach_dir = pd.Series(np.where(approaching, direction, 0), index=close.index)
    approach_speed = (close - close.shift(3)).abs() / close * 100

    # Times tested today — bars within tolerance per day, cumulative count.
    tol = tolerance_pct / 100.0
    touched = ((close - nearest_level_s).abs() <= nearest_level_s * tol).astype(int)
    times_tested = touched.groupby(day_id).cumsum()

    out = pd.concat(
        [
            mapped.reset_index(drop=True).set_index(df_5m.index),
            pd.DataFrame(
                {
                    "dist_to_nearest_pivot_pct": dist_pct,
                    "nearest_pivot_type": nearest_type,
                    "pivot_zone": zone,
                    "pivot_approach_dir": approach_dir,
                    "pivot_approach_speed": approach_speed,
                    "pivot_times_tested_today": times_tested,
                },
                index=close.index,
            ),
        ],
        axis=1,
    )
    return out
