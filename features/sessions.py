"""Cat 7 — Session & Time context (15 features).

Session times in UTC (per Sessions.pine [LuxAlgo]):
  Sydney   21:00-06:00
  Tokyo    00:00-09:00
  London   07:00-16:00
  New York 13:00-22:00
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SESSIONS = {
    "sydney": (21, 6),
    "tokyo": (0, 9),
    "london": (7, 16),
    "new_york": (13, 22),
}


def in_session(hour: pd.Series, start: int, end: int) -> pd.Series:
    if start < end:
        return ((hour >= start) & (hour < end)).astype(int)
    # wraps midnight
    return ((hour >= start) | (hour < end)).astype(int)


def session_id_5min(ts: pd.Series) -> pd.Series:
    """Define the trading session as the UTC date — used for VWAP/range groupings.

    A more granular per-major-session grouping would be possible, but for daily
    pivot/VWAP alignment a UTC-day grouping is the convention used by Pivots Fib
    and most session-VWAP indicators.
    """
    return ts.dt.floor("1D")


def primary_session(hour: int) -> str:
    """Most-recently-opened session at this hour. Used for "minutes_into_session"."""
    # Order by start hour ascending; session whose start is closest to (and <=) hour.
    starts = [
        ("tokyo", 0),
        ("london", 7),
        ("new_york", 13),
        ("sydney", 21),
    ]
    chosen = "sydney"
    for name, h in starts:
        if h <= hour:
            chosen = name
    return chosen


def session_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    hour = ts.dt.hour
    minute = ts.dt.minute
    dow = ts.dt.dayofweek

    s_sydney = in_session(hour, *SESSIONS["sydney"])
    s_tokyo = in_session(hour, *SESSIONS["tokyo"])
    s_london = in_session(hour, *SESSIONS["london"])
    s_ny = in_session(hour, *SESSIONS["new_york"])

    overlap_tk_ld = ((hour >= 7) & (hour < 9)).astype(int)
    overlap_ld_ny = ((hour >= 13) & (hour < 16)).astype(int)
    overlap_ny_sy = (hour == 21).astype(int)
    overlap_sy_tk = ((hour >= 0) & (hour < 6)).astype(int)

    active = s_sydney + s_tokyo + s_london + s_ny

    # Primary session = most recently opened.
    primary = hour.apply(primary_session)
    starts = {"tokyo": 0, "london": 7, "new_york": 13, "sydney": 21}
    ends = {"tokyo": 9, "london": 16, "new_york": 22, "sydney": 6}
    primary_start = primary.map(starts)
    primary_end = primary.map(ends)

    # Minutes into / to close — handle Sydney wraparound (start=21, end=6 next day).
    cur_min = hour * 60 + minute
    start_min = primary_start * 60
    end_min = primary_end * 60
    mins_in = np.where(
        primary == "sydney",
        np.where(hour >= 21, cur_min - start_min, cur_min + (24 * 60 - start_min)),
        cur_min - start_min,
    )
    mins_to = np.where(
        primary == "sydney",
        np.where(hour < 6, end_min - cur_min, (24 * 60 + end_min) - cur_min),
        end_min - cur_min,
    )

    # Session range vs avg of last 20 same-session (use UTC day as session id for simplicity).
    day_id = ts.dt.floor("1D")
    grp = df.groupby(day_id)
    s_high = grp["high"].transform("max")
    s_low = grp["low"].transform("min")
    s_range = s_high - s_low
    # Take the last value per day, then rolling 20-day avg.
    daily_range = s_range.groupby(day_id).last()
    daily_avg20 = daily_range.shift(1).rolling(20, min_periods=20).mean()
    avg_map = daily_avg20.reindex(day_id.values).set_axis(df.index)
    range_vs_avg = s_range / avg_map

    # Previous session (= previous UTC day).
    prev_range_pct = (
        ((daily_range - daily_range) * 0).rename("prev_session_range_pct")  # placeholder
    )
    prev_high = grp["high"].transform("max")
    prev_low = grp["low"].transform("min")
    prev_open = grp["open"].transform("first")
    prev_session_range = (prev_high - prev_low) / prev_open * 100
    prev_session_range_per_day = prev_session_range.groupby(day_id).last().shift(1)
    prev_range_pct = prev_session_range_per_day.reindex(day_id.values).set_axis(df.index)

    return pd.DataFrame(
        {
            "session_sydney": s_sydney,
            "session_tokyo": s_tokyo,
            "session_london": s_london,
            "session_new_york": s_ny,
            "overlap_tokyo_london": overlap_tk_ld,
            "overlap_london_ny": overlap_ld_ny,
            "overlap_ny_sydney": overlap_ny_sy,
            "overlap_sydney_tokyo": overlap_sy_tk,
            "active_session_count": active,
            "minutes_into_session": mins_in,
            "minutes_to_session_close": mins_to,
            "session_range_vs_avg": range_vs_avg,
            "prev_session_range_pct": prev_range_pct,
            "day_of_week": dow,
            "is_monday": (dow == 0).astype(int),
        },
        index=df.index,
    )
