"""Shared math helpers for feature modules.

Design notes:
- Wilder's smoothing (RSI, ATR, ADX) implemented as EMA with alpha=1/period
  (TradingView-equivalent).
- All "safe_*" helpers handle div-by-zero by returning NaN (LightGBM handles NaN
  natively).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num.div(den.replace(0, np.nan))


def pct(num: pd.Series, den: pd.Series) -> pd.Series:
    return safe_div(num, den) * 100.0


def wilder_ema(s: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing — used by RSI/ATR/ADX. Alpha = 1/period, no adjust."""
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def rolling_percentile(s: pd.Series, window: int) -> pd.Series:
    """Percentile rank (0-100) of the current value within trailing `window` bars."""
    return s.rolling(window, min_periods=window).rank(pct=True) * 100.0


def crosses_above(series: pd.Series, level) -> pd.Series:
    """1 on the bar series crosses above `level` (scalar or Series)."""
    if isinstance(level, (int, float)):
        return ((series.shift(1) <= level) & (series > level)).astype(int)
    return ((series.shift(1) <= level.shift(1)) & (series > level)).astype(int)


def crosses_below(series: pd.Series, level) -> pd.Series:
    if isinstance(level, (int, float)):
        return ((series.shift(1) >= level) & (series < level)).astype(int)
    return ((series.shift(1) >= level.shift(1)) & (series < level)).astype(int)


def bars_since(condition: pd.Series) -> pd.Series:
    """For each row, count of bars since `condition` was last True. NaN before first True."""
    cond = condition.astype(bool).to_numpy()
    out = np.full(len(cond), np.nan)
    counter = np.nan
    for i, c in enumerate(cond):
        if c:
            counter = 0
        elif not np.isnan(counter):
            counter += 1
        out[i] = counter
    return pd.Series(out, index=condition.index)


def linreg_value(series: pd.Series, length: int, offset: int = 0) -> pd.Series:
    """Pine ta.linreg(series, length, offset) — value of linear regression at `offset`.

    With offset=0 returns the line at the most recent bar.
    """
    s = series.to_numpy(dtype=float)
    n = len(s)
    out = np.full(n, np.nan)
    if n < length:
        return pd.Series(out, index=series.index)
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(length - 1, n):
        window = s[i - length + 1 : i + 1]
        if np.isnan(window).any():
            continue
        y_mean = window.mean()
        slope = ((x - x_mean) * (window - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        # value at position (length - 1 - offset) inside window
        out[i] = intercept + slope * (length - 1 - offset)
    return pd.Series(out, index=series.index)


def signed_log1p(s: pd.Series) -> pd.Series:
    """log1p that preserves sign — Lorentzian-inspired transform for volatile features."""
    return np.sign(s) * np.log1p(s.abs())
