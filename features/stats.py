"""Cat 9 — Mean Reversion / Stats (8) + Cat 17 — Statistical / Fractal (9)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import safe_div


def mean_reversion_features(df: pd.DataFrame, rsi_series: pd.Series, bb_position: pd.Series) -> pd.DataFrame:
    close = df["close"]
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    sma50 = close.rolling(50, min_periods=50).mean()
    std50 = close.rolling(50, min_periods=50).std(ddof=0)

    z20 = safe_div(close - sma20, std20)
    z50 = safe_div(close - sma50, std50)

    rsi_mean = rsi_series.rolling(50, min_periods=50).mean()
    rsi_std = rsi_series.rolling(50, min_periods=50).std(ddof=0)
    rsi_z = safe_div(rsi_series - rsi_mean, rsi_std)

    return pd.DataFrame(
        {
            "zscore_20": z20,
            "zscore_50": z50,
            "bb_position": bb_position,
            "rsi_zscore": rsi_z,
            "return_5bar": (close / close.shift(5) - 1) * 100,
            "return_20bar": (close / close.shift(20) - 1) * 100,
            "return_60bar": (close / close.shift(60) - 1) * 100,
            "mean_reversion_score": ((z20 + z50) / 2).clip(-3, 3),
        }
    )


# ─── Cat 17 — Fractal / Statistical ─────────────────────────────────────────
def hurst_exponent_window(arr: np.ndarray) -> float:
    """Rescaled-range Hurst estimator on a 1D log-return-like array."""
    if len(arr) < 20 or np.isnan(arr).any():
        return np.nan
    arr = arr - arr.mean()
    z = arr.cumsum()
    r = z.max() - z.min()
    s = arr.std(ddof=0)
    if s == 0 or r == 0:
        return np.nan
    return float(np.log(r / s) / np.log(len(arr)))


def shannon_entropy_window(arr: np.ndarray, bins: int = 10) -> float:
    if len(arr) < 10 or np.isnan(arr).any():
        return np.nan
    hist, _ = np.histogram(arr, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def fractal_dim_window(arr: np.ndarray) -> float:
    """Higuchi-like fractal dimension proxy via path length / range ratio."""
    if len(arr) < 10 or np.isnan(arr).any():
        return np.nan
    n = len(arr)
    path = np.abs(np.diff(arr)).sum()
    rng = arr.max() - arr.min()
    if rng == 0:
        return np.nan
    return float(1 + np.log(path / rng) / np.log(n))


def stats_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]
    log_ret = np.log(close / close.shift(1))

    autocorr_1 = log_ret.rolling(50, min_periods=50).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    autocorr_5 = log_ret.rolling(50, min_periods=50).apply(
        lambda x: pd.Series(x).autocorr(lag=5), raw=False
    )

    skew_20 = log_ret.rolling(20, min_periods=20).skew()
    kurt_20 = log_ret.rolling(20, min_periods=20).kurt()

    h_window = cfg["hurst"]["window"]
    fd_window = cfg["fractal_dim"]["window"]
    pk_window = cfg["parkinson"]["window"]

    hurst = log_ret.rolling(h_window, min_periods=h_window).apply(hurst_exponent_window, raw=True)
    entropy = log_ret.rolling(50, min_periods=50).apply(shannon_entropy_window, raw=True)
    fract_dim = close.rolling(fd_window, min_periods=fd_window).apply(fractal_dim_window, raw=True)

    var_1 = log_ret.rolling(20, min_periods=20).var(ddof=0)
    var_10 = log_ret.rolling(20, min_periods=20).apply(
        lambda x: pd.Series(x).rolling(10).sum().var(ddof=0), raw=False
    )
    variance_ratio = safe_div(var_10, 10 * var_1)

    log_hl = np.log(high / low)
    parkinson = np.sqrt(
        (log_hl ** 2).rolling(pk_window, min_periods=pk_window).mean() / (4 * np.log(2))
    )

    return pd.DataFrame(
        {
            "hurst_exponent": hurst,
            "price_entropy": entropy,
            "autocorrelation_1": autocorr_1,
            "autocorrelation_5": autocorr_5,
            "skewness_20": skew_20,
            "kurtosis_20": kurt_20,
            "fractal_dimension": fract_dim,
            "variance_ratio": variance_ratio,
            "parkinson_vol": parkinson,
        }
    )
