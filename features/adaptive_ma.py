"""Cat 18 — Adaptive MAs (5 features): KAMA, DEMA, TEMA, Parabolic SAR."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import pct, safe_div


def kama(close: pd.Series, period: int, fast_ema: int, slow_ema: int) -> pd.Series:
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period, min_periods=period).sum()
    er = safe_div(change, volatility)
    fast_sc = 2.0 / (fast_ema + 1)
    slow_sc = 2.0 / (slow_ema + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = np.full(len(close), np.nan)
    arr = close.to_numpy()
    sc_arr = sc.to_numpy()
    for i in range(period, len(close)):
        if np.isnan(out[i - 1]):
            out[i] = arr[i]
        else:
            out[i] = out[i - 1] + sc_arr[i] * (arr[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def dema(close: pd.Series, period: int) -> pd.Series:
    e1 = close.ewm(span=period, adjust=False, min_periods=period).mean()
    e2 = e1.ewm(span=period, adjust=False, min_periods=period).mean()
    return 2 * e1 - e2


def tema(close: pd.Series, period: int) -> pd.Series:
    e1 = close.ewm(span=period, adjust=False, min_periods=period).mean()
    e2 = e1.ewm(span=period, adjust=False, min_periods=period).mean()
    e3 = e2.ewm(span=period, adjust=False, min_periods=period).mean()
    return 3 * e1 - 3 * e2 + e3


def parabolic_sar(
    high: pd.Series, low: pd.Series, af_start: float, af_step: float, af_max: float
) -> tuple[pd.Series, pd.Series]:
    n = len(high)
    sar = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)  # +1 long, -1 short
    h = high.to_numpy()
    l = low.to_numpy()
    if n < 2:
        return pd.Series(sar, index=high.index), pd.Series(trend, index=high.index)
    # Initialize
    trend[1] = 1 if h[1] >= h[0] else -1
    sar[1] = l[0] if trend[1] == 1 else h[0]
    ep = h[1] if trend[1] == 1 else l[1]
    af = af_start

    for i in range(2, n):
        prev_sar = sar[i - 1]
        if trend[i - 1] == 1:
            new_sar = prev_sar + af * (ep - prev_sar)
            # Wilder clamp: SAR cannot exceed the lows of the prior two bars.
            new_sar = min(new_sar, l[i - 1], l[i - 2])
            if l[i] < new_sar:
                trend[i] = -1
                sar[i] = ep
                ep = l[i]
                af = af_start
            else:
                trend[i] = 1
                sar[i] = new_sar
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            # Wilder clamp: SAR cannot fall below the highs of the prior two bars.
            new_sar = max(new_sar, h[i - 1], h[i - 2])
            if h[i] > new_sar:
                trend[i] = 1
                sar[i] = ep
                ep = h[i]
                af = af_start
            else:
                trend[i] = -1
                sar[i] = new_sar
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, index=high.index), pd.Series(trend, index=high.index)


def adaptive_ma_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]
    kama_v = kama(close, cfg["kama"]["period"], cfg["kama"]["fast_ema"], cfg["kama"]["slow_ema"])
    dema_v = dema(close, cfg["dema"]["period"])
    tema_v = tema(close, cfg["tema"]["period"])
    sar, trend = parabolic_sar(
        high, low, cfg["psar"]["af_start"], cfg["psar"]["af_step"], cfg["psar"]["af_max"]
    )

    return pd.DataFrame(
        {
            "kama_dist_pct": pct(close - kama_v, close),
            "dema_dist_pct": pct(close - dema_v, close),
            "tema_dist_pct": pct(close - tema_v, close),
            "psar_direction": trend,
            "psar_dist_pct": pct(close - sar, close),
        }
    )
