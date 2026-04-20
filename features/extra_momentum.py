"""Cat 15 — Additional Momentum (9 features): Williams %R, CCI, CMO, ROC, TSI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import safe_div


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    return -100 * safe_div(hh - close, hh - ll)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return safe_div(tp - sma, 0.015 * mad)


def cmo(close: pd.Series, period: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).rolling(period, min_periods=period).sum()
    dn = (-diff.clip(upper=0)).rolling(period, min_periods=period).sum()
    return safe_div(up - dn, up + dn) * 100


def tsi(close: pd.Series, long: int, short: int) -> pd.Series:
    diff = close.diff()
    ema1 = diff.ewm(span=long, adjust=False, min_periods=long).mean()
    ema2 = ema1.ewm(span=short, adjust=False, min_periods=short).mean()
    abs_ema1 = diff.abs().ewm(span=long, adjust=False, min_periods=long).mean()
    abs_ema2 = abs_ema1.ewm(span=short, adjust=False, min_periods=short).mean()
    return 100 * safe_div(ema2, abs_ema2)


def extra_momentum_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    wr = williams_r(h, l, c, cfg["williams_r"]["period"])
    cci_v = cci(h, l, c, cfg["cci"]["period"])
    cmo_v = cmo(c, cfg["cmo"]["period"])
    roc_p = cfg["roc"]["period"]
    roc = (c / c.shift(roc_p) - 1) * 100
    tsi_v = tsi(c, cfg["tsi"]["long"], cfg["tsi"]["short"])

    return pd.DataFrame(
        {
            "williams_r": wr,
            "williams_r_direction": wr.diff(),
            "cci_20": cci_v,
            "cci_direction": cci_v.diff(),
            "cci_extreme": (cci_v.abs() - 100).clip(lower=0),
            "cmo_14": cmo_v,
            "cmo_direction": cmo_v.diff(),
            "roc_10": roc,
            "tsi": tsi_v,
        }
    )
