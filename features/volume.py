"""Cat 4 — Volume / VFI / Buy-Sell Pressure (17 features) + Cat 14 Money Flow (8).

VFI (LazyBear): standard MTF Volume Flow Indicator.
Buy/sell estimation per spec is the candle-shape heuristic (close-low)/(high-low).
The TRUE buy/sell signal arrives in Cat 21 from Hyperliquid trade tape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import crosses_above, crosses_below, pct, safe_div


# ─── VFI (LazyBear) ─────────────────────────────────────────────────────────
def vfi(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int,
    coef: float,
    vcoef: float,
) -> pd.Series:
    """LazyBear's Volume Flow Indicator.

    typical = (high + low + close) / 3
    inter = log(typical) - log(typical[1])
    vinter = stdev(inter, 30)
    cutoff = coef * vinter * close
    vave = sma(volume, period)[1]      # previous bar's avg
    vmax = vave * vcoef
    vc = min(volume, vmax)
    mf = typical - typical[1]
    vcp = vc * sign(mf if abs(mf) > cutoff else 0)
    vfi = sum(vcp, period) / vave
    """
    typical = (high + low + close) / 3.0
    inter = np.log(typical) - np.log(typical.shift(1))
    vinter = inter.rolling(30, min_periods=30).std(ddof=0)
    cutoff = coef * vinter * close
    vave = volume.rolling(period, min_periods=period).mean().shift(1)
    vmax = vave * vcoef
    vc = volume.where(volume <= vmax, vmax)
    mf = typical - typical.shift(1)
    direction = np.where(mf.abs() > cutoff, np.sign(mf), 0.0)
    vcp = vc * direction
    raw = vcp.rolling(period, min_periods=period).sum() / vave
    return raw.rolling(3, min_periods=3).mean()  # LazyBear Pine: sma(..., 3)


def vfi_features_5m(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, cfg: dict
) -> pd.DataFrame:
    v = vfi(close, high, low, volume, cfg["period"], cfg["coef"], cfg["vcoef"])
    sig = v.ewm(span=cfg["signal_period"], adjust=False, min_periods=cfg["signal_period"]).mean()
    hist = v - sig
    cross_up = crosses_above(v, 0).astype(bool)
    cross_dn = crosses_below(v, 0).astype(bool)
    cross = cross_up.astype(int) - cross_dn.astype(int)
    rng_min = v.rolling(50, min_periods=50).min()
    rng_max = v.rolling(50, min_periods=50).max()
    return pd.DataFrame(
        {
            "vfi": v,
            "vfi_signal": sig,
            "vfi_hist": hist,
            "vfi_direction": v.diff(),
            "vfi_hist_direction": hist.diff(),
            "vfi_cross_zero": cross,
            "vfi_range_pct": safe_div(v - rng_min, rng_max - rng_min),
        }
    )


def vfi_features_1h(
    close_1h: pd.Series, high_1h: pd.Series, low_1h: pd.Series, volume_1h: pd.Series, cfg: dict
) -> pd.DataFrame:
    v = vfi(close_1h, high_1h, low_1h, volume_1h, cfg["period"], cfg["coef"], cfg["vcoef"])
    return pd.DataFrame({"vfi_1h": v, "vfi_1h_direction": v.diff()})


# ─── Volume & Buy/Sell pressure (8) ─────────────────────────────────────────
def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    rng = (high - low).replace(0, np.nan)
    buy_pct = ((close - low) / rng).clip(0, 1)
    sell_pct = 1.0 - buy_pct
    buy_vol = volume * buy_pct
    sell_vol = volume * sell_pct
    delta = buy_vol - sell_vol

    sma20 = volume.rolling(20, min_periods=20).mean()
    sma5 = volume.rolling(5, min_periods=5).mean()

    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    # Slope via 10-bar linreg approximated as (obv - obv[10]) / 10
    obv_slope = (obv - obv.shift(10)) / 10

    bs_ratio = safe_div(buy_vol, sell_vol)

    return pd.DataFrame(
        {
            "volume_ratio": safe_div(volume, sma20),
            "volume_trend": safe_div(sma5, sma20),
            "buy_volume_pct": buy_pct,
            "sell_volume_pct": sell_pct,
            "buy_sell_ratio_sma5": bs_ratio.rolling(5, min_periods=5).mean(),
            "cum_delta_5": delta.rolling(5, min_periods=5).sum(),
            "cum_delta_20": delta.rolling(20, min_periods=20).sum(),
            "obv_slope": obv_slope,
        }
    )


# ─── Cat 14 — Money Flow (8) ────────────────────────────────────────────────
def money_flow_features(df: pd.DataFrame, vfi_series: pd.Series, rsi_series: pd.Series) -> pd.DataFrame:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    rng = (high - low).replace(0, np.nan)

    mf_mult = ((close - low) - (high - close)) / rng
    mf_vol = mf_mult * volume

    cmf = mf_vol.rolling(20, min_periods=20).sum() / volume.rolling(20, min_periods=20).sum()

    typical = (high + low + close) / 3
    raw_mf = typical * volume
    pos_mf = raw_mf.where(typical > typical.shift(1), 0.0)
    neg_mf = raw_mf.where(typical < typical.shift(1), 0.0)
    pos_sum = pos_mf.rolling(14, min_periods=14).sum()
    neg_sum = neg_mf.rolling(14, min_periods=14).sum()
    mfr = safe_div(pos_sum, neg_sum)
    mfi = 100 - 100 / (1 + mfr)

    ad = mf_vol.cumsum()
    ad_slope = (ad - ad.shift(10)) / 10

    return pd.DataFrame(
        {
            "cmf_20": cmf,
            "cmf_direction": cmf.diff(),
            "mfi_14": mfi,
            "mfi_direction": mfi.diff(),
            "ad_line": ad,
            "ad_slope_10": ad_slope,
            "mfi_rsi_divergence": mfi - rsi_series,
            "cmf_vfi_agreement": np.sign(cmf) * np.sign(vfi_series),
        }
    )
