"""Cat 3 — Volatility (15 features).

ATR (Wilder), ATR ratio (UT Bot concept), BB/KC widths, squeeze state, realized
vol, candle range, session range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import bars_since, pct, rolling_percentile, safe_div, true_range, wilder_ema


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    return wilder_ema(true_range(high, low, close), period)


def bbands(close: pd.Series, period: int, std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    basis = close.rolling(period, min_periods=period).mean()
    dev = close.rolling(period, min_periods=period).std(ddof=0)
    return basis, basis + std * dev, basis - std * dev


def keltner(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int, mult: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    basis = close.rolling(length, min_periods=length).mean()
    rng = (high - low).rolling(length, min_periods=length).mean()
    return basis, basis + mult * rng, basis - mult * rng


def volatility_features(
    df: pd.DataFrame,
    cfg: dict,
    session_id: pd.Series | None = None,
) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    a_cfg = cfg["atr"]
    bb_cfg = cfg["bb"]
    sq_cfg = cfg["squeeze"]

    atr_v = atr(high, low, close, a_cfg["period"])
    atr_short = atr(high, low, close, a_cfg["short_period"])

    bb_basis, bb_up, bb_lo = bbands(close, bb_cfg["period"], bb_cfg["std"])
    sq_bb_basis, sq_bb_up, sq_bb_lo = bbands(close, sq_cfg["bb_length"], sq_cfg["bb_mult"])
    kc_basis, kc_up, kc_lo = keltner(high, low, close, sq_cfg["kc_length"], sq_cfg["kc_mult"])

    bb_width = pct(bb_up - bb_lo, bb_basis)
    kc_width = pct(kc_up - kc_lo, kc_basis)

    in_squeeze = ((sq_bb_up < kc_up) & (sq_bb_lo > kc_lo)).astype(int)
    just_released = ((in_squeeze.shift(1) == 1) & (in_squeeze == 0)).astype(int)
    squeeze_state = pd.Series(0, index=close.index, dtype=float)
    squeeze_state[in_squeeze == 1] = -1
    squeeze_state[just_released == 1] = 1
    bars_since_release = bars_since(just_released == 1)

    log_ret = np.log(close / close.shift(1))
    rv5 = log_ret.rolling(5, min_periods=5).std(ddof=0)
    rv20 = log_ret.rolling(20, min_periods=20).std(ddof=0)

    out = pd.DataFrame(
        {
            "atr_14": atr_v,
            "atr_pct": pct(atr_v, close),
            "atr_ratio": safe_div(atr_short, atr_v),
            "atr_percentile": rolling_percentile(atr_v, a_cfg["percentile_window"]),
            "atr_roc": pct(atr_v - atr_v.shift(a_cfg["roc_window"]), atr_v.shift(a_cfg["roc_window"])),
            "bb_width": bb_width,
            "bb_width_percentile": rolling_percentile(bb_width, 100),
            "kc_width": kc_width,
            "squeeze_state": squeeze_state,
            "bars_since_squeeze_release": bars_since_release,
            "realized_vol_5": rv5,
            "realized_vol_20": rv20,
            "vol_ratio_5_20": safe_div(rv5, rv20),
            "high_low_range_pct": pct(high - low, close),
        }
    )

    # Session range — needs session id.
    if session_id is not None:
        grp = df.groupby(session_id)
        s_high = grp["high"].transform("max")
        s_low = grp["low"].transform("min")
        s_open = grp["open"].transform("first")
        out["session_range_pct"] = pct(s_high - s_low, s_open)
    else:
        out["session_range_pct"] = pd.Series(np.nan, index=close.index)
    return out
