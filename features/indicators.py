"""Core indicators (Cat 1 Momentum 47 + Cat 2 Trend 19 = 66 features).

Implements RSI/WaveTrend/Stochastic/Squeeze/MACD deep features and ADX/DI + EMA
features. Calculations match TradingView Pine reference where possible:
- RSI: Wilder's smoothing
- WaveTrend (LazyBear): EMA(EMA-distance / 0.015 * EMA(abs)).
- Stochastic: standard fast %K then SMA smoothing.
- Squeeze Momentum (LazyBear): linreg of close - midprice basis.
- MACD: EMA(12)-EMA(26), signal EMA(9).
- ADX/DI: Wilder DM/TR smoothing then DX -> ADX.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import (
    crosses_above,
    crosses_below,
    pct,
    rolling_percentile,
    safe_div,
    true_range,
    wilder_ema,
)


# ─── RSI (12) ───────────────────────────────────────────────────────────────
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = wilder_ema(up, period)
    avg_dn = wilder_ema(down, period)
    rs = safe_div(avg_up, avg_dn)
    return 100 - 100 / (1 + rs)


def rsi_features(close: pd.Series, cfg: dict) -> pd.DataFrame:
    p = cfg["period"]
    sma_p = cfg["sma_period"]
    bb_p = cfg["bb_period"]
    bb_std = cfg["bb_std"]

    r = rsi(close, p)
    sma = r.rolling(sma_p, min_periods=sma_p).mean()
    bb_basis = r.rolling(bb_p, min_periods=bb_p).mean()
    bb_dev = r.rolling(bb_p, min_periods=bb_p).std(ddof=0)
    upper = bb_basis + bb_std * bb_dev
    lower = bb_basis - bb_std * bb_dev
    rng_min = r.rolling(20, min_periods=20).min()
    rng_max = r.rolling(20, min_periods=20).max()

    rsi_dir = r.diff()
    sma_dir = sma.diff()
    cross_up = crosses_above(r, sma)
    cross_dn = crosses_below(r, sma)
    crossed = (cross_up | cross_dn).astype(bool)
    cross_speed = (r - sma).abs().where(crossed, 0.0)

    return pd.DataFrame(
        {
            "rsi_14": r,
            "rsi_sma_14": sma,
            "rsi_minus_sma": r - sma,
            "rsi_direction": rsi_dir,
            "rsi_sma_direction": sma_dir,
            "rsi_sma_roc": (sma - sma.shift(3)) / 3,
            "rsi_accel": rsi_dir.diff(),
            "rsi_cross_speed": cross_speed,
            "rsi_bb_upper": upper,
            "rsi_bb_lower": lower,
            "rsi_bb_position": safe_div(r - lower, upper - lower),
            "rsi_range_pct": safe_div(r - rng_min, rng_max - rng_min),
        }
    )


# ─── WaveTrend (LazyBear) (10 on 5min + 3 on 1H = 13) ───────────────────────
def wavetrend(
    high: pd.Series, low: pd.Series, close: pd.Series, n1: int, n2: int, signal: int
) -> tuple[pd.Series, pd.Series]:
    hlc3 = (high + low + close) / 3.0
    esa = hlc3.ewm(span=n1, adjust=False, min_periods=n1).mean()
    d = (hlc3 - esa).abs().ewm(span=n1, adjust=False, min_periods=n1).mean()
    ci = safe_div(hlc3 - esa, 0.015 * d)
    wt1 = ci.ewm(span=n2, adjust=False, min_periods=n2).mean()
    wt2 = wt1.rolling(signal, min_periods=signal).mean()
    return wt1, wt2


def wavetrend_features(
    high: pd.Series, low: pd.Series, close: pd.Series, cfg: dict, suffix: str = "5min"
) -> pd.DataFrame:
    """suffix='5min' -> full 10-feature set; suffix='1h' -> 3-feature subset (#23-25)."""
    wt1, wt2 = wavetrend(high, low, close, cfg["n1"], cfg["n2"], cfg["signal"])
    hist = wt1 - wt2

    cross_up = crosses_above(wt1, wt2)
    cross_dn = crosses_below(wt1, wt2)
    cross = cross_up - cross_dn

    if suffix == "5min":
        wt1_dir = wt1.diff()
        rng_min = wt1.rolling(20, min_periods=20).min()
        rng_max = wt1.rolling(20, min_periods=20).max()
        return pd.DataFrame(
            {
                "wt1_5min": wt1,
                "wt2_5min": wt2,
                "wt_hist_5min": hist,
                "wt_cross_5min": cross,
                "wt1_direction": wt1_dir,
                "wt1_accel": wt1_dir.diff(),
                "wt_hist_direction": hist.diff(),
                "wt_ob_depth": (wt1 - cfg["ob_level"]).clip(lower=0),
                "wt_os_depth": (cfg["os_level"] - wt1).clip(lower=0),
                "wt_range_pct": safe_div(wt1 - rng_min, rng_max - rng_min),
            }
        )
    # 1H subset
    return pd.DataFrame(
        {
            "wt1_1h": wt1,
            "wt2_1h": wt2,
            "wt_cross_1h": cross,
        }
    )


# ─── Stochastic (6) ─────────────────────────────────────────────────────────
def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, k_smooth: int, d_smooth: int
) -> tuple[pd.Series, pd.Series]:
    ll = low.rolling(k_period, min_periods=k_period).min()
    hh = high.rolling(k_period, min_periods=k_period).max()
    fast_k = safe_div(close - ll, hh - ll) * 100
    k = fast_k.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return k, d


def stoch_features(high: pd.Series, low: pd.Series, close: pd.Series, cfg: dict) -> pd.DataFrame:
    k, d = stochastic(high, low, close, cfg["k_period"], cfg["k_smooth"], cfg["d_smooth"])
    spread = k - d
    return pd.DataFrame(
        {
            "stoch_k": k,
            "stoch_d": d,
            "stoch_k_direction": k.diff(),
            "stoch_kd_spread": spread,
            "stoch_kd_spread_change": spread.diff(),
            "stoch_range_pct": ((k - 20) / 60).clip(0, 1),
        }
    )


# ─── Squeeze Momentum (LazyBear) (8) ────────────────────────────────────────
def squeeze_momentum_value(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int
) -> pd.Series:
    """LazyBear's val = linreg(source - avg(avg(highest, lowest), sma), length, 0)
    where source = close."""
    hh = high.rolling(length, min_periods=length).max()
    ll = low.rolling(length, min_periods=length).min()
    midprice = (hh + ll) / 2
    sma_close = close.rolling(length, min_periods=length).mean()
    basis = (midprice + sma_close) / 2
    diff = close - basis
    # linreg via rolling apply (slope*length-1 + intercept = value at last bar with offset=0)
    # Use simple polyfit - acceptable speed for 5min data of ~150k bars.
    arr = diff.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < length:
        return pd.Series(out, index=close.index)
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(length - 1, n):
        window = arr[i - length + 1 : i + 1]
        if np.isnan(window).any():
            continue
        y_mean = window.mean()
        slope = ((x - x_mean) * (window - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        out[i] = intercept + slope * (length - 1)
    return pd.Series(out, index=close.index)


def squeeze_features(
    high: pd.Series, low: pd.Series, close: pd.Series, cfg_squeeze: dict
) -> pd.DataFrame:
    length = cfg_squeeze["kc_length"]
    val = squeeze_momentum_value(high, low, close, length)
    accel = val.diff()
    direction = np.sign(val)
    abs_val = val.abs()

    zero_cross_up = crosses_above(val, 0).astype(bool)
    zero_cross_dn = crosses_below(val, 0).astype(bool)
    zero_cross = (zero_cross_up | zero_cross_dn).astype(int)

    percentile = rolling_percentile(abs_val, 50)
    slope_5 = (val - val.shift(5)) / 5

    return pd.DataFrame(
        {
            "squeeze_momentum": val,
            "squeeze_mom_accel": accel,
            "squeeze_mom_direction": direction,
            "squeeze_mom_abs": abs_val,
            "squeeze_mom_accel2": accel.diff(),
            "squeeze_mom_zero_cross": zero_cross,
            "squeeze_mom_percentile": percentile,
            "squeeze_mom_slope_5bar": slope_5,
        }
    )


# ─── MACD (8 — 6 on 5min + 2 on 1H) ─────────────────────────────────────────
def macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def macd_features_5m(close: pd.Series, cfg: dict) -> pd.DataFrame:
    line, sig, hist = macd(close, cfg["fast"], cfg["slow"], cfg["signal"])
    hist_dir = hist.diff()
    return pd.DataFrame(
        {
            "macd_line_5min": line,
            "macd_signal_5min": sig,
            "macd_hist_5min": hist,
            "macd_hist_direction": hist_dir,
            "macd_hist_accel": hist_dir.diff(),
            "macd_line_direction": line.diff(),
        }
    )


def macd_features_1h(close_1h: pd.Series, cfg: dict) -> pd.DataFrame:
    _line, _sig, hist = macd(close_1h, cfg["fast"], cfg["slow"], cfg["signal"])
    return pd.DataFrame(
        {
            "macd_hist_1h": hist,
            "macd_hist_1h_direction": hist.diff(),
        }
    )


# ─── ADX/DI (12 — 11 on 5min + 1 on 1H) ─────────────────────────────────────
def adx_di(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=high.index)
    tr = true_range(high, low, close)
    atr = wilder_ema(tr, period)
    di_plus = 100 * safe_div(wilder_ema(plus_dm, period), atr)
    di_minus = 100 * safe_div(wilder_ema(minus_dm, period), atr)
    dx = 100 * safe_div((di_plus - di_minus).abs(), di_plus + di_minus)
    adx = wilder_ema(dx, period)
    return di_plus, di_minus, adx


def adx_features_5m(high: pd.Series, low: pd.Series, close: pd.Series, cfg: dict) -> pd.DataFrame:
    p = cfg["period"]
    roc_w = cfg["di_roc_window"]
    spread_roc_w = cfg["spread_roc_window"]
    di_plus, di_minus, adx = adx_di(high, low, close, p)
    spread = di_plus - di_minus
    adx_dir = adx.diff()
    rng_min = adx.rolling(20, min_periods=20).min()
    rng_max = adx.rolling(20, min_periods=20).max()
    abs_spread = spread.abs()
    convergence = abs_spread - abs_spread.shift(roc_w)

    return pd.DataFrame(
        {
            "di_plus": di_plus,
            "di_minus": di_minus,
            "adx": adx,
            "di_plus_roc": di_plus - di_plus.shift(roc_w),
            "di_minus_roc": di_minus - di_minus.shift(roc_w),
            "di_spread": spread,
            "di_spread_roc": spread - spread.shift(spread_roc_w),
            "adx_direction": adx_dir,
            "adx_accel": adx_dir.diff(),
            "adx_range_pct": safe_div(adx - rng_min, rng_max - rng_min),
            "di_convergence": convergence,
        }
    )


def adx_features_1h(high_1h: pd.Series, low_1h: pd.Series, close_1h: pd.Series, cfg: dict) -> pd.DataFrame:
    _dp, _dm, adx = adx_di(high_1h, low_1h, close_1h, cfg["period"])
    return pd.DataFrame({"adx_1h": adx})


# ─── EMA Stack (7) ──────────────────────────────────────────────────────────
def ema_features_5m(close: pd.Series, periods: list[int]) -> pd.DataFrame:
    out = {}
    for p in periods:
        ema = close.ewm(span=p, adjust=False, min_periods=p).mean()
        out[f"ema{p}_dist_pct"] = pct(close - ema, close)
    return pd.DataFrame(out)


def ema_features_1h(close_1h: pd.Series, periods: list[int]) -> pd.DataFrame:
    out = {}
    emas = {}
    for p in periods:
        emas[p] = close_1h.ewm(span=p, adjust=False, min_periods=p).mean()
        out[f"ema{p}_1h_dist_pct"] = pct(close_1h - emas[p], close_1h)

    # Stack score: +1 each for EMA9>EMA21, EMA21>EMA50, close>EMA9 -> max +3, min -3
    # (-1 for the inverse condition; per spec it's "+1 for each", but a -3 to +3 range
    # implies signed scoring, so we score +1 when up-stacked and -1 when down-stacked.)
    score = pd.Series(0, index=close_1h.index, dtype=float)
    if 9 in emas and 21 in emas:
        score += np.sign(emas[9] - emas[21])
    if 21 in emas and 50 in emas:
        score += np.sign(emas[21] - emas[50])
    if 9 in emas:
        score += np.sign(close_1h - emas[9])
    out["ema_stack_1h"] = score
    return pd.DataFrame(out)
