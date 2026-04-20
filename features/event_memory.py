"""Cat 20 — Event Memory (41 features).

The model needs to know not just "where is RSI now?" but "how recently did
RSI emerge from oversold?" Recency, count, depth, and recovery-speed features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import bars_since, crosses_above, crosses_below


def _bars_in_state(state: pd.Series) -> pd.Series:
    """Count of consecutive bars `state` has been True up to current row.
    Resets when state goes False.
    """
    s = state.astype(bool)
    grp = (~s).cumsum()
    return s.astype(int).groupby(grp).cumsum()


def _last_episode_duration(state: pd.Series) -> pd.Series:
    """Length of the most-recently-completed True episode. Forward-filled."""
    s = state.astype(bool).to_numpy()
    out = np.full(len(s), np.nan)
    last_dur = np.nan
    run = 0
    for i, v in enumerate(s):
        if v:
            run += 1
        else:
            if run > 0:
                last_dur = run
                run = 0
        out[i] = last_dur
    return pd.Series(out, index=state.index)


def _last_episode_extreme(value: pd.Series, state: pd.Series, op: str) -> pd.Series:
    """Forward-filled extreme value (max or min) reached during the most recent True episode."""
    arr = value.to_numpy()
    s = state.astype(bool).to_numpy()
    out = np.full(len(arr), np.nan)
    cur_extreme = np.nan
    last_extreme = np.nan
    in_episode = False
    for i in range(len(arr)):
        if s[i]:
            if not in_episode:
                cur_extreme = arr[i]
                in_episode = True
            else:
                if op == "max":
                    cur_extreme = max(cur_extreme, arr[i])
                else:
                    cur_extreme = min(cur_extreme, arr[i])
        else:
            if in_episode:
                last_extreme = cur_extreme
                cur_extreme = np.nan
                in_episode = False
        out[i] = last_extreme if not in_episode else (cur_extreme if not np.isnan(cur_extreme) else last_extreme)
    return pd.Series(out, index=value.index)


def _bars_in_current_or_last_episode(state: pd.Series) -> pd.Series:
    """Running bar count while in True state; forward-fills last completed duration when False.

    Used so `*_bars_in_ob/os` always reports "most recent OB/OS episode length" per spec —
    whether the episode is still active or just ended.
    """
    s = state.astype(bool).to_numpy()
    out = np.full(len(s), np.nan)
    run = 0
    last_dur = np.nan
    for i, v in enumerate(s):
        if v:
            run += 1
            out[i] = run
        else:
            if run > 0:
                last_dur = run
                run = 0
            out[i] = last_dur
    return pd.Series(out, index=state.index)


def _last_extreme_depth(
    value: pd.Series,
    in_ob: pd.Series,
    in_os: pd.Series,
    ob_level: float,
    os_level: float,
) -> pd.Series:
    """Depth of whichever OB/OS episode happened MOST recently (not just whichever ever occurred).

    During an active episode, tracks the running extreme depth. After exit, forward-fills.
    """
    v = value.to_numpy()
    o = in_ob.astype(bool).to_numpy()
    s = in_os.astype(bool).to_numpy()
    out = np.full(len(v), np.nan)
    cur_kind: str | None = None
    cur_extreme = np.nan
    last_depth = np.nan
    for i in range(len(v)):
        if o[i]:
            cur_extreme = v[i] if cur_kind != "ob" else max(cur_extreme, v[i])
            cur_kind = "ob"
            last_depth = cur_extreme - ob_level
        elif s[i]:
            cur_extreme = v[i] if cur_kind != "os" else min(cur_extreme, v[i])
            cur_kind = "os"
            last_depth = os_level - cur_extreme
        else:
            cur_kind = None
            cur_extreme = np.nan
        out[i] = last_depth
    return pd.Series(out, index=value.index)


def _recovery_speed(value: pd.Series, state_extreme: pd.Series, target: float = 50.0) -> pd.Series:
    """Bars from end of most recent extreme episode until value crosses `target`."""
    arr = value.to_numpy()
    s = state_extreme.astype(bool).to_numpy()
    out = np.full(len(arr), np.nan)
    last_exit_idx = -1
    direction = None  # "up" if recovering from low, "down" if from high
    for i in range(len(arr)):
        if i > 0 and not s[i] and s[i - 1]:
            last_exit_idx = i
            direction = "up" if arr[i - 1] <= target else "down"
        if last_exit_idx >= 0 and direction is not None:
            if (direction == "up" and arr[i] >= target) or (direction == "down" and arr[i] <= target):
                out[i] = i - last_exit_idx
                last_exit_idx = -1
                direction = None
            else:
                # Still recovering; keep last completed value
                pass
    # Forward-fill recoveries between events.
    return pd.Series(out, index=value.index).ffill()


# ─── Stochastic Event Memory (8) ───────────────────────────────────────────
def stoch_event_memory(stoch_k: pd.Series, stoch_d: pd.Series, cfg_em: dict) -> pd.DataFrame:
    ob_level = cfg_em["stoch_ob"]
    os_level = cfg_em["stoch_os"]

    # "Green Bar" = K crossed above D while K < ob (bullish setup in oversold zone)
    green = (crosses_above(stoch_k, stoch_d).astype(bool) & (stoch_k < os_level))
    # "Red Bar" = K crossed below D while K > os (bearish setup in overbought zone)
    red = (crosses_below(stoch_k, stoch_d).astype(bool) & (stoch_k > ob_level))

    in_ob = stoch_k > ob_level
    in_os = stoch_k < os_level

    return pd.DataFrame(
        {
            "stoch_bars_since_green": bars_since(green),
            "stoch_bars_since_red": bars_since(red),
            "stoch_green_count_50": green.astype(int).rolling(50, min_periods=50).sum(),
            "stoch_red_count_50": red.astype(int).rolling(50, min_periods=50).sum(),
            "stoch_last_extreme_depth": _last_extreme_depth(stoch_k, in_ob, in_os, ob_level, os_level),
            "stoch_bars_in_ob": _bars_in_current_or_last_episode(in_ob),
            "stoch_bars_in_os": _bars_in_current_or_last_episode(in_os),
            "stoch_recovery_speed": _recovery_speed(stoch_k, in_ob | in_os, target=50.0),
        }
    )


# ─── WaveTrend Event Memory (8) ────────────────────────────────────────────
def wt_event_memory(wt1: pd.Series, wt2: pd.Series, ob_level: float, os_level: float) -> pd.DataFrame:
    in_ob = wt1 > ob_level
    in_os = wt1 < os_level

    bull_cross = crosses_above(wt1, wt2)
    bear_cross = crosses_below(wt1, wt2)

    last_ob_max = _last_episode_extreme(wt1, in_ob, "max")
    last_os_min = _last_episode_extreme(wt1, in_os, "min")

    return pd.DataFrame(
        {
            "wt_bars_since_ob_exit": bars_since((in_ob.shift(1) == True) & (in_ob == False)),
            "wt_bars_since_os_exit": bars_since((in_os.shift(1) == True) & (in_os == False)),
            "wt_bars_since_bull_cross": bars_since(bull_cross.astype(bool)),
            "wt_bars_since_bear_cross": bars_since(bear_cross.astype(bool)),
            "wt_ob_count_100": in_ob.astype(int).diff().clip(lower=0).rolling(100, min_periods=100).sum(),
            "wt_os_count_100": in_os.astype(int).diff().clip(lower=0).rolling(100, min_periods=100).sum(),
            "wt_last_ob_depth": (last_ob_max - ob_level).clip(lower=0),
            "wt_last_os_depth": (os_level - last_os_min).clip(lower=0),
        }
    )


# ─── RSI Event Memory (6) ──────────────────────────────────────────────────
def rsi_event_memory(rsi_series: pd.Series, cfg_em: dict) -> pd.DataFrame:
    ob = cfg_em["rsi_ob"]
    os = cfg_em["rsi_os"]
    mid = cfg_em["rsi_mid"]
    in_ob = rsi_series > ob
    in_os = rsi_series < os
    cross_50 = crosses_above(rsi_series, mid).astype(bool) | crosses_below(rsi_series, mid).astype(bool)

    return pd.DataFrame(
        {
            "rsi_bars_since_ob": bars_since(in_ob),
            "rsi_bars_since_os": bars_since(in_os),
            "rsi_bars_since_50_cross": bars_since(cross_50),
            "rsi_ob_count_100": in_ob.astype(int).diff().clip(lower=0).rolling(100, min_periods=100).sum(),
            "rsi_os_count_100": in_os.astype(int).diff().clip(lower=0).rolling(100, min_periods=100).sum(),
            "rsi_time_above_50_pct": (rsi_series > mid).astype(int).rolling(50, min_periods=50).mean() * 100,
        }
    )


# ─── MACD Event Memory (5) ─────────────────────────────────────────────────
def macd_event_memory(macd_hist: pd.Series, macd_line: pd.Series) -> pd.DataFrame:
    bull = crosses_above(macd_hist, 0).astype(bool)
    bear = crosses_below(macd_hist, 0).astype(bool)
    zero_line = crosses_above(macd_line, 0).astype(bool) | crosses_below(macd_line, 0).astype(bool)
    return pd.DataFrame(
        {
            "macd_bars_since_bull_cross": bars_since(bull),
            "macd_bars_since_bear_cross": bars_since(bear),
            "macd_bars_since_zero_cross": bars_since(zero_line),
            "macd_hist_max_50": macd_hist.rolling(50, min_periods=50).max(),
            "macd_hist_min_50": macd_hist.rolling(50, min_periods=50).min(),
        }
    )


# ─── Squeeze Event Memory (5) ──────────────────────────────────────────────
def squeeze_event_memory(squeeze_state: pd.Series, squeeze_mom: pd.Series) -> pd.DataFrame:
    # squeeze_state: -1 in squeeze, +1 just released, 0 normal.
    fire = (squeeze_state == 1)
    in_squeeze = (squeeze_state == -1)

    bars_since_fire = bars_since(fire)
    duration_last = _last_episode_duration(in_squeeze)
    count_200 = fire.astype(int).rolling(200, min_periods=200).sum()
    momentum_at_fire_series = squeeze_mom.where(fire).ffill()

    # Peak abs momentum since last fire: reset on fire.
    abs_mom = squeeze_mom.abs()
    peak_since_fire = []
    cur_peak = np.nan
    fa = fire.to_numpy()
    am = abs_mom.to_numpy()
    for i in range(len(fa)):
        if fa[i]:
            cur_peak = am[i]
        else:
            if not np.isnan(cur_peak) and not np.isnan(am[i]):
                cur_peak = max(cur_peak, am[i])
            elif not np.isnan(am[i]):
                cur_peak = am[i] if np.isnan(cur_peak) else cur_peak
        peak_since_fire.append(cur_peak)
    peak_series = pd.Series(peak_since_fire, index=squeeze_mom.index)

    return pd.DataFrame(
        {
            "squeeze_bars_since_fire": bars_since_fire,
            "squeeze_duration_last": duration_last,
            "squeeze_count_200": count_200,
            "squeeze_momentum_at_fire": momentum_at_fire_series,
            "squeeze_mom_peak_since_fire": peak_series,
        }
    )


# ─── ADX/DI Event Memory (4) ───────────────────────────────────────────────
def adx_event_memory(adx: pd.Series, di_plus: pd.Series, di_minus: pd.Series, cfg_em: dict) -> pd.DataFrame:
    trending = adx > cfg_em["adx_trend"]
    weak = adx < cfg_em["adx_weak"]
    di_cross = crosses_above(di_plus, di_minus).astype(bool) | crosses_below(di_plus, di_minus).astype(bool)
    return pd.DataFrame(
        {
            "adx_bars_since_trend": bars_since(trending),
            "adx_bars_since_weak": bars_since(weak),
            "di_bars_since_cross": bars_since(di_cross),
            "adx_peak_50": adx.rolling(50, min_periods=50).max(),
        }
    )


# ─── Price Event Memory (5) ────────────────────────────────────────────────
def price_event_memory(
    df: pd.DataFrame,
    nearest_pivot_dist_pct: pd.Series,
    day_id: pd.Series,
    volume_spike_mult: float,
    pivot_tolerance_pct: float,
) -> pd.DataFrame:
    high, low, volume = df["high"], df["low"], df["volume"]
    new_high_20 = high == high.rolling(20, min_periods=20).max()
    new_low_20 = low == low.rolling(20, min_periods=20).min()

    pivot_touch = nearest_pivot_dist_pct <= pivot_tolerance_pct
    touches_today = pivot_touch.astype(int).groupby(day_id).cumsum()

    sma_vol20 = volume.rolling(20, min_periods=20).mean()
    spike = volume > volume_spike_mult * sma_vol20

    return pd.DataFrame(
        {
            "bars_since_new_high_20": bars_since(new_high_20),
            "bars_since_new_low_20": bars_since(new_low_20),
            "bars_since_pivot_touch": bars_since(pivot_touch),
            "pivot_touch_count_today": touches_today,
            "bars_since_volume_spike": bars_since(spike),
        }
    )
