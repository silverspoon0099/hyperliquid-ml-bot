"""Triple-barrier labeling (Lopez de Prado).

For each 5-min candle, simulate a trade forward with three exit conditions:
  1. Upper barrier (close + tp_atr_mult * ATR) hit -> LONG (0)
  2. Lower barrier (close - sl_atr_mult * ATR) hit -> SHORT (1)
  3. Time barrier (max_holding_bars) reached -> check net P&L vs min_profit_pct

Per Decision #18 + #19: this matches the trader's actual exit style (ATR-based
trailing stops, no fixed take-profit ratio).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    tp_atr_mult: float = 3.0,
    sl_atr_mult: float = 2.0,
    max_holding_bars: int = 48,
    min_profit_pct: float = 0.3,
    classes: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Returns DataFrame indexed like `df` with columns:
        label          0=LONG, 1=SHORT, 2=NEUTRAL
        holding_bars   bars to barrier hit (or max_holding_bars at time-out)
        exit_price
        pnl_pct
    """
    classes = classes or {"LONG": 0, "SHORT": 1, "NEUTRAL": 2}

    n = len(df)
    labels = np.full(n, classes["NEUTRAL"], dtype=np.int8)
    holding = np.full(n, max_holding_bars, dtype=np.int32)
    exit_price = np.full(n, np.nan)
    pnl_pct = np.full(n, np.nan)

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df[atr_col].to_numpy()

    for i in range(n - max_holding_bars):
        entry = close[i]
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(entry):
            continue
        upper = entry + tp_atr_mult * a
        lower = entry - sl_atr_mult * a

        hit = False
        for j in range(1, max_holding_bars + 1):
            hi = high[i + j]
            lo = low[i + j]
            # Pessimistic order — if both touch in the same bar we don't know
            # which came first; pick the closer barrier (tighter risk).
            up_hit = hi >= upper
            dn_hit = lo <= lower
            if up_hit and dn_hit:
                # Tighter side (smaller ATR mult) wins by convention -> SL.
                if sl_atr_mult <= tp_atr_mult:
                    labels[i] = classes["SHORT"]
                    exit_price[i] = lower
                    pnl_pct[i] = (lower - entry) / entry * 100
                else:
                    labels[i] = classes["LONG"]
                    exit_price[i] = upper
                    pnl_pct[i] = (upper - entry) / entry * 100
                holding[i] = j
                hit = True
                break
            if up_hit:
                labels[i] = classes["LONG"]
                exit_price[i] = upper
                pnl_pct[i] = (upper - entry) / entry * 100
                holding[i] = j
                hit = True
                break
            if dn_hit:
                labels[i] = classes["SHORT"]
                exit_price[i] = lower
                pnl_pct[i] = (lower - entry) / entry * 100
                holding[i] = j
                hit = True
                break

        if not hit:
            final = close[i + max_holding_bars]
            move_pct = (final - entry) / entry * 100
            exit_price[i] = final
            pnl_pct[i] = move_pct
            holding[i] = max_holding_bars
            if move_pct > min_profit_pct:
                labels[i] = classes["LONG"]
            elif move_pct < -min_profit_pct:
                labels[i] = classes["SHORT"]
            else:
                labels[i] = classes["NEUTRAL"]

    # The last `max_holding_bars` rows can't be labeled (no forward window).
    labels[n - max_holding_bars :] = -1  # sentinel: drop these rows before training

    return pd.DataFrame(
        {
            "label": labels,
            "holding_bars": holding,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
        },
        index=df.index,
    )
