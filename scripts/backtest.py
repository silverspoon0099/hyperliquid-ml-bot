"""Trade-level backtest with configurable R:R exit geometry.

Labels were trained with symmetric 4:4 ATR triple-barriers (to keep label class
distribution free of the tighter-SL mechanical bias). Real trades use an
asymmetric R:R where TP > SL. This script decouples the two: use the model's
LONG/SHORT confidence as the entry signal, then simulate forward with an
independent TP/SL geometry.

Exit logic per trade:
  - Enter at close of signal bar (`i`).
  - Scan bars i+1 … i+max_holding:
      LONG:  TP hit when high >= entry + tp_dist
             SL hit when low  <= entry - sl_dist
      SHORT: TP hit when low  <= entry - tp_dist
             SL hit when high >= entry + sl_dist
  - Same-bar both-hit → assume SL hit first (pessimistic).
  - Otherwise timeout exit at close of bar i+max_holding.
  - No-stack: cursor jumps to exit_idx+1 (skip signals while in position).

Friction: taker fee + slippage, both paid on entry AND exit. Read from config.

Usage:
    python -m scripts.backtest --tag ema_ext_trim --threshold 0.70 --rr atr_2_1 --oot-start 2026-03-29
    python -m scripts.backtest --tag ema_extoot   --threshold 0.70 --rr pct_0p5_0p25 --oot-start 2026-01-01
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from utils.config import load_config
from utils.logging_setup import get_logger

log = get_logger("backtest")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    side: str
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    exit_reason: str  # "TP" | "SL" | "TIMEOUT"
    tp_price: float
    sl_price: float
    gross_return: float
    net_return: float


RR_SCHEMES = {
    # name: (tp_dist_fn, sl_dist_fn) where each takes (entry, atr) -> absolute distance
    "atr_2_1":      (lambda entry, atr: 2.0 * atr,          lambda entry, atr: 1.0 * atr),
    "atr_4_4":      (lambda entry, atr: 4.0 * atr,          lambda entry, atr: 4.0 * atr),
    "atr_2_2":      (lambda entry, atr: 2.0 * atr,          lambda entry, atr: 2.0 * atr),
    "atr_4_2":      (lambda entry, atr: 4.0 * atr,          lambda entry, atr: 2.0 * atr),
    "pct_0p5_0p25": (lambda entry, atr: 0.005 * entry,      lambda entry, atr: 0.0025 * entry),
    "pct_0p6_0p6":  (lambda entry, atr: 0.006 * entry,      lambda entry, atr: 0.006 * entry),
}


def apply_calibrators(proba: np.ndarray, calibrators: list) -> np.ndarray:
    out = np.zeros_like(proba)
    for c in range(proba.shape[1]):
        out[:, c] = calibrators[c].transform(proba[:, c])
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def simulate_trade(
    i: int, side: str,
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    atr: float, rr: str, max_bars: int,
) -> tuple[int, float, str, float, float]:
    """Forward-scan barriers. Returns (exit_idx, exit_price, reason, tp_price, sl_price)."""
    entry = close[i]
    tp_fn, sl_fn = RR_SCHEMES[rr]
    tp_dist = tp_fn(entry, atr)
    sl_dist = sl_fn(entry, atr)

    if side == "LONG":
        tp_price = entry + tp_dist
        sl_price = entry - sl_dist
    else:  # SHORT
        tp_price = entry - tp_dist
        sl_price = entry + sl_dist

    n = len(close)
    max_j = min(max_bars, n - 1 - i)
    for j in range(1, max_j + 1):
        hi = high[i + j]
        lo = low[i + j]
        if side == "LONG":
            up_hit = hi >= tp_price
            dn_hit = lo <= sl_price
        else:
            up_hit = lo <= tp_price    # TP below entry for short
            dn_hit = hi >= sl_price    # SL above entry for short
        if up_hit and dn_hit:
            return i + j, sl_price, "SL", tp_price, sl_price  # pessimistic
        if up_hit:
            return i + j, tp_price, "TP", tp_price, sl_price
        if dn_hit:
            return i + j, sl_price, "SL", tp_price, sl_price
    return i + max_j, close[i + max_j], "TIMEOUT", tp_price, sl_price


def run(
    symbol: str, cfg: dict, tag: str, threshold: float, rr: str,
    oot_start: str, use_calibrated: bool,
) -> dict:
    suffix = f"_{tag}"
    model_dir = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf{suffix}"
    local_sel = model_dir / "selected_features.json"
    base_sel = PROJECT_ROOT / "model" / "models" / f"{symbol}_wf" / "selected_features.json"
    sel_path = local_sel if local_sel.exists() else base_sel
    selected = json.loads(sel_path.read_text())["features"]
    log.info(f"using {len(selected)} features from {sel_path}")

    df = pd.read_parquet(PROJECT_ROOT / cfg["features"]["output_dir"] / f"{symbol}_features.parquet")
    df = df[df["label"] >= 0].reset_index(drop=True)
    df["_ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    oot_ts = pd.Timestamp(oot_start, tz="UTC")
    oot = df[df["_ts"] >= oot_ts].reset_index(drop=True)
    if len(oot) == 0:
        raise RuntimeError(f"no OOT rows at/after {oot_start}")
    log.info(f"OOT: {len(oot):,} rows  {oot['_ts'].iloc[0]} -> {oot['_ts'].iloc[-1]}")

    fold_files = sorted(model_dir.glob("fold_*.txt"), key=lambda p: int(p.stem.split("_")[1]))
    if not fold_files:
        raise RuntimeError(f"no fold_*.txt in {model_dir}")
    booster = lgb.Booster(model_file=str(fold_files[-1]))
    log.info(f"loaded {fold_files[-1].name}")

    X = oot[selected].to_numpy(dtype=np.float32)
    raw_proba = booster.predict(X)

    if use_calibrated:
        cal_path = model_dir / "calibrators.pkl"
        if not cal_path.exists():
            raise RuntimeError(f"--calibrated requested but {cal_path} missing")
        with open(cal_path, "rb") as f:
            calibrators = pickle.load(f)
        proba = apply_calibrators(raw_proba, calibrators)
        log.info("using CALIBRATED probabilities")
    else:
        proba = raw_proba
        log.info("using RAW probabilities")

    max_p = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    classes = cfg["labeling"]["classes"]
    neutral_idx = classes["NEUTRAL"]
    side_map = {classes["LONG"]: "LONG", classes["SHORT"]: "SHORT"}

    max_holding = cfg["labeling"]["max_holding_bars"]
    fee_rt = 2.0 * cfg["fees"]["default"]["taker"]
    slip_rt = 2.0 * cfg["fees"]["slippage_pct"]
    friction = fee_rt + slip_rt
    log.info(
        f"friction (round-trip): fee={fee_rt*1e4:.1f}bp + slip={slip_rt*1e4:.1f}bp = "
        f"{friction*1e4:.1f}bp"
    )

    close = oot["close"].to_numpy()
    high = oot["high"].to_numpy()
    low = oot["low"].to_numpy()
    atr = oot["atr_14"].to_numpy()
    ts = oot["_ts"].astype(str).to_numpy()
    n = len(oot)

    trades: list[Trade] = []
    skipped_bad_atr = 0
    i = 0
    while i < n:
        if max_p[i] < threshold or pred[i] == neutral_idx:
            i += 1
            continue
        side = side_map[int(pred[i])]
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(close[i]):
            skipped_bad_atr += 1
            i += 1
            continue
        exit_idx, exit_price, reason, tp_price, sl_price = simulate_trade(
            i, side, high, low, close, a, rr, max_holding
        )
        entry_price = float(close[i])
        if side == "LONG":
            gross = (exit_price - entry_price) / entry_price
        else:
            gross = (entry_price - exit_price) / entry_price
        net = gross - friction
        trades.append(Trade(
            entry_idx=int(i), exit_idx=int(exit_idx), side=side,
            entry_ts=str(ts[i]), exit_ts=str(ts[exit_idx]),
            entry_price=entry_price, exit_price=float(exit_price),
            exit_reason=reason, tp_price=float(tp_price), sl_price=float(sl_price),
            gross_return=float(gross), net_return=float(net),
        ))
        i = exit_idx + 1  # no-stack: skip past exit

    log.info(f"trades: {len(trades)}   skipped (bad atr): {skipped_bad_atr}")

    report: dict = {
        "symbol": symbol, "tag": tag, "threshold": threshold, "rr": rr,
        "probabilities": "calibrated" if use_calibrated else "raw",
        "oot_start": str(oot["_ts"].iloc[0]),
        "oot_end":   str(oot["_ts"].iloc[-1]),
        "n_bars": int(n),
        "n_trades": len(trades),
        "friction_bps": friction * 1e4,
        "max_holding_bars": max_holding,
    }
    if trades:
        net = np.array([t.net_return for t in trades])
        gross = np.array([t.gross_return for t in trades])
        wins = (net > 0).sum()
        equity = np.cumprod(1.0 + net)
        roll_max = np.maximum.accumulate(equity)
        dd = (equity - roll_max) / roll_max
        tp_n = sum(1 for t in trades if t.exit_reason == "TP")
        sl_n = sum(1 for t in trades if t.exit_reason == "SL")
        to_n = sum(1 for t in trades if t.exit_reason == "TIMEOUT")

        long_t = [t for t in trades if t.side == "LONG"]
        short_t = [t for t in trades if t.side == "SHORT"]

        report.update({
            "hit_rate": float(wins / len(trades)),
            "n_wins": int(wins),
            "n_losses": int(len(trades) - wins),
            "exit_tp": int(tp_n),
            "exit_sl": int(sl_n),
            "exit_timeout": int(to_n),
            "gross_mean_pct":  float(gross.mean() * 100),
            "net_mean_pct":    float(net.mean() * 100),
            "net_median_pct":  float(np.median(net) * 100),
            "cum_return_pct":  float((equity[-1] - 1.0) * 100),
            "max_drawdown_pct": float(dd.min() * 100),
            "sharpe_per_trade": float(net.mean() / (net.std() + 1e-12)),
            "days_oot": int((oot["_ts"].iloc[-1] - oot["_ts"].iloc[0]).days),
        })
        for label, subset in [("long", long_t), ("short", short_t)]:
            if subset:
                arr = np.array([t.net_return for t in subset])
                report[f"{label}_n"] = len(subset)
                report[f"{label}_hit_rate"] = float((arr > 0).sum() / len(subset))
                report[f"{label}_net_mean_pct"] = float(arr.mean() * 100)

        log.info(f"=== BACKTEST [{tag}  thr={threshold}  rr={rr}  {report['probabilities']}] ===")
        log.info(
            f"  trades={len(trades)}  hit={report['hit_rate']*100:.2f}%  "
            f"exits TP={tp_n} SL={sl_n} TO={to_n}"
        )
        log.info(
            f"  LONG  n={len(long_t):4d}  hit={report.get('long_hit_rate', 0)*100:6.2f}%  "
            f"net/trade={report.get('long_net_mean_pct', 0):+6.3f}%"
        )
        log.info(
            f"  SHORT n={len(short_t):4d}  hit={report.get('short_hit_rate', 0)*100:6.2f}%  "
            f"net/trade={report.get('short_net_mean_pct', 0):+6.3f}%"
        )
        log.info(
            f"  net/trade={report['net_mean_pct']:+.3f}%  "
            f"cum={report['cum_return_pct']:+.2f}%  "
            f"max_dd={report['max_drawdown_pct']:.2f}%  "
            f"over {report['days_oot']}d"
        )
    else:
        log.warning("no trades taken")

    suffix_json = f"backtest_{rr}_t{int(round(threshold*100)):02d}_{'cal' if use_calibrated else 'raw'}"
    (model_dir / f"{suffix_json}.json").write_text(json.dumps(report, indent=2))
    if trades:
        pd.DataFrame([asdict(t) for t in trades]).to_csv(
            model_dir / f"{suffix_json}_trades.csv", index=False
        )
    log.info(f"saved -> {model_dir}/{suffix_json}.json")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--rr", choices=list(RR_SCHEMES.keys()), default="atr_2_1")
    ap.add_argument("--oot-start", required=True)
    ap.add_argument("--raw", action="store_true",
                    help="use raw probabilities (default: calibrated)")
    args = ap.parse_args()
    cfg = load_config()
    run(args.symbol, cfg, tag=args.tag, threshold=args.threshold, rr=args.rr,
        oot_start=args.oot_start, use_calibrated=not args.raw)


if __name__ == "__main__":
    main()
