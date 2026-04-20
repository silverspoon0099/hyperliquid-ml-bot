# ML Trading Bot — Phase 1 Scaffold

Phase 1 implementation per `PROJECT_SPEC.md` (v2.1) and `SESSION_HANDOFF.md`.

## What is built

| Step | Module | Purpose |
|------|--------|---------|
| 1.1 | `data/collectors/fetcher.py` | Binance Futures OHLCV — 18mo backfill, paginated, resumable |
| 1.2 | `data/collectors/hyperliquid_ws.py` | 24/7 WebSocket: L2 book, trades, hourly funding |
| 1.3 | `features/indicators.py`, `features/volatility.py`, `features/volume.py` | RSI/WT/Stoch/Squeeze/MACD/ADX/EMA/ATR/BB/KC/VFI |
| 1.4 | `features/pivots.py`, `features/sessions.py`, `features/vwap.py`, `features/candles.py` | Fib pivots, sessions, VWAP, candle structure |
| 1.5 | `features/{stats,regime,context,divergence,extra_momentum,structure,adaptive_ma,ichimoku}.py` | Cats 9-19 |
| 1.6 | `features/event_memory.py` | Cat 20 — 41 recency / depth / count features |
| 1.7 | `features/builder.py::merge_1h_into_5m` | 1H → 5min map (shifted by 1 to prevent look-ahead) |
| 1.8 | `model/labeler.py` | Triple-barrier labeling |
| 1.9 | `features/builder.py` | Master pipeline → parquet feature matrix |

## Setup

```bash
# System deps (TA-Lib not actually required by current implementation but
# pandas-ta will use it if present)
sudo apt install -y libta-lib0 libta-lib-dev   # optional

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run order

```bash
# 1) Start the Hyperliquid collector — leave running 24/7 (best in tmux/systemd)
python -m data.collectors.hyperliquid_ws

# 2) In a separate terminal: backfill Binance OHLCV (BTC first; spec asset progression)
python -m data.collectors.fetcher --symbol BTC/USDT

# 3) Build the feature matrix (triple-barrier labels included by default)
python -m features.builder --symbol BTC/USDT

# Output: data/storage/features/BTC_features.parquet
```

## Critical rules baked in (do NOT regress)

- 5min entry, 1-4h holds, **trailing-stop exits** — no fixed TP.
- BTC correlation = booster; never used as a filter or to reduce confidence.
- **Bar-close execution only** — features computed on completed candles.
- Multi-TF merge uses **previous** 1H bar (`shift(1)`) to avoid look-ahead.
- Train on Binance (cleanest OHLCV); execute on Hyperliquid.
- Order book has NO historical API — collector starts Day 1 (Decision #25).
- Hyperliquid funding is **hourly** (not 8h) — Decision #26.

## What's left for Phase 2+

- Model training (`model/trainer.py` — LightGBM, Optuna, walk-forward purge=48).
- SHAP explainer (`model/explainer.py`).
- Backtest engine (`model/evaluator.py`).
- Live execution on Hyperliquid (`execution/`).
- Phase 3 features 269-289 — derived from collected Hyperliquid data once 2-3 months accumulate.
- Phase 4 features 290-295 — cross-asset correlation for altcoin transfer learning.
