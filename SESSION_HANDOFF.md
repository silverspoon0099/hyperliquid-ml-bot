# Session Handoff — ML Trading Bot Development

**Purpose:** This document captures EVERYTHING discussed across two planning sessions (2026-04-18 to 2026-04-19) so a new Claude session on the VPS can start Phase 1 implementation with full context. Read this FIRST, then read PROJECT_SPEC.md for the formal specification.

**Project directory:** `trading/ml-bot/`
**Full spec:** `trading/ml-bot/PROJECT_SPEC.md` (Version 2.1, 295 features, 26 decisions logged)
**TradingView indicators:** `trading/TradingView/top_strategies/` (60+ Pine scripts)
**User's custom scripts:** `trading/TradingView/my_scripts/` (5 Pine scripts)

---

## 1. Who Is the User

- **Crypto trader** building an ML-powered trading bot for Hyperliquid exchange
- **ML knowledge:** Base level — not comfortable designing from scratch, prefers existing frameworks (LightGBM, CCXT) with clear guidance
- **Trading experience:** Under 100 trades, 50%+ win rate
- **Domain expertise:** Worked on Bittensor Subnets �� this is why TAO is the final target asset
- **Platform:** Hyperliquid perpetual futures exchange
- **Development setup:** Windows 10 (local dev) + VPS (deployment), VS Code, Git

**How to work with this user:**
- Provide practical, step-by-step guidance — not theoretical ML lectures
- Favor existing frameworks over custom implementations
- Explain ML concepts through trading analogies when possible
- The user knows trading deeply but ML is new territory — bridge the gap

---

## 2. What the Bot Does (NOT a Scalping Bot)

This came up as a critical correction. The user initially described it as "scalping" but it's fundamentally different:

### Trading Style: "Momentum-with-Structure"

```
Pivot Fibonacci  = the structural map (where are targets and supports)
EMA50            = the momentum compass (which direction is the flow)
Oscillators      = timing confirmation (WT, RSI, DI, Squeeze)
```

| Aspect          | Scalping Bot    | THIS Bot                                      |
|-----------------|-----------------|-----------------------------------------------|
| Entry timeframe | 1-5 min         | 5 min (same — precise entry timing)           |
| Hold duration   | 5-30 min        | **1-4 hours typical**, 15 min to 1 day range  |
| Profit target   | Tiny (0.1-0.2%) | **0.5%+ BTC, 1-3%+ TAO**                     |
| Exit logic      | Fixed TP/SL     | **Trailing stop riding between pivot levels** |
| Trades/day      | 10-30           | **1-5** (min 1, max 7 hard cap)               |

### Two Trade Archetypes the Model Must Learn

**Type A: Full Pivot Ride (higher profit)**
- Entry: Bounce at Pivot level (e.g., S1) + momentum confirmation (DI+↑, DI-↓, WT cross)
- Ride: S1 → P → R1 → R2 (trailing stop behind each confirmed level)
- Exit: Trailing stop hit OR momentum exhaustion
- Profit: 0.5-1.5% BTC, 1-5%+ TAO
- Common on TAO (high volatility), rare on BTC (choppy between levels)

**Type B: EMA50 Direction Change (higher win rate)**
- Entry: EMA50 changes slope direction + price near a Pivot level
- Ride: EMA50 direction → next Pivot level
- Exit: Next Pivot level touch OR EMA50 reverses again
- Profit: 0.3-0.8% BTC, 0.5-2% TAO
- Common on BTC (where full Pivot rides are rare)

### Hold Time Distribution
- **Rare** (~15 min): Quick reversal at EMA50
- **Common** (1-4 hours): Ride S1→P→R1 or similar pivot-to-pivot
- **Less common** (full day): Big trend day S1→R3+, happens ~2-3 times/month

### Key User Insight About Entry
The user described their entry approach: "Daily Pivot Fibonacci on 5min, WaveTrend/EMA50 on 1H, DI+/DI- confirmation, ATR accumulation awareness, RSI with SMA+BB."

Critical insight: **"DI+ rising AND DI- flat/falling is the real signal"** — not just DI+ > DI-. This means the RATE OF CHANGE of directional indicators matters more than their absolute position. This directly informed features `di_plus_roc`, `di_minus_roc`, and `di_spread_roc`.

---

## 3. Asset Progression and BTC Correlation (CRITICAL)

### Training Order: BTC → SOL → LINK → TAO → HYPE

| Asset | Why This Order | BTC Correlation |
|-------|---------------|-----------------|
| BTC | Most data, cleanest patterns, base model | — |
| SOL | High BTC correlation, good transfer target | High |
| LINK | High correlation, moderate volatility | High |
| **TAO** | **Final target** — user's primary asset, Bittensor domain expertise | Variable — follows BTC mostly, can surge/dump independently on ecosystem news |
| HYPE | Hyperliquid native token — potential zero/reduced trading fees | Variable |

### BTC Correlation = BOOSTER, NOT FILTER (User Correction — Do NOT Violate)

This was a **strong correction** from the user. I initially wrote "decorrelation safeguards" suggesting reduced confidence and position sizing during BTC-TAO decorrelation. The user strongly disagreed:

> "When TAO is news driven, it can surge over the week, how can I skip this golden opportunity... model should detect the decorrelation then reduce the correlation value and detect its own price movement."

**The rule:**
```
Token's OWN features (268):     The engine — ALWAYS drives the signal
BTC cross-asset features (6):   The turbo — adds power when correlated, ignored when not
```

**What this means for implementation:**
- Each token's model runs primarily on the token's own 268 features
- BTC features are OPTIONAL context — the model naturally ignores them when they're noise
- NEVER build any mechanism that reduces confidence, skips signals, or reduces position size when BTC correlation is low
- Independent altcoin moves (TAO March 2026 rally, TAO April 10 dump) are the BEST trading opportunities
- TAO April 10 dump = GO SHORT (not "stay out" — a dump is a short opportunity)
- `correlation_20bar` is context, not a gate

This applies to ALL altcoins, not just TAO. SOL, LINK, HYPE can all move independently on news.

---

## 4. Data Source Architecture (Major Decision)

### Why Binance for Training, Hyperliquid for Live

The user asked whether to use Binance or Hyperliquid data. After analysis:

**Binance Futures for OHLCV (training data):**
- ~10x daily volume vs Hyperliquid → cleaner candles with less wick noise
- Complete 18+ month history via REST API (CCXT)
- Perpetual futures prices between exchanges differ by < 0.01% on 5-min candles (arbitrage bots)
- All 268 indicator features computed from OHLCV are exchange-agnostic (price is price)
- BTC: Binance ~$30-50B/day vs Hyperliquid ~$2-5B/day
- TAO: Binance ~$200-500M/day vs Hyperliquid ~$30-80M/day

**Hyperliquid for live/exchange-specific data:**
- Order book (L2 Book WebSocket) — THIS is where you're executing, so THIS book matters
- Trade tape (Trades WebSocket) — actual buy/sell data from your execution venue
- Funding rate — Hyperliquid has **hourly funding** (not 8-hour like Binance), unique data
- OI, liquidation data — exchange-specific

**Critical: Order book has NO historical API.** It's live-streaming only. You CANNOT backfill. Start the collector on Day 1 of Phase 1. After 2-3 months, this data becomes trainable.

### Data Collection Architecture

```
Phase 1 Day 1 (parallel):
├── Binance OHLCV fetcher: Backfill 18 months of BTC 5min + 1H candles → parquet
└── Hyperliquid WebSocket collector: Start streaming order book, trade tape, funding → parquet
    (runs continuously in background while you build features)

Phase 2 (weeks 4-6):
└── Train model on Binance OHLCV → 268 features only

Phase 3 (weeks 7-9):
└── Add 21 Hyperliquid microstructure features (now have 2-3 months of live data)
    Retrain model: 268 → 289 features

Phase 4 (weeks 10-14):
└── Add 6 cross-asset features for altcoins
    Fine-tune: BTC → SOL → LINK → TAO → HYPE
```

### 18 Months Minimum Data

User specifically requested 18 months (upgraded from initial 6 months �� 1 year → 18 months). The reasoning:
- BTC moved 68K → 126K rally AND sudden drop to 60K in the past 18 months
- Model needs BOTH bull runs and crash regimes
- Must include trending, ranging, volatile, and crash/recovery regimes

---

## 5. Feature Engineering — 295 Features Across 22 Categories

### Design Philosophy

1. **Raw values, not rules**: Feed `RSI = 65.3`, not `overbought = true`. ML learns its own thresholds.
2. **Percentage-based normalization**: All price-distance features as %. BTC $500 move = SOL $2 move = TAO $15 move in comparable feature values. Enables BTC→altcoin transfer learning.
3. **Depth framework**: Every major indicator gets: value, direction, acceleration, position vs signal line, cross momentum, range context. RSI alone = 12 features, not 1.
4. **Event Memory**: "RSI is at 55" is meaningless. "RSI at 55 but was oversold 8 bars ago" = recovery signal. 41 features track recency of key events.
5. **Log-transform volatile features**: Apply `log(1 + |x|)` to ATR, volume spikes, extreme RSI (Lorentzian-inspired from the ML classification TradingView indicator).

### Feature Phase Breakdown

| Phase | Features | Count | Source |
|-------|----------|-------|--------|
| Phase 1 | Categories 1-20 (indicators, pivots, sessions, event memory, etc.) | 268 | Computed from Binance OHLCV |
| Phase 3 | Category 21 (funding, OI, order book, trade tape, funding deep) | 21 | Hyperliquid live WebSocket (2-3 months collection) |
| Phase 4 | Category 22 (BTC correlation, relative strength, dominance) | 6 | BTC data from Binance |
| **Total** | | **295** | |

### The 22 Categories (count, phase)

1. Momentum (RSI + WT + Stoch + Squeeze + MACD deep) — 47, Phase 1
2. Trend / Direction (ADX/DI deep + EMA) — 19, Phase 1
3. Volatility — 15, Phase 1
4. Volume / Buy-Sell (VFI deep) — 17, Phase 1
5. VWAP — 8, Phase 1
6. S/R Structure (Pivots) — 13, Phase 1
7. Session / Time — 15, Phase 1
8. Price Action / Candle — 9, Phase 1
9. Mean Reversion / Stats — 8, Phase 1
10. Market Regime — 7, Phase 1
11. Previous Context / Memory — 8, Phase 1
12. Lagged Dynamics — 8, Phase 1
13. Divergence Detection — 7, Phase 1
14. Money Flow (CMF/MFI/AD) — 8, Phase 1
15. Additional Momentum (Williams %R/CCI/CMO/TSI) — 9, Phase 1
16. Market Structure (HH/HL/LH/LL) — 10, Phase 1
17. Statistical / Fractal — 9, Phase 1
18. Adaptive MAs (KAMA/DEMA/TEMA/SAR) — 5, Phase 1
19. Ichimoku (partial) — 5, Phase 1
20. Event Memory — 41, Phase 1
21. Microstructure (Hyperliquid) — 21, Phase 3
22. Cross-Asset Correlation — 6, Phase 4

### buy_volume_pct Reliability Note

Category 4 includes `buy_volume_pct` = `(close - low) / (high - low)`. This is a **candle-shape heuristic**, NOT true buy/sell measurement.

Reliability: ~0.5-0.65 correlation with actual trade-level buy/sell data.
- Works well on strong directional candles
- Unreliable on dojis and long-wick candles
- Division by ~0 when high ≈ low

**Decision: Keep them.** LightGBM handles noisy features — it just ignores them if they don't help. SHAP analysis will reveal true importance. The REAL buy/sell data comes from Category 21's `trade_delta_5min` (Hyperliquid trade tape), which gets added in Phase 3.

---

## 6. User's TradingView Indicators (Feature Source)

All features are derived from indicator logic in the user's collected TradingView scripts.

### Currently Used by Trader (on their live chart)
- **Pivots Fibonacci** — S/R structure, the backbone of their trading
- **BB_KC** (custom `my_scripts/BB_KC.pine`) — Bollinger + Keltner squeeze detection
- **CM Stochastic Multi-TF** — Stoch K/D on 5min and 1H
- **WaveTrend with Crosses** — WT1, WT2, histogram, cross signals
- **Squeeze Momentum (LazyBear)** — squeeze state, momentum, acceleration
- **ADX and DI** — trend strength and directional confirmation
- **CM Ultimate MACD MTF** — MACD histogram on 5min and 1H
- **RSI (with SMA + BB)** — RSI value, RSI vs SMA, RSI BB position

### User's Custom Scripts (in `my_scripts/`)

**UT Bot v4.0** — Feature concepts extracted:
- `efficiency_ratio` = abs(close - close[14]) / sum(abs(close - close[1]), 14)
- `atr_ratio` = ATR(5) / ATR(14)
- Momentum normalization approach

**UT Bot v4.0 Design Analysis (NOT a bug, but NOT used for execution):**
The v4.0 uses dynamic `currentMult` that's typically calculated below 1.5 by design, producing a tight trailing stop that generates many signals. Combined with `confThreshold = 0`, every trail touch triggers a signal. This is the v4.0 design intent (NOT a bug, NOT "false signals") — but the high signal frequency makes it unsuitable for execution logic targeting 1-5 trades/day.

**UT Bot v3.1:**
- Uses `confThreshold = 0.15` (more selective than v4.0)
- ADX filter is available but **user set `adx_threshold = 0`** (disabled it)
- Produces fewer, more selective signals

**Volume Flow Indicator MTF (modified by K)** — Multi-timeframe VFI
**BB_KC** — Bollinger + Keltner Channel squeeze detection

### 60+ Collected Indicators (in `top_strategies/`)

ML-Based: Lorentzian Classification, ML Adaptive SuperTrend
Smart Money: ICT Concepts, Buyside & Sellside Liquidity, Market Structure Break, Order Block Detector
S/R: CM_Pivot Points, Volume-based S/R, Support Resistance Dynamic
Trend: Alpha Trend, Chandelier Exit, SuperTrend, Madrid MA Ribbon, Trendlines with Breaks
Momentum: VuManChu Cipher B, Divergence for Many Indicators v4, WaveTrend
Volume: Volume Flow Indicator, Volume-based S/R
Sessions: Sessions [LuxAlgo]
Breakout: Breakout Finder, Breakout Probability, Fibonacci Bollinger Bands

The full analysis of each indicator and what features were extracted is in PROJECT_SPEC.md Section 5.

---

## 7. Labeling Strategy — Triple Barrier (NOT Fixed Lookahead)

### Why Not Fixed Lookahead

Fixed lookahead fails because the user's hold time varies (15 min to 1 day):
- 12-bar (1hr) lookahead: A 3-hour S1→R1 trade gets labeled NEUTRAL (move hadn't finished)
- 48-bar (4hr) lookahead: A quick 15-min reversal uses 3.75 hours of noise in its label

### Triple-Barrier Method (Lopez de Prado)

For each 5-min candle, simulate a trade forward with three exit conditions:
1. **Upper barrier**: price rises `tp_atr_mult × ATR` → LONG (label 0)
2. **Lower barrier**: price falls `sl_atr_mult × ATR` → SHORT (label 1)
3. **Time barrier**: `max_holding_bars` reached → check net P&L → LONG/SHORT/NEUTRAL

| Parameter | BTC | SOL/LINK | Why |
|-----------|-----|----------|-----|
| tp_atr_mult | 3.0x ATR | 3.0x ATR | Multi-pivot moves ≈ 2-3x ATR |
| sl_atr_mult | 2.0x ATR | 2.0x ATR | Matches trader's real stop placement |
| max_holding_bars | 48 (4 hours) | 48 (4 hours) | Covers common 1-4 hour hold range |
| min_profit_pct | 0.3% | 0.6% | Time-barrier minimum for directional label |

The full labeling code is in PROJECT_SPEC.md Section 9.2.

### No Fixed Take Profit — Trailing Stop Only

**Critical decision:** No fixed TP. The trailing stop lets winners ride S1→P→R1→R2. A fixed 1.5x ATR TP would cut these multi-pivot moves short. The trailing stop (2.0x ATR behind price) matches how the user actually exits.

---

## 8. ML Model — LightGBM

### Why LightGBM (Not Deep Learning)

- Fastest training among gradient boosting libraries
- Handles 295 features without dimensionality reduction
- Built-in SHAP integration for feature importance
- Native incremental learning via `init_model` → enables BTC→altcoin transfer
- Works on tabular data (our features are tabular, not sequential)
- No GPU required
- Research shows ensemble models (XGBoost, LightGBM) outperform LSTM/SVR for crypto

### 3-Class Classification

Output: `[P(LONG), P(SHORT), P(NEUTRAL)]`
- If P(LONG) > 0.65 → signal = LONG
- If P(SHORT) > 0.65 → signal = SHORT
- Else → NO TRADE

### Transfer Learning (BTC → Altcoins)

```python
btc_model = lgb.train(params, btc_train_data)
btc_model.save_model('btc_base.model')

# Progressive fine-tuning
sol_model = lgb.train(params, sol_train_data, init_model='btc_base.model')
link_model = lgb.train(params, link_train_data, init_model='btc_base.model')
tao_model = lgb.train(params, tao_train_data, init_model='btc_base.model')
```

### SHAP Analysis

After first training, SHAP reveals what the model actually learned:
- Which features drive predictions globally?
- For THIS specific trade, why did the model say LONG?
- Expected to trim from 295 → 120-150 features after first SHAP analysis

### Walk-Forward Validation

Financial data is time-ordered — random splits cause look-ahead bias. Use walk-forward with a **48-bar purge gap** (matching max_holding_bars) between train and test sets.

---

## 9. Key Corrections and Non-Obvious Decisions

These are things that came up in discussion that might not be immediately obvious from the spec:

### 9.1 Trading Style Corrections
- **NOT scalping** — holds are 1-4 hours, not 5-30 minutes
- Trades per day: **1-5** (was 1-3, user updated). Min 1/day — skip only if momentum is truly dead all session. Max 7 hard cap.
- Big trend days (S1→R3+): Happen **2-3 times per month** (was "~once/month", user corrected)
- Hold days (position for full day) is **rare** — user clarified this

### 9.2 TAO-Specific Insights
- TAO March 2026: Independent rally (not BTC-driven). Model should trade using TAO's own features.
- TAO April 10 2026: Independent dump. Model should **go short** — a dump is a short opportunity, NOT a reason to sit out.
- TAO often moves independently on Bittensor ecosystem news — these are the BEST trades, not risks.

### 9.3 UT Bot Analysis Corrections
- v4.0's `currentMult` typically < 1.5 is the **design intent**, not a bug that produces "false signals that v3.1 correctly filters." It's a different design philosophy (tighter trailing = more signals).
- v3.1's ADX filter: User **disabled it** by setting `adx_threshold = 0`.
- Neither bot's execution logic is used — only feature CONCEPTS (efficiency_ratio, atr_ratio) are extracted.

### 9.4 BTC Correlation (The Biggest Correction)
- I initially wrote "decorrelation safeguards" — user strongly corrected this
- Independent altcoin moves are GOLDEN OPPORTUNITIES, not risks
- Never reduce confidence, skip signals, or reduce position size during low BTC correlation
- Each token's own 268 features are the ENGINE. BTC features are TURBO.
- LightGBM naturally handles this — it ignores low-value features per-context

### 9.5 Order Book Decisions
- `buy_volume_pct` = candle-shape heuristic (~0.5-0.65 correlation with reality). Kept for SHAP evaluation.
- Real buy/sell data from Hyperliquid trade tape (`trade_delta_5min`) is far superior
- Hyperliquid order book is thinner than Binance (lower volume), but it IS the execution environment
- Order book data is live-only — NO historical API — must collect from Day 1

### 9.6 Funding Rate
- Hyperliquid has **hourly funding** (not 8-hour like Binance)
- This gives 8x faster signal on crowding changes
- 3 dedicated funding rate deep features added (287-289)

### 9.7 Option A vs Option C
- **Option A (Pure ML) first**: No rule-based filters. Let the model learn freely.
- **Option C (later)**: After backtesting + live results, consider adding Pivot Fibonacci as a hard filter to reduce overtrading. Decision based on actual results, not assumption.
- The model should learn its own thresholds from raw indicator values, not pre-defined rules.

---

## 10. Preprocessing Pipeline

```
Raw OHLCV data (from Binance Futures)
    │
    ▼
Indicator Calculation (ta-lib / pandas-ta)
    │
    ▼
Percentage Normalization (all price distances as %)
    │
    ▼
Log Transform (volatile features: ATR, volume spikes, extreme RSI)
    │
    ▼
MinMaxScaler(-1, 1) — per-feature scaling
    │
    ▼
VarianceThreshold — remove zero-variance columns
    │
    ▼
NaN Handling — forward-fill then drop first N warmup rows
    │
    ▼
Feature Matrix (DataFrame: rows=candles, cols=295 features)
```

**Multi-timeframe merge:** 1H indicator values are mapped to 5min candles. IMPORTANT: Use the PREVIOUS completed 1H bar to avoid look-ahead bias (`df_1h.shift(1)`).

---

## 11. Execution Architecture

### Bar-Close Only
- Signals evaluated ONLY at the close of each 5-minute candle
- All features require complete candles (close price, volume, wick ratios)
- Intrabar signals are unreliable — partial candles produce noise
- This was a deliberate rejection of UT Bot v4.0's intrabar approach

### Hyperliquid Execution
- REST API for orders and account info
- WebSocket for real-time price updates
- `hyperliquid-python-sdk` for Python integration
- Limit orders preferred (avoid taker fee when possible)
- Stop loss as reduce-only order, updated every 5min bar (trailing)

### Risk Management
- Max risk per trade: 2% of capital
- Max position size: 10% of capital
- Max daily loss: 5% (circuit breaker)
- Max concurrent positions: 1 (Phase 1), 2 (Phase 3)
- ATR trailing stop: 2.0x ATR, only moves in favorable direction
- **No fixed take profit** — trailing stop lets winners ride

### Realistic Backtesting Assumptions
- Trading fee: 0.035% per trade (taker) on Hyperliquid
- Slippage: 0.02% per trade (conservative)
- Funding rate: Applied **every 1 hour** on Hyperliquid (not 8-hour)
- HYPE may have zero/reduced maker fees — make fees asset-specific in config

---

## 12. Tech Stack

```
# Data
pandas >= 2.0, numpy >= 1.24, pyarrow (parquet I/O)

# Technical Analysis
ta-lib (C-based, fast), pandas-ta (fallback)

# ML
lightgbm >= 4.0, scikit-learn, optuna (hyperparameter tuning), shap

# Exchange
ccxt (historical data from Binance), hyperliquid-python-sdk (live trading)

# Utilities
pyyaml, loguru, schedule

# Python 3.11+
```

---

## 13. Directory Structure

```
ml-bot/
├── PROJECT_SPEC.md              # Full 295-feature specification (Version 2.1)
├── SESSION_HANDOFF.md           # This document
├── config.yaml                  # All parameters, thresholds, API keys reference
├── requirements.txt
│
├── data/
│   ├── collectors/
│   │   ├── fetcher.py           # OHLCV from Binance Futures (18 months backfill)
│   │   ├── hyperliquid_ws.py    # Live WebSocket (order book, trade tape, funding)
│   │   └── storage.py           # Parquet read/write utilities
│   └── storage/
│       ├── btc_5m.parquet       # Raw 5min OHLCV (from Binance)
│       ├── btc_1h.parquet       # Raw 1H OHLCV (from Binance)
│       ├── btc_features.parquet # Complete feature matrix
│       └── hyperliquid/         # Live data (order book, trades, funding)
│
├── features/
│   ├── indicators.py            # RSI, WaveTrend, ADX, MACD, Stochastic deep features
│   ├── pivots.py                # Pivot Fibonacci with context features
│   ├── sessions.py              # Session detection, overlaps, session VWAP
│   ├── vwap.py                  # Daily VWAP with bands
│   ├── volume.py                # VFI deep, buy/sell pressure, OBV, cumulative delta
│   ├── money_flow.py            # CMF, MFI, Accumulation/Distribution
│   ├── candles.py               # Price action features (body %, wick %, engulfing)
│   ├── regime.py                # Market regime detection
│   ├── stats.py                 # Z-scores, mean reversion, returns, statistical/fractal
│   ├── structure.py             # Market structure (HH/HL/LH/LL, swing detection)
│   ├── divergence.py            # RSI/MACD/WT/Stoch price divergences
│   ├── event_memory.py          # Event recency features (41 features)
│   ├── adaptive_ma.py           # KAMA, DEMA, TEMA, Parabolic SAR
│   ├── ichimoku.py              # Partial Ichimoku (Tenkan, Kijun, cloud)
│   ├── context.py               # Previous day/session context, lagged dynamics
│   └── builder.py               # Combine all features, multi-TF merge, preprocessing
│
├── model/
│   ├── labeler.py               # Triple-barrier labeling
│   ├── trainer.py               # LightGBM training pipeline
│   ├── tuner.py                 # Optuna hyperparameter optimization
│   ├── evaluator.py             # Backtest simulation + performance metrics
│   ├── explainer.py             # SHAP analysis
│   └── models/                  # Saved model files
│
├── execution/
│   ├── hyperliquid.py           # Exchange connection, order management
│   ├── risk.py                  # Position sizing, trailing stop, risk rules
│   ├── paper_trader.py          # Simulated live trading
│   └── live_trader.py           # Real execution engine
│
├── monitoring/
│   ├── logger.py
│   └── dashboard.py
│
├── tests/
│   ├── test_indicators.py       # Verify calculations match TradingView
│   ├── test_features.py
│   ├── test_labeler.py
│   └── test_risk.py
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_feature_analysis.ipynb
    ├── 03_model_training.ipynb
    └── 04_shap_analysis.ipynb
```

---

## 14. Phase 1 Implementation Guide (START HERE)

### Step 1.1: Binance OHLCV Fetcher

```python
import ccxt
import pandas as pd

# Use Binance Futures — deepest liquidity, cleanest candles
exchange = ccxt.binance({'options': {'defaultType': 'future'}})

# Fetch 18 months: ~157,680 5min candles, ~13,140 1H candles
# CCXT has a limit per request �� must paginate
# Store as parquet files partitioned by month
```

Key considerations:
- CCXT limits per request (usually 1000-1500 candles) — need pagination loop
- Fetch BTC/USDT (not BTC/USDC — Binance Futures uses USDT)
- Also fetch for SOL, LINK, TAO, HYPE while you're at it (parallel fetching)
- Store as parquet (much faster than CSV for large datasets)
- Verify data completeness — check for gaps in timestamps

### Step 1.2: Hyperliquid WebSocket Collector (Start Immediately)

```python
# WebSocket subscriptions needed:
{"method": "subscribe", "subscription": {"type": "l2Book", "coin": "BTC"}}
{"method": "subscribe", "subscription": {"type": "trades", "coin": "BTC"}}
# Also subscribe for TAO, SOL, LINK, HYPE

# Funding rate: poll via REST API every hour
from hyperliquid.info import Info
info = Info(base_url="https://api.hyperliquid.xyz")
```

This collector runs 24/7 in the background. It must be robust (auto-reconnect on disconnect). Store snapshots to date-partitioned parquet files. Order book sampled every 5 seconds, trades stored raw.

### Step 1.3-1.6: Feature Calculation

Implement each category's features in a separate Python file. The full feature definitions (all 268 Phase 1 features with exact calculations) are in PROJECT_SPEC.md Section 6.2.

Libraries to use:
- `ta-lib` for standard indicators (RSI, EMA, MACD, Stochastic, ADX, ATR, Bollinger, etc.)
- `pandas-ta` as fallback for indicators not in ta-lib
- Custom calculations for: Pivot Fibonacci, Event Memory, divergence detection, market structure

### Step 1.7: Multi-Timeframe Merge

Map 1H indicator values to 5min candles. **Use previous completed 1H bar** (shift by 1) to avoid look-ahead bias.

### Step 1.8: Labeling

Implement triple-barrier labeling function. Full code in PROJECT_SPEC.md Section 9.2.

### Step 1.9: Export Feature Matrix

Final output: `btc_features.parquet` — DataFrame where each row = one 5-min candle, columns = 268 features + label + metadata (timestamp, holding_bars, etc.)

---

## 15. Success Criteria

### Phase 2 (Backtest)

| Metric | Target | Hard Minimum |
|--------|--------|--------------|
| Win Rate | > 58% | > 52% |
| Profit Factor | > 1.8 | > 1.3 |
| Max Drawdown | < 8% | < 15% |
| Trades per Day | 1-5 | < 7 |
| Sharpe Ratio | > 2.0 | > 1.0 |
| Avg Trade Duration | 1-4 hours | 30 min - 8 hours |
| Avg Win Duration | > Avg Loss Duration | — |

### Phase 3 (Paper Trading)
- Minimum 2 weeks paper trading before live
- Win rate within 5% of backtest
- Signal latency < 2 seconds from bar close
- System uptime > 99%

---

## 16. Research Sources Used

The ML approach was validated through research:
- Ensemble models (XGBoost, LightGBM) outperform deep learning (LSTM, SVR) for crypto — R² ~0.98 (Springer 2025)
- LightGBM achieved slightly better accuracy than XGBoost for crypto prediction
- FreqAI recommends LightGBM/XGBoost, dropped CatBoost in 2025.12
- The Lorentzian Classification TradingView indicator (14k+ boosts) validated the multi-feature ML approach with distance metrics
- Full reference list with URLs in PROJECT_SPEC.md Section 19

---

## 17. Things NOT to Do

1. **Don't treat it as a scalping bot** — 1-4 hour holds, not 5-30 minute
2. **Don't add decorrelation safeguards** — independent altcoin moves are opportunities
3. **Don't use UT Bot v4.0 execution logic** — only feature concepts
4. **Don't use fixed take profit** — trailing stop only
5. **Don't use intrabar entry** — bar-close execution only
6. **Don't use random train/test split** — walk-forward validation with 48-bar purge gap
7. **Don't hard-code trading rules** — the ML learns from raw feature values (Option A)
8. **Don't skip TAO dump signals** — dumps are short opportunities
9. **Don't use Hyperliquid for OHLCV training data** — use Binance (cleaner, deeper history)
10. **Don't try to backfill order book data** — it doesn't exist. Start live collection now.

---

## 18. Summary of All 26 Design Decisions

| # | Decision | Date |
|---|----------|------|
| 1 | Option A first (Pure ML) — no rule-based filters | 2026-04-18 |
| 2 | Option C later — add Pivot filter after validation | 2026-04-18 |
| 3 | Raw values, not rules | 2026-04-18 |
| 4 | DI rate of change as feature | 2026-04-18 |
| 5 | Custom Python pipeline over FreqAI | 2026-04-18 |
| 6 | LightGBM over deep learning | 2026-04-18 |
| 7 | 295 features, trim after SHAP | 2026-04-18 |
| 8 | Session overlaps as explicit features | 2026-04-18 |
| 9 | Balanced feature categories (22) | 2026-04-18 |
| 10 | Percentage-based features for transfer learning | 2026-04-18 |
| 11 | UT Bot v4.0 — feature concepts only | 2026-04-18 |
| 12 | Bar-close execution only | 2026-04-18 |
| 13 | Depth framework for indicators | 2026-04-18 |
| 14 | Event Memory as dedicated category (41 features) | 2026-04-18 |
| 15 | Divergence detection features | 2026-04-18 |
| 16 | Statistical/Fractal features | 2026-04-18 |
| 17 | NOT scalping — 5min entry, swing-style exit | 2026-04-19 |
| 18 | Triple-barrier labeling over fixed lookahead | 2026-04-19 |
| 19 | Trailing stop only, no fixed take profit | 2026-04-19 |
| 20 | Momentum-with-structure, not strict Pivot Fibonacci | 2026-04-19 |
| 21 | Asset progression: BTC → SOL → LINK → TAO → HYPE | 2026-04-19 |
| 22 | BTC correlation = booster, not filter | 2026-04-19 |
| 23 | Binance OHLCV for training, Hyperliquid for live data | 2026-04-19 |
| 24 | 18 months minimum data collection | 2026-04-19 |
| 25 | Hyperliquid live data: collect from Phase 1, use in Phase 3 | 2026-04-19 |
| 26 | Hyperliquid hourly funding rate features | 2026-04-19 |

---

## 19. Starting a New Session on VPS

When you start a new Claude session on the VPS, provide this context:

```
I'm starting Phase 1 development of my ML trading bot for Hyperliquid.

Read these two documents first:
1. trading/ml-bot/SESSION_HANDOFF.md — full context from planning sessions
2. trading/ml-bot/PROJECT_SPEC.md — formal specification (Version 2.1, 295 features)

The TradingView indicator scripts that features are based on are in:
- trading/TradingView/top_strategies/ (60+ collected indicators)
- trading/TradingView/my_scripts/ (my custom Pine scripts)

Start with Phase 1:
1.1 — Binance Futures OHLCV fetcher (18 months BTC 5min + 1H)
1.2 — Hyperliquid WebSocket live collector (order book, trade tape, funding)
Then proceed through 1.3-1.9 (feature calculation, labeling, export)
```

The new session should read both documents, understand the architecture, and begin writing code for Phase 1 step by step.

---

_End of handoff document. Created 2026-04-19._
