# ML Trading Bot for Hyperliquid — Project Development Specification

**Version:** 2.1
**Date:** 2026-04-19
**Status:** Planning Complete, Ready for Phase 1 Development

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Goals & Constraints](#2-project-goals--constraints)
3. [Architecture Overview](#3-architecture-overview)
4. [Research Findings](#4-research-findings)
5. [TradingView Indicator Analysis](#5-tradingview-indicator-analysis)
6. [Complete Feature Engineering Specification](#6-complete-feature-engineering-specification)
7. [ML Model Design](#7-ml-model-design)
8. [Data Pipeline](#8-data-pipeline)
9. [Labeling Strategy](#9-labeling-strategy)
10. [Training & Validation](#10-training--validation)
11. [Backtesting Framework](#11-backtesting-framework)
12. [Execution Layer (Hyperliquid)](#12-execution-layer-hyperliquid)
13. [Risk Management](#13-risk-management)
14. [Project Phases & Timeline](#14-project-phases--timeline)
15. [Tech Stack](#15-tech-stack)
16. [Directory Structure](#16-directory-structure)
17. [Success Criteria](#17-success-criteria)
18. [Key Design Decisions](#18-key-design-decisions)
19. [References & Sources](#19-references--sources)

---

## 1. Executive Summary

This project builds an ML-powered trading bot for the Hyperliquid perpetual futures exchange, targeting **5-minute entry with swing-style exits** on BTC/USDC with progressive expansion to SOL, LINK, TAO, and HYPE. Positions are held 1-4 hours typically, riding moves between Fibonacci pivot levels — NOT traditional scalping (15-30 min holds).

**Final target assets:** TAO (high volatility, trader has Bittensor subnet domain expertise) and LINK. HYPE as a fee-advantage candidate on Hyperliquid.

The core approach:

- **Feature-driven ML classification** using LightGBM, not rule-based trading
- **295 engineered features** across 22 market dimensions, derived from professional TradingView indicators
- **Multi-timeframe analysis**: 5min entry precision with 1H directional context
- **Train on BTC first**, then incremental learning: BTC → SOL → LINK → TAO → HYPE
- **Momentum-with-structure trading**: Pivot Fibonacci as structural map, EMA50 as momentum compass, oscillators for timing — the model learns both full pivot rides AND EMA50 direction change entries
- **Option A (Pure ML) first**: Let the model learn freely from data without imposing trading rules, then consider Option C (Pivot Fibonacci hard filter) after backtesting validates results

The bot does NOT automate fixed trading rules. Instead, it learns from historical data which combinations of market conditions produced profitable outcomes, and assigns probability scores to current setups.

---

## 2. Project Goals & Constraints

### Goals

- Automate trade signal generation with ML-predicted probabilities
- Achieve higher win rate than manual trading (baseline: 50%+)
- Target 1-5 trades per day (minimum 1 — skip entire day only if momentum is truly dead all session)
- Capture 0.5%+ price movements on BTC, 1%+ on altcoins
- Hold positions for the full move between Fibonacci pivot levels (typical: 1-4 hours)
- Provide SHAP-based transparency into model decisions

### Trading Style Definition

This is **NOT** a traditional scalping bot and **NOT** a strict Pivot Fibonacci bot. It is a **momentum-with-structure** trading system:

- **Pivot Fibonacci** = the structural map (where are targets and supports)
- **EMA50** = the momentum compass (which direction is the flow)
- **Momentum oscillators** (WT, RSI, DI, Squeeze) = timing confirmation

| Aspect          | Scalping Bot    | This Bot                                           |
| --------------- | --------------- | -------------------------------------------------- |
| Entry timeframe | 1-5 min         | 5 min (same — precise entry timing)                |
| Hold duration   | 5-30 min        | **1-4 hours typical**, 15 min to 1 day range       |
| Profit target   | Tiny (0.1-0.2%) | **0.5%+ BTC, 1-3%+ TAO** — multi-pivot-level moves |
| Exit logic      | Fixed TP/SL     | **Trailing stop riding between pivot levels**      |
| Trades/day      | 10-30           | **1-5**                                            |

**Hold time distribution:**

- Rare (quick reversal at EMA50): ~15 min
- **Common (ride S1→P→R1 or similar)**: 1-4 hours
- Less common (big trend day, S1→R3+): Full day, ~2-3 times/month

### Two Trade Archetypes

The model should learn to recognize BOTH entry patterns from the features — it doesn't need explicit "type A vs B" classification, but understanding the trader's behavior informs feature design and labeling.

**Type A: Full Pivot Ride (higher profit)**

```
Entry:  Bounce at Pivot level (e.g., S1) + momentum confirmation (DI+↑, DI-↓, WT cross)
Ride:   S1 → P → R1 → R2 (trailing stop behind each confirmed level)
Exit:   Trailing stop hit OR momentum exhaustion at a Pivot level
Profit: 0.5-1.5% BTC, 1-5%+ TAO
Freq:   Common on TAO (high volatility), rare on BTC (choppy between levels)
```

**Type B: EMA50 Direction Change (higher win rate)**

```
Entry:  EMA50 changes slope direction + price near a Pivot level
Ride:   EMA50 direction → next Pivot level
Exit:   Next Pivot level touch OR EMA50 reverses again
Profit: 0.3-0.8% BTC, 0.5-2% TAO
Freq:   Common on BTC (where full Pivot rides are rare), always available
```

**Asset-specific patterns observed by trader:**

| Asset | Full Pivot Rides                                   | EMA50 Entries          | News Risk                           |
| ----- | -------------------------------------------------- | ---------------------- | ----------------------------------- |
| BTC   | Rare — price often stalls between adjacent levels  | **Primary entry type** | Low                                 |
| SOL   | Moderate                                           | Moderate               | Low-Moderate                        |
| LINK  | Moderate                                           | Moderate               | Moderate                            |
| TAO   | **Frequent — volatility drives multi-pivot moves** | Also available         | **High** — March rally, Apr 10 dump |
| HYPE  | TBD                                                | TBD                    | Moderate (exchange-specific events) |

### Constraints

- Single developer with base-level ML knowledge — must use existing frameworks, not custom architectures
- Trading experience: ~100 trades, 50%+ win rate, 5min entry with swing-style 1H exits
- Budget: Start with small capital, scale after validation
- No GPU required (LightGBM runs on CPU)
- Must paper trade before live deployment

### Non-Goals (explicitly excluded)

- High-frequency trading (sub-second execution)
- Market making or arbitrage strategies
- Deep learning as primary model (may add as secondary vote in future)
- Multi-exchange execution
- Social sentiment or news-based features (Phase 1)

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML Trading Bot                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────────┐     │
│   │   Data    │───>│   Feature    │───>│    LightGBM      │     │
│   │ Collector │    │  Calculator  │    │    Classifier     │     │
│   └──────────┘    └──────────────┘    └────────┬─────────┘     │
│        │                                        │               │
│        │          ┌──────────────┐              │               │
│        │          │   Labeler    │              │               │
│        │          │ (training)   │              │               │
│        │          └──────────────┘              │               │
│        │                                        ▼               │
│   ┌──────────┐                         ┌──────────────────┐    │
│   │ 5min +   │                         │   Signal:        │    │
│   │ 1H OHLCV │                         │   LONG  (0.72)   │    │
│   │ + Volume │                         │   SHORT (0.08)   │    │
│   └──────────┘                         │   NEUT  (0.20)   │    │
│                                        └────────┬─────────┘    │
│                                                  │              │
│                                                  ▼              │
│                                        ┌──────────────────┐    │
│                                        │   Risk Manager   │    │
│                                        │  Position Sizing  │    │
│                                        │  ATR Trailing SL  │    │
│                                        └────────┬─────────┘    │
│                                                  │              │
│                                                  ▼              │
│                                        ┌──────────────────┐    │
│                                        │   Hyperliquid    │    │
│                                        │   Executor       │    │
│                                        │  (REST + WS)     │    │
│                                        └──────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Exchange API ──> Raw OHLCV (5min + 1H)
                    │
                    ▼
              Feature Calculator ──> 295 raw indicator values per candle
                    │
                    ▼
              [Training Mode]              [Live Mode]
                    │                           │
                    ▼                           ▼
              Labeler assigns             LightGBM predicts
              "what happened next"        probability of UP/DOWN
                    │                           │
                    ▼                           ▼
              LightGBM.fit()              Signal + Risk Check
                    │                           │
                    ▼                           ▼
              Saved Model (.txt)          Hyperliquid Order
```

---

## 4. Research Findings

### 4.1 ML Model Selection

Extensive research comparing ML approaches for crypto trading (2025-2026) revealed:

**LightGBM selected as primary model because:**

- Fastest training speed among gradient boosting libraries — critical for periodic retraining on 5min data
- Handles 100+ features without dimensionality reduction
- Built-in feature importance via SHAP, directly integrated into LightGBM codebase
- Native incremental learning via `init_model` parameter — enables BTC-to-altcoin transfer
- Works on tabular/structured data (our features are tabular, not sequential)
- No GPU required for training or inference

**Research evidence:**

- Ensemble models (XGBoost, Gradient Boosting) consistently outperformed deep learning models (LSTM, SVR) across multiple cryptocurrencies, with R2 values ~0.98 (Springer, 2025)
- A 4-model ensemble (LightGBM + XGBoost + LSTM + Transformer) processing 40+ features demonstrated real-world viability for crypto signal generation (Corvino Trading Bot)
- LightGBM achieved slightly better prediction accuracy than XGBoost for cryptocurrency close price prediction
- FreqAI (Freqtrade) officially recommends LightGBM and XGBoost, dropped CatBoost in 2025.12 release

**Why NOT deep learning as primary:**

- Requires 10-100x more data for comparable performance
- Much harder to interpret ("why did it trade?")
- Overfits badly on noisy 5min crypto data
- LSTM/Transformer may be added as secondary ensemble vote in Phase 4

**Rejected alternatives:**

- Genetic algorithms for strategy evolution — research shows they overfit to historical data and fail in live trading
- Pure rule-based systems — markets change regimes; fixed rules break
- Reinforcement learning — requires massive training data and is unstable for financial applications

### 4.2 Framework Assessment

| Framework              | Verdict        | Reason                                                                                          |
| ---------------------- | -------------- | ----------------------------------------------------------------------------------------------- |
| FreqAI (Freqtrade)     | Reference only | Excellent feature engineering patterns but too opinionated for custom Hyperliquid integration   |
| HyperLiquidAlgoBot     | Reference only | Hyperliquid-native but live trading not fully implemented; ML limited to parameter optimization |
| Custom Python pipeline | **Selected**   | Full control over features, training, and Hyperliquid execution                                 |
| Hummingbot             | Not suitable   | Designed for market making, not ML-driven directional trading                                   |

### 4.3 Transfer Learning (BTC → SOL → LINK → TAO → HYPE)

LightGBM supports incremental learning natively:

```python
# Phase 1-2: Train on BTC
btc_model = lgb.train(params, btc_train_data)
btc_model.save_model('btc_base.model')

# Phase 4: Progressive fine-tuning
sol_model = lgb.train(params, sol_train_data, init_model='btc_base.model')
link_model = lgb.train(params, link_train_data, init_model='btc_base.model')
tao_model = lgb.train(params, tao_train_data, init_model='btc_base.model')
hype_model = lgb.train(params, hype_train_data, init_model='btc_base.model')
```

**Asset progression rationale:**

| Order | Asset   | Why This Order                                                                                                            | BTC Correlation                                                                   |
| ----- | ------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 1     | BTC     | Most data, cleanest patterns, base model                                                                                  | —                                                                                 |
| 2     | SOL     | High correlation with BTC, high liquidity, good transfer                                                                  | High                                                                              |
| 3     | LINK    | High correlation, moderate volatility                                                                                     | High                                                                              |
| 4     | **TAO** | **Final target** — trader's primary asset with Bittensor domain expertise. High volatility = bigger moves between pivots. | Variable — follows BTC most of the time, can move independently on ecosystem news |
| 5     | HYPE    | Hyperliquid native token — potential zero/reduced trading fees                                                            | Variable                                                                          |

Research confirms BTC patterns transfer to correlated assets because:

- BTC leads the crypto market — altcoins follow with lag
- Common market microstructure patterns (session behavior, volatility clustering) are shared
- Asset-specific differences (volatility magnitude, liquidity) are captured by fine-tuning

### 4.3.1 BTC Correlation Philosophy — Booster, Not Filter

**Critical design principle:** Each token's model runs primarily on the token's **own** 268 features (its own pivots, its own EMA50, its own momentum). These features work regardless of what BTC is doing. BTC correlation features (Category 22) are **optional boosters**, not required filters.

```
Token's OWN features (268):    The engine — always drives the signal
BTC cross-asset features (6):  The turbo — adds power when correlated, ignored when not
```

**Why NOT reduce confidence during decorrelation:**

Any altcoin (TAO, SOL, LINK, HYPE) can move independently of BTC due to ecosystem news. These independent moves are often the **best trading opportunities** — a TAO week-long surge driven by Bittensor news still respects its own pivots, its own EMA50, and its own momentum patterns. Skipping signals during decorrelation would mean missing these golden opportunities.

**What actually happens during decorrelation:**

- TAO (or any altcoin) moves on its own news → BTC features lose predictive power
- The model naturally ignores low-value BTC features (LightGBM learns feature importance per-context)
- The token's OWN 268 features still work — pivots, EMA50, DI+/DI- all reflect the token's real price action
- `correlation_20bar` serves as **context**, not as a gate — it tells the model "BTC features are relevant right now" or "they're not"

**Examples:**

- TAO March 2026 rally: Independent surge. Model should trade TAO using TAO's own features. BTC features = noise → model ignores them.
- TAO April 10 2026 dump: Independent drop. Model should detect the bearish momentum from TAO's own indicators and go short — a dump is a short opportunity, not a reason to sit out.
- TAO following BTC breakout: Correlated move. BTC features add confirmation → model gets extra confidence.

**HYPE fee advantage:**
Hyperliquid may offer reduced or zero maker fees for HYPE/USDC trading. This changes the risk-reward calculation — lower cost per trade means the model can take slightly lower-probability setups profitably. Fee parameters in `config.yaml` should be asset-specific.

**Critical requirement:** All features must be **percentage-based or normalized**, not raw price values. BTC moving $500, SOL moving $2, and TAO moving $15 must produce comparable feature values.

### 4.4 Lorentzian Classification Insight

The TradingView "Machine Learning: Lorentzian Classification" indicator (14k+ boosts, Top 50 all-time) provides a key architectural insight:

**Lorentzian distance** `log(1 + |a - b|)` reduces outlier influence compared to Euclidean distance `(a - b)^2`. Financial markets experience "warping" during Black Swan events and FOMC meetings — Lorentzian geometry handles this naturally.

**Practical application for our model:**

- Apply log-transform to volatile features (ATR, volume spikes, large RSI moves) before feeding to LightGBM
- The Lorentzian indicator's default features (RSI, WaveTrend, CCI, ADX) with 5 dimensions are a proven starting subset — our model expands to 295 dimensions
- Normalization method: min-max scaling to [-1, 1] range, consistent with FreqAI's approach

---

## 5. TradingView Indicator Analysis

### 5.1 Collected Indicators Inventory

60+ indicators collected in `TradingView/top_strategies/`, analyzed and categorized:

#### Currently Used by Trader (on live chart)

| Indicator                   | Category              | Feature Contribution                                 |
| --------------------------- | --------------------- | ---------------------------------------------------- |
| Pivots Fibonacci            | S/R Structure         | Pivot P, S1-S3, R1-R3, context features              |
| BB_KC (Bollinger + Keltner) | Volatility            | Squeeze state, BB width, KC width, compression ratio |
| CM Stochastic Multi-TF      | Momentum              | Stoch K/D on 5min and 1H                             |
| WaveTrend with Crosses      | Momentum              | WT1, WT2, histogram, cross signals                   |
| Squeeze Momentum (LazyBear) | Momentum + Volatility | Squeeze state, momentum value, acceleration          |
| ADX and DI                  | Trend/Direction       | ADX, DI+, DI-, rate of change                        |
| CM Ultimate MACD MTF        | Momentum              | MACD histogram on 5min, 1H                           |
| RSI (with SMA + BB)         | Momentum              | RSI value, RSI vs SMA, RSI BB position               |

#### Key Reference Indicators (collected but not on live chart)

**ML-Based Indicators:**
| Indicator | Key Insight Extracted |
|-----------|---------------------|
| Machine Learning: Lorentzian Classification (jdehorty) | Lorentzian distance metric, 5-feature KNN architecture, normalization approach, volatility/regime/ADX filters |
| ML Adaptive SuperTrend (AlgoAlpha) | K-Means clustering for volatility regimes (high/medium/low), adaptive ATR multiplier |
| Machine Learning Adaptive SuperTrend | Dynamic factor adjustment based on volatility regime classification |

**Smart Money / Structure Indicators:**
| Indicator | Features Extracted |
|-----------|-------------------|
| ICT Concepts [LuxAlgo] | Order blocks, fair value gaps, market structure breaks |
| Buyside & Sellside Liquidity [LuxAlgo] | Liquidity sweep levels, stop hunt zones |
| Market Structure Break & Order Block | Break of Structure (BOS), Change of Character (ChoCH) |
| Smart Money Concepts [LuxAlgo] | Institutional order flow patterns |
| Order Block Detector [LuxAlgo] | Supply/demand zone identification |

**Support & Resistance Indicators:**
| Indicator | Features Extracted |
|-----------|-------------------|
| CM_Pivot Points M-W-D-4H-1H Filtered | Multi-timeframe pivot calculations |
| Volume-based Support & Resistance | High-volume price zones, fractal-based S/R |
| Support Resistance - Dynamic v2 | Dynamic S/R detection |
| Bjorgum Key Levels | Key structural levels |
| Pivot Point Supertrend | Pivot-based trend following |

**Trend Indicators:**
| Indicator | Features Extracted |
|-----------|-------------------|
| Alpha Trend | Momentum-weighted trend with ATR |
| Chandelier Exit | ATR-based trailing stop and trend direction |
| Super Trend / SuperTrend Strategy | ATR-multiplier trend detection |
| Madrid Moving Average Ribbon | Multi-MA alignment scoring |
| CM Ultimate MA MTF V2 | Multi-timeframe MA analysis |
| Trendlines with Breaks [LuxAlgo] | Trendline break detection |

**Momentum / Oscillator Indicators:**
| Indicator | Features Extracted |
|-----------|-------------------|
| VuManChu Cipher B + Divergences | Multi-oscillator composite with divergence detection |
| Divergence for Many Indicators v4 | RSI, MACD, Stoch divergence detection |
| WaveTrend Oscillator [WT] | Core WaveTrend calculation reference |
| CM_Williams Vix Fix | Market bottom detection |
| CM Sling Shot System | Momentum slingshot breakout |

**Volume Indicators:**
| Indicator | Features Extracted |
|-----------|-------------------|
| Volume Flow Indicator [LazyBear] | Net buying/selling pressure (VFI) |
| Volume-based Support & Resistance | Volume-confirmed price levels |

**Volatility / Breakout Indicators:**
| Indicator | Features Extracted |
|-----------|-------------------|
| Fibonacci Bollinger Bands | Fibonacci-scaled volatility bands |
| Breakout Finder | Volatility compression breakout detection |
| Breakout Probability (Expo) | Statistical breakout probability |
| Squeeze Momentum Indicator [LazyBear] | BB inside KC compression detection |

**Session Indicator:**
| Indicator | Features Extracted |
|-----------|-------------------|
| Sessions [LuxAlgo] | Sydney/Tokyo/London/NY session ranges, VWAP per session, session overlaps |

### 5.2 Coverage Assessment

| Market Dimension  | Coverage                 | Source Indicators                                               |
| ----------------- | ------------------------ | --------------------------------------------------------------- |
| Momentum          | Excellent (over-covered) | RSI, WaveTrend, Stochastic, MACD, Squeeze Mom, VuManChu         |
| Trend/Direction   | Excellent                | ADX/DI, SuperTrend, Alpha Trend, Chandelier, EMAs               |
| S/R Structure     | Excellent                | Pivot Points, Order Blocks, Market Structure, ICT, Fibonacci BB |
| Volatility        | Good                     | ATR (via SuperTrend/UT Bot), Squeeze, BB/KC                     |
| Volume            | Adequate                 | VFI, Volume-based S/R                                           |
| Sessions/Time     | Good                     | Sessions [LuxAlgo]                                              |
| ML Reference      | Excellent                | Lorentzian Classification, ML Adaptive SuperTrend               |
| VWAP              | From Sessions.pine       | Session VWAP calculation available                              |
| Buy/Sell Pressure | Derived                  | Estimated from candle structure + volume                        |

### 5.3 User's Custom Indicators

Located in `TradingView/my_scripts/`:

| Script                                         | Relevance                                                                                                                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UT Bot v4.0.pine                               | **Feature concepts only** — Contains useful calculation patterns (efficiency ratio, ATR ratio, momentum normalization) but execution logic is FLAWED. See critique below. |
| UT Bot v3.1.pine                               | Simpler trailing logic with `confThreshold=0.15`. ADX filter available but user set `adx_threshold=0` (disabled). Fewer signals than v4.0 due to higher confThreshold.     |
| BB_KC.pine                                     | Bollinger + Keltner Channel squeeze detection                                                                                                                             |
| Squeeze Momentum Indicator (modified).pine     | Custom squeeze momentum variant                                                                                                                                           |
| Volume Flow Indicator MTF (modified by K).pine | Multi-timeframe VFI with configurable resolution                                                                                                                          |

**UT Bot v4.0 Design Analysis:**

The UT Bot v4.0 uses a dynamic ATR multiplier (`currentMult`) that adapts based on market conditions. The key design characteristic: `currentMult` is typically calculated below 1.5, which produces a tight trailing stop that generates **more signals** than v3.1's fixed approach. Combined with `confThreshold = 0` (line 16), every trail line touch triggers a signal.

This is not a bug — it's the v4.0 design intent. However, the high signal frequency makes v4.0 unsuitable as an execution logic reference for the ML bot, which targets 1-5 trades/day. v3.1's `confThreshold = 0.15` produces fewer, more selective signals (though the user has disabled the ADX filter with `adx_threshold=0`).

**What IS extracted as ML features (concepts only):**

- Efficiency Ratio (line 121-123) → feature `efficiency_ratio`
- Volatility Ratio atrShort/atrLong (line 128-129) → feature `atr_ratio`
- Momentum normalization (line 117-118) → reference for feature normalization approach

**What is NOT used:**

- v4.0's execution logic — too frequent for 1-5 trades/day target
- Dynamic `currentMult` as trailing stop — the ML model uses its own ATR trailing stop
- Intrabar entry + closebar cancel — ML bot uses bar-close execution only

---

## 6. Complete Feature Engineering Specification

### 6.1 Design Principles

1. **Raw values, not rules**: Feed the model indicator VALUES (RSI = 65.3), not rule outputs (overbought = true). The ML learns its own optimal thresholds.
2. **Percentage-based normalization**: All price-distance features as %, enabling BTC → altcoin transfer.
3. **Rate of change matters**: For key indicators (DI+, DI-, ATR, volume), include the rate of change, not just the current value. Insight from user: "DI+ rising AND DI- falling is the real filter."
4. **Context over snapshot**: For Pivot levels, include approach direction, speed, bounce/break, and test count — not just distance.
5. **Log-transform volatile features**: Apply Lorentzian-inspired `log(1 + |x|)` to ATR, volume spikes, and extreme RSI moves.
6. **Multi-timeframe features must be mapped**: 1H values are assigned to the 5min candle that falls within that 1H bar. No future data leakage.
7. **Signed-direction convention** (project-wide): For any feature encoded as {-1, 0, +1}, **`+1` = bullish / up / long-favoring**, **`-1` = bearish / down / short-favoring**, **`0` = neutral or absent**. This applies to crosses, engulfing, pin bars, divergences (regular and hidden), structure type, PSAR direction, TK cross, session direction, etc. The model learns asymmetries more cleanly when this is consistent across all features.

### 6.2 Complete Feature List (295 Features)

Each major indicator is expanded with the **depth framework**: value, direction, acceleration, position vs signal line, cross momentum, and range context. This gives the ML model full visibility into indicator dynamics, not just snapshots.

#### Category 1: Momentum (47 features)

**RSI Deep Features (12):**

| #   | Feature Name        | Calculation                                                                    | Source  | Timeframe |
| --- | ------------------- | ------------------------------------------------------------------------------ | ------- | --------- |
| 1   | `rsi_14`            | RSI(close, 14)                                                                 | RSI     | 5min      |
| 2   | `rsi_sma_14`        | SMA(RSI, 14)                                                                   | RSI+SMA | 5min      |
| 3   | `rsi_minus_sma`     | RSI - SMA(RSI) — position relative to signal                                   | RSI+SMA | 5min      |
| 4   | `rsi_direction`     | RSI - RSI[1] — rising (+) or falling (-)                                       | RSI     | 5min      |
| 5   | `rsi_sma_direction` | SMA(RSI) - SMA(RSI)[1] — signal line trend                                     | RSI+SMA | 5min      |
| 6   | `rsi_sma_roc`       | (SMA(RSI) - SMA(RSI)[3]) / 3 — signal line rate of change                      | RSI+SMA | 5min      |
| 7   | `rsi_accel`         | rsi_direction - rsi_direction[1] — acceleration (momentum of momentum)         | RSI     | 5min      |
| 8   | `rsi_cross_speed`   | abs(RSI - SMA(RSI)) when RSI just crossed SMA — cross momentum                 | RSI+SMA | 5min      |
| 9   | `rsi_bb_upper`      | SMA(RSI,14) + 2\*StdDev(RSI,14) — RSI overbought boundary                      | RSI+BB  | 5min      |
| 10  | `rsi_bb_lower`      | SMA(RSI,14) - 2\*StdDev(RSI,14) — RSI oversold boundary                        | RSI+BB  | 5min      |
| 11  | `rsi_bb_position`   | (RSI - rsi_bb_lower) / (rsi_bb_upper - rsi_bb_lower) — 0-1 within BB           | RSI+BB  | 5min      |
| 12  | `rsi_range_pct`     | (RSI - min(RSI,20)) / (max(RSI,20) - min(RSI,20)) — percentile in recent range | RSI     | 5min      |

**WaveTrend Deep Features (13):**

| #   | Feature Name        | Calculation                                                                 | Source    | Timeframe |
| --- | ------------------- | --------------------------------------------------------------------------- | --------- | --------- |
| 13  | `wt1_5min`          | EMA(CI, 21) where CI = (HLC3 - EMA(HLC3,10)) / (0.015 \* EMA(abs(...), 10)) | WaveTrend | 5min      |
| 14  | `wt2_5min`          | SMA(WT1, 4)                                                                 | WaveTrend | 5min      |
| 15  | `wt_hist_5min`      | WT1 - WT2 — histogram                                                       | WaveTrend | 5min      |
| 16  | `wt_cross_5min`     | 1 if WT1 crosses above WT2, -1 if below, 0 otherwise                        | WaveTrend | 5min      |
| 17  | `wt1_direction`     | WT1 - WT1[1] — rising or falling                                            | WaveTrend | 5min      |
| 18  | `wt1_accel`         | wt1_direction - wt1_direction[1] — acceleration                             | WaveTrend | 5min      |
| 19  | `wt_hist_direction` | wt_hist - wt_hist[1] — histogram expanding or contracting                   | WaveTrend | 5min      |
| 20  | `wt_ob_depth`       | max(0, WT1 - 60) — how far into overbought territory                        | WaveTrend | 5min      |
| 21  | `wt_os_depth`       | max(0, -60 - WT1) — how far into oversold territory                         | WaveTrend | 5min      |
| 22  | `wt_range_pct`      | (WT1 - min(WT1,20)) / (max(WT1,20) - min(WT1,20)) — percentile in range     | WaveTrend | 5min      |
| 23  | `wt1_1h`            | WaveTrend line 1 on 1H data                                                 | WaveTrend | 1H        |
| 24  | `wt2_1h`            | WaveTrend line 2 on 1H data                                                 | WaveTrend | 1H        |
| 25  | `wt_cross_1h`       | WaveTrend cross direction on 1H                                             | WaveTrend | 1H        |

**Stochastic Deep Features (6):**

| #   | Feature Name             | Calculation                                                            | Source       | Timeframe |
| --- | ------------------------ | ---------------------------------------------------------------------- | ------------ | --------- |
| 26  | `stoch_k`                | Stochastic %K(14, 3)                                                   | CM Stoch MTF | 5min      |
| 27  | `stoch_d`                | Stochastic %D = SMA(%K, 3)                                             | CM Stoch MTF | 5min      |
| 28  | `stoch_k_direction`      | %K - %K[1] — K line rising or falling                                  | CM Stoch MTF | 5min      |
| 29  | `stoch_kd_spread`        | %K - %D — distance between K and D                                     | CM Stoch MTF | 5min      |
| 30  | `stoch_kd_spread_change` | stoch_kd_spread - stoch_kd_spread[1] — spread expanding or contracting | CM Stoch MTF | 5min      |
| 31  | `stoch_range_pct`        | (%K - 20) / 60 clipped to [0,1] — position within useful range (20-80) | CM Stoch MTF | 5min      |

**Squeeze Momentum Deep Features (8):**

| #   | Feature Name             | Calculation                                                                   | Source      | Timeframe |
| --- | ------------------------ | ----------------------------------------------------------------------------- | ----------- | --------- |
| 32  | `squeeze_momentum`       | LinReg(close - avg(avg(highest, lowest), SMA), KC_len, 0)                     | Squeeze Mom | 5min      |
| 33  | `squeeze_mom_accel`      | squeeze_momentum - squeeze_momentum[1] — acceleration                         | Squeeze Mom | 5min      |
| 34  | `squeeze_mom_direction`  | sign(squeeze_momentum) — positive or negative momentum                        | Squeeze Mom | 5min      |
| 35  | `squeeze_mom_abs`        | abs(squeeze_momentum) — momentum magnitude regardless of direction            | Squeeze Mom | 5min      |
| 36  | `squeeze_mom_accel2`     | squeeze_mom_accel - squeeze_mom_accel[1] — jerk (rate of acceleration change) | Squeeze Mom | 5min      |
| 37  | `squeeze_mom_zero_cross` | 1 if squeeze_momentum crosses zero this bar, else 0                           | Squeeze Mom | 5min      |
| 38  | `squeeze_mom_percentile` | Percentile rank of abs(squeeze_momentum) in last 50 bars                      | Squeeze Mom | 5min      |
| 39  | `squeeze_mom_slope_5bar` | (squeeze_momentum - squeeze_momentum[5]) / 5 — medium-term trajectory         | Squeeze Mom | 5min      |

**MACD Deep Features (8):**

| #   | Feature Name             | Calculation                                                           | Source   | Timeframe |
| --- | ------------------------ | --------------------------------------------------------------------- | -------- | --------- |
| 40  | `macd_line_5min`         | EMA(close,12) - EMA(close,26) — MACD line                             | MACD     | 5min      |
| 41  | `macd_signal_5min`       | EMA(macd_line, 9) — signal line                                       | MACD     | 5min      |
| 42  | `macd_hist_5min`         | macd_line - macd_signal — MACD histogram                              | MACD     | 5min      |
| 43  | `macd_hist_direction`    | macd_hist - macd_hist[1] — histogram expanding or contracting         | MACD     | 5min      |
| 44  | `macd_hist_accel`        | macd_hist_direction - macd_hist_direction[1] — histogram acceleration | MACD     | 5min      |
| 45  | `macd_line_direction`    | macd_line - macd_line[1] — MACD line slope                            | MACD     | 5min      |
| 46  | `macd_hist_1h`           | MACD histogram on 1H                                                  | MACD MTF | 1H        |
| 47  | `macd_hist_1h_direction` | macd_hist_1h - macd_hist_1h[1] — 1H histogram trend                   | MACD MTF | 1H        |

#### Category 2: Trend / Direction (19 features)

**ADX/DI Deep Features (12):**

| #   | Feature Name     | Calculation                                                                    | Source | Timeframe |
| --- | ---------------- | ------------------------------------------------------------------------------ | ------ | --------- |
| 48  | `di_plus`        | Smoothed DM+ / Smoothed TR \* 100                                              | ADX/DI | 5min      |
| 49  | `di_minus`       | Smoothed DM- / Smoothed TR \* 100                                              | ADX/DI | 5min      |
| 50  | `adx`            | SMA(abs(DI+ - DI-) / (DI+ + DI-) \* 100, 14)                                   | ADX/DI | 5min      |
| 51  | `di_plus_roc`    | DI+ - DI+[5] — DI+ rate of change over 5 bars                                  | ADX/DI | 5min      |
| 52  | `di_minus_roc`   | DI- - DI-[5] — DI- rate of change over 5 bars                                  | ADX/DI | 5min      |
| 53  | `di_spread`      | DI+ - DI- — directional spread                                                 | ADX/DI | 5min      |
| 54  | `di_spread_roc`  | di_spread - di_spread[3] — spread widening or narrowing                        | ADX/DI | 5min      |
| 55  | `adx_direction`  | ADX - ADX[1] — trend strengthening (+) or weakening (-)                        | ADX/DI | 5min      |
| 56  | `adx_accel`      | adx_direction - adx_direction[1] — ADX acceleration                            | ADX/DI | 5min      |
| 57  | `adx_range_pct`  | (ADX - min(ADX,20)) / (max(ADX,20) - min(ADX,20)) — ADX percentile in range    | ADX/DI | 5min      |
| 58  | `di_convergence` | abs(DI+ - DI-) vs abs(DI+[5] - DI-[5]) — lines converging (-) or diverging (+) | ADX/DI | 5min      |
| 59  | `adx_1h`         | ADX on 1H timeframe — higher TF trend strength                                 | ADX/DI | 1H        |

**EMA & SuperTrend (7):**

| #   | Feature Name        | Calculation                                                             | Source | Timeframe |
| --- | ------------------- | ----------------------------------------------------------------------- | ------ | --------- |
| 60  | `ema9_dist_pct`     | (close - EMA9) / close \* 100                                           | EMA    | 5min      |
| 61  | `ema21_dist_pct`    | (close - EMA21) / close \* 100                                          | EMA    | 5min      |
| 62  | `ema50_dist_pct`    | (close - EMA50) / close \* 100                                          | EMA    | 5min      |
| 63  | `ema9_1h_dist_pct`  | (close - 1H_EMA9) / close \* 100                                        | EMA    | 1H        |
| 64  | `ema21_1h_dist_pct` | (close - 1H_EMA21) / close \* 100                                       | EMA    | 1H        |
| 65  | `ema50_1h_dist_pct` | (close - 1H_EMA50) / close \* 100 — **key level for trader**            | EMA    | 1H        |
| 66  | `ema_stack_1h`      | Score -3 to +3: +1 for each (EMA9 > EMA21, EMA21 > EMA50, price > EMA9) | EMA    | 1H        |

**User's key insight:** "DI+ rising AND DI- flat/falling is the real signal." The `di_plus_roc`, `di_minus_roc`, and `di_spread_roc` features capture this dynamic — not just static crossovers but the _velocity_ of directional change.

#### Category 3: Volatility (15 features)

| #   | Feature Name                 | Calculation                                                              | Source         | Timeframe |
| --- | ---------------------------- | ------------------------------------------------------------------------ | -------------- | --------- |
| 67  | `atr_14`                     | Average True Range, period 14                                            | ATR            | 5min      |
| 68  | `atr_pct`                    | ATR / close \* 100 — normalized volatility                               | ATR            | 5min      |
| 69  | `atr_ratio`                  | ATR(5) / ATR(14) — expansion (>1) or contraction (<1)                    | UT Bot concept | 5min      |
| 70  | `atr_percentile`             | Percentile rank of ATR in last 100 bars (0-100)                          | Derived        | 5min      |
| 71  | `atr_roc`                    | (ATR - ATR[5]) / ATR[5] \* 100 — ATR rate of change ("ATR accumulating") | Derived        | 5min      |
| 72  | `bb_width`                   | (upperBB - lowerBB) / basisBB \* 100 — Bollinger Band width              | BB             | 5min      |
| 73  | `bb_width_percentile`        | Percentile rank of BB width in last 100 bars                             | Derived        | 5min      |
| 74  | `kc_width`                   | (upperKC - lowerKC) / basisKC \* 100 — Keltner Channel width             | KC             | 5min      |
| 75  | `squeeze_state`              | -1 if BB inside KC (squeeze), 0 normal, +1 if just released              | BB_KC          | 5min      |
| 76  | `bars_since_squeeze_release` | Count of bars since squeeze_state changed from -1 to 0/+1                | Derived        | 5min      |
| 77  | `realized_vol_5`             | StdDev of log returns over 5 bars — short-term realized volatility       | Derived        | 5min      |
| 78  | `realized_vol_20`            | StdDev of log returns over 20 bars — medium-term                         | Derived        | 5min      |
| 79  | `vol_ratio_5_20`             | realized_vol_5 / realized_vol_20 — volatility term structure             | Derived        | 5min      |
| 80  | `high_low_range_pct`         | (high - low) / close \* 100 — current candle range as %                  | Derived        | 5min      |
| 81  | `session_range_pct`          | (session_high - session_low) / session_open \* 100 — session activity    | Sessions       | Daily     |

#### Category 4: Volume & Buy/Sell Pressure (17 features)

**VFI Deep Features (9):**

| #   | Feature Name         | Calculation                                                           | Source         | Timeframe |
| --- | -------------------- | --------------------------------------------------------------------- | -------------- | --------- |
| 82  | `vfi`                | Volume Flow Indicator — net accumulation/distribution                 | VFI (LazyBear) | 5min      |
| 83  | `vfi_signal`         | EMA(VFI, 5) — VFI signal line                                         | VFI            | 5min      |
| 84  | `vfi_hist`           | VFI - VFI_signal — VFI histogram                                      | VFI            | 5min      |
| 85  | `vfi_direction`      | VFI - VFI[1] — VFI rising or falling                                  | VFI            | 5min      |
| 86  | `vfi_hist_direction` | vfi_hist - vfi_hist[1] — histogram expanding or contracting           | VFI            | 5min      |
| 87  | `vfi_cross_zero`     | 1 if VFI crosses above 0, -1 if below, 0 otherwise                    | VFI            | 5min      |
| 88  | `vfi_range_pct`      | (VFI - min(VFI,50)) / (max(VFI,50) - min(VFI,50)) — position in range | VFI            | 5min      |
| 89  | `vfi_1h`             | VFI calculated on 1H data                                             | VFI MTF        | 1H        |
| 90  | `vfi_1h_direction`   | VFI_1h - VFI_1h[1] — 1H flow trend                                    | VFI MTF        | 1H        |

**Volume & Buy/Sell Pressure (8):**

| #   | Feature Name          | Calculation                                                        | Source  | Timeframe |
| --- | --------------------- | ------------------------------------------------------------------ | ------- | --------- |
| 91  | `volume_ratio`        | volume / SMA(volume, 20) — is this bar's volume unusual?           | Derived | 5min      |
| 92  | `volume_trend`        | SMA(volume, 5) / SMA(volume, 20) — sustained volume change         | Derived | 5min      |
| 93  | `buy_volume_pct`      | (close - low) / (high - low) — estimated buy volume fraction (0-1) | Derived | 5min      |
| 94  | `sell_volume_pct`     | 1 - buy_volume_pct — estimated sell pressure                       | Derived | 5min      |
| 95  | `buy_sell_ratio_sma5` | SMA(buy_volume_pct / sell_volume_pct, 5) — smoothed ratio          | Derived | 5min      |
| 96  | `cum_delta_5`         | Sum of (buy_vol - sell_vol) over 5 bars — net pressure             | Derived | 5min      |
| 97  | `cum_delta_20`        | Sum of (buy_vol - sell_vol) over 20 bars — sustained flow          | Derived | 5min      |
| 98  | `obv_slope`           | Slope of OBV over 10 bars — money flowing in or out                | Derived | 5min      |

**Buy/sell volume estimation method (candle-shape heuristic):**

```python
# Standard approximation without tick-level data
buy_pct = (close - low) / (high - low)   # 1.0 = buyers dominated
sell_pct = 1.0 - buy_pct                  # 1.0 = sellers dominated
buy_vol = volume * buy_pct
sell_vol = volume * sell_pct
delta = buy_vol - sell_vol                # positive = net buying
```

**Reliability note:** This is a candle-shape heuristic, NOT true buy/sell measurement. Correlation with actual trade-level buy/sell is ~0.5-0.65. It works well for strong directional candles but is unreliable on dojis and long-wick candles. These features are kept for SHAP evaluation — the model will determine their actual value. The TRUE buy/sell data comes from Category 21's trade tape features (`trade_delta_5min` #286), collected from Hyperliquid WebSocket after 2-3 months.

#### Category 5: VWAP (8 features)

| #   | Feature Name            | Calculation                                                                    | Source   | Timeframe |
| --- | ----------------------- | ------------------------------------------------------------------------------ | -------- | --------- |
| 99  | `vwap_daily`            | Cumulative(close \* volume) / Cumulative(volume) from day start                | VWAP     | Daily     |
| 100 | `vwap_dist_pct`         | (close - VWAP) / close \* 100 — above = premium, below = discount              | VWAP     | Daily     |
| 101 | `vwap_session`          | VWAP calculated from current session start (from Sessions.pine logic)          | Sessions | Session   |
| 102 | `vwap_session_dist_pct` | (close - session_VWAP) / close \* 100                                          | Sessions | Session   |
| 103 | `vwap_slope`            | (VWAP - VWAP[5]) / VWAP[5] \* 100 — VWAP trend direction                       | Derived  | Daily     |
| 104 | `vwap_upper_band`       | VWAP + StdDev(close, 20 from session start) — overbought relative to VWAP      | VWAP     | Daily     |
| 105 | `vwap_lower_band`       | VWAP - StdDev — oversold relative to VWAP                                      | VWAP     | Daily     |
| 106 | `vwap_band_position`    | (close - lower_band) / (upper_band - lower_band) — position within bands (0-1) | VWAP     | Daily     |

#### Category 6: S/R Structure — Pivot Fibonacci (13 features)

| #   | Feature Name                | Calculation                                                                | Source    | Timeframe |
| --- | --------------------------- | -------------------------------------------------------------------------- | --------- | --------- |
| 107 | `pivot_P`                   | (prev_high + prev_low + prev_close) / 3                                    | Pivot Fib | Daily     |
| 108 | `pivot_R1`                  | P + 0.382 \* (prev_high - prev_low)                                        | Pivot Fib | Daily     |
| 109 | `pivot_R2`                  | P + 0.618 \* (prev_high - prev_low)                                        | Pivot Fib | Daily     |
| 110 | `pivot_R3`                  | P + 1.000 \* (prev_high - prev_low)                                        | Pivot Fib | Daily     |
| 111 | `pivot_S1`                  | P - 0.382 \* (prev_high - prev_low)                                        | Pivot Fib | Daily     |
| 112 | `pivot_S2`                  | P - 0.618 \* (prev_high - prev_low)                                        | Pivot Fib | Daily     |
| 113 | `pivot_S3`                  | P - 1.000 \* (prev_high - prev_low)                                        | Pivot Fib | Daily     |
| 114 | `dist_to_nearest_pivot_pct` | min(abs(close - level)) / close \* 100 for all 7 levels                    | Derived   | 5min      |
| 115 | `nearest_pivot_type`        | Which level is nearest: 0=S3, 1=S2, 2=S1, 3=P, 4=R1, 5=R2, 6=R3            | Derived   | 5min      |
| 116 | `pivot_zone`                | Which zone price is in: S3-S2=0, S2-S1=1, S1-P=2, P-R1=3, R1-R2=4, R2-R3=5 | Derived   | 5min      |
| 117 | `pivot_approach_dir`        | +1 if price falling toward level (from above), -1 if rising toward it      | Derived   | 5min      |
| 118 | `pivot_approach_speed`      | abs(close - close[3]) / close \* 100 — how fast approaching level          | Derived   | 5min      |
| 119 | `pivot_times_tested_today`  | Count of how many times nearest level was touched today (within 0.1%)      | Derived   | Daily     |

**Pivot context is critical.** Price at S1 can mean four different things:

1. Falling into S1 = potential support hold (LONG setup)
2. Price broke below S1 and retesting from below = resistance flip (SHORT setup)
3. Price bouncing off S1 for 3rd time today = weakening support (breakout risk)
4. Price gapped up to S1 at session open = resistance test (wait for reaction)

The context features (approach direction, speed, test count) allow the ML model to distinguish these scenarios.

#### Category 7: Session & Time Context (15 features)

Based on Sessions [LuxAlgo] indicator with session times in UTC:

| #   | Feature Name               | Calculation                                                                    | Source        |
| --- | -------------------------- | ------------------------------------------------------------------------------ | ------------- |
| 120 | `session_sydney`           | 1 if 21:00-06:00 UTC active                                                    | Sessions.pine |
| 121 | `session_tokyo`            | 1 if 00:00-09:00 UTC active                                                    | Sessions.pine |
| 122 | `session_london`           | 1 if 07:00-16:00 UTC active                                                    | Sessions.pine |
| 123 | `session_new_york`         | 1 if 13:00-22:00 UTC active                                                    | Sessions.pine |
| 124 | `overlap_tokyo_london`     | 1 if 07:00-09:00 UTC (Tokyo + London overlap)                                  | Derived       |
| 125 | `overlap_london_ny`        | 1 if 13:00-16:00 UTC — **highest volume period, best for entries**             | Derived       |
| 126 | `overlap_ny_sydney`        | 1 if 21:00-22:00 UTC                                                           | Derived       |
| 127 | `overlap_sydney_tokyo`     | 1 if 00:00-06:00 UTC                                                           | Derived       |
| 128 | `active_session_count`     | Count of currently active sessions (1, 2, or 3)                                | Derived       |
| 129 | `minutes_into_session`     | Minutes elapsed since primary (most recently opened) session started           | Derived       |
| 130 | `minutes_to_session_close` | Minutes remaining until primary session ends                                   | Derived       |
| 131 | `session_range_vs_avg`     | Current session range / avg of last 20 same-session ranges — unusually active? | Derived       |
| 132 | `prev_session_range_pct`   | Previous session's (high-low)/open \* 100 — context for current session        | Derived       |
| 133 | `day_of_week`              | 0=Monday through 6=Sunday                                                      | Time          |
| 134 | `is_monday`                | Boolean — Monday often has unique gap/range patterns                           | Time          |

**Session behavior patterns for BTC:**
| Session | Typical Behavior | Trading Quality |
|---------|-----------------|-----------------|
| Sydney only (22:00-00:00) | Low volume, thin liquidity | Poor — avoid |
| Sydney-Tokyo overlap (00:00-06:00) | Moderate, range-bound | Moderate |
| Tokyo (06:00-07:00) | Low, pre-London positioning | Poor |
| Tokyo-London overlap (07:00-09:00) | Increasing volume, breakouts begin | Good |
| London (09:00-13:00) | High volume, trending moves | Excellent |
| London-NY overlap (13:00-16:00) | **Highest volume, biggest moves** | **Best** |
| New York (16:00-21:00) | Moderate, end-of-day positioning | Good |
| NY close (21:00-22:00) | Position squaring, unwinding | Moderate |

#### Category 8: Price Action / Candle Structure (9 features)

| #   | Feature Name        | Calculation                                                                      | Source |
| --- | ------------------- | -------------------------------------------------------------------------------- | ------ |
| 135 | `body_pct`          | abs(close - open) / (high - low) — body as fraction of range (0-1)               | Candle |
| 136 | `upper_wick_pct`    | (high - max(open, close)) / (high - low) — upper rejection                       | Candle |
| 137 | `lower_wick_pct`    | (min(open, close) - low) / (high - low) — lower rejection                        | Candle |
| 138 | `is_bullish`        | 1 if close > open, 0 otherwise                                                   | Candle |
| 139 | `consecutive_bull`  | Count of consecutive bullish candles (resets on bearish)                         | Candle |
| 140 | `consecutive_bear`  | Count of consecutive bearish candles (resets on bullish)                         | Candle |
| 141 | `body_vs_prev_body` | current body size / previous body size — expansion or exhaustion                 | Candle |
| 142 | `engulfing`         | +1 if bullish engulfing, -1 if bearish engulfing, 0 otherwise                    | Candle |
| 143 | `pin_bar`           | +1 if bullish pin bar (long lower wick > 2x body at support), -1 bearish, 0 none | Candle |

**Design note:** We intentionally avoid named candlestick patterns (morning star, three white soldiers, etc.). Research shows ML models learn these patterns better from raw candle metrics (body %, wick %, engulfing ratio) than from pre-classified pattern labels. The model discovers its own pattern combinations.

#### Category 9: Mean Reversion / Statistical Position (8 features)

| #   | Feature Name           | Calculation                                                                        | Source  |
| --- | ---------------------- | ---------------------------------------------------------------------------------- | ------- |
| 144 | `zscore_20`            | (close - SMA(20)) / StdDev(close, 20) — standard deviations from 20-period mean    | Derived |
| 145 | `zscore_50`            | (close - SMA(50)) / StdDev(close, 50) — standard deviations from 50-period mean    | Derived |
| 146 | `bb_position`          | (close - lowerBB) / (upperBB - lowerBB) — position within Bollinger Bands (0-1)    | BB      |
| 147 | `rsi_zscore`           | (RSI - mean(RSI, 50)) / StdDev(RSI, 50) — how extreme is RSI vs its recent history | Derived |
| 148 | `return_5bar`          | (close / close[5] - 1) \* 100 — recent % return                                    | Derived |
| 149 | `return_20bar`         | (close / close[20] - 1) \* 100 — medium-term % return                              | Derived |
| 150 | `return_60bar`         | (close / close[60] - 1) \* 100 — session-scale % return (5 hours)                  | Derived |
| 151 | `mean_reversion_score` | Average of zscore_20 and zscore_50, clipped to [-3, 3]                             | Derived |

#### Category 10: Market Regime (7 features)

| #   | Feature Name             | Calculation                                                                   | Source         |
| --- | ------------------------ | ----------------------------------------------------------------------------- | -------------- |
| 152 | `regime_trending`        | 1 if ADX > 25 AND abs(DI+ - DI-) > 10, else 0                                 | ADX/DI         |
| 153 | `regime_ranging`         | 1 if ADX < 20 AND bb_width_percentile < 30, else 0                            | ADX + BB       |
| 154 | `regime_volatile`        | 1 if atr_percentile > 80, else 0                                              | ATR            |
| 155 | `regime_quiet`           | 1 if atr_percentile < 20, else 0                                              | ATR            |
| 156 | `bars_in_current_regime` | Count of consecutive bars where the dominant regime hasn't changed            | Derived        |
| 157 | `efficiency_ratio`       | abs(close - close[14]) / sum(abs(close - close[1]), 14) — from UT Bot concept | UT Bot concept |
| 158 | `choppiness_index`       | 100 \* LOG10(SUM(ATR,14) / (highest(14) - lowest(14))) / LOG10(14)            | Derived        |

**Regime is critical because:** A WaveTrend cross in a trending regime means continuation; the same cross in a ranging regime means noise. The model needs to learn "this signal works in this regime but not that one."

#### Category 11: Previous Context / Market Memory (8 features)

| #   | Feature Name                | Calculation                                                   | Source   |
| --- | --------------------------- | ------------------------------------------------------------- | -------- |
| 159 | `prev_day_range_pct`        | (prev_day_high - prev_day_low) / prev_day_close \* 100        | Daily    |
| 160 | `prev_day_close_vs_pivot`   | (prev_day_close - today_pivot_P) / today_pivot_P \* 100       | Pivot    |
| 161 | `gap_pct`                   | (today_open - prev_day_close) / prev_day_close \* 100         | Daily    |
| 162 | `dist_to_prev_day_high_pct` | (close - prev_day_high) / close \* 100                        | Daily    |
| 163 | `dist_to_prev_day_low_pct`  | (close - prev_day_low) / close \* 100                         | Daily    |
| 164 | `prev_session_direction`    | +1 if prev session close > prev session open, -1 otherwise    | Sessions |
| 165 | `prev_session_volume_rank`  | Percentile of prev session's total volume vs last 20 sessions | Sessions |
| 166 | `daily_open_dist_pct`       | (close - today_open) / today_open \* 100 — intraday bias      | Daily    |

#### Category 12: Lagged Feature Dynamics (8 features)

| #   | Feature Name            | Calculation                                                      | Source    |
| --- | ----------------------- | ---------------------------------------------------------------- | --------- |
| 167 | `rsi_5bar_ago`          | RSI value 5 bars ago — trajectory context                        | RSI       |
| 168 | `wt1_slope_5bar`        | (WT1 - WT1[5]) / 5 — WaveTrend velocity                          | WaveTrend |
| 169 | `adx_slope_5bar`        | (ADX - ADX[5]) / 5 — trend strength building or fading           | ADX       |
| 170 | `atr_slope_5bar`        | (ATR - ATR[5]) / ATR[5] \* 100 — volatility trajectory           | ATR       |
| 171 | `volume_slope_5bar`     | (SMA(vol,3) - SMA(vol,3)[5]) / SMA(vol,3)[5] \* 100              | Volume    |
| 172 | `vwap_slope_5bar`       | (VWAP - VWAP[5]) / VWAP[5] \* 100 — institutional flow direction | VWAP      |
| 173 | `di_spread_change_5bar` | (DI+ - DI-) - (DI+[5] - DI-[5]) — directional acceleration       | ADX/DI    |
| 174 | `squeeze_mom_slope`     | (sqz_mom - sqz_mom[3]) / 3 — squeeze momentum acceleration       | Squeeze   |

#### Category 13: Divergence Detection (7 features)

Inspired by "Divergence for Many Indicators v4" in user's collection. Divergences between price and oscillators signal momentum exhaustion.

| #   | Feature Name             | Calculation                                                                                                                  | Source             |
| --- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| 175 | `rsi_price_divergence`   | +1 if price lower low but RSI higher low (bullish), -1 if price higher high but RSI lower high (bearish), 0 none | RSI + Price        |
| 176 | `macd_price_divergence`  | +1 bullish divergence, -1 bearish divergence (MACD histogram vs price)                                                       | MACD + Price       |
| 177 | `wt_price_divergence`    | +1 bullish, -1 bearish (WaveTrend vs price)                                                                                  | WaveTrend + Price  |
| 178 | `stoch_price_divergence` | +1 bullish, -1 bearish (Stochastic %K vs price)                                                                              | Stochastic + Price |
| 179 | `divergence_count`       | Sum of absolute divergences active — multiple divergences = stronger signal                                                  | Derived            |
| 180 | `hidden_divergence`      | +1 if price higher low + RSI lower low (hidden bullish), -1 hidden bearish                                                   | RSI + Price        |
| 181 | `divergence_freshness`   | Bars since most recent divergence signal (lower = more actionable)                                                           | Derived            |

**Divergence detection method (lookback 14 bars):**

```python
# Bullish regular divergence: price lower low, RSI higher low
price_ll = (low < min(low, lookback))
rsi_hl = (rsi > min(rsi, lookback))
bullish_div = price_ll & rsi_hl
```

#### Category 14: Money Flow (8 features)

CMF, MFI, and Accumulation/Distribution — volume-weighted money flow indicators not covered by VFI.

| #   | Feature Name         | Calculation                                                                                 | Source  |
| --- | -------------------- | ------------------------------------------------------------------------------------------- | ------- |
| 182 | `cmf_20`             | Chaikin Money Flow(20) = SUM(((close-low)-(high-close))/(high-low)\*vol, 20) / SUM(vol, 20) | CMF     |
| 183 | `cmf_direction`      | CMF - CMF[1] — money flow accelerating or decelerating                                      | CMF     |
| 184 | `mfi_14`             | Money Flow Index(14) — RSI but volume-weighted                                              | MFI     |
| 185 | `mfi_direction`      | MFI - MFI[1] — MFI trend                                                                    | MFI     |
| 186 | `ad_line`            | Cumulative ((close-low)-(high-close))/(high-low) \* volume                                  | A/D     |
| 187 | `ad_slope_10`        | Slope of A/D line over 10 bars — sustained accumulation or distribution                     | A/D     |
| 188 | `mfi_rsi_divergence` | MFI - RSI — when volume-weighted momentum diverges from price momentum                      | Derived |
| 189 | `cmf_vfi_agreement`  | sign(CMF) == sign(VFI) → 1 (agreement) or -1 (disagreement)                                 | Derived |

#### Category 15: Additional Momentum Oscillators (9 features)

Williams %R, CCI, CMO — oscillators that capture different aspects of momentum not covered by RSI/WT/Stochastic.

| #   | Feature Name           | Calculation                                                                                       | Source      |
| --- | ---------------------- | ------------------------------------------------------------------------------------------------- | ----------- |
| 190 | `williams_r`           | Williams %R(14) — similar to Stochastic but inverted, range [-100, 0]                             | Williams %R |
| 191 | `williams_r_direction` | Williams_R - Williams_R[1] — velocity                                                             | Williams %R |
| 192 | `cci_20`               | CCI(20) = (TP - SMA(TP,20)) / (0.015 \* MeanDev(TP,20))                                           | CCI         |
| 193 | `cci_direction`        | CCI - CCI[1] — CCI trend                                                                          | CCI         |
| 194 | `cci_extreme`          | max(0, abs(CCI) - 100) — how deep into extreme territory                                          | CCI         |
| 195 | `cmo_14`               | Chande Momentum Oscillator(14) = (sumUp - sumDown) / (sumUp + sumDown) \* 100                     | CMO         |
| 196 | `cmo_direction`        | CMO - CMO[1] — CMO velocity                                                                       | CMO         |
| 197 | `roc_10`               | Rate of Change(10) = (close / close[10] - 1) \* 100                                               | ROC         |
| 198 | `tsi`                  | True Strength Index = 100 \* EMA(EMA(close-close[1],25),13) / EMA(EMA(abs(close-close[1]),25),13) | TSI         |

#### Category 16: Market Structure (10 features)

HH/HL/LH/LL pattern detection — the foundation of all price action analysis.

| #   | Feature Name                 | Calculation                                                                   | Source  |
| --- | ---------------------------- | ----------------------------------------------------------------------------- | ------- |
| 199 | `swing_high_dist_pct`        | (close - last_swing_high) / close \* 100 — distance to most recent swing high | Derived |
| 200 | `swing_low_dist_pct`         | (close - last_swing_low) / close \* 100 — distance to most recent swing low   | Derived |
| 201 | `structure_type`             | +1 if HH+HL (uptrend), -1 if LH+LL (downtrend), 0 if mixed                    | Derived |
| 202 | `bars_since_structure_break` | Bars since last BOS (Break of Structure) event                                | Derived |
| 203 | `swing_range_pct`            | (last_swing_high - last_swing_low) / close \* 100 — current swing size        | Derived |
| 204 | `higher_highs_count_20`      | Count of higher pivot highs in last 20 pivot highs — trend persistence        | Derived |
| 205 | `lower_lows_count_20`        | Count of lower pivot lows in last 20 pivot lows                               | Derived |
| 206 | `swing_ratio`                | Current swing length / previous swing length — expansion or contraction       | Derived |
| 207 | `retrace_depth`              | (swing_high - close) / (swing_high - swing_low) — retracement depth (0-1)     | Derived |
| 208 | `range_position`             | (close - low_20) / (high_20 - low_20) — position in 20-bar range (0-1)        | Derived |

**Swing detection method:** Fractals with lookback 5 bars (2 bars left + 2 bars right confirmation).

#### Category 17: Statistical / Fractal (9 features)

Advanced statistical properties that capture market microstructure patterns invisible to standard indicators.

| #   | Feature Name        | Calculation                                                                              | Source  |
| --- | ------------------- | ---------------------------------------------------------------------------------------- | ------- |
| 209 | `hurst_exponent`    | Hurst exponent estimated over 100 bars — >0.5 trending, <0.5 mean-reverting, =0.5 random | Derived |
| 210 | `price_entropy`     | Shannon entropy of discretized return distribution over 50 bars — high = uncertain       | Derived |
| 211 | `autocorrelation_1` | Autocorrelation of returns at lag 1 — positive = momentum, negative = mean reversion     | Derived |
| 212 | `autocorrelation_5` | Autocorrelation at lag 5 — medium-term persistence                                       | Derived |
| 213 | `skewness_20`       | Skewness of returns over 20 bars — positive = right tail (bullish), negative = left tail | Derived |
| 214 | `kurtosis_20`       | Excess kurtosis of returns over 20 bars — high = fat tails (event risk)                  | Derived |
| 215 | `fractal_dimension` | Fractal dimension via box-counting over 50 bars — complexity of price path               | Derived |
| 216 | `variance_ratio`    | Var(returns, 10) / (10 \* Var(returns, 1)) — tests random walk hypothesis                | Derived |
| 217 | `parkinson_vol`     | Parkinson volatility = sqrt(1/(4n*ln2) * SUM(ln(H/L)^2)) — range-based vol estimator     | Derived |

#### Category 18: Adaptive Moving Averages (5 features)

MAs that adapt to market conditions — KAMA, DEMA, TEMA provide less lag than standard EMAs.

| #   | Feature Name     | Calculation                                                               | Source |
| --- | ---------------- | ------------------------------------------------------------------------- | ------ |
| 218 | `kama_dist_pct`  | (close - KAMA(10,2,30)) / close \* 100 — Kaufman Adaptive MA distance     | KAMA   |
| 219 | `dema_dist_pct`  | (close - DEMA(21)) / close \* 100 — Double EMA (less lag than EMA)        | DEMA   |
| 220 | `tema_dist_pct`  | (close - TEMA(21)) / close \* 100 — Triple EMA (least lag)                | TEMA   |
| 221 | `psar_direction` | Parabolic SAR direction: +1 below price (bullish), -1 above (bearish)     | SAR    |
| 222 | `psar_dist_pct`  | (close - PSAR) / close \* 100 — distance to SAR (larger = stronger trend) | SAR    |

#### Category 19: Ichimoku (5 features)

Partial Ichimoku — using components relevant for 5min entry with swing-style exits.

| #   | Feature Name      | Calculation                                                                   | Source   |
| --- | ----------------- | ----------------------------------------------------------------------------- | -------- |
| 223 | `tenkan_dist_pct` | (close - Tenkan-sen(9)) / close \* 100 — distance to conversion line          | Ichimoku |
| 224 | `kijun_dist_pct`  | (close - Kijun-sen(26)) / close \* 100 — distance to base line                | Ichimoku |
| 225 | `tk_cross`        | +1 if Tenkan > Kijun (bullish), -1 if below                                   | Ichimoku |
| 226 | `tk_spread`       | (Tenkan - Kijun) / close \* 100 — TK spread as %                              | Ichimoku |
| 227 | `cloud_dist_pct`  | (close - Senkou_A) / close \* 100 — distance above/below cloud leading span A | Ichimoku |

#### Category 20: Event Memory (41 features)

Tracks **recency** of key indicator events. Inspired by user's insight about CM_Stoch_MTF red/green bars and WaveTrend OB/OS history. The model needs to know not just "where is RSI now?" but "how recently did RSI emerge from oversold?"

**Stochastic Event Memory (8):**

| #   | Feature Name               | Calculation                                                                     | Source       |
| --- | -------------------------- | ------------------------------------------------------------------------------- | ------------ |
| 228 | `stoch_bars_since_green`   | Bars since last Green Bar (K crossed above D while K < 20)                      | CM Stoch MTF |
| 229 | `stoch_bars_since_red`     | Bars since last Red Bar (K crossed below D while K > 80)                        | CM Stoch MTF |
| 230 | `stoch_green_count_50`     | Count of Green Bars in last 50 bars — oversold frequency                        | CM Stoch MTF |
| 231 | `stoch_red_count_50`       | Count of Red Bars in last 50 bars — overbought frequency                        | CM Stoch MTF |
| 232 | `stoch_last_extreme_depth` | How deep was the last OB/OS penetration: (K_at_event - 80) or (20 - K_at_event) | CM Stoch MTF |
| 233 | `stoch_bars_in_ob`         | Bars spent above 80 during most recent OB episode                               | CM Stoch MTF |
| 234 | `stoch_bars_in_os`         | Bars spent below 20 during most recent OS episode                               | CM Stoch MTF |
| 235 | `stoch_recovery_speed`     | Bars from OB/OS extreme back to 50 — fast recovery = strong counter-momentum    | CM Stoch MTF |

**WaveTrend Event Memory (8):**

| #   | Feature Name               | Calculation                                                        | Source    |
| --- | -------------------------- | ------------------------------------------------------------------ | --------- |
| 236 | `wt_bars_since_ob_exit`    | Bars since WT1 dropped below 60 (exited overbought)                | WaveTrend |
| 237 | `wt_bars_since_os_exit`    | Bars since WT1 rose above -60 (exited oversold)                    | WaveTrend |
| 238 | `wt_bars_since_bull_cross` | Bars since WT1 crossed above WT2                                   | WaveTrend |
| 239 | `wt_bars_since_bear_cross` | Bars since WT1 crossed below WT2                                   | WaveTrend |
| 240 | `wt_ob_count_100`          | Count of OB events (WT1 > 60) in last 100 bars                     | WaveTrend |
| 241 | `wt_os_count_100`          | Count of OS events (WT1 < -60) in last 100 bars                    | WaveTrend |
| 242 | `wt_last_ob_depth`         | max(0, max_WT1_during_last_OB - 60) — how deep was last overbought | WaveTrend |
| 243 | `wt_last_os_depth`         | max(0, -60 - min_WT1_during_last_OS) — how deep was last oversold  | WaveTrend |

**RSI Event Memory (6):**

| #   | Feature Name              | Calculation                                                       | Source |
| --- | ------------------------- | ----------------------------------------------------------------- | ------ |
| 244 | `rsi_bars_since_ob`       | Bars since RSI was last above 70                                  | RSI    |
| 245 | `rsi_bars_since_os`       | Bars since RSI was last below 30                                  | RSI    |
| 246 | `rsi_bars_since_50_cross` | Bars since RSI last crossed 50 (midline)                          | RSI    |
| 247 | `rsi_ob_count_100`        | Count of OB events (RSI > 70) in last 100 bars                    | RSI    |
| 248 | `rsi_os_count_100`        | Count of OS events (RSI < 30) in last 100 bars                    | RSI    |
| 249 | `rsi_time_above_50_pct`   | % of last 50 bars where RSI > 50 — sustained bullish/bearish bias | RSI    |

**MACD Event Memory (5):**

| #   | Feature Name                 | Calculation                                                                 | Source |
| --- | ---------------------------- | --------------------------------------------------------------------------- | ------ |
| 250 | `macd_bars_since_bull_cross` | Bars since MACD histogram turned positive (signal cross up)                 | MACD   |
| 251 | `macd_bars_since_bear_cross` | Bars since MACD histogram turned negative (signal cross down)               | MACD   |
| 252 | `macd_bars_since_zero_cross` | Bars since MACD line crossed zero                                           | MACD   |
| 253 | `macd_hist_max_50`           | Max positive histogram value in last 50 bars — recent bullish momentum peak | MACD   |
| 254 | `macd_hist_min_50`           | Min negative histogram value in last 50 bars — recent bearish momentum peak | MACD   |

**Squeeze Event Memory (5):**

| #   | Feature Name                  | Calculation                                                           | Source  |
| --- | ----------------------------- | --------------------------------------------------------------------- | ------- |
| 255 | `squeeze_bars_since_fire`     | Bars since last squeeze release (BB expanded outside KC)              | Squeeze |
| 256 | `squeeze_duration_last`       | How many bars did the last squeeze compression last                   | Squeeze |
| 257 | `squeeze_count_200`           | Count of squeeze events in last 200 bars — market personality         | Squeeze |
| 258 | `squeeze_momentum_at_fire`    | Squeeze momentum value at the moment of last release — fire direction | Squeeze |
| 259 | `squeeze_mom_peak_since_fire` | Peak absolute momentum since last squeeze release                     | Squeeze |

**ADX/DI Event Memory (4):**

| #   | Feature Name           | Calculation                                                | Source |
| --- | ---------------------- | ---------------------------------------------------------- | ------ |
| 260 | `adx_bars_since_trend` | Bars since ADX was last above 25 (trending)                | ADX/DI |
| 261 | `adx_bars_since_weak`  | Bars since ADX was last below 15 (very weak/ranging)       | ADX/DI |
| 262 | `di_bars_since_cross`  | Bars since DI+ and DI- last crossed                        | ADX/DI |
| 263 | `adx_peak_50`          | Max ADX value in last 50 bars — recent trend strength peak | ADX/DI |

**Price Event Memory (5):**

| #   | Feature Name              | Calculation                                                        | Source         |
| --- | ------------------------- | ------------------------------------------------------------------ | -------------- |
| 264 | `bars_since_new_high_20`  | Bars since price made a new 20-bar high                            | Price          |
| 265 | `bars_since_new_low_20`   | Bars since price made a new 20-bar low                             | Price          |
| 266 | `bars_since_pivot_touch`  | Bars since price last came within 0.1% of any Pivot level          | Price + Pivots |
| 267 | `pivot_touch_count_today` | Number of Pivot level touches today — active S/R testing           | Price + Pivots |
| 268 | `bars_since_volume_spike` | Bars since volume was > 3x its 20-bar SMA — recent liquidity event | Volume         |

#### Category 21: Market Microstructure — Hyperliquid-Specific (21 features, Phase 3)

**Funding & OI (10):**

| #   | Feature Name                   | Calculation                                                            | Source          |
| --- | ------------------------------ | ---------------------------------------------------------------------- | --------------- |
| 269 | `funding_rate`                 | Current funding rate — positive = longs pay shorts                     | Hyperliquid API |
| 270 | `funding_rate_trend`           | Funding rate vs 8-hour avg — crowding detection                        | Hyperliquid API |
| 271 | `open_interest_change_pct`     | OI change over last hour — new money entering or exiting               | Hyperliquid API |
| 272 | `long_short_ratio`             | Long accounts / short accounts — sentiment imbalance                   | Hyperliquid API |
| 273 | `oi_funding_divergence`        | OI rising + funding negative = shorts building (or vice versa)         | Hyperliquid API |
| 274 | `funding_rate_extreme`         | Percentile rank of funding rate vs last 100 periods — extreme crowding | Hyperliquid API |
| 275 | `liquidation_level_dist_long`  | Estimated distance to largest long liquidation cluster as %            | Hyperliquid API |
| 276 | `liquidation_level_dist_short` | Estimated distance to largest short liquidation cluster as %           | Hyperliquid API |
| 277 | `oi_rate_of_change`            | (OI - OI[12]) / OI[12] \* 100 — 1-hour OI change rate                  | Hyperliquid API |
| 278 | `volume_oi_ratio`              | 24h volume / open interest — turnover rate (high = active speculating) | Hyperliquid API |

**Real Order Book & Trade Tape (8):**

Hyperliquid provides **real L2 order book** (up to 100 price levels) and **executed trade tape** via WebSocket. This gives actual buy/sell data — far superior to the candle-based estimation (`(close-low)/(high-low)`) used in Category 4.

| #   | Feature Name                   | Calculation                                                            | Source          |
| --- | ------------------------------ | ---------------------------------------------------------------------- | --------------- |
| 279 | `ob_imbalance`                 | (total_bid_volume - total_ask_volume) / (bid + ask) — order book imbalance, range [-1, 1] | L2 Book WS |
| 280 | `ob_imbalance_top5`            | Imbalance of top 5 price levels only — near-market pressure            | L2 Book WS |
| 281 | `ob_bid_depth_pct`             | Total bid volume within 0.5% of mid price / daily avg volume — buy wall strength | L2 Book WS |
| 282 | `ob_ask_depth_pct`             | Total ask volume within 0.5% of mid price / daily avg volume — sell wall strength | L2 Book WS |
| 283 | `ob_spread_pct`                | (best_ask - best_bid) / mid_price * 100 — spread as % (liquidity indicator) | L2 Book WS |
| 284 | `trade_buy_volume_5min`        | Actual buy volume in last 5min bar (from trade tape, side = buy)       | Trades WS |
| 285 | `trade_sell_volume_5min`       | Actual sell volume in last 5min bar (from trade tape, side = sell)      | Trades WS |
| 286 | `trade_delta_5min`             | trade_buy_volume - trade_sell_volume — real net buying/selling pressure | Trades WS |

**Hyperliquid WebSocket subscriptions used:**
```python
# L2 Order Book — snapshots updated in real-time
{"type": "l2Book", "coin": "TAO"}  # up to 100 price levels

# Trade Tape — every executed trade with side (buy/sell) and size
{"type": "trades", "coin": "TAO"}
```

**Why this matters:** Category 4's buy/sell estimation (`(close-low)/(high-low)`) is a rough approximation (~0.5-0.65 correlation with reality — it's a candle-shape heuristic, not true buy/sell measurement). With real trade tape data, `trade_delta_5min` gives the TRUE net buying/selling pressure — no estimation needed. Order book imbalance (`ob_imbalance`) shows pending demand/supply before it becomes a trade. Category 4's estimation features are kept for SHAP evaluation — if they add no value beyond the real data, they'll be pruned.

**Hyperliquid Funding Rate Deep Features (3):**

Hyperliquid has **hourly funding** (not 8-hour like Binance), providing more granular funding data. These features capture dynamics specific to Hyperliquid's funding mechanism.

| #   | Feature Name                   | Calculation                                                            | Source          |
| --- | ------------------------------ | ---------------------------------------------------------------------- | --------------- |
| 287 | `funding_rate_1h_change`       | Current funding rate - previous hour's funding rate — hourly momentum  | Hyperliquid API |
| 288 | `funding_rate_8h_sum`          | Sum of last 8 hourly funding rates — cumulative holding cost over 8h   | Hyperliquid API |
| 289 | `funding_rate_flip_bars`       | Bars since funding rate last crossed zero — funding regime duration     | Hyperliquid API |

**Why hourly funding matters:** Binance settles funding every 8 hours — by the time you see a shift, it's old news. Hyperliquid's hourly updates give the model **8x faster signal** on crowding changes. `funding_rate_1h_change` detects sentiment shifts within hours, not half-days.

**Data source note:** ALL Category 21 features (269-289) come from **Hyperliquid** — not Binance. Funding rates, OI, and order book differ between exchanges. Since the bot executes on Hyperliquid, it must train on Hyperliquid's market data. These features require 2-3 months of live collection before use (see Section 8.3).

#### Category 22: Cross-Asset Correlation (6 features, Phase 4 — altcoins only)

| #   | Feature Name            | Calculation                                                              | Source       |
| --- | ----------------------- | ------------------------------------------------------------------------ | ------------ |
| 290 | `btc_return_5bar`       | BTC % return over last 5 bars — leader signal                            | BTC data     |
| 291 | `btc_return_1h`         | BTC % return over 1H — BTC trend direction                               | BTC data     |
| 292 | `btc_dominance`         | BTC market cap % — rising = altcoin weakness                             | External API |
| 293 | `correlation_20bar`     | Rolling 20-bar correlation with BTC — decorrelation = potential breakout | Derived      |
| 294 | `beta_to_btc`           | Regression slope vs BTC returns over 100 bars — leverage factor          | Derived      |
| 295 | `alt_relative_strength` | Alt return / BTC return over 20 bars — outperformance detection          | Derived      |

**BTC correlation as context, not filter:** `correlation_20bar` (#293) tells the model whether BTC features are currently useful. When correlation is high → BTC features add prediction power. When correlation drops → the model naturally relies on the token's own 268 features. This applies to ALL altcoins (TAO, SOL, LINK, HYPE), not just TAO. Independent moves are trading opportunities, not risks to avoid.

> **Note on numbering:** Features 269-289 are Phase 3 Microstructure (21: funding/OI + order book/trade tape + funding rate deep). Features 290-295 are Phase 4 Cross-Asset (6). Phase 1 deploys features 1-268 (268 features). Grand total: 268 + 21 + 6 = 295.

### 6.3 Feature Balance Summary

| #   | Category                                          | Count   | %        | Phase |
| --- | ------------------------------------------------- | ------- | -------- | ----- |
| 1   | Momentum (RSI + WT + Stoch + Squeeze + MACD deep) | 47      | 16.5%    | 1     |
| 2   | Trend / Direction (ADX/DI deep + EMA)             | 19      | 6.7%     | 1     |
| 3   | Volatility                                        | 15      | 5.3%     | 1     |
| 4   | Volume / Buy-Sell (VFI deep)                      | 17      | 6.0%     | 1     |
| 5   | VWAP                                              | 8       | 2.8%     | 1     |
| 6   | S/R Structure (Pivots)                            | 13      | 4.6%     | 1     |
| 7   | Session / Time                                    | 15      | 5.3%     | 1     |
| 8   | Price Action / Candle                             | 9       | 3.2%     | 1     |
| 9   | Mean Reversion / Stats                            | 8       | 2.8%     | 1     |
| 10  | Market Regime                                     | 7       | 2.5%     | 1     |
| 11  | Previous Context / Memory                         | 8       | 2.8%     | 1     |
| 12  | Lagged Dynamics                                   | 8       | 2.8%     | 1     |
| 13  | Divergence Detection                              | 7       | 2.5%     | 1     |
| 14  | Money Flow (CMF/MFI/AD)                           | 8       | 2.8%     | 1     |
| 15  | Additional Momentum (Williams %R/CCI/CMO/TSI)     | 9       | 3.2%     | 1     |
| 16  | Market Structure (HH/HL/LH/LL)                    | 10      | 3.5%     | 1     |
| 17  | Statistical / Fractal                             | 9       | 3.2%     | 1     |
| 18  | Adaptive MAs (KAMA/DEMA/TEMA/SAR)                 | 5       | 1.8%     | 1     |
| 19  | Ichimoku (partial)                                | 5       | 1.8%     | 1     |
| 20  | Event Memory                                      | 41      | 14.4%    | 1     |
| 21  | Microstructure (Hyperliquid + Order Book)          | 21      | 7.1%     | 3     |
| 22  | Cross-Asset Correlation                           | 6       | 2.0%     | 4     |
|     | **Total**                                         | **295** | **100%** |       |

Phase 1 features: 268 (all except Microstructure and Cross-Asset)
Phase 3 additions: 21 (Microstructure: funding/OI + order book/trade tape + funding rate deep)
Phase 4 additions: 6 (Cross-Asset correlation)

**Feature diversity improvement:** The original 143-feature set was 40% momentum-biased. The expanded 295-feature set distributes across 22 categories with no single dimension exceeding 17%. The Event Memory category (41 features, 14.4%) is the second largest — this captures temporal dynamics that snapshot features miss (e.g., "RSI is at 55 now, but was oversold 8 bars ago" is very different from "RSI has been at 55 for 50 bars").

LightGBM handles 500+ features efficiently. SHAP analysis after first training will likely trim this to 120-150 most informative features.

### 6.4 Preprocessing Pipeline

```
Raw OHLCV data
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

---

## 7. ML Model Design

### 7.1 Primary Model: LightGBM Classifier

**Task:** 3-class classification — LONG / SHORT / NEUTRAL

**Architecture:**

```python
import lightgbm as lgb

params = {
    'objective': 'multiclass',
    'num_class': 3,            # LONG=0, SHORT=1, NEUTRAL=2
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,          # tuned via Optuna
    'learning_rate': 0.05,
    'feature_fraction': 0.8,   # use 80% of features per tree (reduces overfitting)
    'bagging_fraction': 0.8,   # use 80% of data per tree
    'bagging_freq': 5,
    'min_child_samples': 50,   # minimum samples per leaf (prevents noise fitting)
    'lambda_l1': 0.1,          # L1 regularization
    'lambda_l2': 0.1,          # L2 regularization
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
```

**Output:**

```python
probabilities = model.predict(features)
# Returns: [P(LONG), P(SHORT), P(NEUTRAL)]
# Example: [0.72, 0.08, 0.20]

# Trading decision:
if P(LONG) > 0.65:
    signal = "LONG"
elif P(SHORT) > 0.65:
    signal = "SHORT"
else:
    signal = "NO TRADE"
```

### 7.2 Secondary Model (Phase 4, Optional): Ensemble

If Phase 2 results are promising, add XGBoost as a second vote:

```python
# Weighted ensemble
lgbm_prob = lgbm_model.predict(features)     # weight: 1.0
xgb_prob  = xgb_model.predict(features)      # weight: 0.8

final_prob = (lgbm_prob * 1.0 + xgb_prob * 0.8) / 1.8
```

### 7.3 Feature Importance Analysis (SHAP)

After training, SHAP analysis reveals what the model actually learned:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance plot
shap.summary_plot(shap_values, X_test)

# Per-prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

SHAP answers:

- Which features drive predictions? (global importance)
- For THIS specific trade, why did the model say LONG? (local explanation)
- Are Pivot features actually important, or is it ATR that matters most?
- Does the model use session features? Does London-NY overlap improve predictions?

---

## 8. Data Pipeline

### 8.1 Data Collection

### Data Source Architecture

**Key principle:** Train on the cleanest, deepest data available. Execute on your actual exchange.

| Data Type | Source | Why | History Available |
|---|---|---|---|
| **OHLCV (5min + 1H)** | **Binance Futures** | Deepest liquidity → cleanest candles, least wick noise. 18+ months available. | Yes — full backfill |
| **Indicator features (1-268)** | Computed from Binance OHLCV | Prices between Binance and Hyperliquid perps differ by < 0.01% on 5min candles (arbitrage bots). All indicator calculations are exchange-agnostic. | Yes — derived from OHLCV |
| **Order book + trade tape (279-289)** | **Hyperliquid WebSocket** | You execute on Hyperliquid — its order book is what matters for your fills. | **No** — live streaming only |
| **Funding rate + OI (269-278)** | **Hyperliquid API** | Hyperliquid has unique **hourly** funding (not 8-hour like Binance). Funding rates differ between exchanges. | Partial — some history via API, but collect live for consistency |
| **Cross-asset (290-295)** | Binance OHLCV (BTC data) | Same OHLCV source, just BTC data alongside altcoin data | Yes — full backfill |

**Why Binance for OHLCV, not Hyperliquid:**
- Binance Futures daily volume is ~10x Hyperliquid → cleaner price candles with less noise
- Complete 18-month history available via standard REST API (CCXT)
- Hyperliquid launched late 2023; deep liquidity on all pairs is more recent
- Perpetual futures prices track within basis points across exchanges (arbitrage ensures this)
- For live inference, the bot uses Hyperliquid real-time data (same prices, same features)

```python
import ccxt

# TRAINING DATA: Binance Futures — deepest liquidity, cleanest candles
exchange = ccxt.binance({'options': {'defaultType': 'future'}})

# 5-minute candles: 18 months ≈ 157,680 candles
ohlcv_5m = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=157680)

# 1-hour candles: 18 months ≈ 13,140 candles
ohlcv_1h = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=13140)

# Convert to DataFrames
df_5m = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
```

**Data requirements:**

- **Minimum: 18 months** (~157,680 5min candles) — must capture full market cycle
- Recommended: As much as available (2+ years if exchange has history)
- **Critical:** The last 18 months include BTC 68K→126K rally AND sudden drop to 60K — the model needs BOTH bull runs and crashes to learn regime-dependent behavior
- Must include different market regimes (trending, ranging, volatile, crash/recovery)
- Store as parquet files for efficient loading

### 8.2 Multi-Timeframe Merge

```python
def merge_higher_tf(df_5m, df_1h):
    """Map each 5min candle to its containing 1H candle's indicator values."""
    # Floor 5min timestamps to nearest hour
    df_5m['hour_key'] = df_5m['timestamp'].dt.floor('1H')

    # Merge 1H features onto 5min data
    # IMPORTANT: Use the PREVIOUS completed 1H bar to avoid look-ahead bias
    df_1h_shifted = df_1h.shift(1)  # use previous bar's values
    df_merged = df_5m.merge(df_1h_shifted, left_on='hour_key', right_on='timestamp',
                            suffixes=('', '_1h'))
    return df_merged
```

### 8.3 Live Data Collection (Hyperliquid WebSocket)

Category 21 features (funding, OI, order book, trade tape) require **live data from Hyperliquid**. Order book and trade tape have NO historical API — they must be collected in real-time from day one.

**Start the Hyperliquid collector in Phase 1** (alongside Binance OHLCV fetching). After 2-3 months of collection, this data becomes usable for training in Phase 3.

```python
import asyncio
import websockets
import json

HYPERLIQUID_WS = "wss://api.hyperliquid.xyz/ws"

async def collect_live_data(coin="BTC"):
    async with websockets.connect(HYPERLIQUID_WS) as ws:
        # Subscribe to L2 order book snapshots
        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}))

        # Subscribe to executed trades (buy/sell with side and size)
        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}))

        while True:
            msg = json.loads(await ws.recv())
            # Store snapshots to time-series database (e.g., parquet partitioned by date)
            store_snapshot(msg)
```

**Funding rate collection** (hourly — Hyperliquid-specific):
```python
from hyperliquid.info import Info

info = Info(base_url="https://api.hyperliquid.xyz")

# Poll every hour (Hyperliquid updates funding hourly, not 8-hourly like Binance)
funding_data = info.funding_history(coin="BTC", startTime=start_ts)
```

**Data collection timeline:**

| Data Type | Start Collecting | Usable for Training | Features |
|---|---|---|---|
| Binance OHLCV | Phase 1 Day 1 (backfill) | Immediately | 1-268 |
| Hyperliquid order book | Phase 1 Day 1 (live) | **After 2-3 months** | 279-289 |
| Hyperliquid trade tape | Phase 1 Day 1 (live) | **After 2-3 months** | 279-289 |
| Hyperliquid funding rate | Phase 1 Day 1 (live) | **After 2-3 months** | 269-278 |
| BTC cross-asset data | Phase 4 (backfill from Binance) | Immediately | 290-295 |

**Storage:** Append-only parquet files partitioned by date. Order book snapshots sampled every 5 seconds (aligned to 5min candle aggregation). Trade tape stored raw, aggregated to 5min bars during feature calculation.

---

## 9. Labeling Strategy

### 9.1 Why NOT Fixed Lookahead

A naive approach labels each candle by looking N bars ahead. This fails for our trading style:

```
Fixed lookahead = 12 bars (1 hour):
  - A trade that runs S1→P→R1 over 3 hours → labeled NEUTRAL (move hadn't finished)
  - Model learns to exit too early, misses the multi-pivot rides that generate most profit

Fixed lookahead = 48 bars (4 hours):
  - A quick 15-min reversal trade → label uses noise from 3.75 hours of unrelated price action
  - Model learns from contaminated labels
```

Our hold time varies (15 min to 1 day), so the label must adapt to each trade.

### 9.2 Triple-Barrier Method (Lopez de Prado)

For each 5-minute candle, simulate a trade forward with three exit conditions — whichever hits first determines the label:

```python
def create_labels_triple_barrier(df, atr_col='atr_14',
                                  sl_atr_mult=2.0,
                                  tp_atr_mult=3.0,
                                  max_holding_bars=48,    # 4 hours on 5min
                                  min_profit_pct=0.3):    # minimum move to count as directional
    """
    Triple-barrier labeling — matches how the trader actually exits.

    Three barriers:
      1. Upper barrier (take profit): price rises tp_atr_mult * ATR → LONG
      2. Lower barrier (stop loss):   price falls sl_atr_mult * ATR → SHORT
      3. Time barrier:                max_holding_bars reached → check net P&L

    Returns:
        labels: 0=LONG, 1=SHORT, 2=NEUTRAL
        meta:   holding_bars, exit_price, pnl_pct (for analysis)
    """
    labels = np.full(len(df), 2)  # default NEUTRAL
    holding_bars = np.zeros(len(df))

    for i in range(len(df) - max_holding_bars):
        entry = df['close'].iloc[i]
        atr = df[atr_col].iloc[i]
        upper = entry + tp_atr_mult * atr
        lower = entry - sl_atr_mult * atr

        for j in range(1, max_holding_bars + 1):
            bar = df.iloc[i + j]

            # Check upper barrier (LONG win)
            if bar['high'] >= upper:
                labels[i] = 0  # LONG
                holding_bars[i] = j
                break

            # Check lower barrier (SHORT win — price fell to SL)
            if bar['low'] <= lower:
                labels[i] = 1  # SHORT
                holding_bars[i] = j
                break
        else:
            # Time barrier hit — check net movement
            final_price = df['close'].iloc[i + max_holding_bars]
            pnl_pct = (final_price - entry) / entry * 100

            if pnl_pct > min_profit_pct:
                labels[i] = 0   # LONG (drifted up enough)
            elif pnl_pct < -min_profit_pct:
                labels[i] = 1   # SHORT (drifted down enough)
            else:
                labels[i] = 2   # NEUTRAL (no clear direction)
            holding_bars[i] = max_holding_bars

    return labels, holding_bars
```

**Why this matches the trader's style:**

- Trades that move fast (15-min reversal) → barrier hit quickly, labeled correctly
- Trades that develop slowly (S1→P→R1 over 3 hours) → barrier hit at natural exit point
- Choppy sideways bars → time barrier expires, labeled NEUTRAL
- The ATR-based barriers adapt to current volatility, just like the trader's real stops

### 9.3 Label Parameters

| Parameter          | BTC Value    | SOL/LINK Value | Reasoning                                                   |
| ------------------ | ------------ | -------------- | ----------------------------------------------------------- |
| `tp_atr_mult`      | 3.0x ATR     | 3.0x ATR       | Matches multi-pivot moves (S1→R1 ≈ 2-3x ATR on typical day) |
| `sl_atr_mult`      | 2.0x ATR     | 2.0x ATR       | Matches trader's actual stop placement                      |
| `max_holding_bars` | 48 (4 hours) | 48 (4 hours)   | Covers the common 1-4 hour hold range                       |
| `min_profit_pct`   | 0.3%         | 0.6%           | Time-barrier minimum to distinguish from noise              |

**Tuning note:** These parameters should be validated against manual trade history. The `max_holding_bars` can be extended to 96 (8 hours) or even 288 (24 hours) if backtest shows the model benefits from capturing longer trends. Start conservative at 48.

### 9.4 Label Distribution

Expected approximate distribution for BTC 5min with triple-barrier:

- LONG: ~25-30%
- SHORT: ~25-30%
- NEUTRAL: ~40-50%

NEUTRAL will be higher than with fixed lookahead because the triple-barrier correctly identifies choppy periods. This is desirable — the model learns NOT to trade during chop.

If distribution is heavily skewed, apply class weights:

```python
params['class_weight'] = 'balanced'
# Or manually: {0: 1.2, 1: 1.2, 2: 0.7}  # boost directional classes, reduce NEUTRAL
```

### 9.5 Alternative: Trailing-Stop Labels (Phase 2 experiment)

After initial results with triple-barrier, experiment with trailing-stop-based labels that more closely mimic the trader's exit style:

```python
def create_labels_trailing(df, atr_col='atr_14', trail_mult=2.0, max_bars=96):
    """
    Simulate a trailing stop from each entry point.
    The trail rides with the trend — captures S1→P→R1→R2 continuation moves.
    Label = direction and P&L at trail stop exit.
    """
    # For each candle, simulate LONG trail and SHORT trail
    # LONG: trail stop = max(close) - trail_mult * ATR, exits when low < trail_stop
    # SHORT: trail stop = min(close) + trail_mult * ATR, exits when high > trail_stop
    # Label = whichever trail captured more profit
    ...
```

This second labeling approach would let the model learn to hold through multi-pivot rides, matching the trader's actual behavior of trailing stops behind confirmed levels.

---

## 10. Training & Validation

### 10.1 Walk-Forward Validation

**Critical:** Financial data is time-ordered. Random train/test splits cause look-ahead bias.

```
Month 1-3:  ████████████  Training
Month 4:    ░░░░          Validation (tune hyperparameters)
Month 5:    ▓▓▓▓          Test (evaluate performance)
                          ─── Purge gap (5 bars) ───
Month 2-4:  ████████████  Training (roll forward)
Month 5:    ░░░░          Validation
Month 6:    ▓▓▓▓          Test
```

**Purge gap:** 48 bars (4 hours) between train and validation/test to prevent label leakage. This matches the `max_holding_bars` in the triple-barrier labeling — ensures no label from the training set was influenced by data in the test set.

### 10.2 Hyperparameter Tuning (Optuna)

```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10, log=True),
    }
    # Train and evaluate on validation set
    model = lgb.train(params, train_data, valid_sets=[valid_data], ...)
    return model.best_score['valid_0']['multi_logloss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### 10.3 Overfitting Prevention

| Technique               | Implementation                                                   |
| ----------------------- | ---------------------------------------------------------------- |
| Early stopping          | Stop training when validation loss hasn't improved for 50 rounds |
| Feature subsampling     | `feature_fraction=0.8` — each tree uses random 80% of features   |
| Data subsampling        | `bagging_fraction=0.8` — each tree uses random 80% of data       |
| Minimum leaf samples    | `min_child_samples=50` — prevents fitting to noise               |
| L1/L2 regularization    | `lambda_l1=0.1, lambda_l2=0.1`                                   |
| Walk-forward validation | Never test on data the model has seen                            |
| Purge gaps              | Remove overlapping labels between train/test                     |

---

## 11. Backtesting Framework

### 11.1 Simulation Logic

```python
def backtest(df, model, threshold=0.65, initial_capital=10000):
    capital = initial_capital
    position = None
    trades = []

    for i in range(len(df)):
        features = df.iloc[i][feature_columns]
        proba = model.predict(features.values.reshape(1, -1))[0]

        # Entry logic
        if position is None:
            if proba[0] > threshold:  # P(LONG) > 0.65
                position = open_long(df.iloc[i], capital)
            elif proba[1] > threshold:  # P(SHORT) > 0.65
                position = open_short(df.iloc[i], capital)

        # Exit logic (ATR trailing stop + target)
        elif position is not None:
            position = update_trailing_stop(position, df.iloc[i])
            if should_exit(position, df.iloc[i]):
                trade_result = close_position(position, df.iloc[i])
                capital += trade_result['pnl']
                trades.append(trade_result)
                position = None

    return trades, capital
```

### 11.2 Realistic Assumptions

| Parameter    | Value                                                                                                   | Source                        |
| ------------ | ------------------------------------------------------------------------------------------------------- | ----------------------------- |
| Trading fee  | 0.035% per trade (taker)                                                                                | Hyperliquid fee schedule      |
| Slippage     | 0.02% per trade                                                                                         | Conservative estimate for BTC |
| Max position | 10% of capital per trade                                                                                | Risk management rule          |
| Funding rate | Applied every **1 hour** on Hyperliquid (not 8-hour like Binance) — more frequent but smaller amounts. Relevant for 1-4hr holds, must be modeled accurately | Hyperliquid perpetuals        |

### 11.3 Performance Metrics

```python
metrics = {
    'total_trades': len(trades),
    'trades_per_day': len(trades) / trading_days,
    'win_rate': wins / total_trades * 100,
    'profit_factor': gross_profit / gross_loss,
    'avg_win': mean(winning_trades_pnl),
    'avg_loss': mean(losing_trades_pnl),
    'max_drawdown_pct': max_drawdown / peak_capital * 100,
    'sharpe_ratio': (mean_return - risk_free) / std_return * sqrt(252),
    'calmar_ratio': annual_return / max_drawdown,
    'avg_trade_duration': mean(trade_durations),
}
```

---

## 12. Execution Layer (Hyperliquid)

### 12.1 Connection Architecture

```python
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# REST API: for orders, account info
exchange = Exchange(wallet, base_url=constants.MAINNET_API_URL)

# WebSocket: for real-time price updates
info = Info(base_url=constants.MAINNET_API_URL, skip_ws=False)
info.subscribe({'type': 'candle', 'coin': 'BTC', 'interval': '5m'}, callback)
```

### 12.2 Order Execution Flow

```
ML Signal (LONG, 0.72)
    │
    ▼
Risk Manager
    ├── Position size = f(capital, ATR, Kelly criterion)
    ├── Initial stop loss = entry - 2.0 * ATR
    └── Trailing stop = 2.0 * ATR behind price (rides multi-pivot moves)
    │
    ▼
Hyperliquid Exchange
    ├── Limit order at current price (avoid taker fee when possible)
    └── Stop loss order (reduce-only, updated every 5min bar as trailing stop)
```

### 12.3 Bar-Close Execution

The ML bot uses **bar-close execution only** — signals are evaluated at the close of each 5-minute candle:

```
Bar Close Evaluation:
    1. All 295 features calculated on the completed candle
    2. LightGBM predicts P(LONG), P(SHORT), P(NEUTRAL)
    3. If max probability > threshold (0.65) → generate signal
    4. Risk manager validates position size, stop loss
    5. Order submitted to Hyperliquid

Why NOT intrabar:
    - ML features require complete candles (close price, volume, wick ratios)
    - Intrabar signals are noisy — partial candles produce unreliable features
    - Bar-close execution aligns with how the model was trained (on completed candles)
    - Avoids the false-signal problem seen in UT Bot v4.0's intrabar entry
```

**Note:** The UT Bot v4.0's intrabar entry + closebar cancel pattern was evaluated and rejected. The `confThreshold=0` flaw in v4.0 demonstrated that intrabar triggers without strong momentum confirmation generate excessive noise. The ML model should only act on complete information.

---

## 13. Risk Management

### 13.1 Position Sizing

```python
def calculate_position_size(capital, atr, entry_price, risk_per_trade=0.02):
    """
    Risk-based position sizing.
    Default: risk 2% of capital per trade.
    """
    stop_distance = 2.0 * atr                    # ATR-based stop
    risk_amount = capital * risk_per_trade        # max $ to lose
    position_size = risk_amount / stop_distance   # units to buy
    position_value = position_size * entry_price

    # Cap at 10% of capital
    max_position = capital * 0.10
    position_value = min(position_value, max_position)

    return position_value
```

### 13.2 Risk Rules

| Rule                       | Value                             | Rationale                                                                                             |
| -------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Max risk per trade         | 2% of capital                     | Standard risk management                                                                              |
| Max position size          | 10% of capital                    | Concentration limit                                                                                   |
| Max daily loss             | 5% of capital                     | Circuit breaker                                                                                       |
| Max concurrent positions   | 1 (Phase 1), 2 (Phase 3)          | Simplicity first                                                                                      |
| Max trades per day         | 7                                 | Prevent overtrading (target 1-5, hard cap 7)                                                          |
| Min trades per day         | 1                                 | Bot should find at least 1 setup/day — skip only if momentum is dead all session                      |
| ATR stop loss multiplier   | 2.0x ATR                          | Enough room for noise, tight enough for protection                                                    |
| Trailing stop multiplier   | 2.0x ATR                          | Trail behind price to ride multi-pivot moves                                                          |
| No fixed take profit       | Trailing stop only                | Fixed TP cuts winners short — the 1-4 hour moves between pivot levels are where the profit comes from |
| Pivot-aware exit (Phase 3) | Close partial at next pivot level | Optional: 50% at first pivot, trail rest                                                              |

### 13.3 ATR Trailing Stop

```python
def update_trailing_stop(position, current_bar, atr_multiplier=2.0):
    atr = current_bar['atr_14']

    if position['side'] == 'LONG':
        new_stop = current_bar['close'] - atr_multiplier * atr
        position['stop_loss'] = max(position['stop_loss'], new_stop)  # only move up

    elif position['side'] == 'SHORT':
        new_stop = current_bar['close'] + atr_multiplier * atr
        position['stop_loss'] = min(position['stop_loss'], new_stop)  # only move down

    return position
```

---

## 14. Project Phases & Timeline

### Phase 1: Data + Feature Engineering (Weeks 1-3)

| Step | Task                                                                             | Deliverable                                          |
| ---- | -------------------------------------------------------------------------------- | ---------------------------------------------------- |
| 1.1  | Data collector — fetch 18+ months BTC 5min + 1H OHLCV from **Binance Futures**   | `data/collectors/fetcher.py`, parquet files          |
| 1.2  | **Hyperliquid live collector** — start WebSocket for order book, trade tape, funding rate | `data/collectors/hyperliquid_ws.py` (runs continuously) |
| 1.3  | Indicator calculator — all 268 Phase 1 features in Python                        | `features/indicators.py`                             |
| 1.4  | Pivot Fibonacci engine with context features                                     | `features/pivots.py`                                 |
| 1.5  | Session/time features with overlaps                                              | `features/sessions.py`                               |
| 1.6  | VWAP calculator (daily + session)                                                | `features/vwap.py`                                   |
| 1.7  | Multi-timeframe merge (1H → 5min mapping)                                        | `features/builder.py`                                |
| 1.8  | Labeling function                                                                | `model/labeler.py`                                   |
| 1.9  | Complete feature matrix export                                                   | `data/storage/btc_features.parquet`                  |

### Phase 2: ML Training + Validation (Weeks 4-6)

| Step | Task                                              | Deliverable               |
| ---- | ------------------------------------------------- | ------------------------- |
| 2.1  | Train/test split with walk-forward                | `model/trainer.py`        |
| 2.2  | LightGBM training pipeline                        | Trained model file        |
| 2.3  | Hyperparameter tuning with Optuna                 | Optimal params config     |
| 2.4  | SHAP feature importance analysis                  | Feature importance report |
| 2.5  | Feature trimming (remove low-importance features) | Refined feature set       |
| 2.6  | Backtest simulation with realistic fees           | `model/evaluator.py`      |
| 2.7  | Performance metrics and report                    | Backtest results          |
| 2.8  | Walk-forward retraining validation                | Stability assessment      |

**Key milestone:** End of Week 5 — first SHAP analysis reveals what features the model actually values. This is valuable regardless of whether the bot goes live.

### Phase 3: Execution + Live Trading (Weeks 7-9)

| Step | Task                                                                      | Deliverable                             |
| ---- | ------------------------------------------------------------------------- | --------------------------------------- |
| 3.1  | Hyperliquid API connection (REST + WebSocket)                             | `execution/hyperliquid.py`              |
| 3.2  | Risk management module                                                    | `execution/risk.py`                     |
| 3.3  | Paper trading mode                                                        | Live signal logging without real trades |
| 3.4  | Monitoring and logging                                                    | Position tracker, P&L logger            |
| 3.5  | Add Hyperliquid microstructure features (funding, OI, L2 order book, trade tape, funding rate deep) — requires 2-3 months of live data collected since Phase 1 | 21 new features (real buy/sell data + funding dynamics) |
| 3.6  | Live trading with minimum position size                                   | Real deployment                         |

### Phase 4: Altcoin Expansion (Weeks 10-14)

| Step | Task                                                                                                                               | Deliverable              |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| 4.1  | BTC → SOL incremental learning fine-tune                                                                                           | SOL model                |
| 4.2  | BTC → LINK incremental learning fine-tune                                                                                          | LINK model               |
| 4.3  | Cross-asset correlation features (6 new)                                                                                           | Enhanced alt models      |
| 4.4  | BTC → **TAO** fine-tune — validate that model trades well during both BTC-correlated AND independent moves                         | TAO model                |
| 4.5  | TAO validation — backtest across known independent moves (March rally, Apr 10 dump) to confirm model trades TAO's own price action | TAO validation report    |
| 4.6  | BTC → HYPE fine-tune, configure asset-specific fees (zero/reduced maker)                                                           | HYPE model               |
| 4.7  | Optional: XGBoost ensemble addition                                                                                                | 2-model ensemble         |
| 4.8  | Optional: Option C implementation (Pivot hard filter)                                                                              | Filtered model           |
| 4.9  | Performance monitoring and model retraining schedule                                                                               | Monthly retrain pipeline |

**TAO deployment note:** Backtest must validate that the model performs well during BOTH BTC-correlated periods AND independent TAO moves. The token's own 268 features (pivots, EMA50, momentum) should be sufficient to trade TAO regardless of BTC correlation. BTC features are bonus context when correlated.

---

## 15. Tech Stack

### Core Dependencies

```
# Data
pandas >= 2.0
numpy >= 1.24
pyarrow          # parquet file I/O

# Technical Analysis
ta-lib           # C-based TA library (fast, comprehensive)
pandas-ta        # fallback / additional indicators

# ML
lightgbm >= 4.0
scikit-learn     # preprocessing, metrics
optuna           # hyperparameter tuning
shap             # model interpretability

# Exchange
ccxt             # historical data fetching
hyperliquid-python-sdk  # live trading execution

# Utilities
pyyaml           # configuration
loguru           # logging
schedule         # periodic retraining
```

### Development Environment

```
Python: 3.11+
OS: Windows 10 (development), Linux (optional for deployment)
IDE: VS Code
Version Control: Git
```

---

## 16. Directory Structure

```
ml-bot/
├── PROJECT_SPEC.md              # This document
├── config.yaml                  # All parameters, thresholds, API keys reference
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── collectors/
│   │   ├── fetcher.py           # OHLCV data collection from Binance Futures
│   │   ├── hyperliquid_ws.py    # Live WebSocket collector (order book, trade tape, funding)
│   │   └── storage.py           # Parquet read/write utilities
│   └── storage/
│       ├── btc_5m.parquet       # Raw 5min OHLCV (from Binance)
│       ├── btc_1h.parquet       # Raw 1H OHLCV (from Binance)
│       ├── btc_features.parquet # Complete feature matrix
│       └── hyperliquid/         # Live data (order book snapshots, trades, funding)
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
│   └── builder.py               # Combine all 295 features, multi-TF merge, preprocessing
│
├── model/
│   ├── labeler.py               # Label creation (LONG/SHORT/NEUTRAL)
│   ├── trainer.py               # LightGBM training pipeline
│   ├── tuner.py                 # Optuna hyperparameter optimization
│   ├── evaluator.py             # Backtest simulation + performance metrics
│   ├── explainer.py             # SHAP analysis and feature importance
│   └── models/                  # Saved model files
│       ├── btc_v1.txt           # LightGBM model (text format)
│       └── btc_v1_params.json   # Training parameters record
│
├── execution/
│   ├── hyperliquid.py           # Exchange connection, order management
│   ├── risk.py                  # Position sizing, stop loss, take profit
│   ├── paper_trader.py          # Simulated live trading
│   └── live_trader.py           # Real execution engine
│
├── monitoring/
│   ├── logger.py                # Trade logging, P&L tracking
│   └── dashboard.py             # Optional: simple monitoring UI
│
├── tests/
│   ├── test_indicators.py       # Verify indicator calculations match TradingView
│   ├── test_features.py         # Feature pipeline unit tests
│   ├── test_labeler.py          # Label correctness tests
│   └── test_risk.py             # Risk management edge case tests
│
└── notebooks/
    ├── 01_data_exploration.ipynb    # EDA on collected data
    ├── 02_feature_analysis.ipynb    # Feature distributions and correlations
    ├── 03_model_training.ipynb      # Interactive training and evaluation
    └── 04_shap_analysis.ipynb       # Feature importance deep dive
```

---

## 17. Success Criteria

### Phase 2 (Backtest) — Minimum Viable Performance

| Metric             | Target              | Hard Minimum                                   |
| ------------------ | ------------------- | ---------------------------------------------- |
| Win Rate           | > 58%               | > 52%                                          |
| Profit Factor      | > 1.8               | > 1.3                                          |
| Max Drawdown       | < 8%                | < 15%                                          |
| Trades per Day     | 1-5                 | < 7                                            |
| Sharpe Ratio       | > 2.0               | > 1.0                                          |
| Avg Win / Avg Loss | > 1.2               | > 0.8                                          |
| Avg Trade Duration | 1-4 hours           | 30 min - 8 hours                               |
| Avg Win Duration   | > Avg Loss Duration | — (winners should ride longer than losers cut) |

### Phase 3 (Paper Trading) — Live Validation

| Metric               | Target                              |
| -------------------- | ----------------------------------- |
| Paper trading period | Minimum 2 weeks                     |
| Win rate consistency | Within 5% of backtest               |
| Signal latency       | < 2 seconds from bar close to order |
| System uptime        | > 99% during trading sessions       |

### Phase 3 (Live Trading) — Go/No-Go Criteria

| Criteria               | Requirement                         |
| ---------------------- | ----------------------------------- |
| Paper trading passed   | All metrics within acceptable range |
| Initial capital        | Start with minimum viable amount    |
| Max daily loss trigger | Halt trading if -5% in one day      |
| Monthly review         | Review and retrain model monthly    |

---

## 18. Key Design Decisions

### Decision Log

| #   | Decision                                                                    | Rationale                                                                                                                                                                                                                                                                                               | Date       |
| --- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| 1   | **Option A first (Pure ML)** — no rule-based filters initially              | Let the model learn freely from data; avoid imposing potentially suboptimal human rules. SHAP will reveal if trader's intuitions (Pivot importance, DI dynamics) are confirmed by data.                                                                                                                 | 2026-04-18 |
| 2   | **Option C later** — add Pivot Fibonacci hard filter after validation       | If the model overtrades or trader wants more disciplined entry, filter signals to only occur near Pivot levels. Decision based on backtest + live results, not assumption.                                                                                                                              | 2026-04-18 |
| 3   | **Raw values, not rules** — feed indicator values, not boolean rule outputs | ML learns better from continuous values (RSI=65.3) than binary rules (overbought=true). Allows model to find non-obvious threshold combinations.                                                                                                                                                        | 2026-04-18 |
| 4   | **DI rate of change as feature**                                            | Trader's insight: "DI+ rising AND DI- falling" matters more than "DI+ > DI-". This captures directional momentum, not just position.                                                                                                                                                                    | 2026-04-18 |
| 5   | **Custom Python pipeline over FreqAI**                                      | More control over Hyperliquid integration and feature engineering. FreqAI's patterns used as reference.                                                                                                                                                                                                 | 2026-04-18 |
| 6   | **LightGBM over deep learning**                                             | Faster, interpretable (SHAP), works on tabular data, supports incremental learning for BTC→altcoin transfer.                                                                                                                                                                                            | 2026-04-18 |
| 7   | **295 features, trim after SHAP**                                           | Start wide (295), let SHAP identify the top 120-150 contributors. Better to include and discover than to miss a key feature. LightGBM handles 500+ features efficiently.                                                                                                                                | 2026-04-18 |
| 8   | **Session overlaps as explicit features**                                   | Trader insight: sessions aren't just "Asia/London/NY" — overlaps (especially London-NY) are the highest-quality trading windows.                                                                                                                                                                        | 2026-04-18 |
| 9   | **Balanced feature categories (22 categories)**                             | Initial design was 40% momentum-biased (16 features). Expanded to 22 categories across all market dimensions — no single category exceeds 17%.                                                                                                                                                          | 2026-04-18 |
| 10  | **Percentage-based features for transfer learning**                         | All price-distance features as % to enable BTC model → SOL/LINK fine-tuning without recalibration.                                                                                                                                                                                                      | 2026-04-18 |
| 11  | **UT Bot v4.0 — feature concepts only, NOT execution reference**            | v4.0 has confThreshold=0 (no confidence filter), causing false signals on every trail line touch. v3.1 had confThreshold=0.15 which was better. Extract only efficiency_ratio and atr_ratio concepts.                                                                                                   | 2026-04-18 |
| 12  | **Bar-close execution only, NO intrabar entry**                             | ML features require complete candles. Intrabar signals produce unreliable features from partial data. The UT Bot v4.0 false-signal problem reinforces this decision.                                                                                                                                    | 2026-04-18 |
| 13  | **Depth framework for all major indicators**                                | Each indicator expanded with: value, direction, acceleration, position vs signal, cross momentum, range context. RSI went from 5 to 12 features, WaveTrend from 7 to 13, ADX/DI from 6 to 12. Snapshot values alone miss critical dynamics.                                                             | 2026-04-18 |
| 14  | **Event Memory as dedicated category (41 features)**                        | "RSI is at 55 now" is meaningless without temporal context. "RSI at 55 but was oversold 8 bars ago" is a recovery signal. CM_Stoch red/green bar recency, WaveTrend OB/OS history, squeeze fire timing all captured.                                                                                    | 2026-04-18 |
| 15  | **Divergence detection as explicit features**                               | Price-oscillator divergences (RSI, MACD, WT, Stochastic) signal momentum exhaustion before price reverses. Hidden divergences detect continuation setups. User has "Divergence for Many Indicators v4" in collection but no features were initially mapped.                                             | 2026-04-18 |
| 16  | **Statistical/Fractal features for regime detection**                       | Hurst exponent, entropy, autocorrelation, and variance ratio capture market microstructure patterns invisible to standard TA indicators — trending vs mean-reverting vs random behavior.                                                                                                                | 2026-04-18 |
| 17  | **NOT a scalping bot — 5min entry, swing-style exit**                       | Trader holds 1-4 hours typically (15 min to 1 day range), riding between Fibonacci pivot levels. Fixed 1-hour lookahead would mislabel most winning trades. Triple-barrier labeling adapts to actual hold time.                                                                                         | 2026-04-19 |
| 18  | **Triple-barrier labeling over fixed lookahead**                            | Labels adapt to each trade's natural exit point via ATR-based barriers + time barrier (48 bars = 4 hours). Captures quick reversal trades (15 min) and multi-pivot rides (4 hours) equally well. Trailing-stop labels as Phase 2 experiment.                                                            | 2026-04-19 |
| 19  | **Trailing stop only, no fixed take profit**                                | Fixed TP (1.5x ATR) would cut the S1→P→R1→R2 multi-pivot rides that generate most profit. ATR trailing stop lets winners run to their natural exhaustion point, matching the trader's actual exit behavior.                                                                                             | 2026-04-19 |
| 20  | **Momentum-with-structure, not strict Pivot Fibonacci**                     | Bot uses Pivots as structural map + EMA50 as momentum compass + oscillators for timing. Two archetypes: Full Pivot ride (high profit, TAO) and EMA50 direction change (high win rate, BTC). Model learns both from features.                                                                            | 2026-04-19 |
| 21  | **Asset progression: BTC → SOL → LINK → TAO → HYPE**                        | BTC has most data and cleanest patterns for base model. TAO is the final target (trader's Bittensor expertise) but has news-driven decorrelation risk. HYPE offers potential zero/reduced fees on Hyperliquid.                                                                                          | 2026-04-19 |
| 22  | **BTC correlation = booster, not filter**                                   | Any altcoin can move independently of BTC on news. These moves are OPPORTUNITIES, not risks — the token's own 268 features (pivots, EMA50, momentum) still work. BTC cross-asset features (6) add power when correlated, are naturally ignored when not. Never skip signals due to low BTC correlation. | 2026-04-19 |
| 23  | **Binance OHLCV for training, Hyperliquid for live data**                   | Binance Futures has ~10x Hyperliquid volume → cleanest candles with least noise. Perp prices differ by < 0.01% across exchanges (arbitrage). All 268 indicator features are exchange-agnostic. Live inference uses Hyperliquid real-time data. | 2026-04-19 |
| 24  | **18 months minimum data collection**                                       | Must capture full market cycle: BTC 68K→126K rally + sudden 60K drop. 18 months provides both trending and crash regimes for robust regime-dependent learning. Collect as much as available (2+ years if exchange has history). | 2026-04-19 |
| 25  | **Hyperliquid live data: collect from Phase 1, use in Phase 3**             | Order book and trade tape have NO historical API — must stream live from day one. Funding rate and OI also collected from Hyperliquid (unique hourly funding). After 2-3 months of collection, 21 microstructure features become trainable. | 2026-04-19 |
| 26  | **Hyperliquid hourly funding rate features**                                | Hyperliquid settles funding every hour (not 8-hour like Binance) = 8x faster signal on crowding shifts. Added 3 funding rate deep features (1h_change, 8h_sum, flip_bars) to capture this unique data advantage. | 2026-04-19 |

---

## 19. References & Sources

### Research Papers & Articles

- [ML Models That Actually Work in Crypto Trading](https://medium.com/@laostjen/machine-learning-models-that-actually-work-in-crypto-trading-78a6735b5639) — Ensemble approach analysis, why genetic algorithm strategies failed
- [AI Trading Bot with Real-Time Crypto Signals (Corvino)](https://medium.com/@davide.civid.96/i-built-an-ai-trading-bot-that-generates-real-time-crypto-signals-heres-how-56641e8fd43c) — 4-model ensemble (LightGBM + XGBoost + LSTM + Transformer), 40+ features, ATR-based TP/SL
- [LSTM+XGBoost Crypto Price Prediction (arXiv 2506.22055)](https://arxiv.org/html/2506.22055v1) — Hybrid architecture, XGBoost enhances LSTM for non-linear relationships
- [ML Approaches to Crypto Trading Optimization (Springer)](https://link.springer.com/article/10.1007/s44163-025-00519-y) — Ensemble models outperform deep learning, R2 ~0.98
- [LightGBM Cryptocurrency Forecasting (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1544612318307918) — LightGBM for crypto price trend forecasting
- [Bitcoin Volatility Forecasting with Gradient Boosting & SHAP (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425040199) — SHAP feature importance for crypto models
- [Quantitative Alpha in Crypto Markets (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5225612) — Factor models, momentum/liquidity significance
- [ML-Driven Multi-Factor Quantitative Model for Ethereum](https://dl.acm.org/doi/10.1145/3766918.3766922) — Technical indicators + on-chain metrics + sentiment

### Frameworks & Libraries

- [FreqAI Feature Engineering Documentation](https://www.freqtrade.io/en/stable/freqai-feature-engineering/) — Feature expansion patterns, normalization, multi-timeframe methodology
- [FreqAI Introduction](https://www.freqtrade.io/en/stable/freqai/) — LightGBM/XGBoost integration, recommended models
- [Machine Learning for Algorithmic Trading — Stefan Jansen](https://github.com/stefan-jansen/machine-learning-for-trading) — LightGBM intraday features, SHAP analysis, gradient boosting for trading
- [Incremental Learning in LightGBM and XGBoost](https://medium.com/data-science-collective/incremental-learning-in-lightgbm-and-xgboost-9641c2e68d4b) — init_model for transfer learning
- [LightGBM Official](https://lightgbm.org/) — Documentation and API reference

### Exchange & Execution

- [HyperLiquidAlgoBot (GitHub)](https://github.com/SimSimButDifferent/HyperLiquidAlgoBot) — Hyperliquid-native bot with BB+RSI+ADX strategy, ML optimization, backtesting framework
- [Chainstack Hyperliquid Trading Bot](https://github.com/chainstacklabs/hyperliquid-trading-bot) — WebSocket integration, order management patterns
- [Freqtrade & Bitget Guide 2026](https://www.bitget.com/academy/freqtrade-cryptocurrency-trading-bot-america-2026-comprehensive-guide-to-automated-trading) — Bot deployment architecture

### TradingView Indicators (analyzed for feature extraction)

- [Machine Learning: Lorentzian Classification — jdehorty](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/) — Lorentzian distance metric, 5-feature KNN, normalization
- [LightGBM vs XGBoost vs CatBoost Comparison](https://www.griddynamics.com/blog/xgboost-vs-catboost-vs-lightgbm) — Model selection analysis
- [Fibonacci, VWAP, EMA Confluence in Crypto Scalping](https://www.cryptowisser.com/guides/fibonacci-vwap-ema-crypto-scalping/) — Multi-indicator confluence methodology
- [QuantDataAPI — 250+ Financial Features](https://quantdataapi.com/blog/feature-engineering-ml-trading-models) — Professional feature engineering reference

### Trader's TradingView Indicator Collection

Analyzed 60+ indicators from `TradingView/top_strategies/` directory:

- **ML-based:** Machine Learning Lorentzian Classification, ML Adaptive SuperTrend
- **Momentum:** WaveTrend, Squeeze Momentum, VuManChu Cipher B, RSI, MACD MTF, CM Stochastic MTF
- **Trend:** ADX/DI, SuperTrend, Alpha Trend, Chandelier Exit, Madrid MA Ribbon
- **S/R Structure:** CM Pivot Points, ICT Concepts, Order Blocks, Market Structure Break, Volume-based S/R
- **Volume:** Volume Flow Indicator, Volume-based Support & Resistance
- **Sessions:** Sessions [LuxAlgo] with VWAP per session
- **Breakout:** Breakout Finder, Breakout Probability, Fibonacci Bollinger Bands

### Trader's Custom Scripts (reference implementations)

- `my_scripts/UT Bot v4.0.pine` — **Feature concepts only** (efficiency_ratio, atr_ratio). Execution logic is flawed: confThreshold=0 causes false signals. See Section 5.3 critique.
- `my_scripts/UT Bot v3.1.pine` — Simpler trailing logic, `confThreshold=0.15`, ADX filter available but disabled by user (`adx_threshold=0`).
- `my_scripts/Volume Flow Indicator MTF (modified by K).pine` — Multi-timeframe VFI
- `my_scripts/BB_KC.pine` — Bollinger + Keltner squeeze detection

---

_This specification will be updated as the project progresses through each phase. All design decisions are logged with rationale for future reference._
