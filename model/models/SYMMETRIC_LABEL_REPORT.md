# Symmetric-label re-train report

**Context.** The live-chart insight — "was it right to predict SHORT when direction can't confirm?" — prompted an audit of label construction. Original triple-barrier used tp=4·ATR / sl=3·ATR (asymmetric). Because the SL is closer, SHORT-side barriers hit more often under neutral-to-down regimes, baking a 35/48/17 L/S/N skew into labels. A model minimizing log-loss under that skew learns to bias SHORT — not because of alpha, but because SHORT labels are statistically easier to hit.

The fix: set `sl_atr_mult = 4.0` (symmetric 4/4), relabel, re-extract features, retrain, re-evaluate.

## Smoking-gun evidence (why relabel was correct)

Same OOT period (2026-03-29 → 2026-04-21), same booster file (`tuned_trim120`), evaluated against:

| Label set | raw @ 0.65 fires | dir_hit | Side mix |
|---|---|---|---|
| Old 4:3 (used for training) | 20 | **95%** | near-100% SHORT |
| New 4:4 (re-extracted) | 23 | **8.7%** | 100% SHORT |

*Same model, same market days, same ultra-high-conviction signals.* The only thing that changed is how we count a "win". When SHORT stops having a tighter barrier, 95% turns into 8.7%. That is a label-asymmetry artifact, not alpha.

## Pipeline executed

Order: **labels → features → feature selection → hyperparameters** (never hyperparameters first).

1. Relabel under symmetric 4/4 barriers. ([config.yaml:tp_atr_mult/sl_atr_mult](config.yaml))
2. Re-extract features (no math change; label column only).
3. Retrain 278 full features, default params → `sym278`.
4. SHAP across all 8 folds of `sym278` → aggregated importance.
5. Pick top-N by cumulative coverage (threshold 0.95, clipped [80, 150]) → **97 features** at 95.14% coverage. ([scripts/pick_features.py](scripts/pick_features.py))
6. Retrain the 97-feature model → `sym_trim97`.
7. Fit isotonic calibrators on OOF stack (69,984 rows).
8. Threshold sweeps (raw + calibrated, thresholds 0.40–0.75).
9. OOT holdout evaluation.
10. Optuna tune (50 trials, 60-min timeout) on folds {1,4,7} → `sym_trim97_tuned`. **Tuner hit timeout at 13 completed trials** (pace: ~5 min/trial at TPE-explored param ranges, 1 pruned). Best trial #6, mean val logloss 1.0419.

## Why SHAP before Optuna

Hyperparameters optimize how a model uses the features you give it. They cannot conjure signal from features that are absent, stale, or noisy. If you tune first, you tune against a feature set that may be the wrong one — and you pay 60 minutes per run. Under the new symmetric labels the feature ranking shifted: 24 features dropped (e.g. `adx_direction`, `macd_bars_since_*`, `atr_ratio`), 1 new (`ema21_dist_pct`). Tuning on the stale trim120 would have been optimizing against the wrong information set.

## Feature set changes (sym278 SHAP vs old trim120)

- Selected: **97/268** (was 120)
- Overlap with trim120: 96
- Added: `ema21_dist_pct`
- Dropped (24): `ema50_dist_pct`, `adx_direction`, `atr_ratio`, `macd_hist_slope_3`, `macd_bars_since_zero_cross`, `macd_bars_since_signal_cross`, `rsi_bars_since_50_cross`, `body_range_ratio_mean_5` and 16 others

Note: EMA50 dropped *from this model's view*. Your live-trading observation ("EMA50 > Pivot Fib as signal") is a directional heuristic the model doesn't need as a raw feature when `ema21_dist_pct` plus the dense pivot-fib feature family already span the same information — the model combines them implicitly.

## OOF walk-forward performance (8-fold, 1-month val windows, 48-bar purge)

| model | features | log-loss mean | LONG prec | SHORT prec | NEUTRAL prec |
|---|---|---|---|---|---|
| tuned_trim120 (old 4:3) | 120 | 1.017 | 0.33 | **0.58** | 0.43 |
| sym278 (new 4:4, default) | 278 | 1.0441 | 0.392 | 0.427 | 0.390 |
| sym_trim97 (new 4:4, default) | 97 | **1.0427** | 0.396 | **0.417** | 0.407 |
| sym_trim97_tuned (new 4:4, Optuna) | 97 | 1.0422 | 0.393 | 0.423 | **0.460** |

Observations:
- Under symmetric labels the class priors balance (33/33/33) and per-class precision collapses toward parity. That is the intended outcome — model no longer gets free precision from SHORT having an easier barrier.
- sym_trim97 has a lower log-loss than sym278 (trim is a strict win at default params).

## OOF CAL threshold sweep — `sym_trim97` (pooled across 8 folds)

| thr | fire% | dir_hit% | LONG n / hit% | SHORT n / hit% | folds firing |
|---|---|---|---|---|---|
| 0.50 | 12.75 | 49.88 | 2343 / 51.4 | 5738 / 50.9 | 7/8 |
| 0.55 | 4.65 | 52.40 | 757 / 53.2 | 1881 / 57.1 | 7/8 |
| 0.60 | 1.74 | 54.49 | 287 / **64.5** | 597 / 62.5 | 5/8 |
| **0.65** | **0.91** | **55.80** | 186 / **71.5** | 312 / 66.7 | 5/8 |
| 0.70 | 0.42 | 57.81 | 129 / **77.5** | 117 / 66.7 | 4/8 |
| 0.75 | 0.19 | **79.49** | 68 / 76.5 | 32 / **87.5** | 3/8 |

This is the honest curve: sparse, symmetric between sides, hit-rate climbs monotonically with threshold.

## OOF CAL threshold sweep — `sym_trim97_tuned` vs untuned

| thr | tuned fire% | tuned hit% | untuned fire% | untuned hit% |
|---|---|---|---|---|
| 0.55 | 5.26 | 51.76 | 4.65 | 52.40 |
| 0.60 | 1.96 | 54.37 | 1.74 | 54.49 |
| **0.65** | **0.54** | **73.74** | 0.91 | 55.80 |
| **0.70** | **0.14** | **79.08** | 0.42 | 57.81 |
| 0.75 | 0.05 | 80.00 | 0.19 | 79.49 |

Tuning lifts OOF hit-rate at high thresholds (0.65 from 55.8% → 73.7%, 0.70 from 57.8% → 79.1%) while keeping fire-rate sparse. Looks great on paper. Then OOT tells a different story.

## OOT tail (2026-03-29 → 2026-04-21, 6670 samples, label dist 39/34/26)

**Raw**
| thr | fires | fire% | dir_hit% | L n/hit | S n/hit |
|---|---|---|---|---|---|
| 0.50 | 1589 | 23.82 | 41.66 | 1041/44.1 | 548/37.0 |
| 0.55 | 430 | 6.45 | **49.30** | 331/46.5 | 99/58.6 |
| 0.60 | 34 | 0.51 | 32.35 | 30/23.3 | 4/100 |
| 0.65 | 0 | 0 | — | — | — |

**Calibrated**
| thr | fires | fire% | dir_hit% | L n/hit | S n/hit |
|---|---|---|---|---|---|
| 0.50 | 808 | 12.11 | 45.79 | 427/50.6 | 381/40.4 |
| **0.55** | **86** | **1.29** | **45.35** | 35/28.6 | 51/56.9 |
| 0.60 | 5 | 0.07 | 20.0 | 5/20 | — | 
| 0.65 | 0 | 0 | — | — | — |

Comparison to `tuned_trim120` (old 4:3 model) re-evaluated on the **same OOT tail**:

| model | cal @ 0.65 | best cal threshold |
|---|---|---|
| tuned_trim120 | 18 fires, 5.6% hit (all SHORT, one hit) | 0.50: 25% fire, 43.4% hit |
| sym_trim97 | 0 fires | 0.55: 1.29% fire, 45.4% hit |

`sym_trim97` is a **strict behavioral improvement**: the catastrophic "confidently wrong at 0.65" failure mode of the old model is gone. The old model was spraying high-confidence SHORT calls into a market that had shifted regime; the new model correctly abstains.

## OOT on `sym_trim97_tuned` — tuning overfit the tuning folds

**Calibrated**
| thr | fires | fire% | dir_hit% | L n/hit | S n/hit |
|---|---|---|---|---|---|
| 0.50 | 1503 | 22.53 | 43.90 | 1086/46.6 | 415/36.9 |
| **0.55** | **647** | **9.70** | **43.59** | 494/50.4 | 153/21.6 |
| 0.60 | 219 | 3.28 | 33.33 | 160/45.6 | 59/**0.0** |
| 0.65 | 37 | 0.55 | **18.92** | 16/43.8 | 21/**0.0** |
| 0.70 | 3 | 0.04 | 0.00 | 3/0 | — |

**Head-to-head on OOT (calibrated):**

| thr | untuned fire% / hit% | tuned fire% / hit% | verdict |
|---|---|---|---|
| 0.55 | **1.29% / 45.4%** | 9.70% / 43.6% | tuned fires 7× more often at worse hit |
| 0.60 | 0.07% / 20% | 3.28% / 33.3% | tuned more aggressive, mediocre |
| 0.65 | **0 fires** | 0.55% / **18.9%** (21 SHORT at 0% hit) | tuned starts being *confidently wrong* again |
| log_loss | **1.0572** | 1.0615 | tuned worse on OOT log-loss |

**Diagnosis.** Optuna optimized against folds {1, 4, 7} with val windows Jul/Oct/Jan. Tuned params (smaller trees num_leaves=34, heavier L1/L2 at 0.12/0.15, narrower feature_fraction=0.63) generalize *within* those three slices but lose generalization to the recent-month tail. The smoking gun is cal @ 0.65: 21 SHORT fires at **0% hit** — the same failure pattern the relabel was supposed to eliminate, reintroduced by tuning that only saw pre-Feb data for SHORT-heavy folds.

## Decision: ship `sym_trim97` (untuned), not `sym_trim97_tuned`

| axis | sym_trim97 | sym_trim97_tuned | winner |
|---|---|---|---|
| OOF log_loss | 1.0427 | 1.0422 | tuned (marginal) |
| OOF hit @ 0.65 | 55.80% | 73.74% | tuned |
| OOT cal log_loss | **1.0572** | 1.0615 | **untuned** |
| OOT cal @ 0.55 hit | **45.4%** | 43.6% | **untuned** |
| OOT cal @ 0.65 fires at 0% hit | **0** | 21 SHORT at 0% | **untuned** |
| SHORT-bias artifact returned? | no | **yes** | **untuned** |

The tuning gain was an OOF artifact. Untuned is the honest model.

## Is it tradeable?

Even the winning untuned model: best OOT cal @ 0.55 = 1.29% fire × 45.4% dir_hit — **below the ~50% hit you need for break-even at 1:1 RR barriers and fees.** Three possible reads:

1. **OOT window is 23 days, 6670 bars.** Sample size at any threshold > 0.55 is tiny; the 0/0 at 0.65 is consistent with sparse signal in this particular regime, not proof the model is broken.
2. **Calibration was fit on OOF ending 2026-03-28 and applied to tail immediately after.** The recent-month regime may differ from the preceding 8 months; calibration could be over-tightening.
3. **45.4% hit on 86 fires is genuinely close to random.** If this persists over more OOT, the 97-feature signal under symmetric labels is real but not tradeable at 1:1 RR. Options: asymmetric exit (RR > 1), wider barriers, or accept that *at this feature set* the model discriminates but does not predict well enough to profit under these fees.

My read: the previous SHORT-bias model was *worse*: it looked tradeable at 95% hit under mismatched labels, then collapsed to 8.7% under honest labels. The new model's 45.4% is what the signal is actually worth. That is painful but truthful. **Tuning did not help** — if anything it reintroduced the SHORT-bias failure mode inside the tail.

## Next steps

1. ~~Optuna tune~~ — done. Tuned variant is worse on OOT; do not ship.
2. **Consider this session complete at `sym_trim97`.** The model is honest but marginal.
3. **If you want to push further** (any of these, ranked by expected impact):
   - **RR asymmetric exit** (tp=5·ATR, sl=3·ATR or wider) — a different hypothesis than "symmetric labels are truthful", accepts that 45% hit × 1.67 RR ~ break-even improves.
   - **Pivot-regime features** (pivot-level proximity × trend strength × EMA21 alignment) — directly encodes your live-trading "EMA50 > Pivot Fib" insight. Current features capture each piece separately; a composite may give the model a handle it currently lacks.
   - **Longer OOT window** (push oot_start earlier, e.g. 2026-01-01) — get statistically meaningful fire counts at 0.65+ so the "45.4%" estimate has less noise.
   - **Re-tune with a larger fold subset** (e.g. tune on folds {1, 3, 5, 7} = 4 folds with more coverage) — may reduce overfitting to folds {1, 4, 7} that caused this round's OOT regression. Cost: ~2× tuning time.
4. **Do not** roll back to 4:3 labels. The SHORT-bias precision was an artifact; the 4:4 picture is what the model actually knows.
5. **Do not** ship `sym_trim97_tuned`. Its OOF gains are overfit to the tuning folds.

## Artifacts

- `model/models/BTC_wf_sym278/` — 278-feature baseline + SHAP + selected_features.json (97)
- `model/models/BTC_wf_sym_trim97/` — **[RECOMMENDED]** 97-feature model + calibrators + sweeps + OOT
- `model/models/BTC_wf/best_params_sym.json` — Optuna result (13 trials, best #6, val logloss 1.0419)
- `model/models/BTC_wf_sym_trim97_tuned/` — tuned variant + eval (do not deploy; kept for analysis)
- `/tmp/tune_sym.log` — Optuna completed log
- `/tmp/post_tune.log` — auto-chained post-tuning pipeline log
