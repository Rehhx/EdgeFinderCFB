# Project Handoff — CFB Betting Predictor
*Written 2026-07-29 for continuity across Claude Code sessions/models (e.g. switching to Opus). Short answer: switching the coding model does NOT affect this build — all code, data, and results are on disk; nothing depends on which model drives the session. The `.env` `CLAUDE_MODEL_*` settings only control the in-pipeline news extraction (keep `claude-sonnet-5` there regardless).*

## What this project is
College football betting model targeting spreads, totals, moneylines, player props. Master plan: `CFB_PREDICTOR_PLAN.md` (read §8 build-order for statuses). Data inventory: `DATA_CATALOG.md`. User purchases: `GRAB_LIST.md` (Odds API 5M plan is LIVE).

## Current state (all phases 0–3 + tooling built and backtested)
| Piece | File | Status / headline result |
|---|---|---|
| Ingestion (CFBD, bulk PBP, rosters) | `ingestion/` | 2021–25 PBP, ~21/1000 CFBD calls used |
| EPA ratings | `features/epa_ratings.py` | ridge, recency, FCS pooling |
| Roster priors (portal/recruiting/returning/coach/OL) | `features/roster_priors.py` | walk-forward safe |
| Coach DB + tendencies | `features/coach_db.py`, `coach_tendencies.py` | PROE/pace are coach-sticky (r=.65/.50), 2H-adjust is NOT (r=.06) |
| Spread backtest | `backtest/spread_phase1.py` + `backtest/spread_history.py` (**8 seasons**) | **THE edge is BIG-SPREAD early-season UNDERDOGS, now TIERED: wks 1–5, model on the dog, edge≥6 — PREMIUM (|spread|≥25) 62.3% ATS / +18.8% ROI / 302 bets / 8-of-8 seasons → size 2u; STANDARD (17–25) 56.9% / +8.6% → 1u. Bigger spreads = more public overbet = better edge. (Road dogs 60.0% vs home dogs 54.2%, but spread size is the stronger filter.)** Bettable-range (|spread|≤21) spreads have NO edge (51.5%); blind big-dog also none (52%) — it's the model's *selection* of overvalued big favorites. Old |spread|≤21 guard WRONGLY suppressed this (fixed in edge_report). Spread-only: same dogs lose outright on ML (37%). |
| **Favourite team-total UNDER** ⚠️ *provisional* | `ingestion/historical_team_totals.py`, `picks/team_total_picks.py` | **A STANDALONE MARKET BIAS, not our model's edge** (corrected after control tests). BLIND big-favourite unders at spread ≥25: **64.5% / +24.8% (62 bets, 65/63/65% by season)**; our model-selected subset 63.9% — picks 61 of the same 62 games and **adds nothing**. Wider ≥17 net is weaker and FADING (60.0→58.2→57.1%). On the same games the **spread bet is better (62.8% vs 58.9%)** and outcomes agree 78% → mostly the same position expressed worse. Prefer the spread; use only if spread limits are exhausted; count as the SAME position. Small sample (~20 bets/season). |
| **BIG-DOG 1H — UPGRADED 2026-08-03** | `backtest/h1_saturation.py`, `picks/first_half_picks.py` | Applying the Q1 screen to the 1H line found the **same bounded-quantity mispricing**, and showed the old weeks-1–5 cap was costing us. Books derive 1H at a flat **0.573 × full spread (R²=0.975)** while the dog's realised 1H share **falls 0.647 → 0.413** as the spread grows (a half holds only ~7–8 possessions a side). Over-grant: −0.17 pts at a 13-pt spread, **+3.58 at 23, +7.32 at 37**. **SHIPPED: ≥17 in wks 1–5 PLUS ≥21 in wks 6–15** → 59 bets/szn, **65.7%, +25.9% ROI, t=+3.79**, CI [+12.0,+38.7], all 3 seasons (+29.9/+18.4/+30.3%) → **30.8 u/szn vs 21.3 before (+45%)**. Rejected the simpler ">=17 all weeks" (30.0 u/szn but wks6+ 2024 was −4.6%); the late slice needs the bigger spread. **Unlike Q1, the model filter HELPS here** (63.6%/+21.8% vs 60.4%/+16.0% blind) — keep `model on dog ≥6`. |
| **⚠️ Q1 and 1H OVERLAP — do not bet both** | `backtest/h1_saturation.py` | On the same game, Q1 and 1H outcomes **agree 76.1% (r=+0.51)**; both win 45.8%, both lose 30.3%. **2u Q1 + 2u 1H is closer to a 4u single position than to two bets.** Pick one by spread size: **≥25 → Q1** (+44.0% vs 1H +26.7%); **21–25 → 1H** (+23.1% vs Q1 +11.4%); 17–21 a wash (+8.5 vs +7.1). Warning is printed in both pick reports. |
| **Q1 SPREAD — ⭐ BEST EDGE FOUND** | `ingestion/historical_q1.py`, `backtest/q1_spreads.py`, `picks/q1_picks.py` | **Take the BIG UNDERDOG on the first-quarter spread.** `\|spread\|≥17`: 102 bets/szn, **61.9%, +20.0% ROI**, t=+3.70, boot CI [+9.1,+30.6]. `\|spread\|≥25`: 31/szn, **75.5%, +45.7%**, t=+5.27, CI [+28.5,+62.1]. Positive in **all 3 seasons at every threshold**, monotone in spread size, worst 10-bet run −0.6u, longest losing streak 4. Push rate 1.6%. **MECHANISM — a linear rule against a saturating reality:** books set the Q1 line at ~**0.289 × full spread** (linear, R²=0.85), but the dog's ACTUAL Q1 deficit **plateaus near 4 pts** however big the spread gets (−3.75 at 13, −4.29 at 19, −3.58 at 28, −4.60 at 37) because a quarter holds only ~3–4 possessions a side — full-game margin is unbounded, Q1 margin is not. So the rule **over-grants** the dog +1.91 pts at a 19-pt spread, +3.40 at 28, +4.33 at 37. **⚠️ NOT A MODEL EDGE — blind BEATS model-selected (75.5%/+45.7% vs 73.3%/+42.1%); `picks/q1_picks.py` deliberately uses no ratings.** ⚠️ **Low-limit market** — expect small maximums; that is probably *why* it stays mispriced. 25,196 line rows pulled (2023–25 all weeks, 8 books, ~26k credits). Works all season (wks 6–10 even better), unlike the 1H/spread edges. |
| **Q1 TOTAL — DO NOT BET** | `ingestion/historical_q1.py totals_q1`, `backtest/q1_totals.py` | Tested straight after the Q1 spread win, same playbook. Free mechanism test looked great (Q1 takes only **22.50%** of regulation points across 20,555 games vs a naive 25%; Q2 is 30.44%). **But books already use 0.2152** (sd 0.014) — *below* the true 22.5%, i.e. conservative, not lazy. Over-grant flips sign by bucket (−0.18/−0.75/+0.77/+0.66). BLIND Q1 UNDER: 1,664 bets, 53.7%, **+1.3% ROI, t=+0.58, boot CI [−3.3,+5.9]** → includes zero; seasons −3.4/+5.7/+1.8%. OVER control −12.9% shows the under-lean is real but **fully priced**. 24,388 rows pulled (~24k credits). **THE SHARPENED RULE: a derived constant only breaks when the sub-period quantity is BOUNDED in a way the parent is not.** Q1 *margin* saturates near 4 pts while full-game margin is unbounded → linear rule over-grants, error grows with spread → **edge**. Q1 *points* scale with full-game points (real ratio stable 0.219→0.233) → a flat ratio is the right model and books use a good one → **no edge**. Hunt constants applied to a quantity with a CEILING its parent lacks. |
| **Alternate spreads — DO NOT BET** | `ingestion/historical_alt_spreads.py`, `backtest/alt_spreads.py` | Tested 2026-08-03 as the next derived-line candidate after 1H. **202,550 ladder rows pulled** (2023–25 wks 1–5, 868 games, 5 books, ~8.7k credits). **No rung reliably beats the main number.** Paired test on the 68 games quoted at all 5 rungs: best candidate −3.5 is +10.0pp over MAIN but **t=+1.53, bootstrap 95% CI [−4.0, +21.5] — includes zero**. Decisive tell: **the "best" rung is unstable** — +3.5 wins unpaired, −3.5 wins paired, −7.0 swings −2.4%→+13.2% between views. Blind ladder has the same shape → market structure, not our selection. **Coverage limit: FBS-vs-FCS games are 46% of the big-dog universe (149 of 323) and have NO alt ladder**, so this only ever addressed half the play. **WHY IT FAILED WHERE 1H WORKED (the transferable rule): the 1H line applies a CONSTANT ratio (56.8%) to something that genuinely varies (front-loaded dog covers) — a real formula error. The alt ladder is derived from the margin distribution, which books price well (3/7/10/14 are the most-studied numbers in football). A derived line is soft when the derivation holds something CONSTANT that reality varies — not merely because it is derived. Hunt constants, not derivations.** |
| **Anytime TD — DO NOT BET** | `models/td_model.py`, `ingestion/historical_td.py`, `backtest/td_vs_book.py` | Built to add volume (~70 quoted players/game, all season). Model is **well calibrated OOS** (0.140 pred vs 0.141 actual; 0.259 vs 0.258 overall) via red-zone-share × expected-team-TDs → Poisson → logistic recalibration. **But it does not beat the price.** Blend weight fits at 0.05; a market-only control with no model does as well or better (EV≥10%: +31.7% vs +28.9%) picking 95% the same bets; **at MEDIAN prices the model finds ZERO +5% EV bets**; winners sit at 1.67x median price vs a 1.02x typical gap → stale lines/data mismatches. Blind control −16.8% (vig: 35% implied vs 27% actual). 77k lines pulled and kept for research. |
| **Moneyline — DO NOT BET** | `backtest/ml_value_history.py` (8 seasons, REAL prices + line shopping) | **No edge. Model prob −5.0% ROI (1/5 seasons +); 30% blend −6% to −8.6% (0-1/5); "pure line shopping" +12% at EV≥10% is a mirage — +828 avg longshots, seasons +25%/+7.6%/−16.1%.** Probabilities ARE calibrated (65%→63.7%, 74.5%→72.2%, 88.7%→87.4%) but calibration ≠ profit: the market forecasts better and ML vig eats the rest. `ml_value.py` is now INFORMATIONAL ONLY; paper tracker logs ML at **stake 0**; parlay builder **bans ML legs** (MAX_ML_LEGS=0). |
| **1st-half spread** | `features/first_half.py`, `backtest/first_half.py`, `picks/first_half_picks.py` | **NEW BEST PLAY — "BIG-DOG 1H" (2u): take the big-dog selection (full-game \|spread\|≥17, model on dog ≥6) but bet the 1H line → 64.8% ATS / +23.7% ROI, 125 bets, 3/3 seasons. Beats the same games on the full game (62.4%/+19.1%) and the generic 1H edge (57.8%).** Mechanism: big-dog covers are front-loaded (~3.6 pts in 1H vs ~0.9 in 2H) while books set the 1H line at a flat 56.8% of the full spread — a derived-line shortcut. STANDARD 1H (any side, edge≥2, wks 1–5) stays 57.8% / +10.3% at 1u. — derived 1H lines are soft early. Wks 6–15 dead; 1H totals/ML no edge. Live picks via per-event `spreads_h1` (posts near kickoff). 1H lines in `historical_1h_lines.parquet`. |
| Totals sim | `models/game_sim.py` | edge≥5: 54.1% (+3.3%); coach priors worth +2pts |
| Props stack | `features/player_stats.py`, `models/props.py` | priors incl. transfers; beats naive all stats; NB receptions |
| Props vs real lines | `backtest/props_vs_book.py` + **`features/matchups.py`** (cal 2023, OOS 2024+2025) | **UPGRADED 2026-07-30 with offense-vs-defense matchup features: EV≥5% now 54.6% / +3.6% ROI over 1,255 OOS bets, positive BOTH seasons (2024 +4.3%, 2025 +2.8%) — was −0.1% before.** Matchup = CFBD `/stats/season/advanced` pass/rush splits (off ppa/success/explosiveness vs what the opp D allows), percentile-ranked, walk-forward via `endWeek`. Enters the recalibration as `c*(matchup−0.5)*proj`. Then two more upgrades: **gamma tails** for yardage (right-skewed: rec 1.22 / rush 1.07 / pass 0.00 — a normal tail overstates P(over); gamma fixed pass_yds −2.1%→+6.5%) and **rec_yds EXCLUDED from betting** (loses under both distributions, both seasons). **FINAL: EV≥5% → 56.3% / +7.0% ROI over 780 OOS bets (~390/season), positive both seasons; by stat receptions +11.8%, pass_yds +6.5%, rush_yds +6.0%.** **Now BAND-TIERED (2026-07-31)** — EV bands are NON-monotone: 3–5% = 54.2%/+2.2% but wildly inconsistent (2024 +18.8, 2025 −8.8) → **skip**; **5–8% = 59.5%/+12.7%, both seasons + → PRIME 2u**; 8%+ = 54.0%/+3.1%, inconsistent → 1u. Very high "EV" usually signals a stale line or model error, so the top band is staked DOWN not up. Props are the volume engine (~390 bets/szn vs ~70 for spreads). **SEASON-ARC FILTER ADDED 2026-08-02 (see below).** ⚠️ **ALL ROI/UNIT FIGURES IN THIS ROW PRE-DATE THE 2026-08-04 SIGN-BUG FIX AND ARE ~2.2x TOO HIGH — see the 🚨 section above. Corrected: ~227 bets/szn, +23.5 u/szn.** The *shape* of every finding here (band tiering, gamma tails, rec_yds exclusion, matchup features) survived the fix; only the magnitudes were inflated. |
| **Props season arc** ⭐ NEW | `backtest/props_vs_book.py::season_arc()`, `picks/prop_picks.py::current_week()` | **The props edge RISES through the season — and weeks 1–4 have none.** OOS production selection: wk1–4 **−2.3%** (51.0%, negative BOTH seasons), wk5–8 +4.1%, wk9–15 **+16.2%** (61.2%, +14.8/+17.6 by season). Monotone, consistent, and it survives the **median-price control** (wk1–4 −6.1% vs wk9–15 +12.8%) so it is not a line-shopping artifact. Slope +1.44pp ROI/week (t=1.79); permutation test on the wk≥5 vs wk≤4 gap **p=0.028**. The loss is concentrated in the **STANDARD band**: wk1–4 STANDARD −4.6% (both seasons −) vs wk1–4 PRIME +2.6% (n=79, noise). **Change shipped: STANDARD band suppressed before week 5, PRIME kept year-round** — the minimal cut the evidence supports. 390→303 bets/szn, +7.0%→**+10.3%** ROI, 47.7→**51.7 units/szn** (beats cutting all of wk1–4, which gives higher ROI but fewer units: 264 bets/+11.5%/+49.7u). ⚠️ **Unit figures here are pre-sign-bug and ~2.2x too high (corrected +23.5 u/szn); the arc itself is unaffected — and note the bug was CONCENTRATED in weeks 1–5, so it was part of what made early-season props look the way they did. The week-5 STANDARD gate should be re-derived on the fixed data before the season.** **Mechanism (best read, not proven):** our projections are EWMA-weighted on current-season form while the book's prop line is anchored to a season-long average, so our edge GROWS as in-season role changes accumulate; in September neither side has data and the book's prior is at least as good as ours. |
| **Game-script usage shares — REJECTED** | `backtest/script_shares.py`, `models/props.py::SCRIPT_MODE` (left `"off"`) | Hypothesis: a starter's share is not a constant — he sits in a blowout win, gets fed when behind — so the trailing EWMA carries whatever scripts he happened to draw into every projection. Two separable fixes tested: `descript` (divide each PAST game's share by its realised script factor before the EWMA — uses a margin we KNOW) and `full` (also re-script forward by E[factor \| spread]). **OOS 2024–25: off (production) +39.6u, full +34.5u, descript +26.1u → rejected.** ⚠️ **THE MEASUREMENT TRAP that nearly shipped this:** a raw `share / trail_share` ratio makes the effect look huge (RB 1.154 behind → 0.944 won-by-25+, a ~20% band) but that compares **different players** — good teams both blow opponents out and run deeper committees, so most of the swing is **roster composition, not script**. A player-fixed-effects fit (within-player demeaning of log share, quadratic in margin) collapses it to a **~5% band** (`rush beta=[-0.011,-0.002]` per 10 pts → 0.950x at a 30-pt win). Real, far too small to survive pricing. **RULE: for any "X affects Y" claim about players, demean within player before believing the size.** The forward half was always riskier — the spread leaves a **21.1-pt residual sd on margin**, so E[factor\|spread] is weak and risks merely agreeing with the book (cf. the opp-adjusted defense EPA rejection). |
| **Fade the favourite's starters in mismatches — REJECTED** | `backtest/starter_rest.py` | Found by tracing the accidental edge the sign bug was exploiting. Blind P(stat lands UNDER), favourite vs dog: 7–14 **46.7% / 54.4%**, 14–21 **47.7% / 54.6%**, 21–28 54.5% / **58.1%**, 28+ 58.0% (+2.10sd). **DIES ON THE MECHANISM TEST: the DOGS go under more than the favourites at 7–21.** If this were starters resting it would be specific to the favourite — his backups take the snaps. It is not; it is a general "props land under in mismatch games" tendency. Money agrees: flat-betting favourite unders at ≥21 gives 2023 +5.2%, 2024 +14.7%, **2025 +0.0%** — fails the both-OOS-seasons rule, t=+1.30 at 28+ after 5 spread buckets × 2 sides of looking. **TRANSFERABLE RULE: check whether the effect is specific to the side your mechanism names. If both sides show it, the mechanism is wrong even when the headline number is significant.** Our model already leans 78–82% under on favourites there and grades fine, so nothing is left on the table. |
| Calibration | `backtest/confidence_report.py` | **ML is under-confident (stated 70–80% → 83.5% actual); spread/prop raw confidence is inflated** |
| Live picks | `picks/edge_report.py`, `ml_value.py`, `prop_picks.py` | run weekly in-season; CLV log auto-appends; each report leads with **best-edge / best-confidence** callouts |
| Paper tracker | `picks/paper_trades.py` | logs ML + spreads/totals + **props**; settles vs CFBD finals & 2026 player logs (DNP props void); CLV tracked |
| Transfer tier translation | `features/transfer_elo.py` | G5→P4 RB keeps 66% share/86% eff; QBs travel best; **star-RB residual bump +2.0 share pts** |

## 🚨 PROPS EDGE WAS 2.2x OVERSTATED — corrected 2026-08-04 (`backtest/props_stability.py`)
**A sign bug inflated every props number published before 2026-08-04.**
`models/props.py` took `home_id` from play-by-play, missing on 4.9% of
player-game rows. When it came back NaN, `np.where(team_id == home_id, spread,
-spread)` was False for **both** sides, so both teams got the away-side spread —
144 games had their two teams recorded as the same-size underdog. Affected rows
average a **39.7-point spread and 81% sit in weeks 1–5**: the early-season FCS
buy games, where the **favourite was fed to the volume model as a 39.7-point
dog**. Now sourced from the CFBD games table (`models/props.py::game_margins`),
and a failed merge yields NaN instead of a silently flipped sign.

| props edge, 2024–25 OOS | u/szn |
|---|---|
| claimed pre-fix (buggy) | +51.7 |
| **corrected** | **+21.7** (bootstrap median; baseline draw +23.5) |

`backtest/props_stability.py` — 40 bootstrap refits of the team-volume model on
resampled training team-games: median **+21.7, sd 3.5**, 5th %ile +14.3, 95th
+26.4, **P(unprofitable) 0.0%**. **The edge is real and robustly positive, just
half the advertised size.** Two side-findings:
1. **The volume model was fitted on one arbitrary side of each game** —
   `train.groupby("game_id").first()` over *player* rows, so row order decided
   which team trained it. Now one row per team-game
   (`models/props.py::fit_volume_coefs`). Coefficient noise alone is only
   ±3.5 u/szn, so this was **not** what caused the swing — the sign bug was.
2. **The bug was accidentally profitable**: it under-projected the favourite's
   rush attempts by 7.7 in those mismatches, i.e. it was unknowingly betting
   "fade the favourite's RB in a blowout". Tested deliberately and rejected —
   see the starter-rest row below.

## SEASON-ARC GATE RE-DERIVED 2026-08-04 (`picks/prop_picks.py`)
The old gate (`MIN_WEEK_STANDARD=5`, PRIME year-round) was **fitted on the
region the sign bug corrupted** — 81% of bugged rows sat in weeks 1–5. On clean
data, corrected 2024–25 OOS ROI by band:

| | wk1–4 | wk5–8 | wk9–15 |
|---|---|---|---|
| PRIME | **−5.2%** | +14.2% | +11.7% |
| STANDARD | **−4.1%** | −6.2% | +10.9% |

**wk1–4 is now negative in all four band×season cells** (PRIME −7.8/−3.0,
STANDARD −6.2/−3.2). Pre-fix PRIME read +2.6% there and survived only on the
bug's help. STANDARD in wk5–8 splits −39.8%/+22.1% by season = no signal.

| gate | bets/szn | u/szn | per-season |
|---|---|---|---|
| STANDARD wk5+ (old) | 226 | +23.5 | +8.3 / **+38.7** |
| drop all wk1–4 | 189 | +27.4 | +13.8 / +41.1 |
| **drop wk1–4, STANDARD wk9+ (SHIPPED)** | **147** | **+30.0** | **+28.9 / +31.2** |

**Chosen for SEASON BALANCE, not for being the maximum** — the old gate drew 82%
of its units from one season, the new one is near-identical in both. ⚠️ wk1–4
pooled is only t=−0.66 on its own; the case rests on 4-of-4 cell consistency,
not significance. Re-check after 2026. `MIN_WEEK_PRIME=5`, `MIN_WEEK_STANDARD=9`.

## STABILITY AUDIT of the derived-line plays (`backtest/play_stability.py`, 2026-08-04)
Props have fitted coefficients, so `props_stability.py` resamples their training
data. These plays mostly do not — Q1 is deliberately model-free. What *was*
fitted is the **thresholds and week splits**, chosen by looking at the data, so
that is where the overfitting risk lives. Four tests each: threshold sweep,
leave-one-season-out, median-vs-best price, week-block bootstrap.

**The decisive test is the sweep: a structural edge is a PLATEAU, a fitted one
is a SPIKE.** Nothing physical happens at exactly 25.0.

**Q1 PREMIUM — passes emphatically.** Perfectly monotone, and the ramp is
*predicted by the mechanism* (the Q1 line over-grants more as the spread grows
because the dog's deficit saturates), not fitted:
| Q1 cut | bets | win% | ROI | t |
|---|---|---|---|---|
| ≥19 | 232 | 63.4% | +23.3% | +3.75 |
| ≥21 | 188 | 64.9% | +26.4% | +3.86 |
| ≥23 | 142 | 71.8% | +39.7% | +5.36 |
| **≥25 (shipped)** | 94 | 75.5% | +45.7% | +5.27 |
| ≥27 | 70 | 77.1% | +48.0% | +4.91 |
| ≥31 | 34 | 85.3% | +64.1% | +5.33 |

LOSO +37.4/+51.6/+47.7%; each season alone +61.0/+33.1/+41.6%. Median price
+45.7% vs best +49.3% (shopping adds only 3.7%). Bootstrap **+29.4 u/szn, 90%
CI [+11.0,+49.1], P(losing season) 0.3%.** **The biggest contributor is also the
most robust play** — the opposite of what the props audit found.

| play | sweep | LOSO | median price | bootstrap u/szn |
|---|---|---|---|---|
| Q1 ALL ≥17 | monotone 13→21 | +21.2/+19.4/+19.3 | +20.0% (best +23.2) | +20.2, CI [+5.6,+35.0] |
| BIG-DOG 1H wks1–5 | plateau from 17 | +24.2/+27.8/+22.6 | +24.8% (best +26.8) | +21.4, CI [+10.4,+32.1] |
| SPREAD PREMIUM | flat +22–27% over 19–27 | +28.1/+22.9/+23.7 | **+25.0% = best** (flat −110, no shopping component at all) | — |

### ⚠️ THE WEAK LINK: BIG-DOG 1H weeks 6–15 (the 2026-08-03 extension)
**n=49 bets total across 3 seasons.** Bootstrap **+8.6 u/szn, 90% CI
[+0.7,+17.6]** — lower bound barely clears zero. Every LOSO t-stat < 2.4;
individual seasons have <25 bets. **The BLIND version has a HIGHER t than the
filtered one** (+2.47 vs +2.24) — the model filter lifts ROI (+29.0% vs +22.9%)
but on half the sample. Positive in every view and the mechanism matches the
plays that work, so **kept — but it is the first thing to cut if the season
starts poorly, and must not be sized as if it were as solid as Q1.**

### Found but deliberately NOT shipped
**Q1 at ≥29 returns +57.3% and ≥31 returns +64.1%** — tempting as a 3u
"super-premium" tier. **Do not add it.** n=34–40 is the thinnest slice in the
book, and staking UP on the smallest sample of a **low-limit market** is exactly
where the $50 cap bites hardest. The monotone ramp is real; the right response
is confidence in ≥25, not a tier mined from the tail.

## EV THRESHOLDS RE-DERIVED + new PROBE tier (`backtest/props_ev_sweep.py`, 2026-08-04)
Applying discipline rule 10 (sweep every grid-searched hyperparameter like a
threshold) to the props EV cuts, which had never had that test.

**The reassuring half:** ROI is a genuine **PLATEAU** across EV 0.02–0.05
(+10.8% to +14.4%). That partly offsets the blend-weight fragility above — the
**selection rule is sound**, only the probability **scale** is uncertain.

**The actionable half:** the old 0.05 floor was justified by "3–5% band = +2.2%,
2025 −8.8%, unreliable" — measured on **sign-bug-corrupted data**. Clean OOS
sub-bands (weeks 5+, 2+ books):
| EV band | bets | ROI | t | 2024 | 2025 |
|---|---|---|---|---|---|
| 0.02–0.03 | 121 | −1.6% | −0.19 | +7.1% | **−10.2%** |
| 0.03–0.04 | 99 | +11.8% | +1.26 | +37.3% | **−4.1%** |
| **0.04–0.05** | 98 | **+21.4%** | **+2.32** | **+29.1%** | **+14.8%** |
| 0.05–0.06 | 84 | +18.1% | +1.78 | +24.9% | +12.0% |
| 0.08–0.12 | 113 | **−5.9%** | −0.66 | −15.9% | +2.6% |

The 0.04–0.05 slice grades **better than the band above it** and is positive in
both seasons; 0.03–0.04 fails 2025, so **the floor stops at 0.04 — set by the
both-seasons rule, not by maximising units.** Paired week-block bootstrap of the
change: **+21.1 u/szn, 90% CI [+6.7, +35.4], P(gain)=99%**.

**SHIPPED as its own PROBE tier at 1u, not folded into PRIME at 2u.** Two
independent reasons: (a) it is the least-validated slice in the book — two
seasons, and only visible after today's sign-bug fix; (b) the bankroll maths
agrees — 0.05→0.04 at 1u buys +$46 of median for +0.9pp P(losing season), at 2u
a further +$45 for +1.3pp, a strictly worse trade on the newest evidence.
Keeping it a **separate tier means 2026 results can grade it on its own** and
promote or drop it without contaminating PRIME.

Also confirmed and left alone: **`n_books>=2` is optimal** (+30.0 u/szn vs +27.6
at ≥1, +15.0 at ≥3), and **the non-monotonicity is real** — EV 0.08–0.12 grades
−5.9%, so staking the top band DOWN stays correct.

⚠️ **Three separate filters turned out to have been fitted on the corrupted
region today** (the week gate, PRIME-year-round, the EV floor). Assume there are
more: **any threshold tuned before 2026-08-04 is unverified until swept.**

## Bankroll projection — $1,000 start (`backtest/bankroll_sim.py`, **recomputed 2026-08-04**)
De-duplicated book (one derived bet per game — Q1/1H/spread all ride the same
dog, plus 2-team teasers): **409 bets/season, +126.9 units/season** after the
2026-08-05 prop-line pull widening (was 354 / +110.5 on the smaller sample; was
432 / +114.7 pre-fix, but
that number was inflated; this one is clean *and* better-gated).
| play | bets/szn | win% | ROI | units/szn |
|---|---|---|---|---|
| Q1 PREMIUM (2u) | 31 | 75.5% | +45.7% | **+28.6** |
| PROPS PRIME (2u) | 96 | 59.4% | +12.7% | **+24.4** |
| BIG-DOG 1H (2u) | 35 | 65.4% | +25.4% | +17.6 |
| SPREAD PREMIUM (2u) | 18 | 71.7% | +36.9% | +13.0 |
| **PROPS PROBE (1u)** | 49 | 64.3% | **+21.4%** | **+10.5** |
| PROPS STANDARD (1u) | 52 | 58.3% | +10.9% | +5.6 |
| **TEASER 1H (1u)** | 27 | **81.2%** | +26.4% | **+7.0** |
| Q1 STANDARD (1u) | 39 | 56.8% | +9.4% | +3.7 |
| SPREAD STANDARD (1u) | 7 | 52.4% | −0.0% | **0.0** — redundant after dedupe |

Week-block bootstrap (resamples whole weeks, preserving same-game/same-week
correlation), Q1 stakes capped at $50 for its low limits. **Haircut = share of
the backtested edge assumed not real; 50% is the base case** (overfitting, line
movement, limits, decay). At **1u = 2% = $20**:
**Props take `PROPS_EXTRA_HAIRCUT = 0.20` ON TOP of the base row**, because their
edge is *measurably* less certain than the audited derived-line plays (see the
pricing-fragility section below). So the "50%" row is 50% on Q1/1H/spread and
70% on props.
| haircut | median | 5th %ile | P(loss) | median max DD |
|---|---|---|---|---|
| 0% | $2,876 | $2,077 | 0.0% | −4.8% |
| 30% | $2,278 | $1,563 | 0.2% | −8.4% |
| **50% (base)** | **$1,881** | **$1,188** | **2.0%** | **−11.0%** |

**TEASER INTEGRATION — the assumption was wrong.** "replace" (teasers instead of
the 1H singles whose legs they use) looked disciplined and is the WORST option:
| mode | u/szn | median | P(loss) |
|---|---|---|---|
| off | +103.5 | $1,814 | 2.8% |
| replace | +97.6 | $1,748 | 3.0% |
| **add (SHIPPED)** | **+110.5** | **$1,883** | **2.1%** |
Capital efficiency, not edge: a 1u teaser covers TWO legs (~0.13 u/leg) vs the
2u single's 0.51 u/leg — swapping shrinks the book. "add" carries real extra
exposure on those games and the week-block bootstrap prices it in (both
outcomes land in the same resampled week); it still wins because an 83.8% bet
lifts the median AND the 5th percentile. ⚠️ Modelled optimum keeps improving to
3u (+124.6 u/szn, P(loss) 1.4%) — **shipped at 1u anyway**, same principle as
PROPS PROBE: n=80 tickets, price and availability both unconfirmed.
**⚠️ The 0% row's P(loss) is an artifact of treating a measured edge as
certain — never quote it as a forecast.** Recommended: **1u = 2% ($20)**.
Per-season at 0% haircut: 2024 416 bets +135.2u, 2025 409 bets +121.1u.

## 🚨 PROPS PRICING IS THE FRAGILE PART (`props_pricing_stability.py`, `props_blend_sweep.py`)
`props_stability.py` audited the team-VOLUME model and found it contributes only
±3.5 u/szn. That was not the scary parameter. **recal (a/b/sigma), NB dispersion
r and the market-blend weight w are all fitted on the 2023 rows alone**, and
sigma+w decide which bets clear EV≥5% at all. 60 week-block bootstraps, refitting
all three **together** (they are coupled):

| | u/szn |
|---|---|
| median | +27.0 |
| **sd** | **15.6** (volume model: **3.5**) |
| 5th–95th %ile | +1.4 to +48.7 |
| bet count | median 414, **range 33–642** |

**RE-MEASURED on the +26% dataset (2026-08-05): sd 15.6 → 14.4 (−8%), and the
3-season-vs-1-season comparison STILL gives exactly 10%.** Prediction held: more
data shrinks the sampling component roughly as sqrt would predict (26% more data
→ 8% less sd) but the LEVEL stays ~4x the volume model's 3.5, because the
grid-search component does not shrink with data. **The tail improved far more
than the spread: 5th %ile +1.4 → +11.6 u/szn and P(unprofitable) 3.3% → 0.0%.**
That is the real payoff from the pull — not a tighter distribution, a much
better worst case.

2024 alone goes unprofitable in **13.3%** of draws. **More data barely helps** —
a 3-season calibration cuts sd only 15.6→14.1 (**10%**, vs the ~42% sqrt(3)
predicts), so it is not a sample-size problem.

**`props_blend_sweep.py` found the cause: w is a COARSE GRID SEARCH**
(`np.arange(0, 0.65, 0.05)`) and the book is wildly sensitive to the grid point:
| w | bets/szn | units/szn | 2024 | 2025 |
|---|---|---|---|---|
| 0.25 | 92 | **+0.4** | **−8.0** | +8.3 |
| **0.30 (fitted)** | 147 | **+30.0** | +14.4 | +15.6 |
| 0.35 | 203 | **+54.2** | +30.9 | +23.3 |

**One grid step swings the season 30–54 units and the fitted value sits directly
beside a near-zero outcome.** The opposite of the Q1 threshold plateaus.

**Tested and REJECTED as a fix:** requiring EV≥5% at *every* w in
{0.25,0.30,0.35,0.40}. Lower w pulls toward the market so it is strictly the
binding constraint — the "robust" intersection is just the w=0.25 book
(+0.4 u/szn). **Do not chase w=0.35 either — that is fitting the grid harder.**
Log-loss is a proper scoring rule and stays the principled choice.

### ✅ RE-SWEEP AFTER THE PULL — all five thresholds HELD (2026-08-05)
`backtest/props_resweep.py` (new, repeatable — run after ANY data change).
Shipped config on the +26% dataset: **252 bets/szn, +15.2% ROI, +56.9 u/szn,
2024 +57.9 / 2025 +55.9.** Every fitted threshold survived independent data:
| threshold | shipped | best alternative | verdict |
|---|---|---|---|
| EV floor | 0.04 → +56.9u | 0.03 → +60.6u | **keep 0.04** |
| PROBE/PRIME | 0.05 → +56.9u | 0.045 → +62.9u | **keep 0.05** |
| PRIME/STANDARD | 0.08 | — (0.08 wins) | ✅ |
| week gates | wk5/wk9 | — (wins, best balance) | ✅ |
| min books | ≥2 → +56.9u | ≥1 → +65.2u | **keep ≥2** |

**Why the three richer alternatives were declined:**
- **EV 0.03**: the pooled total improves but the MARGINAL band 0.03-0.04 grades
  +6.1% with **2025 at −1.2%** — fails a season. Same stopping rule that set
  the floor originally. (0.04-0.05 re-earns its place: +14.2%, both seasons +.)
- **PROBE 0.045**: a 0.005 move worth +6u, justified by splitting a ~119-bet
  band into ~60-bet halves. Noise-mining.
- **n_books≥1**: the marginal single-book bets are ~102/szn for ~8.3u, i.e.
  **~+2.4% ROI against a +15.2% book average** — the least reliable lines (no
  consensus to check staleness against) and the least placeable.

⚠️ **CORRECTION to the 2026-08-04 entry: EV 0.08-0.12 is now +4.3%, not −5.9%.**
The non-monotonicity is milder than reported — the top band is no longer
NEGATIVE, just clearly worse than the 0.05-0.06 sweet spot (+21.5%, t=+2.30).
Staking it down at 1u stays right; calling it a losing band does not.

### A SECOND CALIBRATION SEASON — tested, marginal (2026-08-05)
There is no earlier season to buy: **The Odds API historical props start May
2023**, so 2022 does not exist. The only free way to add calibration data is
WALK-FORWARD — price each season on everything that preceded it (2024 on 2023,
2025 on 2023+2024). `backtest/props_vs_book.py::walkforward`,
`backtest/props_walkforward_ab.py`. 2025 is the only season that differs:
| 2025 | bets | ROI | units | cal MAE |
|---|---|---|---|---|
| 1 cal season | 206 | +12.1% | **+39.0** | 30.979 |
| 2 cal seasons | 132 | **+13.5%** | +28.3 | **30.934** |
**Better per bet, more conservative, fewer units.** The MAE gain is **0.15%** —
confirming OUT-OF-SAMPLE what props_pricing_stability.py saw in-sample: the
fragility is a **coarse-grid-search artifact, not a sample-size problem**. Not
adopted for the backtest (costs the 2023 season and loses units on the only
comparable season). **Production is unaffected — `fit_production()` already
calibrates on all three seasons.**

✅ **DONE 2026-08-05 — pull widened, +26% data for 9,288 credits (0.19% of
budget; I had estimated 40-60k, so the estimate was 5x high).** Saturday window
12h→24h plus dedicated Thu/Fri 22:00Z snapshots; snapshots/week 1 → 2-4.
Rows 119,961 → **151,464**, events 1,066 → **1,342**, players 1,840 → 2,243.
Props went **+40.5 → +56.9 u/szn from data alone, no model change.**
⚠️ We now REQUEST 55-67 Saturday events but only ~29 return prop markets —
**books simply do not quote props on the smaller half of the slate**, so the
gain came from weeknight games and late kickoffs, not from the wider Saturday
window. That caps how much more coverage is buyable.
⚠️ **A guard I added silently dropped a week**: `MIN_GAMES_PER_DAY=2` killed
2023 wk15 (Army-Navy plays ALONE). Caught only by diffing week coverage, not
row counts — rows were +25% and looked unambiguously good. Fixed to 1.
**Filters that skip "oddities" are exactly where real data goes missing.**

**Historic note — the original finding:** We pull **ONE snapshot per
week** (main Saturday 14:00Z) and keep only events kicking off within **12
hours**, giving **~24 events/week of the ~60+ FBS plays** — Thursday/Friday
games and late West Coast kickoffs are missed entirely. Current spend was
~1,066 events x 4 markets x 10 credits = **~42.6k**. Widening `WINDOW_H` 12→24
and adding a Thu/Fri snapshot could add 30-50% more calibration rows for
**~20k credits — 0.4% of the 4.95M remaining.** That is the cheapest real
increase in props calibration data available.

### ✅ BLEND WEIGHT FIXED 2026-08-05 — stop estimating it
Called a "coarse grid" for two days; **measuring the objective proved otherwise.**
On the 2023 rows (n=4,048): continuous argmax **w=0.310** (0.05 grid picks 0.30
— no real difference), and moving ±0.05 costs 0.523 log-loss out of ~2,781 =
**0.000129 PER ROW**. The surface is nearly FLAT: w is barely identified while
the book is violently sensitive. A finer grid buys precision on noise.

`backtest/blend_mode_ab.py`, 40 bootstrap refits per mode:
| mode | u/szn | 2024 / 2025 | **boot sd** | 5th %ile |
|---|---|---|---|---|
| grid (old) | +62.7 | +66.0/+59.3 | **15.3** | +11.3 |
| **fixed w=0.30 (SHIPPED)** | +56.9 | **+57.9/+55.9** | **7.0** | +32.8 |
| average 0.20–0.45 | +63.1 | +84.2/+42.0 | 7.7 | +40.1 |

**Estimating w cost >half the stability for nothing.** Clincher: grid at 0.01
picks **0.31** vs fixed **0.30** — that ONE HUNDREDTH moves the season by
**5.8 units and 30 bets**. "average" wins on units but splits 84.2/42.0 by
season and needs a NEW discretionary `BLEND_BAND`; rejected on season balance.
⚠️ **`bankroll_sim` does NOT show this gain** — it bootstraps outcomes given a
FIXED bet list, while the fix reduces uncertainty in *which bets exist*. Median
stays $1,956; the forward uncertainty behind it is lower.
**RULE: before refining an estimator, measure whether the objective can
IDENTIFY the parameter. A flat surface means precision buys noise.**

### Shipped in response
1. **`fit_production()` in `backtest/props_vs_book.py`.** The LIVE coefficients
   were fitted on **2023 only** while 2024–25 sat unused — a *backtest*
   constraint (keep 2024-25 OOS) had leaked into *production*, where it buys
   nothing. `__main__` now runs the backtest with `write_coefs=False`, then
   refits on all three seasons and writes those for live picks.
   ⚠️ That production book is IN-SAMPLE; never quote its ROI.
2. **`PROPS_EXTRA_HAIRCUT = 0.20`** in `bankroll_sim.py` (above).

**RULE: a hyperparameter fitted by grid search deserves the same
plateau-vs-spike sweep as any threshold.**

**Net of the bug fix AND the re-derived gate, the projection is barely lower
than the old inflated one on the median and STRICTLY BETTER on risk:**
median $1,989 → $1,849, but 5th %ile **$1,183 → $1,254**, P(losing season)
**2.1% → 1.0%**, typical drawdown **−13.0% → −9.1%**. Fewer, better bets.
**Q1 PREMIUM is now the largest single contributor** (+28.6 u from 31 bets) and
is limit-constrained; the $50 cap is the assumption this projection is most
sensitive to. If books cut Q1 to $25 the base case drops toward $1,600 —
**find out early which books take size on Q1 lines.**

## PARLAYS + COMPOUNDING — the growth plan (`backtest/growth_paths.py`, 2026-08-04)
Built to answer "can $1,000 reach $5,000 in a season" with real graded outcomes.

**Parlays are genuinely +EV — the edge MULTIPLIES**: `EV=(1+EV1)(1+EV2)-1`, so
two +45.7% Q1 legs price near +112%.
| | bets/szn | win% | ROI | t | per-season |
|---|---|---|---|---|---|
| **2-leg (recommended)** | 38 | 47.8% | **+77.3%** | +4.43 | +103/+82/+86% |
| 3-leg | 23 | 32.9% | +130.2% | +3.28 | +234/+154/+99% |

**Composition — Q1 is the best ingredient** (our best single squared):
| composition | /szn | win% | ROI | t |
|---|---|---|---|---|
| **Q1 PREM × Q1 PREM** | 19 | 55.9% | **+103.8%** | +4.34 |
| Q1 PREM × any derived | 31 | 50.0% | +84.5% | +4.39 |
| any derived × any derived | 116 | 43.1% | +59.4% | +6.02 |
| 1H × 1H | 23 | 37.1% | +37.3% | +1.73 |

All positive in all 3 seasons. **`picks/parlay_builder.py` never gathered Q1
legs until 2026-08-04** — fixed (PREMIUM tier only), `MAX_LEGS` cut 6→4.
⚠️ **Legs are NOT independent** — weekly win-rate variance is **1.14×** what
independence predicts, so true joint prob is slightly BELOW p1×p2; the `PROB`
haircut is load-bearing. ⚠️ **The 116/szn row is not a licence to bet that
many** — it re-uses each leg ~3×, so one loss takes down several parlays AND
the single on that game.

**COMPOUNDING SHIPPED** (`picks/paper_trades.py`): stakes are now a % of the
CURRENT bankroll. `UNIT_PCT=0.04`, `current_bankroll()`, `unit_usd()`, and a
`unit_usd` column recorded per row **at placement time** so history stays
gradeable as the unit moves. New CLI: `bankroll`, `backfill-units`.

| strategy | median | 5th %ile | P($5k) | P(ruin) |
|---|---|---|---|---|
| flat 2% (old plan) | $1,814 | $1,121 | **0.0%** | 0.1% |
| compound 4% | $2,644 | $651 | 20.8% | 1.4% |
| **compound 4% + 2-leg parlays** | $3,916 | $733 | **46.6%** | 1.6% |
| ↑ with Q1 limit at $200 | **$4,939** | $802 | **57.0%** | 1.5% |
| compound 8%, no haircut | $18,660 | $250 | 82.8% | **29.8%** |

🚨 **THE HIGHEST-VALUE ACTION IS NOT MODELLING — IT IS Q1 LIMITS.** P($5k) runs
**42.1% at a $25 cap, 46.6% at $50, 57.0% at $200**, 57.9% at $500. Spread Q1
across ~4 books to reach $200; past that it stops mattering. Do this before
2026-08-29.

⚠️ **8% staking busts 3 times in 10** — that is where aggression stops, because
busting ends the compounding the whole plan depends on.
⚠️ **"70-80% hit rate" and parlays are mathematically opposed.** Q1 PREMIUM
already hits 75.5% but is 31 bets/szn and cannot 5x alone; parlays buy edge by
SPENDING hit rate (2-leg ~51%, 3-leg ~33%). No configuration gives both.

## ⭐ TEASERS — the high-HIT-RATE play (`backtest/teasers.py`, 2026-08-04)
Parlays raise EV by SPENDING hit rate. Teasers do the opposite: buy points, hit
rate goes UP. That is unusually cheap for us **because our edge IS a saturation
effect** — the dog's sub-period deficit plateaus near 4 pts however big the full
spread — so the distribution we buy points against is already compressed.
**Teasers are the natural instrument for a saturation edge.**

Validated on **REAL same-week, different-game pairs** (not an independence
assumption — our legs run 1.14x correlated):
| tease | n | TRUE hit | EV | t | by season |
|---|---|---|---|---|---|
| 1H +6.0 | 80 | 71.2% | +36.0% | +3.71 | 64/71/78% |
| 1H +7.0 | 80 | 75.0% | +32.7% | +3.79 | 72/75/78% |
| **1H +10.0 (SHIPPED)** | 80 | **83.8%** | **+30.3%** | **+4.69** | **80/82/89%** |
| Q1 +7.0 | 37 | 83.8% | +48.2% | +4.44 | 86/75/91% |
| Q1 +10.0 | 38 | 92.1% | +43.3% | +6.28 | 86/92/100% |

**The independence assumption HELD** (83.8% true vs 82.8% assumed) — at 91% per
leg there is little room left for correlation to bite, unlike parlays.
**Margin of safety is huge:** at 83.8%, a −180 price needs only 64.3%, so the
play survives down to about **−400**. Nothing else in the book tolerates that.

Shipped: `picks/teaser_picks.py` (report) + `paper_trades import-teasers`.
Legs are stored as JSON in a new `legs` ledger column and each is graded
against **its own teased number** — the parlay settler cannot be reused,
because it matches leg text against settled singles and a teased line is not a
line any single was placed at. Settlement math unit-checked (5/5 cover cases,
push-voids-ticket).

⚠️ **AVAILABILITY IS THE BINDING CONSTRAINT, NOT THE EDGE.** Confirm the book
(a) offers 1H teasers, (b) allows teasing a line this large, (c) quotes at or
better than break-even. **Q1 teasers are the better play but are rarely
offered** — treat those rows as the ceiling, not a placeable bet. Q1 +6.0 also
splits 93/58/58% by season on n=38, so do not take small Q1 teases.
⚠️ **Teasers are NOT in `bankroll_sim` / `growth_paths` yet**, so the $5k
projection excludes them (conservative on return, but it also means the added
leg-correlation with the 1H singles is unmodelled). Do not bet the same legs
as singles *and* teasers without accounting for the doubled exposure.

### FULL-GAME teasers — REJECTED, and the rejection confirms the theory
Full-game teasers are the MOST widely offered version, so they were the most
placeable candidate. **Our own derived-line rule predicted they would be weak:
full-game margin is UNBOUNDED, so the distribution you buy points against is
wide, not compressed.** Confirmed:
| | leg hit | EV |
|---|---|---|
| 1H +10 (margin SATURATES) | 91.0% | **+30.3%** |
| Full-game +10 (UNBOUNDED) | 84.5% | +11.2% |

**The saturating market pays ~3x more for the same 10 points.** On real
same-week pairs only +10 is positive in all 3 seasons (t=+1.77, n=52); +6.0,
+6.5 and +7.0 each FAIL a season with t<1. **Not shipped** — logged as a
paper-trial candidate. **Transferable: hunt teasers in BOUNDED sub-period
markets, not full-game ones.**

### Sub-period TOTALS — thesis confirmed, but PRICE is the constraint
`backtest/total_teasers.py`. Residual sd and what +6 buys:
| market | resid sd | +6 buys | +6 UNDER leg | 2-team EV @std price |
|---|---|---|---|---|
| **Q1 total** | 6.99 | **0.86sd** | 81.1% | +25.9% |
| **1H total** | 10.74 | 0.56sd | 70.0% | ~break-even |
| **Full total** | 15.94 | 0.38sd | 63.1% | **−19.5%** |

Monotone in boundedness — full-game total teasers strongly NEGATIVE where Q1
is strongly positive. **⚠️ THE REFINEMENT: boundedness tells you where points
are worth MOST; it does NOT mean the book SELLS them cheaply.** Full-game
*spread* teaser prices are not a real quote for a Q1 *total*. Break-evens
(UNDER): **−194 (+6), −240 (+7), −472 (+10)** 2-team; **−429 / −975** as single
alt-lines. **Needs a new pull** — our Q1 totals data has exactly one point per
game/book/side, so no alt lines to test. **Only the 1H SPREAD teaser survives
real pricing**, because books apply standard teaser prices to it.
⚠️ **ZERO-FLOOR ARTIFACT** — teasing an OVER down on a small total becomes
unloseable (+10 → line ≤0 on 33.8% of Q1 games, +14 → 97.7%). Arithmetic, not
edge. Only the UNDER side is meaningful.

### VACATED SHARE — real 38% effect, but REJECTED in betting (2026-08-05)
`models/props.py` uses the portal ONLY for ARRIVALS. **Departures just delete
the stale prior row — the vacated share evaporates**, so a returning WR2 whose
room lost its #1 target keeps a prior computed while that player was there.

Measured (wks 1-4, returning players, prior_tgt_share>0.05, n=6,140):
| team vacated tgt share | n | actual/prior | actual | prior |
|---|---|---|---|---|
| 0–39% | 1539 | **0.844** | 0.089 | 0.101 |
| 39–54% | 1537 | 0.915 | 0.094 | 0.094 |
| 54–66% | 1533 | 1.003 | 0.094 | 0.087 |
| 66–99% | 1531 | **1.169** | 0.100 | 0.078 |

**38% monotone swing** (corr 0.176). The tell is the last two columns: **actual
share is FLAT (0.089→0.100) while the PRIOR falls steeply (0.101→0.078)** — the
outcome is fine, the prior is wrong. Matters most in **weeks 1-4, which we skip
entirely**, i.e. exactly where the season prior does all the work. This is the
most plausible route to making early-season props bettable.

❌ **REJECTED 2026-08-05.** Fitted on 2023 (slopes tgt +0.630, rush +0.762,
QB +0.755 — consistent). `backtest/vacated_ab.py`, identical pricing:
| variant | bets | ROI | u/szn | 2024 | 2025 |
|---|---|---|---|---|---|
| **off (production)** | 393 | **+14.4%** | **+40.5** | +42.0 | +39.0 |
| on | 398 | +10.0% | +24.2 | +37.7 | **+10.7** |
**−16 u/szn.** MAE barely moves (rush 30.88→30.75, pass 75.54→**76.75 worse**):
the adjustment MOVES projections without making them more ACCURATE, and props
selection is exquisitely sensitive to that. Early season does not rescue it —
pooled ROI +4.1%→+4.7% but the season split WORSENS (2024 +0.8% → −2.7%).
⚠️ **Trap avoided:** raw multipliers were ALL <1.0 (0.53–0.74) because the
intercept carries a big LEVEL correction (priors over-project share 25–47%).
Shipping it would double-count the `actual ~ a + b*proj` recal that
props_vs_book already fits. **Only the slope was kept, normalised to mean 1** —
change the SHAPE, not the level. `VACATED_MODE="off"`; feature + A/B harness
kept for a 4th-season revisit.

✅ **Rosters pulled 2026-08-04 and worth keeping regardless.** `run_ingest.py` now pulls
`/roster` for 2022-2026. Cost: **5 CFBD calls (5/1000 this month)**.
| source of "who left" | vacated targets captured | corr with truth |
|---|---|---|
| portal only | 17% | 0.188 |
| **roster** | **97%** | **0.969** |
Name matching verified: **98.1%** of 2024 log players appear in the 2024
roster; real retention is 43.4% of >5%-share contributors — portal-era churn,
not a matching artifact.
🚨 **`roster_2026` came back EMPTY — not published yet. RE-PULL BEFORE WEEK 1**
or vacated share cannot be computed for the live season. Added to the August
refit checklist below.
⚠️ **Do NOT apply a naive 1/(1-vacated) boost** — mean vacated is 0.678 (a 3.1x
implied boost) against a measured swing of just 1.38x; returning players absorb
only part of the vacancy, freshmen and incoming transfers take the rest. Fit the
adjustment on the train season.

### ALT Q1/1H TOTAL PULL — scoped (2026-08-04)
**Credits are NOT the constraint: 46,014 used of 5,000,000 (4.95M left).** A
full alt-totals pull mirrors the Q1 totals pull (~24k rows, ~25k credits) =
**0.5% of budget.** The real uncertainty is whether the market EXISTS —
The Odds API documents `alternate_spreads`/`alternate_totals` (full game) and
period markets `totals_q1`/`totals_h1`, but `alternate_totals_q1` is not
confirmed. **Scope: spend ~20 credits on a SINGLE-EVENT probe first**
(`ingestion/historical_q1.py` already takes a market argument), and only commit
the 25k if the probe returns rows. Target numbers to beat are the break-evens
in the total-teasers section: −429 (+6) and −975 (+10) as single alt-unders.

### Props volume model + game total — REJECTED
Volume model R² is only ~0.20 (trailing attempts + spread). Adding `book_total`
and `|spread|` moves it to 0.191 / 0.223 (rush / pass) — **+0.005 R² for the
loss of 35% of rows** (only 65.4% of team-games carry a book total). Bad trade.
Coefficients point the right way (total +, |spread| − i.e. lopsided games run
clock), the effect is just negligible. **Team attempt counts are close to
irreducibly noisy at R²≈0.20 — that is a ceiling on props projection quality,
and part of why props are near-efficient and the PRICING layer dominates.**

## SAME-GAME Q1 + 1H — correlation as an ASSET, not just a risk
We have always treated the Q1/1H overlap as a hazard ("never bet both"). That
is right for two *separate* bets. As a **single same-game parlay** the sign
flips: Q1 leg 74.3%, 1H leg 66.2%, **both 56.8% vs 49.2% if independent —
a +7.5% correlation lift**. A book pricing it as independent quotes ~2.03
decimal, where the true rate is **+15.3% EV**. Most books correlation-adjust
same-game combos, so **verify the actual quoted price** — this is the size of
the prize if you find one that does not, not a confirmed play.

## Weekly in-season routine (season opens 2026-08-29)
```
# ⏱️ RUN SPREAD PICKS SUNDAY/MONDAY — betting the OPENER beats the close in
# every season tested (PREMIUM 65.9% vs 62.9%; STANDARD 60.0% vs 58.2%).
python -m ingestion.run_ingest --all           # Tue: data refresh
python -m ingestion.scrapers.ourlads_depth     # Tue: depth snapshot
python -m ingestion.news_injuries              # daily-ish: injuries + Claude extraction
python -m picks.edge_report                    # Wed: spreads/totals board + CLV log
python -m picks.ml_value                       # Wed: ML value picks
python -m picks.q1_picks                       # Fri/Sat: Q1 dog, big mismatches (ALL season)
python -m picks.first_half_picks               # Fri/Sat: 1H spreads (wks 1-5!)
python -m picks.prop_picks                     # Fri/Sat: props post late
python -m picks.paper_trades import-ml         # log the week's ML picks
python -m picks.paper_trades import-edges      # log flagged spreads/totals
python -m picks.paper_trades import-props      # log flagged props (Fri/Sat)
python -m picks.paper_trades import-1h          # log flagged 1H spreads (wks 1-5)
python -m picks.paper_trades import-q1          # log Q1 big-dog plays (all season)
python -m picks.teaser_picks                    # 2-team 10pt teasers on 1H big dogs
python -m picks.paper_trades import-teasers     # log them (83.8% hit, +30.3% EV)
python -m picks.paper_trades bankroll           # current bankroll + unit size
python -m picks.ml_spread_picks                 # ML early-season big-dog spreads
python -m picks.paper_trades import-ml-spread   # paper-trial ML vs linear spread
python -m ingestion.scrapers.splits_capture     # (auto Wed/Fri/Sat) DIY splits
python -m picks.parlay_builder                  # 2-6 leg parlay ladder
python -m picks.paper_trades import-parlay      # log the recommended parlay
# Sun/Mon settlement — prop grading needs fresh 2026 player logs first:
python -m ingestion.bulk_pbp 2026              # refresh 2026 play-by-play
python -m features.player_stats                # rebuild player game logs
python -m picks.paper_trades settle            # grade + CLV (DNP props void)
python -m picks.paper_trades report            # season P&L summary
```
Paper ledger: `warehouse/paper_trades.parquet` (flat 1u stakes). Transfer
tier multipliers: `features/transfer_elo.py` (fitted: G5→P4 RBs keep 66%
share/−14% eff; QBs travel best; P4→G5 gets a usage bump) — auto-applied in
player priors via model_coefs.json. Coach deployment: `features/coach_usage.py`
(usage concentration is NOT coach-sticky, r≈0.1–0.2 — talent drives touches).
**Star-RB effect** fitted as a *residual* in `transfer_elo.py`: the raw
team-level "+13 pts with a star" was mostly selection bias; the causal bump
above what prior share predicts is **+2.0 share pts** (stored in
model_coefs.json `star_rb`, applied in `models/props.py` player_priors).

## Model coefficients — now automated
All fitted coefficients live in `warehouse/model_coefs.json`, written by the
fitting backtests and auto-loaded by `picks/` (in-file values are fallbacks
only). **The August refit is a registered Windows scheduled task**: "CFB
August Refit", weekly Mondays 09:03 starting 2026-08-17, runs
`python -m scripts.august_refit` (logs to `warehouse/refit_task.log`,
reports to `reports/refit_<date>.md`). It pulls 2026 returning
production/rosters/portal when published (fails gracefully + retries weekly
until they are), re-scrapes staff pages, rebuilds coach DB and depth
snapshot, and re-fits everything. **The weekly refit report now leads with
the big-dog spread picks** (short & sweet: "pick Toledo spread vs Michigan
State (+11.5) — edge 20.3 pts") via `big_dog_picks()`. Delete the task after
the season starts: `Unregister-ScheduledTask -TaskName "CFB August Refit"`.

## Loss post-mortem — WHY bets lose (`backtest/loss_review.py`, 2026-08-03)
- **PROPS losses are VOLUME, not efficiency, and not defense.** The projection
  is share × team-volume × efficiency, so a miss decomposes exactly. On losing
  bets the volume term is **64% of the miss for rush_yds and 81% for pass_yds**.
- **Team volume is projected almost perfectly** — team rush attempts −3.4%,
  team pass attempts +0.1%. The `trail_att + spread` volume model is fine.
- **The entire error is the player's SHARE**: trailing share understates
  realised share by **+18.2% (rush), +14.5% (QB), +21.7% (targets)**. Books only
  quote props for players they expect to be featured, while an EWMA over prior
  games includes the committee/injury games the quoted set selects against.
- **ROOT CAUSE OF THE SEASON ARC.** The share bias DECAYS with sample —
  1.448× at ≤2 prior games → 1.210 → 1.092 → 1.075× at 8+; by week 1.339 →
  1.197 → 1.078 — and ROI rises in lockstep (+1.3%→+11.6%; −2.3%→+16.2%).
  **The props season arc is not a calendar effect, it IS trailing-share
  convergence.** Two findings from this session are one finding.
- **A uniform share correction CANNOT help**: `proj_cal = a + b*proj`, so
  scaling every projection by k just remaps b→b/k and changes no bet. Only the
  *varying* part of the bias is exploitable — which is what the week filter
  already harvests. Don't "fix" the level and expect a gain.
- **DEFENCE DOES NOT EXPLAIN PROP LOSSES.** Losers vs winners faced identical
  defences: pass_def_epa t=**0.00**, rush_def_epa t=−0.01, sack_rate t=−1.02,
  pass_havoc t=−1.74 (and pointing the *wrong* way), expl_pass_allowed t=+0.96.
  Overs by opponent sack-rate quartile are non-monotone (59.6/55.3/52.2/63.8%)
  — the HIGHEST sack-rate quartile is the best for overs. Consistent with the
  earlier result that wiring opp-adjusted defence into props HURT. **Stop
  looking for a scheme/matchup explanation; there isn't one in this data.**
- **Game script is real but NOT bettable.** By final margin, overs go 68.4% in
  ≤7-pt games and 49.4% in 22+ blowouts; unders invert. But the same split on
  the **pregame total** is non-monotone and season-unstable (<45: −0.7% in 2024
  vs +25.1% in 2025; 59+: +5.7% vs −8.7%). Same shape as "blowouts go OVER" —
  only visible after the game. Do not bet it.
- **Q1 losses are clean variance, no predictor.** We lose exactly when the
  favourite wins Q1 by 7+: cover rate is 100% when the dog leads, 98.2% at a
  0–6 favourite lead, 15.1% at 7–13, 0% at 14+. Losing games needed the
  favourite to score ~3 TDs more in one quarter. Full spread on losses 22.0 vs
  24.3 on wins (bigger spread = *safer*, matching the monotone edge); week 6.5
  vs 6.7 (nothing). **The losses look random, which is what a real edge's
  losses should look like.**

## FUTURES / season win totals — dead end (2026-08-03)
- **The Odds API has NO NCAAF season win totals.** Only `americanfootball_ncaaf`
  (game markets) and `americanfootball_ncaaf_championship_winner`. If we ever
  want win totals they must be captured manually or bought elsewhere.
- **Championship-winner futures: hold measured at 26.9–32.5%** (FanDuel 26.9,
  DK 28.6, BetOnline 30.2, BetMGM 32.5). For scale we rejected the moneyline at
  ~5% and anytime TD at ~8%. Beating a 30% hold needs a 30% edge. **Never bet
  outright futures.**
- **⚠️ A SELECTION ARTIFACT that nearly became a "finding" — read this.**
  Aggregating the market's game-level win probabilities to team-seasons showed
  top-tier teams (expected wins ≥9) beating expectation by **+1.11 wins, t=+4.9,
  positive in all 7 seasons**, with a perfectly monotone tier gradient
  (−0.52/−0.39/+0.04/+0.41/+1.11). It survived a non-parametric mapping control
  (+1.216, t=+4.18) and the tail calibration was perfect (±0.05pp). **It is
  still fake.** The tell: within those teams' own games, implied 0.337 → actual
  0.692 in coin-flip spots. No market is 35pp wrong on a pick'em. **Cause:
  team-seasons were selected on `exp_wins`, which is computed from IN-SEASON
  closing spreads — and spreads respond to results. Teams that win get bigger
  spreads, inflating both their summed expectation and their actual wins.
  Conditioning on that quantity conditions on the season's outcomes.**
  RULE: never bucket a season-level aggregate by a quantity derived from
  in-season market prices; the market updates on the same outcomes you grade.
  A real futures test needs PRESEASON market numbers, which we do not have.

## Can we predict our losses? NO — tested two ways (2026-08-03)
- **Share-trend correction — REJECTED** (`backtest/props_trend.py`). The loss
  review said losses come from realised share > trailing EWMA, and an EWMA lags
  a trend, so a walk-forward share trend should fix it. The trend genuinely
  predicts the residual (rush b=+0.177 t=+14.6; QB b=+0.481 t=+38.1; targets
  b=−0.060 t=−4.8; trend autocorr 0.41–0.58) — **but correcting for it makes the
  BETTING worse**: 56.3%/+7.0% → 55.7%/+6.1%, units 51.7 → 50.7. Per stat:
  rush +6.0→+4.7%, pass +6.5→+3.7%, receptions +11.8→+16.4%. **The two that got
  worse are the two with POSITIVE momentum** — rising rush/QB share is readable
  off a box score, so the book already has it and correcting just moves us
  toward the line. Targets MEAN-REVERT (opposite sign) and improved in both
  seasons, but that is one cell of three on n=116 (1 SE ≈ 9pp) → **not shipped,
  logged as a paper-trial candidate** with ml_spread.
- **Direct loss classifier — REJECTED, anti-predictive.** Gradient booster on
  22 pregame features (trailing shares + trends, usage, opp defence EPA,
  matchup, spread, EV, our blended prob, side, line), train 2023–24 → test 2025.
  **OOS AUC 0.470** (below coin-flip). The quartile it liked LEAST won **60.0%
  (+13.7%)**; the one it liked MOST won 56.4% (+6.4%). Dropping the "worst"
  quartile LOWERED ROI +7.3%→+5.1%; flipping it returned **−23.6%**.
  **Interpretation: bets surviving the EV≥5% filter are already the ones where
  we disagree with a market that prices everything observable — what remains is
  variance. If our losses were predictable pregame, the market would price
  that too.** Do not rebuild this.
- **What DOES work is regime avoidance, not loss prediction.** The week filter
  (shipped) doesn't predict individual losses — it avoids the regime where the
  projection *input* is immature (trailing share 1.45× off at ≤2 prior games vs
  1.08× at 8+). Fix the input regime, don't try to forecast the outcome.

## Modeling experiments — findings (2026-07-29, Opus session)
- **Early-season transfer props — HYPOTHESIS FALSIFIED, and backwards (2026-08-02).**
  The analyst hypothesis was that books anchor a transfer's week-2 line to his
  old school's numbers, so weeks 1–4 transfer props should be the softest spot
  on the board (it would have sat at the intersection of our two proven veins:
  derived shortcuts + early-season prior staleness). **The data says the exact
  opposite.** Player provenance was computed from `player_game_logs` (returning
  = same team last season, transfer = different FBS team, newcomer = no FBS
  production) and it is **not a usable filter at all**: returning +7.3%,
  transfer +6.3%, newcomer +6.1% — all within noise of each other, and dropping
  newcomers moves the book-graded edge by +0.1pp. What IS real is the calendar:
  weeks 1–4 lose regardless of provenance. **Read: in September we have no
  current-season data on transfers either, so the book's prior is at least as
  good as ours — we were not going to out-guess them on players nobody has seen
  yet.** The useful output of this test was the season arc (see the props row
  in the table above), which is now shipped. Rule learned: *early-season
  softness is a SPREAD phenomenon (public money on brand favourites), not a
  PROPS phenomenon — do not assume an edge mechanism transfers between markets.*
- **Defense profiles** (`features/defense_profiles.py`): opponent-adjusted,
  facet-split pass/rush defense EPA + sack/havoc rates, walk-forward as-of
  table (`defense_asof.parquet`). Face-valid (TTU/Oregon/Miami top 2025 pass
  D); pass-D EPA y/y r=0.33 (defense less sticky than offense — real).
  Built by the August refit.
- **ML sim** (`models/ml_sim.py`): HistGradientBoosting for margin/total vs
  the linear game_sim, head-to-head on identical games. **ML did NOT beat
  linear** (totals 52.9% vs 54.1% OU ROI; ATS ~wash). Kept linear. Defense
  features gave the ML a small OU-selection lift (51.6%→52.9% at edge≥5) but
  no central-accuracy gain.
- **Deep learning props** (`models/dl_props.py`, `backtest/dl_props_vs_book.py`,
  torch 2.13 CPU installed): player-embedding MLP predicting full conditional
  DISTRIBUTIONS (Gamma for yards, NegBinom for receptions) via NLL — targeting
  the hypothesis that per-player variance (boom/bust vs steady) would sharpen
  P(over). **Result: marginal and hypothesis FALSIFIED.** 2025 OOS at EV≥3%:
  DL 54.0%/+2.1% vs production 53.1%/+0.9% — better, but ~1pp ROI over 556
  bets is inside the noise band (SE≈4%). Ablation: predicted dispersion
  (+2.1%) ≈ constant dispersion (+1.8%), so the variance modeling adds
  nothing; DL's edge is in the mean. Fitted blend weight **0.05** = the market
  supplies 95% of the signal. DL MAE is WORSE than production on every stat
  (it optimizes likelihood, not MAE). **Not adopted as production** — kept as
  a parallel candidate to paper-trial. Props stay near-efficient; no DL
  breakthrough. (Deep learning DID help early-season spreads — see ml_spread.)
- **1H tiering attempt — no tier found (2026-07-30).** Unlike the full-game
  big-dog play, 1H does NOT tier. Tested and rejected: dog-side filter
  (55.7%, worse than 57.8% base); favourite-side (73.2% but n=56 and
  declining 86/77/65% by season); 1H spread size and edge size (no monotone
  pattern); book 1H/full ratio anomaly (48.6%/55.3%/62.1% — non-monotone,
  per-season 54.8/59.8/51.6%). **Mechanism test that killed the fav-side
  idea: favourites really do front-load (~58% of final margin comes in 1H,
  n=4,400) — but books already set 1H at 56.8% of the full spread, so it's
  priced.** Keep 1H single-tier, all sides, edge≥2, weeks 1–5 only.
- **Weather & rest/travel — both REJECTED (2026-07-30).** Data now on disk
  (`weather.parquet` from CFBD `/games/weather` — temp/wind/precip/indoors,
  9,650 games; `situational.parquet` from `features/situational.py` — rest
  days, bye, travel km via haversine, tz shift, elevation gain, dome change,
  23,467 games). **Both have real physical effects but no betting edge:**
  wind ≥15 mph costs passing props 11 yds vs +9 under 8 mph (a 20-yd swing we
  don't price) — yet adding a wind term DROPPED props +7.5%→+5.7% and
  pass_yds +6.5%→+0.9%. Rest/travel/altitude/tz correlate with ATS at only
  0.005–0.022 and don't refine the big-dog play (60.9% vs 58.6%, noise).
  **The market prices weather, rest and travel properly.** Kept for research.
- **BET THE OPENER, not the close** (2026-07-30, `spreadOpen` in CFBD lines —
  74% coverage, previously unused): grading the big-dog play against the
  OPENING line beats the closing line in **every season tested** —
  PREMIUM 65.9% / +25.9% (vs 62.9% / +20.2%), STANDARD 60.0% / +14.5% (vs
  58.2% / +11.2%), ALL 17+ 62.8% vs 60.5%. Mean line movement in these spots
  is only +0.04 pts, so the gain is **selection** (which games clear the
  edge≥6 bar when measured off the opener), not a better number. Practical
  change: run `picks.ml_spread_picks` Sun/Mon when lines post, not midweek.
  CLV caveat: our edge-vs-opener predicts the line moving our way only ~45%
  of the time — this edge is NOT "beat the close", it's "both numbers are
  wrong on big mismatches and the opener is wronger in our favour."
- **PFF-adjacent free player data** (`features/player_advanced.py`, 2026-07-30):
  two untapped CFBD endpoints — `/player/usage` (usage split by down and
  standard/passing downs) and `/ppa/players/games` (per-player per-game PPA,
  113k rows, the closest free analogue to a PFF grade). **Both tested in the
  props recalibration and REJECTED**: situational third-down usage HURT
  (+6.0%→+3.5%); player PPA looked like a gain (+6.9%) in an isolated test but
  with the production pipeline gives **+6.4% vs +7.0% without** it. Data is
  pulled and available for future use; neither is in the production model.
  **Bug worth remembering:** joining PPA on (season, week, player) silently
  DUPLICATED 2,234 projection rows — 4,263 player-week keys collide across
  teams. `ppa_trailing()` now keys on team_id too and de-duplicates. Always
  verify projection row count (46,487) after adding a merge.
- **DIY betting-splits capture** (`ingestion/scrapers/splits_capture.py`):
  scrapes the FREE public Action Network NCAAF public-betting page (no
  paywall/login — same category as Ourlads/Wikipedia; we still refuse
  paywalled PFF/AN-Pro data). Parses `__NEXT_DATA__` →
  `games[].markets.<book>.event.{moneyline,spread,total}[].bet_info` giving
  **tickets.percent (bet share) AND money.percent (handle share)** per side.
  Appends timestamped snapshots to `betting_splits.parquet` + raw JSON.
  Preseason returns empty (no action yet) — by design. **Scheduled task "CFB
  Splits Capture"** runs Wed/Fri/Sat from 2026-08-26. Goal: own a
  backtestable splits history by 2027 so we can implement the direct
  fade-the-public angle (>75% handle → ~47% ATS) instead of our big-spread
  proxy. `public_fade_candidates()` lists heavy-handle sides from the newest
  snapshot.
- **Parlay builder** (`picks/parlay_builder.py`): combines ONLY validated
  +EV legs (big-dog spreads, 1H spreads, ML value) — never props/totals (no
  edge → vig compounds → guaranteed loser). One leg per game (independence),
  win rates haircut from backtest (0.565/0.555), longshot guard (leg prob
  ≥0.45) and max 1 ML leg — without those, the EV-maximizer stacked +400 dogs
  into 0.1%-hit "lottery" parlays. Prints a 2–6 leg ladder w/ true joint
  prob, payout, EV, ¼-Kelly. Math: parlaying +EV legs raises EV% but crushes
  hit rate (2-leg 32%/+16% → 6-leg 3.3%/+57%); recommends by EV×hit-prob
  (usually the 2-leg). `import-parlay` logs it as one bet; settles only when
  every leg's single has graded. **Singles still grow a bankroll faster.**
- **SP+ ensemble** (2026-07-30): CFBD provides SP+/FPI/Elo ratings free
  (`sp_ratings.parquet`, refreshed by refit). Prev-season SP+ persists r=0.75
  y/y — a strong early-season prior. SP+ ALONE is slightly worse than EPA, but
  **blending SP+ + EPA beats either**: margin MAE 13.92→13.35, big-dog edge
  59.1%→59.3%/+13.2%. Added `sp_diff` to ml_spread FEATS (MAE 13.72→13.63).
  Ensemble > single model. Dead angle checked: home underdogs 50.9% (no edge).
- **ML early-season spread** (`models/ml_spread.py`): the ONE place ML wins.
  Wks 1-5 only, features = as-of EPA + prev-season net + returning PPA +
  portal + recruiting. **ML beats linear: MAE 14.53→13.72, ATS 55.2%→55.7%,
  +5.3%→+6.3% ROI** (walk-forward 2023-25, ~400 bets). Mechanism: early
  season EPA is data-starved, so roster priors + nonlinear interactions help.
  Modest & only 3 seasons (roster priors need 2022+) — paper-trial alongside
  linear before adopting as the production spread model.
- **TOTALS are a dead market** (`why totals flop`): pregame spread barely
  correlates with over/under (r=−0.03). Blowouts go OVER (62%) not under, but
  only visible AFTER the game — not forecastable pregame. "Under on big
  favorites" tested across 10 seasons = 48% (−8% ROI), only 1/10 seasons
  positive; the 2-season +1.7% was noise. Don't bet totals (full or 1H).
- **Loss investigation** (why props/ML lose): PROPS — bigger model-vs-line
  disagreements do NOT win more (all buckets ~52-55%), so better projections
  won't help; the thin edge is in segments (unders 56.7% vs overs 52.0%, low
  lines 58.3% vs mid 49.4%, rush_yds best). MONEYLINE — model picks winners
  69.7% vs always-favorite 74.7%; model is WORSE than the spread at picking
  winners, dog-picks win only 37%. ML "value" must come from calibrated price
  edges + the market blend, never the model's raw opinion.
- **Defense in props**: residual test proved raw allowed UNDER-adjusts for D
  quality (resid corr +0.10..+0.20). But wiring opp-adjusted D into props
  **hurt** the betting edge (EV≥5 +5.2%→−5.3%) — the market already prices
  unit defense, so a more defense-aware projection just agrees with the book.
  Reverted; opp-adj columns remain in the table for research only.
- **Player-vs-player coverage (shadow corner) is NOT possible** from CFBD
  PBP — no coverage charting (who covered whom). True matchup data needs
  PFF or SIS (only two vendors; see GRAB_LIST 6/6b). We do NOT scrape their
  paywalled consumer tiers — ToS/ban risk + it's licensed commercial data.
- **Free player-level defense** (`features/defense_players.py`): CFBD season
  box stats (PD, INT, sacks, hurries, TFL — 1 API call/season) → per-team
  key coverage DBs + pass-rushers (`defense_players_key.parquet`). Can't do
  who-covered-whom, but identifies each team's shutdown DB by name. The real
  free edge: `coverage_injury_flags()` crosses this with the live injury
  feed → when a team's top-PD DB is OUT, lean opposing WR/QB overs. Surfaced
  in the prop_picks report as a watch. **NOW BACKTESTED** (`backtest/
  coverage_injury.py`, 75k CFBD tier): used per-game defensive box scores as
  a DNP proxy (season-regular DB absent from a game = sat out), 298 DB-out
  team-games in 2024-25. **On-field effect is REAL** — opposing pass EPA
  +0.047/play higher when the top DB is out. **But NO betting edge**: opposing
  pass-catcher overs hit 47.0% with the DB out vs 48.2% in (both lose blind).
  The market fully prices announced DB injuries into prop lines. → The
  coverage-injury angle does NOT beat the book; the watch stays informational
  only. Implication: **do not buy PFF/SIS expecting this specific edge** —
  any residual value is late-breaking-news timing (operational, not data).

## Hard-won gotchas (do not relearn these)
1. **Grade bets only on official finals** (CFBD homePoints) — PBP scores are garbage-time-truncated (once produced a fake 64.8% ATS).
2. 2021–22 PBP: bools are object dtype, down/distance 94% null → coalesce from `start.down`; 2025 PBP: yardage 43% null → coalesce from `statYardage`.
3. **Never average American odds across the ±100 gap** — aggregate in payout/probability space (once produced fake +40% ROI). Pair over/under at the SAME point per book (alt-line ladders poison consensus).
4. Walk-forward self-calibrating models need a **warm-up season** before the first test season (INIT coefs once caused an 88%-overs disaster).
5. Team-name matching must be exact + alias table (prefix matching mapped FCS Arkansas-Pine Bluff → Arkansas, fake 43-pt edge). Aliases live in `picks/edge_report.py` and `ingestion/scrapers/*`.
6. Raw model probabilities vs a market line are overconfident — always blend toward no-vig market prob (0.2/0.8 validated). Exception: ML pick win-prob vs *outcomes* is genuinely calibrated.
7. Wikipedia: use the batched query API (50 titles/req); single-page `action=parse` gets 429-throttled.
8. Odds API historical = 10 credits × market × region × event; live is cheap. Usage log: `warehouse/raw/odds/odds_usage.jsonl`.
9. Portal players must lose their stale "returning" row at the origin school (models/props.py handles it).
10. **A failed merge must yield NaN, never a default.** `np.where(team_id == home_id, spread, -spread)` silently hands BOTH teams the away sign when `home_id` is NaN. Cost: a 2.2x overstatement of the props edge that stood for months. Any two-way sign assignment needs an explicit third branch for "unknown".
11. **Never source a game attribute from play-by-play when a games table has it.** PBP is missing for ~5% of games and those absences are not random — they concentrate in FCS buy games, i.e. the biggest spreads and the earliest weeks.
12. **Fit on one row per unit of analysis, not on whatever the join produced.** `train.groupby("game_id").first()` over *player* rows silently trained the team-volume model on one arbitrary side of each game, making the fit depend on row order.
13. **Demean within player before believing any "X affects player Y" effect size.** A raw ratio-to-trailing-average conflates between-player composition with the within-player effect and overstated a game-script effect 4x (~20% vs a true ~5%).
14. **Check that an effect is specific to the side your mechanism names.** "Favourite's starters rest in blowouts" looked significant until the dogs showed the same skew — which means the mechanism was wrong regardless of the t-stat.

## Memory
Claude Code persistent memory for this project lives at `~/.claude/projects/C--Users-pcagm-PycharmProjects-PythonProject14/memory/` — survives model switches and new sessions. `MEMORY.md` is the index; `cfb-predictor-project.md` mirrors much of this file.

## Keys (.env — never commit)
CFBD ✓, The Odds API (5M/mo) ✓, OpenAI ✓ (embeddings, not yet wired), Anthropic ✓ (news extraction, working). `.env.example` is a blank template (verified 2026-07-31 — the CFBD key that used to be in it is gone; keep every key line blank) and now also carries `ODDS_API_KEY` + `RAW_DIR`, which `config.py` reads but the template had been missing. Project `README.md` covers architecture with collapsible zoom levels.
