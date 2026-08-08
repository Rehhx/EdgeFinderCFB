# Player Props — full report, 2026-08-06

Props are **~78% of the season book** and the only play in the system whose
edge is positive in both out-of-sample seasons with every control passed.
Measured on the de-contaminated data (see `HANDOFF.md` §1, §19).

## Headline (OOS 2024–25, the shipped selection)

| | |
|---|---|
| bets / season | **250** |
| hit rate | **61.2%** |
| ROI | **+16.2%** |
| units / season | **+66.5** (book contribution +62.7 after de-duplication) |
| t-stat | **+3.90** |
| bootstrap 95% CI | **[+8.0%, +24.4%]** — excludes zero |

## ⭐ The edge is NOT line shopping

| pricing | ROI | units/szn |
|---|---|---|
| best quoted price (shipped) | +16.2% | +66.5 |
| **median price across books** | **+15.0%** | **+62.5** |

Line shopping contributes only **+1.1pp of ROI**. The edge survives almost
intact at the median quote, which is the single most important control here —
it is what separates a real model edge from stale-line harvesting. (Compare
the anytime-TD model, which was rejected precisely because its "edge"
evaporated at median prices.)

## By stat — one engine, two passengers

| stat | bets/szn | hit | ROI | u/szn | t | boot CI |
|---|---|---|---|---|---|---|
| **rush_yds** | 139 | **63.3%** | **+19.5%** | **+51.8** | **+3.56** | **[+8.5%, +29.7%]** |
| pass_yds | 80 | 59.6% | +12.4% | +10.1 | +1.69 | [−1.8%, +26.6%] |
| receptions | 30 | 55.7% | +11.1% | +4.6 | +0.86 | [−13.9%, +35.3%] |
| ~~rec_yds~~ | — | — | — | EXCLUDED | — | 2025 −4.2%, fails both-seasons |

**rush_yds is the only stat whose CI excludes zero**, and it carries ~78% of
the props units. This is why the per-stat stake tilt (rush 1.25x, others
0.85x) was shipped — and why it was validated out-of-sample first (ranked on
2023–24, applied to 2025 alone: +15.4% → +16.6%).

The mechanism agrees with the money: rush share (corr 0.596) and rush attempts
(0.541) are the best-predicted components in the entire model.

## By tier — the bands are flatter than the staking implies

| tier | bets/szn | stake | hit | ROI | 2024 / 2025 |
|---|---|---|---|---|---|
| PRIME (EV 5–8%) | 119 | 2.0 | 60.9% | +15.4% | +31.4 / +53.8 |
| PROBE (EV 4–5%) | 59 | 1.0 | 61.9% | +17.1% | +16.1 / +6.0 |
| STANDARD (EV 8%+) | 72 | 1.5 | 61.1% | +16.7% | +19.4 / +6.2 |

All three tiers hit 60.9–61.9% at +15.4% to +17.1% ROI — **the EV bands barely
differentiate.** Worth watching: PRIME *improved* in 2025 (+31.4 → +53.8) while
PROBE and STANDARD both fell hard (+16.1 → +6.0, +19.4 → +6.2). If that
persists in 2026 the tier structure needs re-deriving, not just re-sizing.

## By week — the season arc holds

| block | bets/szn | hit | ROI | u/szn |
|---|---|---|---|---|
| wk 5–8 | 70 | 58.6% | +10.6% | +16.9 |
| **wk 9–11** | 76 | **64.2%** | **+21.6%** | +27.2 |
| wk 12–15 | 104 | 60.8% | +16.0% | +22.3 |

Confirms the shipped gate: nothing before week 5, STANDARD only from week 9.
The edge is weakest early and strongest mid-late, consistent with the stated
mechanism — our trailing EWMA accumulates in-season role information the
book's season-long anchor does not.

## Where the projection error actually lives

| stat | attempts | efficiency | inside the attempts error |
|---|---|---|---|
| rush_yds (MAE 29.2) | 36% | 28% | **player share 55%**, team volume 10% |
| pass_yds (MAE 73.8) | 29% | 22% | share 31%, team volume 34% |

⚠️ **The market out-predicts us on every stat** (pass_yds MAE 69.4 vs 63.9).
Our edge is not better central estimates — it is the recalibration, the
gamma/NB tails, and a 30% model weight that disagrees in the right places.

Single-game efficiency is essentially unpredictable (ypc corr 0.104, ypa 0.122)
and that quarter of the error is **irreducible**.

## Improvement attempts — 7 tested, 1 shipped

| attempt | result |
|---|---|
| Per-stat stake tilt | ✅ **SHIPPED** — validated OOS, +7.8 u/szn |
| Shrink efficiency harder | ❌ shipped constants already optimal |
| Recalibrate player share | ❌ worse OOS on all three stats |
| Availability / staleness (`gap`) | ❌ real bias, but −11.6 u/szn |
| Filter out high-line props | ❌ top quintiles still positive both seasons |
| Include `rec_yds` | ❌ +$22 median for +3.9pp P(loss) |
| Accept single-book lines | ❌ those bets grade 52.0% / +1.3% / t=+0.19 |
| **Per-stat blend weight** | ❌ **new 2026-08-06** — see below |

### Blend weight: global 0.30 re-confirmed

| w (weight on our model) | bets/szn | ROI | u/szn | 2024 / 2025 |
|---|---|---|---|---|
| 0.25 | 168 | +8.5% | +15.0 | +9.6 / +20.4 |
| 0.28 | 220 | +15.1% | +47.5 | +51.3 / +43.8 |
| **0.30 (shipped)** | 250 | **+16.2%** | +66.5 | **+67.0 / +66.0** |
| 0.32 | 277 | +14.6% | +75.1 | +90.1 / +60.1 |
| 0.35 | 318 | +11.9% | +68.5 | +91.6 / +45.5 |
| 0.40 | 370 | +8.5% | +45.8 | +73.0 / +18.6 |

0.30 has the best ROI **and by far the best season balance (67.0 / 66.0)**.
0.32 shows more total units but splits 90.1 / 60.1 — the repo's rule prefers
balance over the maximum. Per-stat weights (rush .40 / pass .20 / rec .30) were
also tested: **+50.3 u/szn, worse than global**.

⚠️ Note the fragility this exposes: one 0.05 step down (0.30 → 0.25) costs
**51 units**. The weight is barely identifiable from the data — which is
exactly why it is FIXED rather than re-estimated each run.

## Round 2 of improvement attempts — 3 more tested 2026-08-06, ALL REJECTED

| attempt | result |
|---|---|
| Per-stat market blend weight | ❌ global 0.30 wins on ROI (+16.2%) and season balance (+67.0/+66.0); per-stat +50.3 u/szn |
| **Opponent pace** in the team-volume model | ❌ rush MAE 6.440 -> 6.439, pass 6.682 -> 6.677. Nothing. |
| **Weather** (wind/precip/cold) on the run-pass mix | ❌ right signs, worse OOS |

### Opponent pace adds nothing — and that is informative

Team volume is 34% of the pass-attempt error with corr only 0.425, and the
model uses ONLY own trailing pace + spread. Adding the opponent's trailing
plays (and their rush/pass split) moved OOS MAE by <0.1%. **CFB possessions are
near-symmetric**, so opponent pace is already embedded in the spread. The
shipped volume model is at its ceiling.

### Weather: real mechanism, far too small and too rare

Free mechanism test on 4,552 outdoor team-games (pass rate, sd 0.122):

| condition | team-games | pass rate | vs calm |
|---|---|---|---|
| calm | 3,672 | 48.0% | — |
| wind 10-15 | 994 | 47.7% | **-0.2%** |
| wind 15-20 | 210 | 47.7% | **-0.3%** |
| wind 20+ | 88 | 48.7% | **+0.8%** |
| any precip | 362 | 45.4% | -2.6% |
| precip > 0.1 | 36 | 43.2% | -4.8% |
| temp < 40F | 258 | 44.9% | -3.1% |

**WIND HAS NO EFFECT AT ALL** — not even monotone. That contradicts the
conventional "wind kills the passing game" and is worth remembering before
anyone reaches for it again. Precipitation and cold DO shift the mix in the
expected direction, but by ~0.2-0.4 sd on 6-8% of games; fitted coefficients
had the right signs (precip: rush +25.2, pass -36.2) yet made OOS predictions
WORSE, because they are estimated from a handful of extreme observations.

### Also checked: the disabled PPA term is correctly disabled

`props_vs_book.py:226` hard-sets `ppa_c = 0.0`. The HANDOFF row claiming
player-PPA was validated (+6.0% -> +6.9%) is STALE — the in-code comment
records that after a join bug was fixed it gives +6.4% vs +7.0% WITHOUT it,
and 2025 falls 7.3% -> 5.8%. Not a free win; the code is right and the doc row
is out of date.

**Running total: 11 props improvement attempts, 1 shipped.** The model is at
the ceiling of the data currently available — efficiency is irreducible, team
volume is saturated, and player share needs as-of availability data that does
not exist historically (see [[depth-chart-capture]]).

## Operational status for 2026

- Live picks now run **the same estimator the backtest measures**
  (`models.props.project_upcoming`); verified identical on a past week
  (corr 1.0000, max diff 0.0000 over 1,069 players).
- Gates verified: nothing before wk 5, STANDARD from wk 9, **postseason
  suppressed** (bowl rosters are gutted by opt-outs — unmodellable).
- Lines are pinned to the correct player AND game (the CFBD week-0 collision
  that double-graded 117 rows is fixed).
- **No props are on the board yet** — verified 2026-08-06, zero prop markets
  across `us` and `us2`. They post nearer kickoff; first bets are week 5.

## What would change the picture

1. **2026 grading of the tier structure** — PROBE and STANDARD both halved in
   2025. Two seasons is not enough to know if that is decay or noise.
2. **Depth-chart history** (capture started 2026-08-06) — player share is 55%
   of the rush-attempt error and the only component with signal left to
   sharpen. It needs as-of knowledge of who plays, which no current data has.
3. **DFS volume markets** — rush/pass attempts are predicted *better* than the
   yards we bet, but no book we can backtest posts them. Forward grading via
   `ingestion/dfs_lines.py` starts week 1.
