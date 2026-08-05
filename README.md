# College Football Betting Predictor

A walk-forward backtested system for finding — and honestly rejecting — betting edges in
college football markets: spreads, 1st-half spreads, player props, totals, and moneylines.
All ~134 FBS teams, 2016–2025 history, pregame only.

The headline result of this project is not a model. It's a **short list of edges that
survived control tests**, and a much longer list of plausible-looking edges that didn't.

---

## Validated plays (backtested, controls passed)

| Play | Rule | Win% | ROI | Size | Sample |
|---|---|---|---|---|---|
| **Q1 PREMIUM** ⭐ | Full \|spread\| ≥25 → take the dog on the **first-quarter** line | **75.5%** | **+45.7%** | 2u | 94 bets, 3/3 seasons |
| **Q1 STANDARD** ⭐ | Full \|spread\| 17–25 → take the dog on the Q1 line | 61.9% | +20.0% | 1u | 307 bets, 3/3 seasons |
| **BIG-DOG 1H** | \|spread\| ≥17 (wks 1–5) or ≥21 (wks 6+), model on dog ≥6, **bet the 1H line** | **65.7%** | **+25.9%** | 2u | 178 bets, 3/3 seasons |
| **Big-dog spread PREMIUM** | \|spread\| ≥25, weeks 1–5, model on dog ≥6 | 62.3% | +18.8% | 2u | 302 bets, 8/8 seasons |
| **Props PRIME band** | EV 5–8% (not higher — see below), **weeks 5+** | 61.0% | +15.5% | 2u | ~120/season |
| **Props PROBE band** ⚠️ *new* | EV 4–5%, **weeks 5+** — only 2 seasons old, so staked down | 60.5% | +14.2% | 1u | ~60/season |
| **1H standard** | Any side, edge ≥2, weeks 1–5 | 57.8% | +10.3% | 1u | 3/3 seasons |
| **Big-dog spread STANDARD** | \|spread\| 17–25, weeks 1–5 | 56.9% | +8.6% | 1u | 8/8 seasons |
| **Props STANDARD** | EV ≥8%, **weeks 9+ only** | 60.4% | +15.4% | 1u | ~72/season |

Realistic volume: **~252 props + ~70 Q1 + ~60 spread/1H bets per season**, de-duplicated to
**409 bets and +126.9 units/season**. From $1,000 at 1u = 2% and a 50% edge haircut (70% on
props): **median ~$1,956**, 5th percentile $1,167, 2.5% chance of a losing season.

> ✅ **Widening the prop-line pull (2026-08-05) added 26% more data for 9,288 credits and
> lifted props +40.5 → +56.9 units/season with no model change** — the single cheapest
> improvement in the project. All five fitted props thresholds were then re-swept on the
> enlarged sample and **every one held**, which is the first independent evidence they were
> signal rather than fitted noise. From $1,000 at 1u = 2% and a 50% edge haircut (70% on
props, which are measurably less certain): **median ~$1,811**, 5th percentile $1,132, 2.7%
chance of a losing season, ~11% typical drawdown.

> ⚠️ **PROBE is staked at 1u on purpose.** It grades best of any props tier (+21.4%, t=+2.32,
> positive both seasons) but rests on two seasons and only became visible after the
> 2026-08-04 sign-bug fix. It is a *separate tier* rather than part of PRIME precisely so the
> 2026 results can grade it alone and promote or drop it without contaminating the band that
> is already validated.

> 🚨 **Corrected 2026-08-04.** Every props figure published before that date was inflated
> ~2.2x by a sign bug: `home_id` came from play-by-play, missing on 4.9% of rows, and a
> two-way `np.where` handed **both** teams the away-side spread when it was NaN. Those rows
> average a **39.7-point spread and 81% sit in weeks 1–5** — so in the season's biggest
> mismatches the *favourite* was modelled as a 39.7-point dog. The props edge is real and
> robustly positive (40 bootstrap refits: median **+21.7 u/szn, sd 3.5**, P(unprofitable)
> 0.0%) but roughly **half** the old headline. It rebounds to **+30.0 u/szn** once the
> season-arc gate is re-derived on clean data. Details in `HANDOFF.md`.

> ✅ **Stability-audited 2026-08-04** (`backtest/play_stability.py`). Every play was swept
> across neighbouring thresholds, re-run leaving each season out, re-priced at the *median*
> quote, and block-bootstrapped. **A real structural edge is a plateau; a fitted one is a
> spike at the exact number you chose.** Q1 is textbook — ROI climbs monotonically from
> +23.3% at ≥19 to +64.1% at ≥31, which is what the saturation mechanism *predicts* rather
> than something fitted. Q1 PREMIUM: LOSO +37.4/+51.6/+47.7%, bootstrap **+29.4 u/szn, 90%
> CI [+11.0, +49.1]**. The biggest contributor is also the most robust play.
>
> ⚠️ **The one weak link: BIG-DOG 1H in weeks 6–15** — only 49 bets in three seasons, 90% CI
> [+0.7, +17.6] barely clearing zero, and the blind version scores a *higher* t-stat than the
> model-filtered one. Kept (positive in every view, mechanism matches), but it is the first
> thing to cut if the season starts badly, and must not be sized like Q1.

> ⚠️ The Q1 plays are a **market-structure edge, not model skill** — blind selection *beat*
> model-selected in the backtest, so `picks/q1_picks.py` uses no ratings at all. They are
> also **low-limit markets**; expect small maximums, which is likely why they stay
> mispriced. Size accordingly and don't assume the spread play's limits carry over.
>
> ⚠️ **Never bet Q1 and 1H on the same game.** Their outcomes agree 76% of the time
> (r=+0.51), so 2u on each is closer to a 4u single position than to two bets. Pick one by
> spread size: **≥25 → Q1** (+44.0% vs +26.7%), **21–25 → 1H** (+23.1% vs +11.4%).

**The two markets are mirror images, and this is the most useful structural fact we have.**
Spread edges live in **weeks 1–5** and die after. Props are the opposite — they have *no*
edge in weeks 1–4 (**−4.5%, negative in all four band×season cells**) and their best from
week 9 (+10.9% even in the weak band). The correction sharpened this rather than weakening
it: with the sign bug gone, weeks 1–4 lost the artificial support that had kept the PRIME
band bettable that early, and props are now **skipped entirely before week 5**. Spreads pay
early
because the market has no data and public money floods brand-name favorites; props pay late
because our projections weight current-season form while the book's prop line stays anchored
to a season-long average. Don't assume an edge mechanism transfers between markets — we
tested exactly that assumption and it was backwards.

**Bet the opener, not the close.** Grading the big-dog play against `spreadOpen` beat the
closing line in every season tested (PREMIUM 65.9% vs 62.9%). Run spread picks Sunday/Monday.

### Two counter-intuitive results worth internalizing

- **Props EV bands are non-monotone.** The 5–8% EV band (59.5%) beats the 8%+ band (54.0%).
  A very large computed edge usually means a stale line or a model error, so the top band is
  staked **down**, not up.
- **Fewer bets can mean more profit.** Tightening the props gate from "STANDARD after week 5"
  to "nothing before week 5, STANDARD after week 9" cut volume from 226 to 147 bets/season
  and *raised* units from +23.5 to +30.0. Optimize the bankroll, not the win percentage — but
  pick the gate on **season balance**, not on the best total: the old rule drew 82% of its
  units from a single season, the new one is +28.9 / +31.2 across the two.
- **An accidental edge is still evidence — of something.** The sign bug was profitable: by
  modelling big favourites as huge dogs it under-projected their rush volume and effectively
  bet "fade the favourite's RB in a blowout". Tested deliberately, that play **fails** — the
  underdogs' props go under just as often, so the starter-rest mechanism isn't what drives
  it. Worth chasing anyway; a bug that makes money is telling you where to look.
- **Bigger spreads are better, not riskier.** An old `|spread| ≤ 21` "blowout guard" was
  suppressing the single best edge in the system. Spreads now use the full board.

## ⭐ Teasers — the high-hit-rate play

Parlays raise EV by *spending* hit rate. Teasers do the opposite: you buy points, so the hit
rate goes up. That is unusually cheap for us **because our edge is itself a saturation
effect** — the dog's sub-period deficit plateaus near 4 points however large the full spread —
so the distribution we're buying points against is already compressed. A teaser is the natural
instrument for a saturation edge.

Validated on **real same-week, different-game pairs** (not an independence assumption):

| Teaser | n | **True hit rate** | EV | t | Per-season |
|---|---|---|---|---|---|
| 1H +6.0 | 80 | 71.2% | +36.0% | +3.71 | 64/71/78% |
| 1H +7.0 | 80 | 75.0% | +32.7% | +3.79 | 72/75/78% |
| **1H +10.0 (shipped)** | 80 | **83.8%** | **+30.3%** | **+4.69** | **80/82/89%** |
| Q1 +10.0 ⚠️ rarely offered | 38 | 92.1% | +43.3% | +6.28 | 86/92/100% |

The independence assumption *held* (83.8% true vs 82.8% assumed) — at 91% per leg there is
little room left for the 1.14× leg correlation to bite, unlike parlays. **Margin of safety is
huge:** at 83.8% a −180 price needs only 64.3%, so this survives down to roughly **−400**.

> ⚠️ **Availability is the binding constraint, not the edge.** Confirm your book offers 1H
> teasers, allows teasing a line this large, and quotes at or better than break-even. Q1
> teasers grade better still but are rarely offered — treat them as a ceiling, not a bet.
>
**Full-game teasers are rejected, and the rejection confirms the mechanism.** They're the most
widely offered version, so they'd have been the most placeable — but full-game margin is
*unbounded*, so there's no compressed distribution to buy points against. Measured: 1H +10
pays **+30.3%**, full-game +10 pays **+11.2%** — the saturating market pays ~3× more for the
same 10 points, and on real pairings only the +10 size survives all three seasons (t=+1.77).
**Hunt teasers in bounded sub-period markets, not full-game ones.**

Teasers are now **in** the bankroll projection. Integrating them corrected an assumption:
"replace the 1H singles whose legs they use" looked disciplined and is the *worst* option
(+97.6 u/szn vs +103.5 doing nothing), because a 1u teaser covers two legs and so deploys a
quarter of the capital per leg that a 2u single does. Betting them **in addition** wins
(+110.5 u/szn, P(losing season) 2.8% → 2.1%) — an 83.8% bet added to a book lifts the median
*and* the 5th percentile.

**Same-game Q1 + 1H** flips the overlap from hazard to asset: as two *separate* bets they're
one position (never do it), but as a **single same-game parlay** the true joint rate is 56.8%
vs 49.2% if independent — a **+7.5% correlation lift**, worth +15.3% EV against a book that
prices it independently. Verify the quoted price; most books correlation-adjust.

## Growth plan: compounding + parlays

Flat 2% staking is the honest way to *state* an edge, but it is not how you grow a bankroll.
Two levers, both validated on real graded outcomes (`backtest/growth_paths.py`):

**Compounding** — stake a % of the *current* bankroll, not the starting one. Free, no new
bets, no extra per-bet risk. Median $1,814 → $1,990 at the same 2%. Shipped in
`picks/paper_trades.py`; every row records the dollar value of 1u at placement time so the
history stays gradeable as the unit moves (`bankroll` / `backfill-units` commands).

**Parlays** — the edge *multiplies*: `EV = (1+EV₁)(1+EV₂) − 1`, so two +45.7% Q1 legs price
near +112%. Measured on same-week, different-game pairs:

| Composition | /szn | Win% | ROI | t | Per-season |
|---|---|---|---|---|---|
| **Q1 PREM × Q1 PREM** | 19 | 55.9% | **+103.8%** | +4.34 | +163/+61/+80% |
| Q1 PREM × any derived | 31 | 50.0% | +84.5% | +4.39 | +93/+59/+101% |
| any derived × any derived | 116 | 43.1% | +59.4% | +6.02 | +62/+70/+46% |
| 1H × 1H | 23 | 37.1% | +37.3% | +1.73 | +55/+31/+30% |

> ⚠️ **Legs are not independent.** Weekly win-rate variance is **1.14×** what independence
> predicts — our big dogs win and lose together, because they are all the same structural
> bet. True joint probability is slightly *below* p₁×p₂, so the win-rate haircut in
> `parlay_builder.PROB` is load-bearing, not decoration.
>
> ⚠️ **The 116/season row is not permission to bet that many.** It re-uses each leg ~3×, so
> one leg losing takes down several parlays *and* the single on that game.

**$1,000 → $5,000 is roughly a coin flip**, not a plan: compound 4% + 2-leg parlays gives a
$3,916 median and **46.6%** chance of $5k — rising to **57.0%** if Q1 limits reach $200 across
books. Getting Q1 down at four books instead of one is worth more than any model change
(P($5k): 42.1% at $25 → 57.0% at $200, then flat). Compounding at 8% busts **3 times in 10**,
which is where aggression stops paying.

**"70–80% hit rate" and parlays are mathematically opposed.** Q1 PREMIUM already hits 75.5%,
but it is 31 bets a season and cannot 5x anything alone. Parlays buy edge by *spending* hit
rate — 2-leg ~51%, 3-leg ~33%. No configuration delivers both.

## Rejected after honest testing

Moneyline (−5% ROI), totals (dead across 10 seasons), anytime TD, `rec_yds` props, weather,
rest/travel/altitude, coverage-injury (real on-field effect, zero betting edge),
opponent-adjusted defense EPA in props, situational usage, player PPA, deep-learning
distributional props, home underdogs, team totals (demoted to a standalone market bias),
**player provenance** (returning +7.3% vs transfer +6.3% vs newcomer +6.1% — all noise;
the calendar matters, who the player is doesn't), **alternate spreads** (no rung beats the
main number; the "best" rung moves depending on which subset you look at), **Q1 totals**
(books price the ratio at 0.2152 against a true 0.2250 — conservative, not lazy),
**season win totals and outright futures** (no data from our provider; outrights hold
**27–33%**), **loss prediction** (a classifier trained to spot our own losing bets scored
AUC 0.470 — worse than a coin flip), **game-script usage shares** (real effect, but ~5% once
fitted within-player rather than across players — de-scripting the trailing EWMA loses to
production, +26.1u vs +39.6u OOS), **fading the favourite's starters in mismatches** (the
underdogs go under just as often, so the mechanism is wrong; 2025 ROI +0.0%).

The recurring finding: **the market prices everything observable** — weather, rest, defense
quality, player quality, announced injuries. Every surviving edge comes from
**derived-line shortcuts** (books setting 1H at a flat 56.8% of the full spread, team totals
as `(total ± spread)/2`) applied to big early-season mismatches.

But "derived" alone isn't enough, and the alternate-spread test is what sharpened this:
**a derived line is soft when the derivation holds something *constant* that reality varies.**
The 1H line applies a flat 56.8% ratio to something that genuinely moves (big-dog covers are
front-loaded, ~3.6 pts in the 1H vs ~0.9 in the 2H) — that constant is a real formula error.
The alt-spread ladder is derived from the margin *distribution*, which books price well; 3,
7, 10 and 14 are the most-studied numbers in football betting, so walking the ladder just
walks a correctly-priced curve. **Hunt constants, not derivations.**

Applying that rule immediately produced the best edge in the project. Books set the
first-quarter line at **~0.289 × the full spread** — a straight line, R²=0.85. But the
underdog's *actual* Q1 deficit **plateaus near 4 points no matter how large the spread gets**:

| Full spread | Book's Q1 line | Dog's actual Q1 margin | Book over-grants |
|---|---|---|---|
| 13.5 | 4.10 | −3.75 | +0.36 |
| 19.1 | 6.20 | −4.29 | **+1.91** |
| 27.6 | 6.98 | −3.58 | **+3.40** |
| 37.0 | 8.93 | −4.60 | **+4.33** |

A quarter holds only ~3–4 possessions a side, so you *cannot* lose by 30 in it. Full-game
margin is unbounded; Q1 margin is not. A linear rule against a saturating reality must
over-grant at the extremes — and the error grows exactly as the spread does.

Testing the **Q1 total** immediately afterwards sharpened the rule further, by failing.
Q1 takes only 22.5% of a game's points (Q2 is the big quarter at 30.4%), so a naive 25%
derivation would be ~1.3 points too high. But books actually use **0.2152** — already below
the true share. The under returns +1.3% with a CI spanning zero: correctly priced.

The difference is the whole lesson. **A derived constant only breaks when the sub-period
quantity is bounded in a way its parent is not.** Q1 *margin* saturates while full-game margin
doesn't, so a linear rule must fail at the extremes. Q1 *points* scale with full-game points —
the real ratio is stable from 0.219 to 0.233 — so a flat ratio is the correct model and the
book uses a good one. Hunt constants applied to a quantity with a **ceiling its parent lacks**.

---

## The discipline (read this before adding any edge)

These rules were each paid for with a fake result. See `HANDOFF.md` for the full list.

1. **Always run a blind / no-model control.** If betting every qualifying game with no model
   does as well, you found a market bias, not an edge. This demoted team totals and killed
   anytime TD.
2. **Test at median price, not best price.** An edge that exists only at the single best
   quoted number is stale lines and data mismatches, not skill.
3. **Check the fitted blend weight.** A weight near 0.05 means the market is supplying 95%
   of the signal and your model is decoration.
4. **Calibration ≠ edge.** The anytime-TD model calibrated beautifully (0.259 predicted vs
   0.258 actual) and was still unbettable.
5. **Grade only on official CFBD finals** (`homePoints`/`awayPoints`). PBP scores are
   garbage-time truncated — that once produced a fake 64.8% ATS.
6. **Never average American odds across the ±100 gap.** Aggregate in payout space; pair
   over/under at the same point per book. Ignoring this produced a fake +40% props ROI.
7. **Self-calibrating models need a warm-up season** before the first test season.
8. **Team names: exact + alias table only.** Prefix matching mapped FCS Arkansas–Pine Bluff
   to Arkansas and invented a 43-point edge.
9. **Assert row counts after any merge** (projections must stay at 46,487) — a PPA join on
   `(season, week, player)` silently duplicated 2,234 rows because keys collide across teams.
10. **Before refining an estimator, check the objective can identify the parameter.** The
    props blend weight looked like a coarse-grid problem for two days. Measuring the log-loss
    surface showed it was flat — ±0.05 costs 0.000129 per row — so `w` was barely identified
    while the book swung wildly with it. A finer grid would have estimated noise more
    precisely. Not estimating it at all **halved the bootstrap sd (15.3 → 7.0)**.
11. **Sweep every grid-searched hyperparameter the way you sweep a threshold.** The props
    market-blend weight is fitted over `arange(0, 0.65, 0.05)`. Moving it ONE step swings the
    season **+0.4 → +30.0 → +54.2 units**, and the fitted 0.30 sits directly beside the
    near-zero outcome. A parameter chosen by grid search is a threshold in disguise, and
    deserves the same plateau-vs-spike test. (Requiring bets to clear EV at *every* w does not
    fix it — the intersection is just the most conservative w.)
12. **A backtest constraint must not leak into production.** Props calibration is fitted on
    2023 alone so 2024–25 stay out-of-sample — correct for *measuring*. But the live picks
    read those same coefficients, so the running system was pricing 2026 bets off one season
    and ignoring two. Backtest and production want different fits; give them different fits.
13. **A failed merge must yield NaN, never a default.** `np.where(team_id == home_id, spread,
    -spread)` hands *both* teams the away sign when `home_id` is NaN. That single missing
    branch overstated the props edge 2.2x for months. Any two-way sign assignment needs an
    explicit third case for "unknown".
14. **Never take a game attribute from play-by-play when a games table has it.** PBP is
    missing for ~5% of games and the gaps are *not random* — they cluster in FCS buy games,
    i.e. the biggest spreads and the earliest weeks, which is exactly where it hurts.
15. **Demean within player before believing any "X affects player Y" effect.** A raw
    ratio-to-trailing-average conflates roster composition with the within-player effect and
    overstated a game-script effect 4x (an apparent 20% band; the honest fixed-effects fit
    says ~5%). Good teams blow opponents out *and* run deeper committees.
16. **Check that an effect is specific to the side your mechanism names.** "Favourite's
    starters rest in blowouts" looked real until the *underdogs* showed the same skew — which
    falsifies the mechanism no matter what the t-stat says.
17. **Re-derive every filter that was fitted on a region you later fixed.** The props week
    gate was tuned on weeks 1–5, which is where 81% of the sign-bug rows lived; on clean data
    the old rule was wrong and the corrected one is worth +6.5 units/season.

---

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate                      # Windows;  source .venv/bin/activate on POSIX
pip install pandas numpy pyarrow scikit-learn scipy requests python-dotenv \
            anthropic tabulate torch --index-url https://download.pytorch.org/whl/cpu

cp .env.example .env                        # then fill in your keys
```

Required keys: `CFBD_API_KEY` (free, [collegefootballdata.com/key](https://collegefootballdata.com/key))
and `ODDS_API_KEY` ([the-odds-api.com](https://the-odds-api.com) — props and historical
markets need a paid tier). `ANTHROPIC_API_KEY` powers news/injury extraction;
`OPENAI_API_KEY` is reserved for embeddings and not yet wired.

> `.env` and `warehouse/` are gitignored. `.env.example` is a template — every key line in it
> is blank, and it must stay that way.

Then build the warehouse and confirm the backtests reproduce:

```bash
python -m ingestion.run_ingest --cfbd        # CFBD games, lines, rosters, ratings
python -m ingestion.run_ingest --pbp 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
python -m features.epa_ratings               # opponent-adjusted EPA
python -m backtest.spread_history            # 8-season spread walk-forward
python -m backtest.props_vs_book             # props vs real book prices
```

`--pbp` takes an explicit list of seasons (it does not expand ranges); `--all` is shorthand
for `--cfbd` plus PBP for 2023–2025.

---

## Weekly in-season routine

```bash
# Sun/Mon — bet the opener
python -m picks.ml_spread_picks              # early-season big-dog spreads
python -m picks.paper_trades import-ml-spread

# Tue — data refresh
python -m ingestion.run_ingest --all
python -m ingestion.scrapers.ourlads_depth
python -m ingestion.news_injuries            # daily-ish

# Wed — main board
python -m picks.edge_report                  # spreads/totals + CLV log
python -m picks.paper_trades import-edges

# Fri/Sat — derived + props markets (they post late)
python -m picks.q1_picks                     # Q1 dog on big mismatches (all year)
python -m picks.first_half_picks             # 1H spreads, weeks 1-5 only
python -m picks.prop_picks                   # PRIME all year; STANDARD wks 5+
python -m picks.parlay_builder
python -m picks.paper_trades import-1h
python -m picks.paper_trades import-q1
python -m picks.paper_trades import-props
python -m picks.paper_trades import-parlay

# Sun/Mon — settle (prop grading needs fresh player logs first)
python -m ingestion.bulk_pbp 2026
python -m features.player_stats
python -m picks.paper_trades settle          # DNP props void
python -m picks.paper_trades report
```

Paper ledger: `warehouse/paper_trades.parquet` (flat 1u stakes, CLV tracked).

**Offseason:** the "CFB August Refit" scheduled task runs `python -m scripts.august_refit`
weekly from 2026-08-17, pulling 2026 returning production/rosters/portal as CFBD publishes
them and re-fitting every coefficient. Its report leads with the big-dog picks. Delete it
once the season starts:
`Unregister-ScheduledTask -TaskName "CFB August Refit"`.

---

## Architecture

*This section zooms. Read the diagram for the shape of the system, then expand only the
layer you're touching.*

### Zoom 0 — the whole system in one picture

```
                 ┌──────────────────────────────────────────────┐
                 │  warehouse/  (parquet — the layer interface)  │
                 └──────────────────────────────────────────────┘
                    ▲          ▲          ▲              ▲
                    │          │          │              │
  SOURCES  ──►  INGESTION ──► FEATURES ──► MODELS ──► BACKTEST ──► PICKS ──► reports/
  CFBD          cache-first   as-of       fair        walk-        live       + paper
  Odds API      quota-logged  tables      lines/      forward      board      ledger
  sportsdataverse                         probs       validation
  public pages                                            │            ▲
                                                          │            │
                                                          └────────────┘
                                              warehouse/model_coefs.json
                                       backtests WRITE it · picks READ it
```

Data flows left to right. **One arrow flows backward**, and it is the most important edge in
the design: backtests are the *only* writers of fitted coefficients, and the live pick
modules are pure readers. That loop is what stops the live board from drifting away from
what was actually validated.

<details>
<summary><b>Zoom 1 — the six layers</b> (what each one owns, and its contract)</summary>

<br>

| Layer | Owns | Reads | Writes | Invariant |
|---|---|---|---|---|
| **`ingestion/`** | Talking to the outside world | APIs, public pages | `warehouse/raw/*` + `warehouse/parquet/*` | Every response cached to disk with an `as_of` stamp; every call quota-logged. Re-runs are free and reproducible. |
| **`features/`** | Turning raw data into as-of tables | warehouse parquet | one parquet per feature family | **Nothing may use data from after the game it describes.** |
| **`models/`** | Fair lines and probabilities | feature tables | projections parquet | Output is a *fair line*, never a bet. No price awareness. |
| **`backtest/`** | Deciding what is real | models + historical prices | results parquet + **`model_coefs.json`** | Walk-forward only; warm-up season before the first test season. |
| **`picks/`** | Applying validated rules to live prices | `model_coefs.json` + live odds | `reports/*.md` + paper ledger | Reads coefficients — never re-fits them. |
| **`scripts/`** | Unattended orchestration | all of the above | `reports/refit_*.md` | Idempotent; failed steps log and retry next run. |

**The parquet warehouse is the interface.** Layers communicate through files on disk, not
through each other's internals, so any stage can be re-run in isolation and each expensive
step (a 17k-credit odds pull, a 113k-row PPA join) is paid for once.

</details>

<details>
<summary><b>Zoom 2 — inside each layer</b> (module-by-module)</summary>

<br>

<details>
<summary><code>ingestion/</code> — cache-first, quota-aware</summary>

<br>

Both API clients hash endpoint+params into a disk cache and log every real network hit, because
the quotas are the binding constraint: CFBD is metered per month, and Odds API *historical*
endpoints cost **10 credits × market × region × event** — a single careless backfill can burn
tens of thousands of credits.

| Module | Role |
|---|---|
| `config.py` | Loads `.env`, defines every warehouse path, and owns `save_coefs()` / `load_coefs()` — the read/write API for `model_coefs.json` |
| `cfbd_client.py` | CollegeFootballData client; cache + `usage_log.jsonl` |
| `odds_client.py` | The Odds API client; cache + `odds_usage.jsonl`, `remaining()` to check before big pulls |
| `bulk_pbp.py` | sportsdataverse play-by-play parquet, 2016–2025 |
| `historical_props.py`, `historical_1h.py`, `historical_team_totals.py`, `historical_td.py` | Per-event historical odds pulls (period and prop markets 422 on the bulk endpoint, so they must be walked event by event) |
| `news_injuries.py` | ESPN injury feed → Claude structured extraction |
| `scrapers/` | `ourlads_depth` (depth charts), `wiki_staff` (batched query API, 50 titles/req), `splits_capture` (public betting splits) |

</details>

<details>
<summary><code>features/</code> — the as-of discipline lives here</summary>

<br>

Every table is keyed by `(season, week)` and computed using only prior information. The three
mechanisms, in the order you'll meet them:

- **`endWeek` on the API call** — `matchups.py` asks CFBD for season stats *as they stood
  before week N*, so leakage is impossible by construction.
- **`.shift(1)` before the EWMA** — trailing player form (`player_stats.py`, `td_model`) always
  drops the current game before averaging.
- **Fit on strictly earlier seasons** — `roster_priors.py` fits its mapping on prior seasons
  only, then predicts the target season from offseason inputs known before kickoff.

| Module | Produces |
|---|---|
| `epa_ratings.py` | Opponent-adjusted offense/defense EPA via sparse ridge, recency half-life, FCS pooling |
| `roster_priors.py` | Preseason team prior from returning production + portal + recruiting |
| `matchups.py` | **The props breakthrough.** Offense pass/rush profile vs what the opponent defense allows, percentile-ranked |
| `coach_db.py`, `coach_tendencies.py`, `coach_usage.py` | Staff history; PROE and pace are coach-sticky (r=.65/.50), 2H-adjustment is **not** (r=.06) |
| `player_stats.py`, `player_advanced.py` | Per-game logs; usage and per-player PPA |
| `defense_profiles.py`, `defense_players.py` | Unit defense EPA; per-team key coverage DBs and pass-rushers |
| `transfer_elo.py` | Tier translation — G5→P4 RBs keep 66% share / 86% efficiency; QBs travel best; star-RB residual +2.0 share pts |
| `ol_continuity.py`, `first_half.py`, `situational.py`, `availability.py` | Line continuity, 1H splits, rest/travel, depth availability |

</details>

<details>
<summary><code>models/</code> — fair lines, price-blind on purpose</summary>

<br>

Models never see a price. They emit a fair line or a probability; deciding whether that's a
*bet* is the backtest's and pick layer's job. Keeping this boundary clean is why a
well-calibrated model (anytime TD) could be caught as unbettable instead of shipped.

| Module | Approach | Status |
|---|---|---|
| `game_sim.py` | Linear per-team points → spread, total, team total; self-calibrating | **production** |
| `props.py` | Usage share × team volume × efficiency; Gamma tails for yardage, NegBinom for receptions | **production** |
| `ml_spread.py` | Gradient boosting, weeks 1–5 (MAE 14.53→13.72 vs linear) | paper-trial |
| `ml_sim.py` | Gradient boosting margin/total | **did not beat linear** — kept linear |
| `dl_props.py` | Torch MLP, player embeddings, full conditional distributions via NLL | not adopted — variance hypothesis falsified |
| `td_model.py` | Red-zone share × expected team TDs → Poisson → logistic recalibration | ⚠️ **DO NOT BET** — calibrated but doesn't beat the price |

Note the distribution shapes: yardage is right-skewed, so a normal tail **overstates**
P(over). Switching pass yards to Gamma moved it from −2.1% to +6.5% ROI.

</details>

<details>
<summary><code>backtest/</code> — where claims get killed</summary>

<br>

This layer is the source of truth for every number in this README. Each validator walks
forward season by season, grades against official CFBD finals, and — if the edge survives —
writes its fitted coefficients to `model_coefs.json`.

Every validator is expected to run its own controls: a **blind / no-model** version, a
**market-only** version, and a **median-price** version. Three of the six candidate edges
died at exactly this step.

| Module | Validates |
|---|---|
| `spread_history.py` | 8-season spread walk-forward — the big-dog tiering |
| `first_half.py` | The BIG-DOG 1H play |
| `props_vs_book.py` | Props vs real book prices, incl. the EV band tiering |
| `ml_value_history.py` | Moneyline, 8 seasons with real prices — **rejected** |
| `td_vs_book.py` | Anytime TD, incl. the controls that rejected it |
| `coverage_injury.py` | Coverage-injury angle — real effect, **no edge** |
| `q1_spreads.py` | The Q1 big-dog edge, incl. the linear-vs-saturating mechanism table |
| `alt_spreads.py` | Alternate ladders — **rejected**, the "best" rung moves by subset |
| `loss_review.py` | Post-mortem: *why* losing bets lost (volume vs efficiency vs defense) |
| `props_trend.py` | Share-momentum correction — **rejected**, better projection ≠ better bet |
| `confidence_report.py` | Calibration: ML under-confident, spread/prop raw confidence inflated |

</details>

<details>
<summary><code>picks/</code> and <code>scripts/</code> — live application</summary>

<br>

The pick modules apply already-validated rules to live prices. They load thresholds and
coefficients from `model_coefs.json`; the values hardcoded in these files are **fallbacks
only**, so a refit propagates without touching code.

| Module | Emits |
|---|---|
| `ml_spread_picks.py` | Early-season big-dog spreads — run **Sun/Mon**, off the opener |
| `first_half_picks.py` | 1H spreads incl. the BIG-DOG 1H tier (2u) |
| `prop_picks.py` | Band-tiered props (PRIME 5–8% at 2u, standard 8%+ at 1u) |
| `edge_report.py` | Spread/total board; appends to the CLV log every run |
| `parlay_builder.py` | 2–6 leg ladder from validated legs only; longshot guard, no ML legs |
| `paper_trades.py` | The ledger: import → settle vs official finals → season P&L, CLV tracked |
| `scripts/august_refit.py` | Unattended: pull new data → rebuild → re-fit → `reports/refit_<date>.md`, led by the big-dog picks |

`edge_report.py` also owns the **team-name alias table**, which is load-bearing: exact +
alias matching only, because prefix matching once mapped FCS Arkansas–Pine Bluff onto
Arkansas and invented a 43-point edge.

</details>

</details>

<details>
<summary><b>Zoom 3 — tracing one bet end to end</b></summary>

<br>

How "pick Toledo +11.5 vs Michigan State, 1H" actually gets produced:

```
1. ingestion/run_ingest --cfbd        → games + lines parquet     (spreadOpen: -11.5)
2. features/epa_ratings               → as-of team ratings        (through last week only)
3. features/roster_priors             → preseason prior           (week 1-5: prior-weighted)
4. models/game_sim                    → fair line                 (Toledo +5.2 → edge 6.3)
5. backtest/spread_history            → is this pattern real?     (|spread|≥17 + edge≥6:
                                                                   56.9-62.3%, 8/8 seasons)
   └─ writes thresholds ──────────────→ warehouse/model_coefs.json
6. backtest/first_half                → same games, 1H line?      (64.8% — better still)
7. picks/first_half_picks             → reads coefs + live 1H line
                                      → "BIG-DOG 1H, 2 units"     → reports/first_half_*.md
8. picks/paper_trades import-1h       → ledger row, CLV logged
9. picks/paper_trades settle          → graded on official CFBD finals
```

Steps 5 and 6 are the ones that don't exist in most betting systems, and they're the reason
this one has a short list of plays instead of a long one. **The model proposes; the backtest
disposes.**

</details>

### Repo map

```
ingestion/    API clients (disk-cached, quota-logged), bulk PBP, historical odds pulls
  scrapers/   Ourlads depth charts, Wikipedia staff, Action Network public splits
features/     EPA ratings, roster priors, coach DB + tendencies, OL continuity,
              defense profiles, player stats, transfer Elo, matchups
models/       game_sim (linear), ml_sim / ml_spread (gradient boosting),
              props, dl_props (torch), td_model
backtest/     walk-forward validators — the source of truth for every claim above
picks/        weekly live reports, parlay builder, paper-trade ledger
scripts/      august_refit.py (unattended)
warehouse/    parquet data + model_coefs.json   (gitignored)
reports/      dated markdown output
```

## Docs

| File | Contents |
|---|---|
| **`HANDOFF.md`** | **Start here.** Current state of every component, all findings, all rejections, every gotcha |
| `CFB_PREDICTOR_PLAN.md` | Master plan and phase statuses |
| `DATA_CATALOG.md` | 34 data items with free sources |
| `GRAB_LIST.md` | Purchase list and vendor notes |

## Data sources & ethics

CFBD API (Tier 3, 75k calls/mo), The Odds API (5M credits), sportsdataverse bulk parquet,
plus scrapes of **free public pages only** — Ourlads, Wikipedia, Action Network's public
betting page. Raw responses are saved with `as_of` timestamps, rate-limited and cached.

Paywalled data (PFF, SIS, Action Network Pro) is **not** scraped — it's licensed commercial
data and doing so violates ToS. If true player-vs-player coverage charting is ever needed,
it has to be bought.

---

*Research and backtesting tool. Backtested edges are historical and carry no guarantee;
bet only what you can afford to lose.*
