# College Football Betting Predictor

A walk-forward backtested system for finding — and honestly rejecting — betting edges in
college football markets: spreads, 1st-half spreads, player props, totals, and moneylines.
All ~134 FBS teams, 2016–2025 history, pregame only.

The headline result of this project is not a model. It's a **short list of edges that
survived control tests**, and a much longer list of plausible-looking edges that didn't.

---

## 🚨 Read this first — the 2026-08-05/06 audit

**A sample-selection artifact had manufactured most of this project's headline edges,
and it is now fixed.** The garbage-time play filter (`wp_before` ∈ [0.04, 0.96]) fed the
*game universe*, not just the ratings, so any blowout in which every play was filtered
**vanished from the sample entirely**. Membership was conditioned on the outcome.

Coverage was **56.6%** at |spread| ≥25, and the deleted games were the ones where the dog
got buried (mean final **−40.5** vs **−27.9** for those kept). That single filter
produced the "the dog's Q1 deficit plateaus near 4 points" mechanism: flat at −3.8 → −4.1
inside the sample, but **−3.4 → −10.2** across the real universe. The quarter never
saturated.

`game_table` now builds from the CFBD games table — play-by-play decides ratings, never
membership. **Coverage is 100% in every spread band.** Full detail: `HANDOFF.md` §1–14.

Everything below is re-derived on the clean universe. Old figures are void.

## Validated plays (re-derived on the clean sample, 2026-08-06)

| Play | Rule | u/season | Status |
|---|---|---|---|
| **Props PRIME** ⭐ | EV 5–8% (not higher), weeks 5+ | **+36.6** | the book — and the only play with stable, controlled evidence |
| **BIG-DOG 1H** ⭐ | \|spread\| ≥17 (wks 1–5) / ≥21 (6+), model on dog ≥6, bet the 1H line | **+14.6** | best derived play; **inherited all the ≥25 games** from the retired Q1 tier |
| **Props STANDARD** | EV ≥8%, weeks 9+ only | **+12.0** | |
| **Props PROBE** | EV 4–5%, weeks 5+ | **+10.1** | staked down: newest evidence |
| **Spread STANDARD** | \|spread\| 17–25, wks 1–5, model on dog ≥6 | +7.9 | ⚠️ 8-season read is +0.7%, t=+0.14, 4-of-8 — **unproven**, kept for diversification |
| **Q1 STANDARD** | Full \|spread\| **17–25** → dog on the Q1 line | +5.8 | ⚠️ unproven, kept for diversification |
| ~~Q1 PREMIUM ≥25~~ | ~~dog on the Q1 line~~ | **RETIRED** | −0.7 u/szn; CI [−7.3, +23.8], 2025 negative. Games handed to 1H |
| ~~TEASER 1H~~ | ~~2-team +10 on 1H legs~~ | **OUT OF BOOK** | −0.0 u/szn realised; "83.8% / +30.3%" quoted the **joint** rate as the leg rate |
| ~~Big-dog spread PREMIUM~~ | ~~\|spread\| ≥25~~ | **REMOVED** | 48.6% / −7.0% / t=−1.81 / 2-of-8 over 8 clean seasons |

**405 bets and +97.6 units/season.** From $1,000 at 1u = 2% with a 50% edge haircut (70%
on props): **median $2,006**, 5th percentile $1,090, **3.6%** chance of a losing season,
median drawdown −16.4%. *(Pre-audit this claimed $1,956 and 2.5%.)*

> ⭐ **Retiring a play can be worth more than the play was.** Dropping Q1 PREMIUM did not
> lose those games — |spread| ≥25 now flows to BIG-DOG 1H, which grades **+9.5% / 57.1%**
> on exactly that population. 1H went **+3.7 → +14.6 u/szn** and the book **+73.7 → +85.3**,
> on **46 fewer bets**. Units per bet rose 0.163 → 0.215 (+32%).
>
> ⚠️ **But do not strip the book down to props.** Props have the best per-bet ROI, yet a
> props-only book is *worse*: median $1,372 and **P(loss) 23.1%**, versus $1,641 and 12.5%
> for the full book. The unproven-but-positive plays (Q1 STANDARD, SPREAD STANDARD) pay for
> themselves in diversification. **"Fails the significance bar" ≠ "remove from the
> portfolio" — remove plays with negative or zero expectation, keep positive-but-unproven
> ones at reduced size.**

> ✅ **Line-to-game integrity, verified 2026-08-06.** Prop lines are genuinely
> player-specific: 2024 wk12 pass_yds had 433 quotes across 48 players and **105 distinct
> lines**, correctly ordered by quality. `corr(line, actual)` runs +0.45–0.50 and
> `corr(line, proj)` up to +0.84. **One real bug was found and fixed**: CFBD tags its
> week-0 kickoff weekend as `week = 1`, so 267 player-weeks carried *two* game_ids and a
> single quoted line was graded against **both** (117 rows, 0.83%). Lines are now pinned
> to the game whose kickoff they were quoted for — 0 duplicates, and the book rose
> +85.3 → **+87.0 u/szn** because the double-graded rows were net negative.
> **A week number is not a game key.**

### 🆕 Volume markets — projections shipped, PAPER ONLY

**We predict attempts better than the yards we bet** (rush att corr **0.541** vs 0.448;
pass att **0.454** vs 0.358; targets **0.491** vs 0.462) — because yards = attempts ×
efficiency, and efficiency is the noisy factor bolted on. `models/props.py` now exposes
`proj_rush_att`, `proj_pass_att`, `proj_pass_comp`; these numbers were always inside the
yardage projections, just never surfaced.

⚠️ **Recalibrate before use** — raw projections run systematically low (pass att: proj 23.9
vs actual 25.3), which on a 50/50 line would push us to the under every time. Fitted
2024→2025: `rush_att` b=0.939, `pass_att` b=0.936, `pass_comp` b=0.968; residual bias falls
to +0.13…+0.35 but MAE barely beats naive, so a real edge is **plausible, not proven**.

⚠️ **They cannot be backtested.** Traditional US books rarely post NCAAF attempts, and The
Odds API's historical archive has **none** (3 mid-season 2025 events, `us` + `us2`:
`player_pass_tds` returned 4–5 books every time; attempts/completions returned **zero**).
That is a limit of *our data source*, not proof the market is absent — **DFS pick'em books
(PrizePicks, Underdog, Sleeper) post exactly these lines**, near the median at roughly even
money. Those are set for engagement rather than sharp balance, which is where a model edge
can live. Plan: **forward paper grading** — log line + projection, grade from
`player_game_logs` (which now carries attempts, completions and TDs).

**Passing TDs are quoted and were rejected**: 4,227 QB games, mean 1.57 TD, observed sd
1.25, √1.57 = 1.25 — the variance is *entirely* Poisson counting error. Best predictor
R²=0.039 vs 0.097–0.101 for the markets we beat. The probe cost ~70 credits and avoided a
~54,000-credit pull.

### Why the projections "fail" — and four fixes that didn't work

**The market out-predicts us on every stat** (pass_yds MAE 69.4 vs 63.9; rush 31.3 vs 30.0;
receptions 1.73 vs 1.72). Our edge was never better central estimates — it's the
recalibration, the gamma/NB tails, and a 30% model weight that disagrees in the right
places. The error tree:

| stat | attempts | efficiency | of the attempts error |
|---|---|---|---|
| rush_yds (MAE 29.2) | 36% | 28% | **player share 55%**, team volume 10% |
| pass_yds (MAE 73.8) | 29% | 22% | share 31%, team volume 34% |

Single-game efficiency is essentially unpredictable (ypc corr **0.104**, ypa **0.122**) and
that quarter of the error is **irreducible**. Rejected, with evidence: shrinking efficiency
harder (shipped constants already optimal), recalibrating player share (worse out-of-sample
on all three), filtering out high-line props (top quintiles still positive both seasons),
and an availability adjustment (below).

> ⚠️ **Depth charts and injury news cannot be backtested here.** `depth_charts.parquet` has
> **two as-of dates, both 2026-07-28/29** — one preseason snapshot, no history.
> `news_extractions.parquet` has **5 rows**. The leak-free substitute — weeks since a
> player's last appearance — was built and the bias it targets is **real and stable in every
> season** (rush-share error at gap ≥3: −0.0704 / −0.0411 / −0.0237). **Correcting it still
> cost −11.6 u/szn** (+87.0 → +75.4), tested both raw and normalised. `GAP_MODE = "off"`.
>
> ⭐ **The lesson: a real bias in an input does not mean correcting it improves betting.**
> The recalibration and market blend were fitted *around* the uncorrected projection, so
> changing the input shifts the whole EV surface. **Judge a projection change by the book,
> never by the projection's own accuracy.**

> ✅ **All five props thresholds re-swept 2026-08-06 and every one held** (rule 17 — the
> game-pinning fix changed the dataset). EV floor 0.04, PROBE/PRIME 0.05, PRIME/STANDARD
> 0.08, week gates 5/9, min books ≥2. Second time they've survived a data change.
>
> ⚠️ **The `n_books` trap.** Accepting single-book lines reads **+65.2 u/szn** vs +58.7,
> positive both seasons — apparently free units. Graded *alone* those bets are **52.0% hit,
> +1.3% ROI, t=+0.19, 2025 negative**: 102 bets/season of near-zero edge. The pooled total
> rises only because the bet *count* does. With one book, the market probability we price
> against comes from **the same quote we'd bet** — no consensus to disagree with.
> **Grade an expansion on its own, never by the pooled total.**

### ⭐ The edge is decaying — the most important table here

| play | 2023 | 2024 | 2025 |
|---|---|---|---|
| Q1 PREMIUM | +25.3 | +5.6 | **−6.9** |
| BIG-DOG 1H | +16.3 | +1.9 | +5.6 |
| Spread (removed) | +4.0 | −11.1 | **−22.9** |
| TEASER 1H | +3.2 | +2.9 | **−2.9** |
| **PROPS (all)** | n/a | **+57.9** | **+56.0** |

Every derived-line play peaks in 2023 and decays. **Props are the only stable one** — and
now ~78% of the book. Treat every derived play as provisional and re-check it in 2026.

### Why the big-dog spread play was removed

22 pre-specified variants, 8 clean seasons, bar = positive every season **and** bootstrap
CI excluding zero. **Nothing passed.** Blind (no model filter) *beat* model-selected
(−4.6% vs −7.0%); raising the edge threshold made it **worse** (≥12 → −10.5%, and the bulk
of the sample covers just 46.8% while low-edge games sit near 53%) — the model's confidence
is anti-predictive there. Fading the favourite also fails (51.4% vs a 52.4% break-even), so
it is not a sign error, it is no edge. One slice looked great (home dogs 65.1%, +24.3%) on
**n=43** with a CI spanning zero — the 1-in-22 you expect from 22 looks.

> ✅ **Widening the prop-line pull (2026-08-05) added 26% more data for 9,288 credits and
> lifted props +40.5 → +56.9 units/season with no model change** — the single cheapest
> improvement in the project, and it survived the audit intact because props are graded
> against real prop lines rather than the contaminated game sample. All five fitted props
> thresholds were re-swept on the enlarged sample and **every one held**.
> *(The bankroll figures once quoted here — $1,811 median — are superseded; see the table
> above.)*

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

> 🚨 **The 2026-08-04 stability audit is VOID — and it is the cautionary tale of this
> project.** It swept thresholds, left each season out, re-priced at median, and
> block-bootstrapped, then concluded Q1 was "textbook": ROI climbing monotonically from
> +23.3% at ≥19 to +64.1% at ≥31, LOSO +37.4/+51.6/+47.7%, bootstrap +29.4 u/szn with a 90%
> CI of [+11.0, +49.1]. Every one of those tests passed, and every one of them ran on a
> sample that had already deleted the games where big dogs got buried.
>
> **No amount of resampling can detect a bias in which rows exist.** Bootstraps, LOSO, median
> price and threshold plateaus all condition on the sample you hand them. The rule that would
> have caught it is #18 — build the universe from a table of records — not another control
> test. Q1 PREMIUM's real value on the clean sample is **−0.7 u/szn**.

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

> ⚠️ **Numbers below predate the 2026-08-05/06 audit** and were measured on the contaminated game sample. The reasoning holds; the magnitudes do not. Current book is in *Validated plays* above; full detail in `HANDOFF.md` §1–14.

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

> ⚠️ **Numbers below predate the 2026-08-05/06 audit** and were measured on the contaminated game sample. The reasoning holds; the magnitudes do not. Current book is in *Validated plays* above; full detail in `HANDOFF.md` §1–14.

Flat 2% staking is the honest way to *state* an edge, but it is not how you grow a bankroll.
Two levers, both validated on real graded outcomes (`backtest/growth_paths.py`):

**Compounding** — stake a % of the *current* bankroll, not the starting one. Free, no new
bets, no extra per-bet risk. Median $1,814 → $1,990 at the same 2%. Shipped in
`picks/paper_trades.py`; every row records the dollar value of 1u at placement time so the
history stays gradeable as the unit moves (`bankroll` / `backfill-units` commands).

**Parlays — re-derived 2026-08-06 on props legs.** The old version of this section was
built on Q1 legs ("two +45.7% Q1 legs price near +112%"); the audit put Q1 PREMIUM at
−0.7 u/szn, so that is gone. Props legs replace it, and they are *cleaner*:

**Cross-game props legs are essentially exactly independent** — joint hit **36.8%** vs
**36.8%** predicted from the marginals, **lift 0.999** across 6,212 pairs. Parlay EV
**+30.0%**, week-block bootstrap CI **[+11.3%, +48.6%]**, positive both seasons. (The old
"legs are not independent, 1.14× variance" warning applied to big-dog legs, which really
were all the same structural bet. Props are not.)

> ⚠️ **But parlays do not add units — they add capital efficiency.** Pairing the *same*
> legs at the same unit size:
>
> | strategy | tickets | staked | won | /szn | ROI |
> |---|---|---|---|---|---|
> | all singles (shipped) | 504 | 745u | +113.9u | **+56.9** | +15.3% |
> | pairs + leftovers | 258 | 261u | +67.9u | **+34.0** | **+26.0%** |
>
> Half the tickets deploys a third of the capital, so profit falls even as ROI nearly
> doubles. **Parlays double EV per dollar staked, not per leg.**

**When to use them.** The props schedule needs **34u/week average, 54u peak** — at 1u=2% of
$1,000 that peak week risks **108% of bankroll**, so capital really does bind. Median final
bankroll by weekly-exposure cap:

| weekly cap | singles | pairs |
|---|---|---|
| 15% | $1,326 | **$1,443** |
| 25% | $1,532 | **$1,629** |
| 40% | **$1,838** | $1,680 |
| uncapped | **$2,160** | $1,678 |

**Crossover ≈ 30% of bankroll per week.** Running 40%+ through props weekly → bet singles.
Capping exposure below ~30% → parlay. One leg per game either way.

**$1,000 → $5,000 is roughly a coin flip**, not a plan: compound 4% + 2-leg parlays gives a
$3,916 median and **46.6%** chance of $5k — rising to **57.0%** if Q1 limits reach $200 across
books. Getting Q1 down at four books instead of one is worth more than any model change
(P($5k): 42.1% at $25 → 57.0% at $200, then flat). Compounding at 8% busts **3 times in 10**,
which is where aggression stops paying.

**"70–80% hit rate" and parlays are mathematically opposed.** Q1 PREMIUM already hits 75.5%,
but it is 31 bets a season and cannot 5x anything alone. Parlays buy edge by *spending* hit
rate — 2-leg ~51%, 3-leg ~33%. No configuration delivers both.

## Rejected after honest testing

**New markets tested 2026-08-05, all rejected** — each killed by a free mechanism test
before spending a credit: **Q1 tie / 3-way** (the atom is real, P(Q1 margin=0) 18.3% vs
4.6% for a smooth normal, but priced — the dog +0.5 covers 56.9% against a 55.7%
break-even); **team totals** (the line is literally (total ∓ spread)/2, corr 0.9951);
**1H team totals** (+2.36 pts of apparent over-grant on the contaminated sample, **+0.25**
on the full universe); **1H totals** (books *over*-ramp rather than hold a constant);
**1H moneyline** (mechanism real, fully priced); **synthetic Q2** (Q2 takes 32.8% of the
spread vs Q1's 26.6%, non-monotone, two vigs).

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
18. **Never let a row filter decide which units of observation EXIST.** ⭐ The costliest rule
    here. A garbage-time play filter also chose the game sample, so blowouts vanished and
    every big-dog edge inflated. Guarding the *values* is not enough — the code even warned
    that filtered plays give truncated scores, while the same truncation silently decided
    which games were in the study. Build the universe from a table of records; use the
    filtered data only for what it is for.
19. **A backtest result is a claim about a specific estimator — make the live path IMPORT
    it, never re-implement it.** Production ran a preseason-only props projection while the
    backtest measured a trailing-EWMA one (~+11.6 u/szn shipped vs +56.9 measured), and live
    ratings were frozen preseason while backtests refit weekly (corr 0.61). Two copies of a
    decision always drift. When you cannot share code, diff the two outputs on the same input.
20. **Grade against the authoritative column, and check orientation on the minority side.**
    `cover > 0` means the *home* team covered; the dog is home in only 7% of |spread| ≥25
    games, so grading dogs that way measured the favourite. It inverted both spread tiers in
    a report. The parquet's own `won` column settled it.
21. **An empty upstream response is a MISSING answer — never a cached fact, never a reason
    to abandon unrelated work.** A cached `[]` froze two 2026 feeds permanently, and one
    unpublished preseason stat blocked the schedule and betting-line pulls entirely. When a
    guard's failure message is indistinguishable from correct behaviour, make it say how long
    it has been failing.
22. **Reads must not write.** Pricing the board appended to the CLV log — four callers per
    cycle, 2,036 rows describing 86 games. And when you do log, **de-duplicate on the signal,
    not the clock**: keep line *moves*, so a real move is always recorded and an unchanged
    board costs nothing.
23. **Make the report and the simulation describe the same population.** The bankroll sim
    resampled week-blocks from a pool where props existed in only 2 of 3 seasons, so it
    simulated a +54.9 u book while printing a +72.6 one.
24. **Count your looks before believing the best slice.** 22 variants were tested to rescue
    the big-dog spread play; the winner was home dogs at 65.1% on **n=43** with a CI spanning
    zero. That is exactly what 22 looks buys you by chance.
25. **Confirm an API's default before "fixing" a parameter you assume is missing.** An audit
    reported that bowls could never enter the schedule for want of `seasonType=postseason`;
    the endpoint returns them by default. The real December bug was elsewhere — CFBD labels
    postseason games week 1/13/14, which reset every week-gated rule in the repo.

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

# Tue — data refresh. bulk_pbp MUST run weekly: the live season's file grows
# every Saturday and the ratings refit as-of the upcoming week off it.
python -m ingestion.run_ingest --all
python -m ingestion.bulk_pbp 2026            # or: python -m scripts.august_refit
python -m ingestion.scrapers.ourlads_depth
python -m ingestion.news_injuries            # daily-ish

# Wed — main board.  `edge_report` is the ONLY thing that writes the CLV log,
# and only for games whose number actually moved. Everything else reads.
python -m picks.edge_report                  # spreads/totals + CLV log
python -m picks.paper_trades import-edges

# Thu-Sat — DFS pick'em lines (rush/pass attempts + completions live here,
# NOT at traditional books). Append-only, logs line MOVES, safe to re-run.
python -m ingestion.dfs_lines fetch

# Thu/Fri — DEPTH CHART SNAPSHOT is AUTOMATED (scheduled task "CFB Depth
# Charts", Thu+Fri 10:00). Nothing to run by hand. Check it is still alive:
python -m ingestion.scrapers.ourlads_depth history   # shouts if >10 days stale
python -m ingestion.dfs_lines import-pp pp_export.csv   # PrizePicks, manual

# Fri/Sat — derived + props markets (they post late)
python -m picks.q1_picks                     # Q1 dog on big mismatches
python -m picks.first_half_picks             # 1H: BIG-DOG all season, STANDARD wks 1-5
python -m picks.prop_picks                   # nothing before wk5; STANDARD from wk9
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
python -m ingestion.dfs_lines grade          # forward-grade the DFS volume markets
```

**Automatic gates you do not have to remember** — all verified:
`prop_picks` emits nothing before week 5 and no STANDARD before week 9;
`first_half_picks` drops STANDARD after week 5; `ml_spread_picks` emits nothing
after week 5 and is **paper only**; and **all props and 1H suppress the
postseason entirely** (bowl rosters are gutted by opt-outs and the portal —
unmodellable, and outside every threshold's fitted window).

`current_week()` reads the CFBD schedule, takes the first game **strictly
ahead**, and maps the postseason to week 20. Both details were bugs: a 2-day
lookback made 12 of 15 Sunday/Monday runs report the week that had just
*ended*, and CFBD's literal postseason week numbers (1/13/14) made December
report **week 1**, which silently disabled every gate above.

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
                                                                   17-25 only; >=25 REMOVED,
                                                                   48.6%/-7.0% on clean data)
   └─ writes thresholds ──────────────→ warehouse/model_coefs.json
6. backtest/first_half                → same games, 1H line?      (still the better of the two)
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
