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
| **BIG-DOG 1H** | Full-game \|spread\| ≥17, model on dog by ≥6, **bet the 1H line** | **64.8%** | **+23.7%** | 2u | 125 bets, 3/3 seasons |
| **Big-dog spread PREMIUM** | \|spread\| ≥25, weeks 1–5, model on dog ≥6 | 62.3% | +18.8% | 2u | 302 bets, 8/8 seasons |
| **Props PRIME band** | EV 5–8% (not higher — see below) | 59.5% | +12.7% | 2u | ~200/season |
| **1H standard** | Any side, edge ≥2, weeks 1–5 | 57.8% | +10.3% | 1u | 3/3 seasons |
| **Big-dog spread STANDARD** | \|spread\| 17–25, weeks 1–5 | 56.9% | +8.6% | 1u | 8/8 seasons |
| **Props STANDARD** | EV ≥8% | 54.0% | +3.1% | 1u | ~190/season |

Realistic volume: **~390 props + ~110 spread/1H bets per season.** Spread edges are
confined to weeks 1–5; props run all year.

**Bet the opener, not the close.** Grading the big-dog play against `spreadOpen` beat the
closing line in every season tested (PREMIUM 65.9% vs 62.9%). Run spread picks Sunday/Monday.

### Two counter-intuitive results worth internalizing

- **Props EV bands are non-monotone.** The 5–8% EV band (59.5%) beats the 8%+ band (54.0%).
  A very large computed edge usually means a stale line or a model error, so the top band is
  staked **down**, not up.
- **Bigger spreads are better, not riskier.** An old `|spread| ≤ 21` "blowout guard" was
  suppressing the single best edge in the system. Spreads now use the full board.

## Rejected after honest testing

Moneyline (−5% ROI), totals (dead across 10 seasons), anytime TD, `rec_yds` props, weather,
rest/travel/altitude, coverage-injury (real on-field effect, zero betting edge),
opponent-adjusted defense EPA in props, situational usage, player PPA, deep-learning
distributional props, home underdogs, team totals (demoted to a standalone market bias).

The recurring finding: **the market prices everything observable** — weather, rest, defense
quality, player quality, announced injuries. Every surviving edge comes from
**derived-line shortcuts** (books setting 1H at a flat 56.8% of the full spread, team totals
as `(total ± spread)/2`) applied to big early-season mismatches.

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
python -m picks.first_half_picks             # 1H spreads, weeks 1-5 only
python -m picks.prop_picks
python -m picks.parlay_builder
python -m picks.paper_trades import-1h
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
