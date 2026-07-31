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
| **Anytime TD — DO NOT BET** | `models/td_model.py`, `ingestion/historical_td.py`, `backtest/td_vs_book.py` | Built to add volume (~70 quoted players/game, all season). Model is **well calibrated OOS** (0.140 pred vs 0.141 actual; 0.259 vs 0.258 overall) via red-zone-share × expected-team-TDs → Poisson → logistic recalibration. **But it does not beat the price.** Blend weight fits at 0.05; a market-only control with no model does as well or better (EV≥10%: +31.7% vs +28.9%) picking 95% the same bets; **at MEDIAN prices the model finds ZERO +5% EV bets**; winners sit at 1.67x median price vs a 1.02x typical gap → stale lines/data mismatches. Blind control −16.8% (vig: 35% implied vs 27% actual). 77k lines pulled and kept for research. |
| **Moneyline — DO NOT BET** | `backtest/ml_value_history.py` (8 seasons, REAL prices + line shopping) | **No edge. Model prob −5.0% ROI (1/5 seasons +); 30% blend −6% to −8.6% (0-1/5); "pure line shopping" +12% at EV≥10% is a mirage — +828 avg longshots, seasons +25%/+7.6%/−16.1%.** Probabilities ARE calibrated (65%→63.7%, 74.5%→72.2%, 88.7%→87.4%) but calibration ≠ profit: the market forecasts better and ML vig eats the rest. `ml_value.py` is now INFORMATIONAL ONLY; paper tracker logs ML at **stake 0**; parlay builder **bans ML legs** (MAX_ML_LEGS=0). |
| **1st-half spread** | `features/first_half.py`, `backtest/first_half.py`, `picks/first_half_picks.py` | **NEW BEST PLAY — "BIG-DOG 1H" (2u): take the big-dog selection (full-game \|spread\|≥17, model on dog ≥6) but bet the 1H line → 64.8% ATS / +23.7% ROI, 125 bets, 3/3 seasons. Beats the same games on the full game (62.4%/+19.1%) and the generic 1H edge (57.8%).** Mechanism: big-dog covers are front-loaded (~3.6 pts in 1H vs ~0.9 in 2H) while books set the 1H line at a flat 56.8% of the full spread — a derived-line shortcut. STANDARD 1H (any side, edge≥2, wks 1–5) stays 57.8% / +10.3% at 1u. — derived 1H lines are soft early. Wks 6–15 dead; 1H totals/ML no edge. Live picks via per-event `spreads_h1` (posts near kickoff). 1H lines in `historical_1h_lines.parquet`. |
| Totals sim | `models/game_sim.py` | edge≥5: 54.1% (+3.3%); coach priors worth +2pts |
| Props stack | `features/player_stats.py`, `models/props.py` | priors incl. transfers; beats naive all stats; NB receptions |
| Props vs real lines | `backtest/props_vs_book.py` + **`features/matchups.py`** (cal 2023, OOS 2024+2025) | **UPGRADED 2026-07-30 with offense-vs-defense matchup features: EV≥5% now 54.6% / +3.6% ROI over 1,255 OOS bets, positive BOTH seasons (2024 +4.3%, 2025 +2.8%) — was −0.1% before.** Matchup = CFBD `/stats/season/advanced` pass/rush splits (off ppa/success/explosiveness vs what the opp D allows), percentile-ranked, walk-forward via `endWeek`. Enters the recalibration as `c*(matchup−0.5)*proj`. Then two more upgrades: **gamma tails** for yardage (right-skewed: rec 1.22 / rush 1.07 / pass 0.00 — a normal tail overstates P(over); gamma fixed pass_yds −2.1%→+6.5%) and **rec_yds EXCLUDED from betting** (loses under both distributions, both seasons). **FINAL: EV≥5% → 56.3% / +7.0% ROI over 780 OOS bets (~390/season), positive both seasons; by stat receptions +11.8%, pass_yds +6.5%, rush_yds +6.0%.** **Now BAND-TIERED (2026-07-31)** — EV bands are NON-monotone: 3–5% = 54.2%/+2.2% but wildly inconsistent (2024 +18.8, 2025 −8.8) → **skip**; **5–8% = 59.5%/+12.7%, both seasons + → PRIME 2u**; 8%+ = 54.0%/+3.1%, inconsistent → 1u. Very high "EV" usually signals a stale line or model error, so the top band is staked DOWN not up. Props are the volume engine (~390 bets/szn vs ~70 for spreads). |
| Calibration | `backtest/confidence_report.py` | **ML is under-confident (stated 70–80% → 83.5% actual); spread/prop raw confidence is inflated** |
| Live picks | `picks/edge_report.py`, `ml_value.py`, `prop_picks.py` | run weekly in-season; CLV log auto-appends; each report leads with **best-edge / best-confidence** callouts |
| Paper tracker | `picks/paper_trades.py` | logs ML + spreads/totals + **props**; settles vs CFBD finals & 2026 player logs (DNP props void); CLV tracked |
| Transfer tier translation | `features/transfer_elo.py` | G5→P4 RB keeps 66% share/86% eff; QBs travel best; **star-RB residual bump +2.0 share pts** |

## Weekly in-season routine (season opens 2026-08-29)
```
# ⏱️ RUN SPREAD PICKS SUNDAY/MONDAY — betting the OPENER beats the close in
# every season tested (PREMIUM 65.9% vs 62.9%; STANDARD 60.0% vs 58.2%).
python -m ingestion.run_ingest --all           # Tue: data refresh
python -m ingestion.scrapers.ourlads_depth     # Tue: depth snapshot
python -m ingestion.news_injuries              # daily-ish: injuries + Claude extraction
python -m picks.edge_report                    # Wed: spreads/totals board + CLV log
python -m picks.ml_value                       # Wed: ML value picks
python -m picks.first_half_picks               # Fri/Sat: 1H spreads (wks 1-5!)
python -m picks.prop_picks                     # Fri/Sat: props post late
python -m picks.paper_trades import-ml         # log the week's ML picks
python -m picks.paper_trades import-edges      # log flagged spreads/totals
python -m picks.paper_trades import-props      # log flagged props (Fri/Sat)
python -m picks.paper_trades import-1h          # log flagged 1H spreads (wks 1-5)
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

## Modeling experiments — findings (2026-07-29, Opus session)
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

## Memory
Claude Code persistent memory for this project lives at `~/.claude/projects/C--Users-pcagm-PycharmProjects-PythonProject14/memory/` — survives model switches and new sessions. `MEMORY.md` is the index; `cfb-predictor-project.md` mirrors much of this file.

## Keys (.env — never commit)
CFBD ✓, The Odds API (5M/mo) ✓, OpenAI ✓ (embeddings, not yet wired), Anthropic ✓ (news extraction, working). `.env.example` still contains a real CFBD key the user should blank.
