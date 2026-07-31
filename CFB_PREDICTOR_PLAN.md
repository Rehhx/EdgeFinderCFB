# College Football Betting Predictor — Master Plan

**Goal:** Build a data + ML system that finds edge against sportsbooks on **team spreads, moneylines, totals, and player props** by modeling things the market prices slowly or crudely: roster turnover (transfer portal + recruiting), coaching/staff changes, offensive line quality, and coach play-calling tendencies including in-game/script adaptation.

**North star metric:** Closing Line Value (CLV). If our bets consistently beat the closing line, we have real edge even before long-run ROI is provable.

---

## 1. Where the Edge Comes From (Thesis)

Books are extremely sharp on NFL. College football is softer because:

1. **~134 FBS teams, massive roster churn.** The transfer portal + recruiting turnover means preseason and early-season numbers lag reality. Books lean on power ratings that update slowly.
2. **Player props are thin markets.** Limits are low, lines are derived from team projections + last-season usage. A good usage/role model (especially after transfers and OC changes) beats them.
3. **Coaching changes are priced as a vibe, not a model.** A new OC's scheme change (e.g., air raid → wide zone) shifts pace, pass rate, and player usage in predictable ways the market underweights.
4. **O-line continuity is a known but under-quantified predictor.** Returning starts on the OL correlates with rushing efficiency and sack avoidance; we can quantify it per-team, per-scheme.
5. **Situational play-calling is exploitable for live/derivative markets.** Coaches have measurable tendencies when leading/trailing by quarter, down-and-distance, and field position — useful for 1H/2H lines, team totals, and live betting.

---

## 2. Prediction Targets (Markets)

| Market | Model output | Notes |
|---|---|---|
| Spread (full game, 1H) | Margin distribution (mean + variance) | Bet when our fair line differs from book by threshold (e.g., ≥ 2 pts CFB) |
| Moneyline | Win probability | Best value on underdogs where variance is high |
| Total / team totals | Points distribution | Pace + efficiency + weather driven |
| Player props | Per-player stat distributions (pass yds, rush yds, rec, receptions, TDs) | Usage share × team volume × efficiency; biggest edge area |
| Derivatives (1H/2H, alt lines) | From the same simulation | Coach script/tendency features matter most here |

Everything flows from one core engine: **a game simulator** that produces full score + box-score distributions, so all markets are priced consistently from the same simulation.

---

## 3. Data Sources

> **See `DATA_CATALOG.md`** for the full item-by-item inventory (34 data items) with free GitHub/scrape sources per item. Net result: Phases 0–3 can run at **$0/mo**; paid feeds only when real-money prop betting starts.

### Free / cheap core
- **CollegeFootballData API (collegefootballdata.com)** — the backbone. Play-by-play, drives, box scores, SP+/FPI/Elo ratings, recruiting rankings (247 composite), transfer portal entries, returning production, coaching records, betting lines history, roster data. Free tier with API key.
- **ESPN public endpoints** — schedules, injuries (limited), depth charts.
- **The Odds API / OddsJam / Unabated** — live odds across books for line shopping, prop lines, and CLV tracking. (The Odds API has a free tier; props coverage costs more.)
- **247Sports / On3 / Rivals** — recruiting + portal rankings (scrape or manual CSV; On3 has consensus ratings and NIL valuations).
- **NWS / Open-Meteo API** — weather (wind matters most for totals; free).

### Scraped / curated by us (this is where proprietary edge lives)
- **Depth charts + OL starters**: weekly, per team. Track *returning starts* per lineman and combos.
- **Coaching staff database**: HC/OC/DC history, scheme tags (air raid, spread option, pro-style, wide zone, 3-3-5, 4-2-5...), play-caller identity (HC vs OC).
- **Portal impact grades**: not just star ratings — snaps at previous school, PFF-style production if available, position scarcity.
- **Beat-writer / insider text**: injury news, position battles, "sources say" — embedded and searched with LLMs (see §7).

### Chosen paid stack (decided 2026-07-28, ~$109/mo)
1. **CFBD Patreon Tier 3 — $10/mo**: raises the free 1,000 calls/mo to 75,000 and unlocks the GraphQL API with realtime subscriptions. This is the stats/PBP/recruiting/portal backbone; the paid tier is required at our all-FBS scope.
2. **The Odds API Business — $99/mo**: 50+ books, **NCAAF player props (US books)**, and **historical odds snapshots** (game lines since June 2020, props since May 2023 at 5-min intervals) — the props history is the scarce asset that makes prop backtesting possible at all.
3. **Free historical line archives** for pre-2020 spread/total backtests: CFBD's consensus closing lines (2013+), sportsbookreviewsonline.com CSV archives (~2007+).
4. **Deferred**: OddsJam / OpticOdds / Unabated (real-time props across 100–200+ books, ~$500+/mo) — the upgrade path only if we prove CLV and need faster/wider prop shopping. PFF college OL grades also deferred — we derive OL metrics (line yards, stuff rate, adjusted sack rate) from play-by-play ourselves first.

### Storage
- Postgres (or SQLite to start) + Parquet files for play-by-play. One ingestion job per source, versioned, with `as_of` timestamps so we can backtest **without lookahead bias** (critical — only use data available before kickoff).

---

## 4. Feature Engineering (the heart of the project)

### 4.1 Roster turnover: transfer portal + recruiting class
- **Returning production** (offense/defense splits — CFBD provides Bill Connelly's metric) — historically one of the strongest preseason predictors.
- **Net portal talent delta**: Σ(incoming player rating × expected snap share) − Σ(outgoing). Weight by position value (QB ≫ EDGE ≫ OT ≫ ...).
- **QB change flag + QB quality projection**: transfer QB with 500+ prior attempts ≠ redshirt freshman. Model these separately.
- **Recruiting talent base**: 2-year and 4-year weighted 247 composite (blue-chip ratio). Recruiting predicts a team's *ceiling*; returning production predicts its *this-year* level.
- **Chemistry/integration cost**: teams with >40% roster turnover underperform talent early season, converge by mid-October → a *time-decaying turnover penalty* feature.

### 4.2 Coaching & staff changes
- **New HC / new OC / new DC flags** with scheme-transition vector (e.g., pro-style→air raid: expected +8% pass rate, +6 plays/game).
- **Coach priors**: career pace (sec/play), pass rate over expected (PROE), 4th-down aggressiveness, red-zone tendencies, blowout behavior (do they call off the dogs? matters for spreads/totals).
- **Play-caller identity**: when HC hires an OC but keeps play-calling, weight the HC's prior, not the OC's.
- **Coach-change interaction terms**: new OC × returning QB (bad), new OC × transfer QB who followed him (good — e.g., coach brings "his guy").

### 4.3 Offensive line → run game & pass game
- **OL continuity score**: total returning career starts, number of returning starters, and *combo continuity* (did these 5 start together?).
- **Run-game features**: line yards before contact, stuff rate, opportunity rate (from play-by-play), power success rate — regressed on OL continuity + recruiting talent of the 5 starters.
- **Pass-game features**: sack rate allowed, pressure-adjusted (opponent pass-rush quality), time-in-pocket proxy (sack rate + scramble rate).
- **Scheme fit**: wide-zone OLs transferring into gap schemes lose value — interact OL features with new-OC scheme tag.
- **Propagation to props**: bad OL → lower RB rush yds props value, higher QB sack/INT variance, more checkdowns (RB receptions over).

### 4.4 Play-calling tendencies & game-state adaptation
From play-by-play, build per-coach (not per-team — coaches move) tendency profiles:
- **Baseline**: pass rate, pace, personnel proxies, deep-shot rate, by down & distance.
- **Game-state conditioned**: pass rate and pace when leading/trailing by score bucket (0–7, 8–14, 15+) × quarter. Some coaches go turtle with a 10-point lead in Q3 (fade their team total live); others keep the foot down (Lincoln Riley-types).
- **Script vs adaptation**: 1st-15-plays tendencies (scripted) vs rest of game. Useful for 1H markets.
- **2nd-half adjustments**: does this coach's offense improve or decline in 2H vs 1H, opponent-adjusted? Persistent per-coach signal → 2H line edges.
- Represent each coach as a **tendency embedding** (learned vector, §5) so new coach-team pairs generalize.

### 4.5 Standard opponent-adjusted efficiency core
- EPA/play and success rate, offense/defense, rush/pass splits, opponent-adjusted via ridge regression (like SP+ methodology), with recency weighting.
- Special teams, turnover luck regression (fumble recovery %, INT rate vs pressure), garbage-time filtering.
- Home field (team-specific, altitude, travel distance, time zones, weeknight games), rest differential, bye weeks, rivalry/letdown/lookahead spots.

---

## 5. Modeling Stack (Deep Learning + Classical)

**Principle:** classical models set the baseline; deep learning earns its place only if it beats the baseline in backtest. Both feed one simulator.

### Layer 1 — Team strength model
- Baseline: ridge-regression opponent-adjusted EPA ratings + Elo (fast, interpretable).
- DL upgrade: **sequence model (Transformer/GRU) over each team's game history**, where each game is a feature vector (efficiency stats, opponent embedding, context). Outputs a latent team-strength state that updates weekly — naturally handles "team improving after roster integration."

### Layer 2 — Game outcome model
- Inputs: both teams' latent states + matchup features (OL vs DL embeddings, scheme matchup, coach tendencies, weather, rest, travel).
- Output: **joint distribution of (home points, away points)** — e.g., a neural net parameterizing a bivariate distribution (negative binomial marginals + copula), or Monte Carlo drive simulation with NN-predicted drive outcomes.
- This directly prices spread, ML, total, team totals, and derivatives.

### Layer 3 — Player prop model
- **Usage model**: snap share / target share / carry share per player — hierarchical (player within position-room within team), heavy priors from recruiting rating, portal history, depth chart, and coach tendencies. This is where portal + OC-change features pay off most.
- **Volume model**: team plays, pass/rush split from Layer 2 simulation + coach game-state tendencies (blowout scripts change RB carries!).
- **Efficiency model**: yards per carry/target distributions conditioned on OL score, opponent unit strength.
- Combine by simulation → full prop distributions → compare to book lines, compute EV including vig.

### Layer 4 — Embeddings & LLM-assisted features (your OpenAI + Claude keys)
- **OpenAI embeddings** (`text-embedding-3-large` or `-small`): embed beat-writer articles, injury reports, presser transcripts, depth chart notes → semantic search + change detection ("starting LT questionable" reaches the model before the market fully adjusts).
- **Claude API** (`claude-sonnet-5` for volume, `claude-fable-5`/Opus for weekly deep analysis): structured extraction from unstructured news (JSON: player, status, severity, source reliability), coach scheme classification from articles, and a weekly "analyst memo" per slate that sanity-checks model picks against qualitative info the numbers miss (suspensions, weather chatter, motivation spots).
- LLM outputs become **features with confidence scores**, never direct picks.

### Training & evaluation
- Walk-forward backtesting only (train ≤ week N, predict week N+1), seasons 2016–2024 train, 2025 holdout.
- Metrics: log-loss / CRPS for distributions, then **betting sim vs actual historical closing lines** (CFBD has them): ROI, CLV, max drawdown, flat vs Kelly staking.
- Calibration plots per market. A model that's 55% accurate but well-calibrated beats a 58% model that's overconfident.

---

## 6. Bet Selection & Bankroll
- **Edge threshold**: bet only when model fair price beats best available line by a margin exceeding estimated model error (start: ≥3% EV props, ≥2 pts spread).
- **Fractional Kelly (¼ Kelly)** sizing, capped per-game and per-slate exposure; correlation-aware (don't max-bet a team's spread AND its QB over AND its team total — they're the same bet).
- **Line shopping** across books via odds API; log every bet with line, close, result → CLV dashboard.
- Paper-trade a full season segment before real money.

---

## 7. System Architecture

```
ingestion/           # CFBD, odds, weather, scrapers (daily + gameday cron)
  cfbd_client.py
  odds_client.py
  scrapers/          # depth charts, portal, coaching DB
warehouse/           # Postgres/SQLite + parquet play-by-play, as_of versioned
features/            # feature pipelines (roster, coach, OL, tendencies, efficiency)
models/
  team_strength/     # ridge baseline + transformer
  game_sim/          # score distribution / drive sim
  props/             # usage, volume, efficiency
  embeddings/        # OpenAI embed store (pgvector or FAISS)
llm/                 # Claude extraction + weekly memo prompts
backtest/            # walk-forward harness, betting sim, CLV tracking
picks/               # weekly edge report generator
app/                 # (later) dashboard — Streamlit to start
```

## 8. Build Order (Phases)

1. **Phase 0 (wk 1–2):** CFBD ingestion + warehouse + historical odds. Reproduce a basic opponent-adjusted rating; backtest naive spread model vs closing lines to establish baseline (expect ~-EV — that's the bar to beat).
   - ✅ **Done 2026-07-28.** Walk-forward 2024–2025 (weeks 4–15, 1,262 games): EPA-ridge model MAE 13.34 vs book 11.87, corr 0.765, ATS 51.5% (−1.8% ROI at −110). No edge from EPA alone vs closing lines, as expected — this is the bar. Code: `features/epa_ratings.py`, `backtest/spread_baseline.py`; results parquet in `warehouse/parquet/backtest_spread_baseline.parquet`.
2. **Phase 1 (wk 3–5):** Roster features (returning production, portal delta, recruiting) + coach database + OL continuity. Retrain; measure lift, especially **weeks 1–5 of a season** where these features should shine.
   - ✅ **Roster features done 2026-07-28** (`features/roster_priors.py`, `backtest/spread_phase1.py`). Preseason prior = ridge on [prev-season net, returning PPA%, returning usage, 4-yr recruiting z, portal net] → season net; prior fades to 0 by week 7. Findings, weeks 1–15 of 2024–25 (1,647 games): roster priors cut early-season MAE 14.4→13.8 and raised corr-with-book 0.64→0.78, but ATS was best for the *plain EPA* model in weeks 1–5: 57.2% (+9.1% ROI, 509 bets @ edge≥2); replicated on 2022–23 at 53.8–54.1% (+2.6–3.4%, ~450–500 bets). Weeks 6–15 remain no-edge (~51%). **Interpretation:** the market itself is slow early season; roster priors mostly replicate info books already price (they improve accuracy, not edge). Pooled early-season signal ≈55.5% over ~1,000 bets — promising, not yet bankable.
   - ✅ **Coach DB + OL continuity done 2026-07-28.** Coach DB (`features/coach_db.py` + `ingestion/scrapers/wiki_staff.py`): HC per team-season 2013+ from CFBD; OC/DC 2021–2025 scraped from Wikipedia season-page infoboxes via batched query API (690/690 pages, 98–100% field coverage; ~22%/yr new HCs, ~50%/yr OC turnover). OL continuity (`features/ol_continuity.py`): returning-OL share + OL room experience from bulk cfbfastR rosters 2020–2025, plus per-team stuff/sack rates from PBP (`ol_performance.parquet`). Added [new_hc, oc_change, ret_ol_share, ol_exp] to the preseason prior (standardized ridge). Effect: high-conviction early bets improved — roster variant weeks 1–5 edge≥4: **57.2%, +9.3% ROI (407 bets)** vs 56.0%/+6.9% without; full-season edge≥4 now 54.1%/+3.3%. But season-level coefficients on coach/OL features are small and unstable across train windows — their real payoff is expected at game level (scheme-transition tendencies, OL vs pass-rush matchups) in Phases 2–3.
3. **Phase 2 (wk 6–8):** Play-by-play tendency profiles + game simulator → totals, team totals, 1H lines.
   - ✅ **Tendency profiles done 2026-07-28** (`features/coach_tendencies.py` → `coach_tendencies.parquet`, 671 playcaller-seasons 2021–2025, keyed to OC-else-HC so profiles follow the coach). Metrics: PROE, neutral pass rate, pace (sec/play from drive clocks), explosive pass rate, turtle/chase deltas (pass-rate shift when up/down 9+ in 2H), 4th-down go rate, 2H EPA adjustment, scripted-opening delta. Face validity: triple-option academies are the 3 most run-heavy; air-raid programs top PROE. **Stability findings (335 consecutive-season pairs, 77 with coach moving teams):** PROE (r=0.65 all / 0.29 across moves), pass rate (0.63/0.22), and pace (0.50/0.35) are genuine coach traits → use as priors when staff changes, core inputs for totals. Turtle/chase deltas are weakly-but-consistently sticky (~0.16–0.21) → usable with heavy shrinkage for 2H/live later. **Not sticky:** 2H EPA adjustment (r=0.06 — "halftime-adjustments" lore doesn't persist), script delta, 4th-down go rate across moves, EPA/play (that's personnel, not coach). Remaining Phase 2: game simulator wiring these into totals/team-total projections.
   - ✅ **Game simulator done 2026-07-28** (`models/game_sim.py` → `backtest_totals_sim.parquet`). Per-team points model: [off−opp def rating, avg play-caller pace, own play-caller PROE, |rating mismatch|, home]; coach priors use only pre-season-s profile seasons (new OC carries his old-job profile). Self-calibrating with a **2023 warm-up season** (initial hand-set coefs caused an early-2024 over-bias disaster — 88% overs, 45% win — until warm-up fixed it). Results 2024–25 wks 1–15 (1,646 games): total MAE 13.17 vs book 12.60, corr 0.626, residual σ 16.6. O/U monotone in conviction: edge≥4 52.1%, **edge≥5 54.1% +3.3% ROI (562 bets)**. Ablation without coach priors: corr drops to 0.577, edge≥5 falls to 51.9%/−0.9% — **pace+PROE priors are worth ~2pts of win rate on totals**, the first direct edge from coach data. Not yet significant (~0.8σ) — needs 2022–23 out-of-sample validation and more seasons. Sim also outputs team totals + spreads (spread MAE 13.67, worse than the dedicated spread model — keep spreads on Phase 1 stack).
4. **Phase 3 (wk 9–12):** Player prop stack (usage/volume/efficiency) + live prop odds feed.
   - ✅ **v1 stack done 2026-07-28** (`features/player_stats.py` → 102k player-game logs from PBP, ~1% of official stats; `models/props.py` → `prop_projections.parquet`, 22.6k projections 2024–25 wks 4–15). Projection = trailing usage share × spread-adjusted team volume × shrunken efficiency × opponent-defense factor; recalibration layer (slopes 0.69–0.83 confirm usage regression-to-mean). **Results:** distributions are calibrated (78–85% coverage vs 80% target — good enough to price over/unders); point accuracy beats naive trailing average on rush yds (32.7 vs 33.3 MAE) and pass yds, ties/trails slightly on receptions/rec yds. Data fix en route: 2025 PBP yardage 43% null → coalesced from `statYardage`. **Next for props:** (a) buy prop lines (The Odds API Business) — edge vs book is untestable without them and books lean on naive trailing stats; (b) depth-chart/injury awareness — role changes are the biggest projection misses; (c) wire game-sim team totals into the volume model.
   - ✅ **Props v2 + confidence calibration 2026-07-29.** Upgrades: (a) **cross-season player priors** — returning players seed EWMAs with last season's usage/efficiency (3 pseudo-games), **portal transfers carry their origin-school profile** (2 pseudo-games, share ×0.8) → 6,793 new week-1–3 projections; beats naive on all 4 stats now; (b) **negative-binomial receptions** (r=25) → 45.5%→49.9% OOS; (c) weeks 1–3 prop lines pulled (+2.7k credits); (d) **market-blend calibration** — confidence report proved raw model probs are overconfident vs lines (props 70–80% stated → 54% actual; spreads 80%+ → 51%) while **ML is under-confident (70–80% stated → 83.5% actual!)**; blending 0.20·model+0.80·market logits, fitted on 2024: **2025 OOS props +4.2% ROI @ EV≥3% (291 bets, 54.6%) and +5.6% @ EV≥5% (136 bets)** — first positive OOS props result, one season of evidence, needs live confirmation. Calibration tables in `backtest/confidence_report.py`. Odds API live (5M plan; key working). Pulled **2 seasons of historical Saturday prop snapshots** (2024–25 wks 4–15, 1,024 games, 48k outcome rows, FanDuel/Bovada/BetOnline — cost ~20k of 5M credits) via `ingestion/historical_props.py`; backtest in `backtest/props_vs_book.py` (6,519 matched player-lines). **Honest verdict: v1 does not beat main prop lines.** Accuracy duel: book closer on every stat (e.g., rush 29.8 vs our 31.1 MAE). Betting sim 2025 out-of-sample: ~51%, −4% ROI ≈ vig; line shopping (best price) recovers ~1% → still −3%. The "CFB prop lines are naive" hypothesis is **falsified for main lines** at these books. Two graded-data bugs fixed en route: alt-line price mixing, and median-of-American-odds across ±100 (aggregate in payout space, never price space). **Path to prop edge now must come from information the books lack:** depth-chart/injury timing (layer built but not in historical projections), coach game-script → player volume, discrete model for receptions (worst market: 45% OOS), opening-line CLV selection, and thinner derivative markets. rush_yds is the closest market (52.4% OOS).
   - ✅ **Depth-chart/injury layer done 2026-07-28.** Three feeds, all free: (1) **Ourlads depth charts** (`ingestion/scrapers/ourlads_depth.py`) — 138 teams, 3,713 slots, dated snapshots for history, 137/138 mapped to CFBD ids; (2) **ESPN structured injuries endpoint** (`ingestion/news_injuries.py`) — free JSON with player/status/comment, stale-filtered (empty in July, fills in-season); (3) **Claude news extraction** — ESPN news feed → `claude-sonnet-5` → structured availability facts (worked live: extracted 5 real facts incl. QB battles won and portal moves). `features/availability.py` merges all three and `apply_availability()` adjusts prop projections: OUT players zeroed, direct backup inherits 60% of the starter's projected volume (heuristic until fitted on historical replacements). See **`GRAB_LIST.md`** for the items only the user can grab — bottom line: the only purchase needed today is The Odds API Business ($99/mo).
5. **Phase 4:** Deep learning upgrades (team sequence model, coach embeddings) — only after classical baselines are solid.
6. **Phase 5:** LLM/news layer, weekly memo, dashboard, paper-trading through the 2026 season, go-live decision.
   - ✅ **Weekly edge report live 2026-07-29** (`picks/edge_report.py` → `reports/edge_report_<date>.md`). Prices the whole live board (78 FBS games currently posted for 2026): fair spread (EPA ratings decayed to wk1 + 2026 roster prior at full weight), fair total (game-sim + coach priors incl. 2026 staff scrape), ML win-prob + EV at best quoted price. Guards: exact-only team matching (a prefix fallback once mapped FCS Arkansas–Pine Bluff onto Arkansas), blowout filter (|spread|>21 unplayable — model unvalidated there), FCS games skipped. Every run appends fair-vs-market to `warehouse/parquet/clv_log.parquet` — **CLV grading becomes possible the moment lines close**. 2026 caveats: returning production/rosters unpublished until Aug → prior is partial; 21 G5 wiki pages missing; re-run late Aug. First board flags 24 spreads / 11 totals at backtested thresholds (±4 / ±5), incl. openers like Toledo +11.5 at MSU and Coastal +21 at WVU.

---

## 9. Decisions (locked 2026-07-28)

| Decision | Choice | Implication |
|---|---|---|
| Data budget | **$100+/mo** | Paid odds plan with player-prop lines across books, plus room for premium OL/coaching data (e.g., PFF). We'll subscribe when Phase 2/3 needs it — no reason to burn budget during Phase 0. |
| Team scope | **All ~134 FBS teams** | Feature pipelines must be fully automated (no hand-curated depth charts as a dependency). We'll add hand-verification only for teams we actually bet. |
| First market | **Spreads/totals** | Phase order stays as written: validate vs 8+ years of historical closing lines before trusting props. |
| Live betting | **Pregame only (incl. 1H/2H pre-kick)** | Coach game-state tendency features still get built; live engine deferred to v2. |

### Still open
1. **Which sportsbooks/state** do you bet at? Determines which books' lines we shop and which prop menus matter.
2. **CFBD API key** — register free at collegefootballdata.com/key; it's the first dependency for Phase 0.
3. **Compute**: local CPU is fine through Phase 3; confirm you're OK with occasional cloud GPU rental for Phase 4 transformer training.

## 10. Reality Check
- Books hold ~4.5–5% on spreads and much more on props/parlays; long-run winning means beating the close consistently, not picking winners.
- Expected realistic outcome of a good v1: small edges in early-season spreads and mid-tier player props, low limits. CLV positive first; profit second.
- Legal note: this is for markets where you can legally bet; keep stakes within entertainment bankroll until 500+ tracked bets show positive CLV.
