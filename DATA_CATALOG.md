# Data Catalog — Every Data Item We Need, and Where to Get It Free

**TL;DR:** Almost everything except *live odds* is available free via GitHub repos, the CFBD free tier, or light scraping. The one thing with no reliable free source is **real-time player-prop lines across books** — sportsbook scrapers exist but break constantly and violate book ToS. Strategy: build 100% free through backtesting (Phases 0–3), pay only for The Odds API when we start betting real lines.

---

## 1. Core Game & Play-by-Play Data ✅ FREE (solved by GitHub)

| # | Data item | Granularity | Free source | Notes |
|---|---|---|---|---|
| 1 | Play-by-play (all FBS) | every play, 2002–present | **[cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data)** — pre-compiled Parquet/CSV, bulk download, no API key, no rate limits | THE key repo. Includes EPA/WPA already computed. Saves ~all our CFBD API quota |
| 2 | Drives | per drive | cfbfastR-data / CFBD free tier | start/end field position, result |
| 3 | Box scores (team + player) | per game | cfbfastR-data / CFBD | player-level needed for prop labels |
| 4 | Schedules, results, venues | per game | CFBD free tier (cheap calls) | includes neutral site, kickoff time |
| 5 | Team season stats + advanced (SP+, FPI, Elo, PPA) | per team-week | CFBD free tier | opponent-adjusted ratings for baseline sanity checks |
| 6 | Player season/game stats (pass/rush/rec lines) | per player-game | cfbfastR-data / CFBD | training labels for the prop models |
| 7 | Garbage-time & win-probability context | per play | included in cfbfastR-data (WP column) | needed to filter stats |

## 2. Roster Turnover: Portal + Recruiting ✅ FREE

| # | Data item | Granularity | Free source | Notes |
|---|---|---|---|---|
| 8 | Recruiting class rankings + individual recruit ratings (247 composite) | per player, 2000–present | CFBD free tier (`/recruiting`) | already aggregates 247 composite — no scraping needed |
| 9 | Transfer portal entries (origin, destination, rating) | per player per season | CFBD free tier (`/player/portal`) | the portal delta feature comes straight from this |
| 10 | Returning production (off/def) | per team-season | CFBD (`/player/returning`) | Bill Connelly's metric, precomputed |
| 11 | Rosters (height/weight/class/position) | per player-season | CFBD / cfbfastR-data | join key for everything player-level |
| 12 | Blue-chip ratio, 4-yr talent composite | per team | derive from #8 | trivial aggregation |
| 13 | Deeper recruit detail (position rank, state, camp data) | per player | scrape: [247-recruiting-ranking-history-scraper](https://github.com/scottenriquez/247-recruiting-ranking-history-scraper), [CFBRecruits-Selenium](https://github.com/yogermee/CFBRecruits-Selenium) | optional enrichment; CFBD covers the essentials |
| 14 | NIL valuations (On3) | per player | scrape On3 (no maintained repo; Apify actors exist) | nice-to-have, low priority |

## 3. Coaching & Staff ⚠️ MOSTLY FREE + our own curation

| # | Data item | Granularity | Free source | Notes |
|---|---|---|---|---|
| 15 | Head coach history + records | per coach-season | CFBD (`/coaches`) | HC only |
| 16 | OC/DC hires, play-caller identity | per team-season | **no clean source anywhere — build ourselves** | scrape Wikipedia "20XX coaching changes" pages + team pages; ~134 rows/yr, LLM-assisted extraction (Claude) makes this cheap |
| 17 | Scheme tags (air raid, wide zone, 3-3-5…) | per coordinator | build ourselves (LLM classification from articles + our labels) | proprietary edge — nobody sells this clean |
| 18 | Coach tendency profiles (PROE, pace, 4th-down, game-state behavior) | per coach | **derived from #1** (play-by-play) — free | our own computation, keyed to coach not team |

## 4. O-Line & Depth Charts ⚠️ SCRAPE + DERIVE

| # | Data item | Granularity | Free source | Notes |
|---|---|---|---|---|
| 19 | Weekly depth charts | per team-week | scrape **Ourlads** (ourlads.com/ncaa-football-depth-charts) — clean HTML tables, all FBS; ESPN depth-chart JSON endpoints as backup | no maintained GitHub repo; ours to build (~1 page/team/week) |
| 20 | OL starters per game (who actually started) | per game | derive from CFBD roster + participation, cross-check Ourlads | needed for continuity combos |
| 21 | OL returning career starts / continuity score | per team-week | **derived from #19/#20 history** | the OL feature |
| 22 | OL performance metrics (line yards, stuff rate, opp rate, sack rate allowed) | per team-game | **derived from #1** — free | replaces PFF grades for v1 |
| 23 | PFF OL/player grades | per player-game | ❌ no free source (paid, expensive, ToS-protected) | skip; revisit only if derived metrics underperform |

## 5. Betting Lines — Historical (backtesting) ✅ FREE

| # | Data item | Granularity | Free source | Notes |
|---|---|---|---|---|
| 24 | Historical spreads/totals/ML closing lines | per game, 2013+ | CFBD free tier (`/lines` — consensus + several books) | primary backtest labels |
| 25 | Historical lines 2007–2013 (+ openers) | per game | [sportsbookreviewsonline.com](https://www.sportsbookreviewsonline.com/scoresoddsarchives/ncaafootball/ncaafootballoddsarchives.htm) free CSV/XLSX archives | one-time download |
| 26 | Historical **player prop** lines | per player-market | ❌ effectively none free (The Odds API paid, May 2023+) | mitigation: backtest props against *our own* simulated fair lines + realized stats; buy history later if needed |

## 6. Betting Lines — Live (when we start betting) ⚠️ THE REAL GAP

| # | Data item | Granularity | Free source | Notes |
|---|---|---|---|---|
| 27 | Live spreads/totals/ML across books | per game, near-realtime | The Odds API free tier (500 credits/mo — enough for a weekly CFB snapshot) | fine for v1 game lines |
| 28 | Live player prop lines across books | per player-market | ⚠️ scrapers only: [DKscraPy](https://github.com/agad495/DKscraPy) (DraftKings internal API), [sportsbook-odds-scraper](https://github.com/declanwalpole/sportsbook-odds-scraper), [sbrscrape](https://github.com/nkgilley/sbrscrape) | **caution:** violates book ToS, breaks on redesigns, can get accounts limited/flagged. Usable for research; when real money is on the line, pay The Odds API Business ($99/mo) |
| 29 | Line movement / CLV tracking | open→close per market | log our own snapshots (cron) from #27/#28 | build from day one — it's our north-star metric |

## 7. Context Data ✅ FREE

| # | Data item | Granularity | Free source | Notes |
|---|---|---|---|---|
| 30 | Weather (wind, temp, precip) forecast + historical | per venue-hour | Open-Meteo API (free, no key) | wind is the totals signal |
| 31 | Venue lat/long, elevation, dome flag | per venue | CFBD `/venues` | join for weather + altitude |
| 32 | Injuries / suspensions / availability | per player-week | no good free feed for CFB — scrape team beat news + LLM extraction (Claude) into structured rows | this is where the OpenAI embeddings + Claude pipeline earns its keep |
| 33 | News/pressers text corpus | articles | RSS feeds + scraping (ESPN, 247 team sites) → embed with OpenAI | feature source, not labels |
| 34 | Polls, talent composite, conference/rivalry flags | per team-week | CFBD free tier | cheap context features |

---

## Scraping ground rules
1. **Prefer bulk repos over APIs, APIs over scraping.** cfbfastR-data Parquet first; CFBD free tier (1,000 calls/mo) reserved for what's not in the dumps (lines, portal, recruiting — low call volume).
2. **Every scraper writes raw HTML/JSON to disk with an `as_of` timestamp** before parsing — reparse later without re-fetching, and no lookahead bias in backtests.
3. Rate-limit (1 req/2–3s), identify politely, cache aggressively. Ourlads/Wikipedia/247 public pages are low-risk; **sportsbook scraping is ToS-violating** — acceptable for personal research snapshots, not as production infrastructure.
4. R repos (cfbfastR) don't force us into R — **cfbfastR-data is just Parquet files on GitHub**; we read them with pandas/polars directly.

## Revised spend plan
- **Phases 0–3 (build + backtest): $0/mo.** Everything needed is free (rows 1–25, 27, 29–34).
- **CFBD Patreon Tier 3 ($10/mo):** only if we exceed 1,000 calls/mo after bulk downloads — likely optional for months.
- **The Odds API Business ($99/mo):** subscribe when we start betting props with real money and need clean multi-book prop lines + their historical prop archive. Not before.
