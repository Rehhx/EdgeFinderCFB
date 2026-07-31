# Grab List — Everything You Need to Get (as of 2026-07-28)

The model stack (Phases 0–3) is built and runs on free data. These are the
remaining items only **you** can grab — accounts, subscriptions, keys.

## 1. Must grab before betting real lines

| # | Item | Cost | Why | Action |
|---|------|------|-----|--------|
| 1 | **The Odds API — 100K plan** | $59/mo | The one real paid need: NCAAF **player prop lines** (US books), line shopping across books, and **historical prop snapshots back to May 2023** — the day you subscribe we can backtest the Phase-3 prop projections against 2+ seasons of real lines. Props/historical requests burn credits at a multiplied rate, so skip the $30/20K tier; upgrade to $119/5M only if credits run short in-season. | Sign up at **https://the-odds-api.com** ("Get API Key →", account page at the-odds-api.com/account/) → 100K tier → paste key into `.env` as `ODDS_API_KEY=`. **Before paying**: email their support and ask exactly which NCAAF player-prop markets are included (pass yds / rush yds / receptions) — the site doesn't list them. Don't confuse it with lookalike sites (oddsapi.io, oddspapi.io, sportsgameodds.com — different companies). |
| 2 | **Sportsbook accounts** (2–4 books) | free (deposits) | Line shopping is worth ~1–2% ROI by itself; props limits vary by book. | Whichever are legal in your state — DraftKings, FanDuel, BetMGM, Caesars, ESPN Bet. Note which you open; the odds feed covers all of them. |

## 1b. Betting splits (bets% / handle%) — the next data buy

**Why:** the one angle in the betting literature we can't yet do directly. When
>75% of *handle* is on one side it goes ~47% ATS (fade-the-public). Our big-dog
spread edge is a *proxy* for this; real splits would let us target it directly
and likely sharpen the 59% further. Confirmed our Odds API plan does NOT carry
splits (`betting_splits`/`bet_percentage` markets → 422).

| Option | Cost | Notes |
|---|---|---|
| **SportsDataIO** (recommended to price first) | contact sales (~$100s/mo tier) | Only vendor found with **pre-match betting splits in a real API** (moneyline/spread/total). Ask specifically: NCAAF splits endpoint, **bets% AND handle%**, and **historical** splits (we need history to backtest before betting). |
| **Action Network** subscription | ~$8–15/mo | Cheapest source of the same numbers, but it's a *website*, no API. Scraping their paywalled data = ToS violation (same call we made on PFF). Usable to eyeball, not to model. |
| **VSiN / Cleatz** | free-ish web pages | Public splits pages, update every 5–15 min. VSiN is free to view. No API/history; manual capture only. |
| **DIY capture** | $0 | Log the free public splits pages weekly ourselves during 2026 → build our OWN splits history. Slow (needs a season) but free, legal, and creates a proprietary dataset. |

**Recommendation:** email SportsDataIO for NCAAF splits pricing + history. If
history isn't included or it's expensive, do the DIY capture this season (a
weekly scrape of a free public splits page) so we have data to backtest by 2027.
Do NOT pay for a splits feed without historical data — we can't validate an edge
we can't backtest.

## 2. Grab when the trigger hits (not yet)

| # | Item | Cost | Trigger |
|---|------|------|---------|
| 3 | **CFBD Patreon Tier 3** | $10/mo | When `warehouse/raw/cfbd/usage_log.jsonl` shows >800 calls in a month (we're at ~21/1000 — bulk parquet downloads keep usage tiny). Also unlocks their GraphQL/realtime API for in-season automation. | 
| 4 | **OddsJam or OpticOdds** | ~$500+/mo | Only after 500+ tracked bets show positive CLV and prop volume justifies faster/wider line coverage. Skip until then. |
| 5 | **PFF College (consumer)** | ~$40/mo | OL grades — only if our PBP-derived OL metrics (stuff rate, line yards, sack rate — already built) prove insufficient. |
| 6 | **PFF ELITE / coverage & matchup data** ⚠️ **BACKTEST SAYS RECONSIDER** — the coverage-injury edge this would power was tested (`backtest/coverage_injury.py`, 2024-25, 298 DB-out games): the on-field effect is real (+0.047 pass EPA/play with the top DB out) but there is **NO prop betting edge** (opposing overs hit 47% out vs 48% in — the market prices announced DB injuries). Buy PFF/SIS only if you specifically want healthy-game WR-vs-CB matchup modeling AND can show it beats the market; do NOT buy it expecting the injury edge. | $200/yr consumer; charting/API is enterprise (contact sales) | **The one thing our data genuinely cannot do: player-vs-player coverage (shadow-corner shadowing a WR).** CFBD play-by-play has no who-covered-whom charting. PFF has WR/CB matchup + coverage grades. Buy this to build the player-level defensive matchup model (CB rating vs WR, slot vs outside, coverage scheme). Consumer tier gives grades to eyeball; the actual per-snap matchup/coverage data for modeling needs PFF's data/API licensing (enterprise, pricier). Confirm which tier exposes downloadable coverage/matchup data before paying. |
| 6b | **Sports Info Solutions (SIS) DataHub** — the PFF alternative | NCAA $149.99/mo or $999.99/yr (download+API tier via sales) | Only other credible vendor charting all FBS coverage (computer vision + human video scouts). DataHub has CSV downloads + API. **Same caveat as PFF: confirm the tier actually exposes defender-in-coverage at the target level (who-covered-whom), not just team/player grades — email sales@sportsinfosolutions.com.** Get quotes from BOTH PFF and SIS and compare; ask each: "downloadable/API access to the defender(s) in coverage per target + per-player coverage snaps, all FBS, 3+ seasons history?" ESPN has a WR/CB "Shadow Report" but it's published content, not a licensable dataset. |

## 3. Verify (you already have these — 2 minutes)

| # | Item | Check |
|---|------|-------|
| 6 | **Anthropic API billing** | Key is in `.env` and worked today (news extraction ran on `claude-sonnet-5`). Confirm billing/credits are set at console.anthropic.com. In-season cost estimate: ~$5–15/mo for daily news extraction. |
| 7 | **OpenAI API billing** | Key is in `.env` (embeddings layer, not yet wired). Confirm billing at platform.openai.com. Cost trivial (~$1–5/mo for embedding news corpus). |
| 8 | **CFBD key** | Working (in `.env`). Also **blank the copy of the real key you pasted into `.env.example`** — that file is the shareable template. |

## 4. Nothing to buy — already automated & free

- Play-by-play 2002–present (sportsdataverse bulk parquet)
- Rosters, schedules, lines history, recruiting, portal, returning production (CFBD free tier)
- **Depth charts** — Ourlads scraper (`python -m ingestion.scrapers.ourlads_depth`), weekly snapshots
- **Injuries** — ESPN structured endpoint (`python -m ingestion.news_injuries`); sparse in July, fills in-season
- **News → availability facts** — ESPN news feed + Claude extraction (same command)
- Coach staff DB — Wikipedia batch scraper (re-run each offseason: `python -m ingestion.scrapers.wiki_staff 2021 2026`)
- Weather — Open-Meteo, no key needed (not yet wired; free when we need it)

## 5. In-season weekly routine (once the season starts, late August 2026)

```
# Tuesday (after Week N results land):
python -m ingestion.run_ingest --all          # refresh CFBD + check PBP updates
python -m ingestion.scrapers.ourlads_depth    # depth chart snapshot
python -m ingestion.news_injuries             # injuries + news extraction

# Wednesday-Saturday (line shopping window):
#   odds pulls + edge reports — to be built once ODDS_API_KEY is live (item 1)
```

**Bottom line: the only check to write today is The Odds API ($99/mo). Everything else is either free, already working, or waiting on a trigger.**
