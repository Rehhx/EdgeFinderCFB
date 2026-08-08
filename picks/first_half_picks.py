"""Live first-half spread picks — the validated 1H edge (wks 1-5).

Backtest (backtest/first_half.py, 3 seasons 2023-25): 1H spread ATS in
weeks 1-5 hits 58% (+10.5% ROI) at edge>=2, POSITIVE ALL 3 SEASONS
(54.9/59.6/58.9%) — the strongest validated edge in the system. Weeks 6-15
and 1H totals/ML show no edge. So this only flags EARLY-SEASON 1H spreads.

1H lines post near kickoff, so this returns nothing until game week.

  python -m picks.first_half_picks
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from features.epa_ratings import POSTSEASON_WEEK, SEASON_STRIDE, fit_ratings
from features.first_half import load_1h_obs
from ingestion.config import PARQUET_DIR, PROJECT_ROOT, load_coefs
from ingestion.odds_client import NCAAF, OddsClient
from picks.edge_report import team_matcher

SEASON = 2026
PBP_SEASONS = list(range(2021, 2026))
EDGE_FLAG = 2.0
COEFS = load_coefs("first_half", {"b": 40.0, "hfa": 1.2})

# 1H is deliberately SINGLE-TIER (1u), unlike the tiered full-game big-dog
# play. Segment analysis (3 seasons, 469 bets) found NO robust tier:
#   dog-side filter      55.7%  (WORSE than taking all sides at 57.8%)
#   favourite-side       73.2%  but n=56, declining 86/77/65% by season, and
#                        the mechanism fails: books set 1H at 56.8% of the
#                        full spread vs a true 58% front-load — i.e. they
#                        already price it, so that sample is noise
#   1H spread size       no monotone pattern
#   edge size            no monotone pattern (2-3:64%, 3-5:53%, 5-8:64%, 8+:55%)
#   book 1H/full ratio   non-monotone (48.6% / 55.3% / 62.1%), per-season
#   anomaly              inconsistent (54.8/59.8/51.6%)
# Take every side at edge>=2 in weeks 1-5: 57.8% ATS, +10.3% ROI, and
# positive in all three seasons (55%/60%/59%). Don't filter it further.
TIER, UNITS = "STANDARD", 1.0

# BIG-DOG 1H (added 2026-07-30): the highest-win-rate play in the system.
# Take the big-dog selection (full-game |spread|>=17, our full-game model on
# the dog by >=6) but bet it on the 1H line instead of the full game.
# 3 seasons, same 125 games: 1H 64.8% / +23.7% ROI vs full-game 62.4% /
# +19.1%; positive all 3 seasons; and far above the generic 1H edge (57.8%),
# so it is the COMBINATION that wins. Mechanism: in these spots the dog
# covers ~3.6 pts in the 1H but only ~0.9 in the 2H, while books set the 1H
# line at a flat 56.8% of the full spread — a derived-line shortcut.
# STAKE FOLLOWS EVIDENCE (re-derived 2026-08-06). Measured on the
# complete-book seasons, the old assignment was INVERTED: BIG-DOG 1H carried 2u
# on the weakest case in the book (+9.5% ROI, t=+1.24, bootstrap CI spanning
# zero) while PROPS STANDARD carried 1u at +16.7% with a CI that excludes zero.
#   play              stake   ROI     t      CI
#   BIG-DOG 1H         2.00  +9.5%  +1.24  [-5.3%,+24.4%]  spans 0
#   PROPS PRIME        2.16 +15.4%  +2.56  [+3.8%,+26.5%]
#   PROPS PROBE        1.07 +17.1%  +2.01  [+0.5%,+33.2%]
#   PROPS STANDARD     1.06 +16.7%  +2.14  [+1.1%,+32.0%]
#   Q1 STANDARD        1.00 +13.7%  +1.31  spans 0
#   SPREAD STANDARD    1.00 +22.0%  +2.02  [+0.8%,+43.2%]
# Moving 1H 2u->1.5u and PROPS STANDARD 1u->1.5u holds TOTAL STAKE EXACTLY
# constant (629u) while lifting the book +94.8 -> +97.6 u/szn, median $1,985 ->
# $2,014, 5th %ile $1,023 -> $1,090 and P(losing season) 4.6% -> 3.6%. Pure
# reallocation, not leverage.
# ⚠️ SPREAD STANDARD was deliberately NOT raised despite the best 2-season ROI:
# its EIGHT-season read is +0.7% / t=+0.14 / 4-of-8. Never let a short-window
# CI outvote a longer one.
BIGDOG_TIER, BIGDOG_UNITS = "BIG-DOG 1H", 1.5
BIGDOG_MIN_SPREAD, BIGDOG_MIN_EDGE = 17.0, 6.0
# The BIG-DOG 1H play is NOT capped at week 5 (the STANDARD 1H tier above is).
# backtest/h1_saturation.py showed the same bounded-quantity mispricing as the
# Q1 spread: books derive the 1H line at a flat 0.573 x full (R^2 0.975) while
# the dog's REALISED 1H share falls 0.647 -> 0.413 as the spread grows, because
# a half holds only ~7-8 possessions a side. Over-grant: -0.17 pts at a 13-pt
# spread, +3.58 at 23, +7.32 at 37.
#   >=17 wks1-5           43/szn  65.1%  +24.8%   21.3 u/szn   (old rule)
#   >=17 all weeks        69/szn  63.6%  +21.8%   30.0 u/szn   (wks6+ 2024 -4.6%)
#   >=17 wk1-5 + >=21 wk6+ 59/szn 65.7%  +25.9%   30.8 u/szn   <- SHIPPED
# The late-season slice needs the bigger spread: >=21 in weeks 6-15 is +29.0%
# with all three seasons positive, whereas 17-21 late is the weak link.
BIGDOG_LATE_WEEK = 6
BIGDOG_LATE_SPREAD = 21.0


def _full_game_context() -> dict:
    """(home_id, away_id) -> full-game book spread and our model edge, used to
    identify BIG-DOG 1H plays."""
    try:
        from picks.ml_spread_picks import run as fg_run
        return {(int(r.home_id), int(r.away_id)):
                {"book_spread": float(r.book_spread), "edge": float(r.edge)}
                for r in fg_run().itertuples()}
    except Exception as e:
        print(f"(full-game context unavailable: {e})")
        return {}


def run() -> pd.DataFrame:
    # Refit AS OF the upcoming week, including the live season once its
    # play-by-play lands (features/epa_ratings.live_asof).
    from features.epa_ratings import asof_seasons, live_asof
    from picks.prop_picks import current_week
    obs = load_1h_obs(asof_seasons(PBP_SEASONS, SEASON))
    model = fit_ratings(obs, asof_week_idx=live_asof(SEASON, current_week()))
    match = team_matcher()
    full_game = _full_game_context()

    client = OddsClient()
    # period markets need the per-event endpoint (bulk /odds returns 422)
    events = client.get(f"/sports/{NCAAF}/events", refresh=True)
    rows = []
    for e in events:
        hid, aid = match(e["home_team"]), match(e["away_team"])
        if not hid or not aid or hid not in model.ratings.index \
                or aid not in model.ratings.index:
            continue
        try:
            od = client.get(f"/sports/{NCAAF}/events/{e['id']}/odds",
                            {"markets": "spreads_h1", "regions": "us",
                             "oddsFormat": "american"}, refresh=True)
        except Exception:
            continue
        pts = [o["point"] for bk in od.get("bookmakers", [])
               for m in bk.get("markets", []) if m["key"] == "spreads_h1"
               for o in m["outcomes"] if o["name"] == e["home_team"]]
        if not pts:
            continue
        book = float(np.median(pts))
        pred = COEFS["b"] * (model.net(hid) - model.net(aid)) + COEFS["hfa"]
        # full-game context for the BIG-DOG 1H play
        fg = full_game.get((hid, aid), {})
        rows.append({"commence": e["commence_time"][:10],
                     "away": e["away_team"], "home": e["home_team"],
                     "home_id": hid, "away_id": aid,
                     "fair_1h_spread": round(-pred, 1), "book_1h_spread": book,
                     "edge": round(pred - (-book), 1),
                     "fg_spread": fg.get("book_spread", np.nan),
                     "fg_edge": fg.get("edge", np.nan)})
    df = pd.DataFrame(rows)
    if not df.empty:
        fav_home = df.fg_spread < 0
        dog_side = (df.fg_edge > 0) != fav_home
        from picks.prop_picks import current_week
        wk = current_week() or 1
        # weeks 6+ require the bigger spread (see BIGDOG_LATE_SPREAD note)
        min_spread = (BIGDOG_LATE_SPREAD if wk >= BIGDOG_LATE_WEEK
                      else BIGDOG_MIN_SPREAD)
        df["big_dog"] = (dog_side & (df.fg_spread.abs() >= min_spread)
                         & (df.fg_edge.abs() >= BIGDOG_MIN_EDGE)).fillna(False)
        df.attrs["week"], df.attrs["min_spread"] = wk, min_spread
    print(f"1H spreads on board: {len(df)} | credits left {client.remaining()}")
    return df


STD_MAX_WEEK = 5     # STANDARD 1H is validated in weeks 1-5 ONLY


def tiered(df: pd.DataFrame) -> pd.DataFrame:
    """Tier and stake 1H plays. THE single source of truth for both the report
    and the paper ledger.

    Previously `main()` tiered inline and `paper_trades.import_1h` re-derived
    its own selection, so the ledger staked BIG-DOG 1H at 1u instead of 2u
    (`getattr(r, "units", 1.0)` on a frame that has no `units` column) and
    under-recorded a +17.6 u/szn play by half — which also slowed compounding,
    since unit size grows off realised ledger P&L.

    It also applies the STANDARD week gate. The docstring and the emitted
    report both say "weeks 1-5 only, no edge weeks 6-15", but nothing enforced
    it and 1u STANDARD plays were emitted all season.
    """
    if df.empty:
        return df
    wk = df.attrs.get("week", 1)
    if wk >= POSTSEASON_WEEK:
        # Bowls/CFP are out of sample: every threshold here was fitted on the
        # regular season, and bowl rosters are reshaped by opt-outs and the
        # portal in ways the ratings cannot see. Emit nothing.
        return df.iloc[0:0].assign(tier=TIER, units=UNITS)
    bigdog = df[df.get("big_dog", False)].copy()
    bigdog["tier"], bigdog["units"] = BIGDOG_TIER, BIGDOG_UNITS
    if wk <= STD_MAX_WEEK:
        std = df[df.edge.abs() >= EDGE_FLAG]
        std = std[~std.index.isin(bigdog.index)].assign(tier=TIER, units=UNITS)
    else:
        std = df.iloc[0:0].assign(tier=TIER, units=UNITS)
    return pd.concat([bigdog, std])


def main() -> None:
    df = run()
    day = datetime.now(timezone.utc).date().isoformat()
    path = PROJECT_ROOT / "reports" / f"first_half_{day}.md"
    if df.empty:
        path.write_text(f"# 1H Spread Picks — {day}\n\nNo 1H lines posted yet "
                        "(they appear near game day).", encoding="utf-8")
        print("no 1H lines yet — run on game week.")
        return
    plays = tiered(df).sort_values(
        ["tier", "edge"], key=lambda c: c if c.name != "edge" else c.abs(),
        ascending=[True, False])
    path.write_text(
        f"# 1H Spread Picks — {day}\n\n"
        f"{len(bigdog)} **BIG-DOG 1H** ({BIGDOG_UNITS:g}u) + {len(std)} "
        f"{TIER} ({UNITS:g}u).\n\n"
        "| tier | filter | 3-season ATS | ROI | size |\n|---|---|---|---|---|\n"
        f"| BIG-DOG 1H | full spread ≥{BIGDOG_MIN_SPREAD:g} (wks 1–5) or "
        f"≥{BIGDOG_LATE_SPREAD:g} (wks {BIGDOG_LATE_WEEK}+) & model on dog ≥6 | "
        "**65.7%** (178 bets, 3/3 seasons) | +25.9% | 2u |\n"
        "| STANDARD | any side, 1H edge ≥2 | 57.8% (469 bets) | +10.3% | 1u |\n\n"
        "*Books derive the 1H line at a flat 0.573 × the full spread "
        "(R²=0.975), but the dog's realised 1H share FALLS from 0.647 to 0.413 "
        "as the spread grows — a half holds only ~7–8 possessions a side, so "
        "the 1H margin saturates while the full-game margin does not. The "
        "book over-grants the dog +3.6 pts at a 23-point spread and +7.3 at "
        "37.*\n\n"
        "⚠️ **Do not also bet the Q1 line on the same game.** Q1 and 1H "
        "outcomes agree 76% of the time (r=+0.51); 2u on each is closer to a "
        "4u single position than to two bets. Pick one — Q1 grades better at "
        "spreads ≥25 (+44.0% vs +26.7%), 1H better at 21–25 (+23.1% vs "
        "+11.4%).\n\n"
        "**The STANDARD tier below is weeks 1-5 only — 57.8% ATS, +10.3% ROI, positive in "
        "all 3 seasons (55%/60%/59%). Take EVERY side; segment tests found "
        "no robust sub-tier (a dog-side filter actually lowers it to 55.7%). "
        "No edge weeks 6-15 — stop betting these after week 5.**\n\n"
        "positive edge = model likes the HOME 1H side.\n\n"
        + plays.to_markdown(index=False), encoding="utf-8")
    print(f"report -> {path}")
    print(plays.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
