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

from features.epa_ratings import SEASON_STRIDE, fit_ratings
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
BIGDOG_TIER, BIGDOG_UNITS = "BIG-DOG 1H", 2.0
BIGDOG_MIN_SPREAD, BIGDOG_MIN_EDGE = 17.0, 6.0


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
    obs = load_1h_obs(PBP_SEASONS)
    model = fit_ratings(obs, asof_week_idx=SEASON * SEASON_STRIDE + 1)
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
        df["big_dog"] = (dog_side & (df.fg_spread.abs() >= BIGDOG_MIN_SPREAD)
                         & (df.fg_edge.abs() >= BIGDOG_MIN_EDGE)).fillna(False)
    print(f"1H spreads on board: {len(df)} | credits left {client.remaining()}")
    return df


def main() -> None:
    df = run()
    day = datetime.now(timezone.utc).date().isoformat()
    path = PROJECT_ROOT / "reports" / f"first_half_{day}.md"
    if df.empty:
        path.write_text(f"# 1H Spread Picks — {day}\n\nNo 1H lines posted yet "
                        "(they appear near game day).", encoding="utf-8")
        print("no 1H lines yet — run on game week.")
        return
    bigdog = df[df.get("big_dog", False)].copy()
    bigdog["tier"], bigdog["units"] = BIGDOG_TIER, BIGDOG_UNITS
    std = df[df.edge.abs() >= EDGE_FLAG]
    std = std[~std.index.isin(bigdog.index)].assign(tier=TIER, units=UNITS)
    plays = pd.concat([bigdog, std]).sort_values(
        ["tier", "edge"], key=lambda c: c if c.name != "edge" else c.abs(),
        ascending=[True, False])
    path.write_text(
        f"# 1H Spread Picks — {day}\n\n"
        f"{len(bigdog)} **BIG-DOG 1H** ({BIGDOG_UNITS:g}u) + {len(std)} "
        f"{TIER} ({UNITS:g}u).\n\n"
        "| tier | filter | 3-season ATS | ROI | size |\n|---|---|---|---|---|\n"
        "| BIG-DOG 1H | full-game spread ≥17 & model on dog ≥6, bet the 1H | "
        "**64.8%** (125 bets, 3/3 seasons) | +23.7% | 2u |\n"
        "| STANDARD | any side, 1H edge ≥2 | 57.8% (469 bets) | +10.3% | 1u |\n\n"
        "*Big-dog covers are front-loaded (~3.6 pts in the 1H vs ~0.9 in the "
        "2H) while books set the 1H line at a flat 56.8% of the full spread — "
        "a derived-line shortcut. Betting these on the 1H beats the full game "
        "(64.8% vs 62.4% on the same games).*\n\n"
        "**Validated: weeks 1-5 only — 57.8% ATS, +10.3% ROI, positive in "
        "all 3 seasons (55%/60%/59%). Take EVERY side; segment tests found "
        "no robust sub-tier (a dog-side filter actually lowers it to 55.7%). "
        "No edge weeks 6-15 — stop betting these after week 5.**\n\n"
        "positive edge = model likes the HOME 1H side.\n\n"
        + plays.to_markdown(index=False), encoding="utf-8")
    print(f"report -> {path}")
    print(plays.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
