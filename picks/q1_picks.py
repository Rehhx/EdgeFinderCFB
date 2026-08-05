"""First-quarter spread picks: take the BIG UNDERDOG on the Q1 line.

Validated in backtest/q1_spreads.py (2023-25, real prices, official CFBD
quarter scores):
  |spread|>=17  102 bets/szn  61.9%  +20.0% ROI  t=+3.70
  |spread|>=25   31 bets/szn  75.5%  +45.7% ROI  t=+5.27
Every threshold positive in all three seasons; every bootstrap CI excludes
zero; monotone in spread size exactly as the mechanism predicts.

WHY: books derive the Q1 line at ~0.289 x the full spread (linear, R^2 0.85),
but the dog's ACTUAL Q1 deficit plateaus near 4 points however big the spread
gets — a quarter holds only ~3-4 possessions a side, so you cannot lose by 30
in it. Full-game margin is unbounded; Q1 margin is not. The linear rule
therefore over-grants the dog +1.9 pts at a 19-pt spread and +4.3 at 37.

⚠️ NO MODEL IS USED, deliberately. Blind selection BEAT model-selected in the
backtest (75.5% vs 73.3%). This is a market-structure edge; our ratings add
nothing here and adding them would only add a failure mode.
⚠️ LIMITS: Q1 spreads on huge mismatches are low-limit markets. Expect small
maximums — that is probably why they stay mispriced. Size accordingly.

Q1 lines post near game day, like the 1H lines.

  python -m picks.q1_picks
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ingestion.config import PROJECT_ROOT
from ingestion.odds_client import NCAAF, OddsClient
from picks.edge_report import payout, team_matcher

MIN_SPREAD = 17.0          # base threshold; below this the book's rule is fair
PREMIUM_SPREAD = 25.0      # where the linear rule breaks down badly
PREMIUM_UNITS, STANDARD_UNITS = 2.0, 1.0
MAX_EVENTS = 80


def _full_spreads(client: OddsClient) -> dict:
    """(home, away) -> median full-game home spread, from the bulk endpoint."""
    board = client.get(f"/sports/{NCAAF}/odds",
                       {"markets": "spreads", "regions": "us",
                        "oddsFormat": "american"}, refresh=True)
    out = {}
    for e in board:
        pts = [o.get("point") for bk in e.get("bookmakers", [])
               for m in bk.get("markets", []) if m["key"] == "spreads"
               for o in m["outcomes"] if o["name"] == e["home_team"]]
        pts = [p for p in pts if p is not None]
        if pts:
            out[e["id"]] = float(np.median(pts))
    return out


def run() -> pd.DataFrame:
    client = OddsClient()
    match = team_matcher()
    full = _full_spreads(client)
    events = client.get(f"/sports/{NCAAF}/events", refresh=True)

    rows = []
    for e in events[:MAX_EVENTS]:
        fs = full.get(e["id"])
        if fs is None or abs(fs) < MIN_SPREAD:
            continue                      # only big mismatches
        dog = e["home_team"] if fs > 0 else e["away_team"]
        try:  # period markets need the per-event endpoint (bulk returns 422)
            od = client.get(f"/sports/{NCAAF}/events/{e['id']}/odds",
                            {"markets": "spreads_q1", "regions": "us",
                             "oddsFormat": "american"}, refresh=True)
        except Exception:
            continue
        quotes = [(bk["key"], o.get("point"), o.get("price"))
                  for bk in od.get("bookmakers", [])
                  for m in bk.get("markets", []) if m["key"] == "spreads_q1"
                  for o in m["outcomes"] if o["name"] == dog
                  and o.get("point") is not None]
        if not quotes:
            continue
        # books disagree on the Q1 number — take the modal line, then the best
        # price AMONG BOOKS AT THAT LINE (never pair a line with another
        # line's price; that bug inflated the first backtest run)
        pts = pd.Series([q[1] for q in quotes])
        line = float(pts.mode().iloc[0])
        at = [q for q in quotes if q[1] == line]
        best = max(at, key=lambda q: payout(pd.Series([q[2]])).iloc[0])
        hid, aid = match(e["home_team"]), match(e["away_team"])
        rows.append({
            "commence": e["commence_time"][:10],
            "away": e["away_team"], "home": e["home_team"],
            "home_id": hid, "away_id": aid,
            "dog_side": "home" if fs > 0 else "away",
            "fg_spread": fs, "dog": dog, "q1_line": line,
            "price": best[2], "book": best[0], "n_books": len(at),
            "ratio": round(line / abs(fs), 3),
            "tier": "PREMIUM Q1" if abs(fs) >= PREMIUM_SPREAD else "STANDARD Q1",
            "units": (PREMIUM_UNITS if abs(fs) >= PREMIUM_SPREAD
                      else STANDARD_UNITS)})
    df = pd.DataFrame(rows)
    print(f"big-mismatch games with Q1 lines: {len(df)} "
          f"| credits left {client.remaining()}")
    return df


def main() -> None:
    df = run()
    day = datetime.now(timezone.utc).date().isoformat()
    path = PROJECT_ROOT / "reports" / f"q1_picks_{day}.md"
    if df.empty:
        path.write_text(
            f"# Q1 Spread Picks — {day}\n\nNo Q1 lines on big mismatches yet "
            "(they post near game day).", encoding="utf-8")
        print("no Q1 lines yet — run on game week.")
        return
    df = df.sort_values(["tier", "fg_spread"],
                        key=lambda c: c if c.name != "fg_spread" else -c.abs())
    n_prem = int((df.tier == "PREMIUM Q1").sum())
    path.write_text(
        f"# Q1 Spread Picks — {day}\n\n"
        f"**Take the underdog on the first-quarter spread.**\n\n"
        f"{n_prem} PREMIUM ({PREMIUM_UNITS:g}u) + {len(df)-n_prem} STANDARD "
        f"({STANDARD_UNITS:g}u)\n\n"
        "| tier | filter | 3-season | ROI | size |\n|---|---|---|---|---|\n"
        f"| PREMIUM Q1 | full spread ≥{PREMIUM_SPREAD:g} | **75.5%** "
        "(94 bets, 3/3 seasons) | +45.7% | 2u |\n"
        f"| STANDARD Q1 | full spread {MIN_SPREAD:g}–{PREMIUM_SPREAD:g} | "
        "61.9% (307 bets) | +20.0% | 1u |\n\n"
        "*Books derive the Q1 line at ~0.289 × the full spread (linear, "
        "R²=0.85), but the dog's actual Q1 deficit plateaus near 4 points "
        "however big the spread — a quarter holds ~3–4 possessions a side, so "
        "you cannot lose by 30 in it. The linear rule over-grants the dog "
        "+1.9 pts at a 19-point spread and +4.3 at 37.*\n\n"
        "**No model is used — blind selection beat model-selected in the "
        "backtest (75.5% vs 73.3%). This is market structure, not our "
        "ratings.**\n\n"
        "⚠️ *Q1 spreads on big mismatches are low-limit markets; expect small "
        "maximums. That is likely why they stay mispriced.*\n\n"
        "⚠️ **Do not also bet the 1H line on the same game.** Q1 and 1H "
        "outcomes agree 76% of the time (r=+0.51); 2u on each is closer to a "
        "4u single position than to two bets. Q1 grades better at spreads "
        "≥25 (+44.0% vs +26.7% for 1H); 1H is better at 21–25 (+23.1% vs "
        "+11.4%).\n\n"
        + df.to_markdown(index=False), encoding="utf-8")
    print(f"report -> {path}")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
