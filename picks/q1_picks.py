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
# ⚠️ Q1 PREMIUM (>= 25) RETIRED 2026-08-06. On the de-contaminated sample it is
# the one Q1 slice that fails outright — 56.0% / +8.0% pooled but bootstrap CI
# [-7.3, +23.8] with 2025 NEGATIVE, and -0.7 u/szn realised. Those games are
# NOT dropped: |spread| >= 25 is now bet as BIG-DOG 1H, which grades better on
# exactly that population (9.5% ROI / 57.1%). Handing them over lifted 1H from
# 3.7 to 14.6 u/szn and the whole book from +73.7 to +85.3.
MAX_SPREAD = 25.0          # bet Q1 only in 17-25; >=25 belongs to 1H
STANDARD_UNITS = 1.0
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
        if fs is None or not (MIN_SPREAD <= abs(fs) < MAX_SPREAD):
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
        # payout() takes a SCALAR american price. Wrapping it in a Series made
        # `if price < 0` ambiguous and raised for EVERY slate with a Q1 line,
        # including single-quote ones — the best play in the book had never run
        # against a live board.
        best = max(at, key=lambda q: payout(q[2]))
        hid, aid = match(e["home_team"]), match(e["away_team"])
        rows.append({
            "commence": e["commence_time"][:10],
            "away": e["away_team"], "home": e["home_team"],
            "home_id": hid, "away_id": aid,
            "dog_side": "home" if fs > 0 else "away",
            "fg_spread": fs, "dog": dog, "q1_line": line,
            "price": best[2], "book": best[0], "n_books": len(at),
            "ratio": round(line / abs(fs), 3),
            "tier": "STANDARD Q1",
            "units": STANDARD_UNITS,
        })
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
            f"# Q1 Spread Picks - {day}\n\nNo Q1 lines on big mismatches yet "
            "(they post near game day).", encoding="utf-8")
        print("no Q1 lines yet - run on game week.")
        return
    df = df.sort_values("fg_spread", key=lambda c: -c.abs())
    path.write_text(
        f"# Q1 Spread Picks - {day}\n\n"
        "**Take the underdog on the first-quarter spread.**\n\n"
        f"{len(df)} plays at {STANDARD_UNITS:g}u "
        f"(full spread {MIN_SPREAD:g}-{MAX_SPREAD:g})\n\n"
        "> **The >=25 PREMIUM tier was RETIRED 2026-08-06.** On the "
        "de-contaminated sample it is the one Q1 slice that fails: 56.0% / "
        "+8.0% pooled, bootstrap CI [-7.3, +23.8], **2025 negative**, -0.7 "
        "u/szn realised. Those games are NOT dropped - bet them as BIG-DOG "
        "1H, which grades better on exactly that population (+9.5% ROI). "
        "Moving them lifted 1H from 3.7 to 14.6 u/szn and the book from "
        "+73.7 to +85.3 units/season.\n\n"
        "*Mechanism: books derive the Q1 line at ~0.289x the full spread, "
        "but a quarter holds only ~3-4 possessions a side. NOTE the "
        "saturation claim was overstated by the old contaminated sample - on "
        "the full universe the dog Q1 deficit keeps falling (-3.4 to -10.2), "
        "so the remaining 17-25 edge is smaller and less certain than once "
        "advertised.*\n\n"
        "**No model is used - blind selection beat model-selected.**\n\n"
        "> Low-limit market; expect small maximums.\n\n"
        "> **Do not also bet the 1H line on the same game** - Q1 and 1H "
        "outcomes agree 76% of the time (r=+0.51).\n\n"
        + df.to_markdown(index=False), encoding="utf-8")
    print(f"report -> {path}")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
