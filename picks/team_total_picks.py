"""Favourite TEAM-TOTAL UNDER — a STANDALONE market bias, NOT our model's edge.

⚠️ CORRECTED 2026-07-30 after control tests. This was first written up as
"the big-dog edge transferring to a third market". That was WRONG:

  BLIND all big favourites |spread|>=25   62 bets 64.5% / +24.8% (65/63/65)
  our MODEL-selected subset               61 bets 63.9% / +23.6% (64/63/65)

The model picks 61 of the same 62 games and adds NOTHING (-0.6 pts). What
exists is an independent market bias: books set big favourites' team totals
too high and they go under ~64% of the time. Wider net |spread|>=17 is
weaker AND fading: 60.0% -> 58.2% -> 57.1% by season.

Also: on the same games the SPREAD bet beats the team total (62.8% vs
58.9%), and the two outcomes agree 78% of the time — so this is mostly the
same position expressed worse. Prefer the spread; use this only when spread
limits are exhausted, and size it as part of the SAME position.

Sample is small (62 bets / 3 seasons ~ 20 per season). Treat as provisional.

  python -m picks.team_total_picks
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ingestion.config import PROJECT_ROOT
from ingestion.odds_client import NCAAF, OddsClient
from picks.edge_report import team_matcher
from picks.ml_spread_picks import PREMIUM_SPREAD, run as fg_run, validated_plays


def run() -> pd.DataFrame:
    plays = validated_plays(fg_run())
    if plays.empty:
        print("no big-dog games on the board.")
        return plays
    match = team_matcher()
    client = OddsClient()
    events = client.get(f"/sports/{NCAAF}/events", refresh=True)
    by_pair = {(match(e["home_team"]), match(e["away_team"])): e
               for e in events}

    rows = []
    for r in plays.itertuples():
        e = by_pair.get((r.home_id, r.away_id))
        if not e:
            continue
        try:
            od = client.get(f"/sports/{NCAAF}/events/{e['id']}/odds",
                            {"markets": "team_totals", "regions": "us",
                             "oddsFormat": "american"}, refresh=True)
        except Exception:
            continue
        # the favourite is the side NOT backed by the big-dog play
        fav_name = e["away_team"] if r.edge > 0 else e["home_team"]
        quotes = [(o.get("point"), o.get("price"))
                  for bk in od.get("bookmakers", [])
                  for m in bk.get("markets", []) if m["key"] == "team_totals"
                  for o in m.get("outcomes", [])
                  if o.get("description") == fav_name and o["name"] == "Under"]
        quotes = [(p, pr) for p, pr in quotes if p is not None]
        if not quotes:
            continue
        line = float(np.median([p for p, _ in quotes]))
        best = max(pr for p, pr in quotes if p == line) if any(
            p == line for p, _ in quotes) else max(pr for _, pr in quotes)
        rows.append({
            "commence": r.commence, "game": f"{r.away} @ {r.home}",
            "bet": f"{fav_name} team total UNDER {line:g}",
            "price": int(best), "fg_spread": r.book_spread,
            "tier": "PREMIUM" if abs(r.book_spread) >= PREMIUM_SPREAD
            else "STANDARD",
            "units": 2.0 if abs(r.book_spread) >= PREMIUM_SPREAD else 1.0})
    df = pd.DataFrame(rows)
    print(f"team-total unders flagged: {len(df)} | credits {client.remaining()}")
    return df


def main() -> None:
    df = run()
    day = datetime.now(timezone.utc).date().isoformat()
    path = PROJECT_ROOT / "reports" / f"team_totals_{day}.md"
    if df.empty:
        path.write_text(f"# Team-Total Unders — {day}\n\nNo qualifying games "
                        "(team-total lines post near game day).",
                        encoding="utf-8")
        print("nothing to flag yet.")
        return
    path.write_text(
        f"# Favourite Team-Total UNDERs — {day}\n\n{len(df)} plays.\n\n"
        "⚠️ **This is a standalone MARKET BIAS, not our model's edge.** Blind "
        "big-favourite unders (spread ≥25) hit 64.5% over 3 seasons; our "
        "model-selected subset hits 63.9% — the model adds nothing. The wider "
        "spread ≥17 net is weaker and fading (60.0% → 58.2% → 57.1%).\n\n"
        "**Prefer the spread bet:** on these same games the spread wins 62.8% "
        "vs 58.9% for the team total, and the two outcomes agree 78% of the "
        "time. Use this only if spread limits are exhausted, and count it as "
        "the SAME position (never both, never parlayed together).\n\n"
        "*Small sample: 62 bets over 3 seasons (~20/season). Provisional.*\n\n"
        + df.to_markdown(index=False), encoding="utf-8")
    print(f"report -> {path}")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
