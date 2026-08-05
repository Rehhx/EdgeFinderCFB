"""Pull historical FIRST-QUARTER spread lines (spreads_q1) for NCAAF.

Why this market: the 1H edge exists because books derive the 1H line with a
CONSTANT (56.8% of the full spread) while the dog's actual 1H share is ~50.4%.
Quarter scores show the dog's outperformance is concentrated in Q1 and Q4, with
essentially nothing in Q2/Q3 — so Q1 should carry the same broken-constant
mispricing in sharper form.

Free mechanism test first (scratchpad, no API): the dog's Q1 margin is ~21% of
the full number, and the bet break-evens if books price Q1 at >=19.4% of the
full line. Every plausible constant (0.20-0.284) clears it, so the data is
worth buying. 8 books quote spreads_q1 pregame (more than alt spreads' 5).

All 15 weeks so the week arc is available as a control — that split is what
made the props finding. ~2,600 events x 10 credits ~= 26k.

  python -m ingestion.historical_q1
"""
import pandas as pd

from ingestion.config import PARQUET_DIR
from ingestion.historical_props import main_saturday
from ingestion.odds_client import NCAAF, OddsClient

WEEKS = [(s, w) for s in (2023, 2024, 2025) for w in range(1, 16)]
SNAP_HOUR = "14:00:00Z"
WINDOW_H = 12
MARKETS = "spreads_q1"
DEST = PARQUET_DIR / "historical_q1_lines.parquet"
# totals_q1 is a SEPARATE 10-credit market; pulled on demand via
#   python -m ingestion.historical_q1 totals_q1
# Free mechanism test says Q1 takes only 22.5% of regulation points (n=20,555)
# while a naive derivation uses 25% — ~1.3 pts too high at every total level.
TOTALS_DEST = PARQUET_DIR / "historical_q1_totals.parquet"


def pull(markets: str = MARKETS, dest=DEST) -> pd.DataFrame:
    c = OddsClient()
    rows = []
    for season, week in WEEKS:
        day = main_saturday(season, week)
        if not day:
            continue
        snap = f"{day}T{SNAP_HOUR}"
        try:
            ev = c.get(f"/historical/sports/{NCAAF}/events", {"date": snap})
        except Exception as e:
            print(f"{season} wk{week}: events {e}")
            continue
        t0 = pd.Timestamp(snap)
        events = [e for e in ev["data"]
                  if t0 <= pd.Timestamp(e["commence_time"])
                  <= t0 + pd.Timedelta(hours=WINDOW_H)]
        got = 0
        for e in events:
            try:
                resp = c.get(
                    f"/historical/sports/{NCAAF}/events/{e['id']}/odds",
                    {"date": snap, "markets": markets, "regions": "us",
                     "oddsFormat": "american"})
            except Exception:
                continue
            g = resp["data"]
            if not g.get("bookmakers"):
                continue
            got += 1
            for bk in g["bookmakers"]:
                for mkt in bk.get("markets", []):
                    for o in mkt.get("outcomes", []):
                        rows.append({
                            "season": season, "week": week,
                            "event_id": g["id"],
                            "commence": g["commence_time"],
                            "home": g["home_team"], "away": g["away_team"],
                            "book": bk["key"], "name": o["name"],
                            "point": o.get("point"), "price": o["price"]})
        print(f"{season} wk{week} ({day}): {got}/{len(events)} games w/ Q1 "
              f"| rows {len(rows):,} | credits {c.remaining()}")
    df = pd.DataFrame(rows)
    df.to_parquet(dest, index=False)
    print(f"\n{dest.stem}: {len(df):,} rows -> {dest.name}")
    return df


if __name__ == "__main__":
    import sys
    mk = sys.argv[1] if len(sys.argv) > 1 else MARKETS
    pull(mk, TOTALS_DEST if "totals" in mk else DEST)
