"""Pull historical first-half lines (spreads_h1, totals_h1, h2h_h1) for NCAAF.

Uses the bulk historical odds endpoint (all games at a snapshot in one call)
rather than per-event — far cheaper. One Saturday-morning snapshot per week.

  python -m ingestion.historical_1h
"""
import pandas as pd

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR
from ingestion.historical_props import main_saturday
from ingestion.odds_client import NCAAF, OddsClient

WEEKS = [(s, w) for s in (2023, 2024, 2025) for w in range(1, 16)]
SNAP_HOUR = "14:00:00Z"
MARKETS = "spreads_h1,totals_h1,h2h_h1"


WINDOW_H = 12


def pull() -> pd.DataFrame:
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
            try:  # period markets need the per-EVENT historical endpoint
                resp = c.get(
                    f"/historical/sports/{NCAAF}/events/{e['id']}/odds",
                    {"date": snap, "markets": MARKETS, "regions": "us",
                     "oddsFormat": "american"})
            except Exception:
                continue
            g = resp["data"]
            if not g.get("bookmakers"):
                continue
            got += 1
            for bk in g.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    for o in mkt.get("outcomes", []):
                        rows.append({
                            "season": season, "week": week,
                            "event_id": g["id"], "commence": g["commence_time"],
                            "home": g["home_team"], "away": g["away_team"],
                            "book": bk["key"], "market": mkt["key"],
                            "name": o["name"], "point": o.get("point"),
                            "price": o["price"]})
        print(f"{season} wk{week} ({day}): {got}/{len(events)} games w/ 1H, "
              f"credits left {c.remaining()}")
    df = pd.DataFrame(rows)
    dest = PARQUET_DIR / "historical_1h_lines.parquet"
    df.to_parquet(dest, index=False)
    print(f"historical_1h_lines: {len(df):,} rows -> {dest.name}")
    return df


if __name__ == "__main__":
    pull()
