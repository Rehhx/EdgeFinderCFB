"""Pull historical anytime-TD prop lines (player_anytime_td).

Highest-volume prop market: ~70 quoted players per game vs ~20 for yardage
props, and it runs all season (our spread edges are stuck in weeks 1-5).

  python -m ingestion.historical_td
"""
import pandas as pd

from ingestion.config import PARQUET_DIR
from ingestion.historical_props import main_saturday
from ingestion.odds_client import NCAAF, OddsClient

WEEKS = [(s, w) for s in (2023, 2024, 2025) for w in range(1, 16)]
SNAP_HOUR = "14:00:00Z"
WINDOW_H = 12
DEST = PARQUET_DIR / "historical_td_lines.parquet"


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
            print(f"{season} wk{week}: {e}")
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
                    {"date": snap, "markets": "player_anytime_td",
                     "regions": "us", "oddsFormat": "american"})
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
                            "home": g["home_team"], "away": g["away_team"],
                            "book": bk["key"], "player": o.get("description"),
                            "name": o.get("name"), "price": o.get("price")})
        print(f"{season} wk{week}: {got}/{len(events)} events "
              f"| credits {c.remaining()}")
    df = pd.DataFrame(rows)
    df.to_parquet(DEST, index=False)
    print(f"historical_td_lines: {len(df):,} rows -> {DEST.name}")
    return df


if __name__ == "__main__":
    pull()
