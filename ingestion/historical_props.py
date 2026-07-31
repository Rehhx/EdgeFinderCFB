"""Pull historical NCAAF player-prop lines (The Odds API) for backtesting.

For each (season, week) in the projection window, snapshot the main
Saturday at 14:00Z (~2h before the noon-ET kicks) and pull the four core
prop markets for every game commencing in the following 12 hours.
~40 credits per event; a two-season pull is ~60k credits of the 5M plan.
Everything is disk-cached, so re-runs are free.

  python -m ingestion.historical_props
"""
import pandas as pd

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR
from ingestion.odds_client import NCAAF, PROP_MARKETS, OddsClient

WEEKS = [(s, w) for s in (2023, 2024, 2025) for w in range(1, 16)]
SNAP_HOUR = "14:00:00Z"
WINDOW_H = 12

MARKET_TO_STAT = {
    "player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds",
    "player_receptions": "receptions", "player_reception_yds": "rec_yds",
}


def main_saturday(season: int, week: int) -> str | None:
    g = pd.read_parquet(CFBD_PARQUET_DIR / f"games_{season}.parquet")
    g = g[(g.week == week) & (g.seasonType == "regular")].copy()
    g["date"] = pd.to_datetime(g.startDate).dt.tz_convert("UTC").dt.date
    g["dow"] = pd.to_datetime(g.startDate).dt.tz_convert("UTC").dt.dayofweek
    sats = g[g.dow == 5]
    if sats.empty:
        return None
    return str(sats.date.value_counts().idxmax())


def pull() -> pd.DataFrame:
    client = OddsClient()
    rows = []
    for season, week in WEEKS:
        day = main_saturday(season, week)
        if not day:
            continue
        snap = f"{day}T{SNAP_HOUR}"
        try:
            ev = client.get(f"/historical/sports/{NCAAF}/events",
                            {"date": snap})
        except Exception as e:
            print(f"{season} wk{week}: events failed ({e})")
            continue
        t0 = pd.Timestamp(snap)
        events = [e for e in ev["data"]
                  if t0 <= pd.Timestamp(e["commence_time"])
                  <= t0 + pd.Timedelta(hours=WINDOW_H)]
        got = 0
        for e in events:
            try:
                odds = client.get(
                    f"/historical/sports/{NCAAF}/events/{e['id']}/odds",
                    {"date": snap, "markets": ",".join(PROP_MARKETS),
                     "regions": "us", "oddsFormat": "american"})
            except Exception:
                continue
            d = odds["data"]
            for bk in d.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    stat = MARKET_TO_STAT.get(mkt["key"])
                    if not stat:
                        continue
                    for o in mkt.get("outcomes", []):
                        rows.append({
                            "season": season, "week": week, "snap": snap,
                            "event_id": e["id"], "home": e["home_team"],
                            "away": e["away_team"],
                            "commence": e["commence_time"], "book": bk["key"],
                            "stat": stat, "player": o.get("description"),
                            "side": o["name"], "point": o.get("point"),
                            "price": o.get("price"),
                        })
            got += 1
        print(f"{season} wk{week} ({day}): {got}/{len(events)} events, "
              f"remaining credits: {client.remaining()}")

    df = pd.DataFrame(rows)
    dest = PARQUET_DIR / "historical_prop_lines.parquet"
    df.to_parquet(dest, index=False)
    print(f"historical_prop_lines: {len(df):,} outcome rows -> {dest}")
    return df


if __name__ == "__main__":
    pull()
