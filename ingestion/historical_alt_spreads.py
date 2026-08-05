"""Pull historical ALTERNATE spread ladders (alternate_spreads) for NCAAF.

Hypothesis under test: books derive the alt ladder from the main number with a
fixed price-per-point formula. Our validated big-dog edge says the main number
is wrong by ~6 points on early-season mismatches, so the error should propagate
into every rung — and if the book's per-point pricing is flatter or steeper
than the true margin distribution, some rung should pay better than the main
line does. Same family as the 1H derived-line shortcut that DID pay.

Weeks 1-5 only (that is where the big-dog spread edge lives), all games — the
non-big-dog games are the control group. Period/alt markets 422 on the bulk
endpoint, so this walks event by event: 10 credits x event.

  python -m ingestion.historical_alt_spreads
"""
import pandas as pd

from ingestion.config import PARQUET_DIR
from ingestion.historical_props import main_saturday
from ingestion.odds_client import NCAAF, OddsClient

WEEKS = [(s, w) for s in (2023, 2024, 2025) for w in range(1, 6)]
SNAP_HOUR = "14:00:00Z"
WINDOW_H = 12
MARKETS = "alternate_spreads"
DEST = PARQUET_DIR / "historical_alt_spreads.parquet"


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
            try:
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
        print(f"{season} wk{week} ({day}): {got}/{len(events)} games w/ alt "
              f"ladders | rows {len(rows):,} | credits {c.remaining()}")
    df = pd.DataFrame(rows)
    df.to_parquet(DEST, index=False)
    print(f"\nhistorical_alt_spreads: {len(df):,} rows -> {DEST.name}")
    return df


if __name__ == "__main__":
    pull()
