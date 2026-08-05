"""Pull historical NCAAF player-prop lines (The Odds API) for backtesting.

Snapshot every GAME DAY in the week and pull the four core prop markets for
games kicking off shortly after. ~40 credits per event; everything is
disk-cached, so re-runs of already-pulled snapshots are free.

WIDENED 2026-08-05. The original pull took ONE snapshot (main Saturday 14:00Z)
and kept only kickoffs within 12 hours, which captured ~24 events/week of the
~60+ FBS plays: Thursday and Friday games were missed entirely, and the window
closed at 02:00Z (~9pm ET) so late West Coast kickoffs fell outside it. Since
the props CALIBRATION is the fragile part of that chain
(backtest/props_pricing_stability.py: sd 15.6 u/szn vs 3.5 for the volume
model), a bigger and more representative calibration sample is the cheapest
real improvement available — there is no earlier season to buy, as The Odds API
historical props start May 2023.

  python -m ingestion.historical_props
"""
import pandas as pd

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR
from ingestion.odds_client import NCAAF, PROP_MARKETS, OddsClient

WEEKS = [(s, w) for s in (2023, 2024, 2025) for w in range(1, 16)]
# Saturday keeps its 14:00Z snapshot but a 24h window (was 12) so late West
# Coast kickoffs are included. Thursday/Friday get their own 22:00Z snapshot
# with a short window — those are standalone night games.
SNAP_HOUR = "14:00:00Z"
WINDOW_H = 24
WEEKNIGHT_HOUR = "22:00:00Z"
WEEKNIGHT_WINDOW_H = 6
MIN_GAMES_PER_DAY = 1          # 1, not 2: 2023 wk15 is Army-Navy
                               # ALONE and a >=2 guard silently
                               # dropped the whole week

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


def game_days(season: int, week: int) -> list[tuple[str, str, int]]:
    """(date, snapshot_iso, window_hours) for every day this week has games.

    Returns the Saturday slate plus any Thursday/Friday night games, which the
    single-Saturday snapshot never saw.
    """
    g = pd.read_parquet(CFBD_PARQUET_DIR / f"games_{season}.parquet")
    g = g[(g.week == week) & (g.seasonType == "regular")].copy()
    if g.empty:
        return []
    ts = pd.to_datetime(g.startDate).dt.tz_convert("UTC")
    g["date"], g["dow"] = ts.dt.date, ts.dt.dayofweek
    out = []
    for (date, dow), grp in g.groupby(["date", "dow"]):
        if len(grp) < MIN_GAMES_PER_DAY:
            continue
        if dow == 5:                                   # Saturday
            out.append((str(date), f"{date}T{SNAP_HOUR}", WINDOW_H))
        elif dow in (3, 4):                            # Thu / Fri night
            out.append((str(date), f"{date}T{WEEKNIGHT_HOUR}",
                        WEEKNIGHT_WINDOW_H))
    return out


def pull() -> pd.DataFrame:
    client = OddsClient()
    rows = []
    seen_events: set[str] = set()
    for season, week in WEEKS:
      for day, snap, window_h in game_days(season, week):
        try:
            ev = client.get(f"/historical/sports/{NCAAF}/events",
                            {"date": snap})
        except Exception as e:
            print(f"{season} wk{week}: events failed ({e})")
            continue
        t0 = pd.Timestamp(snap)
        events = [e for e in ev["data"]
                  if t0 <= pd.Timestamp(e["commence_time"])
                  <= t0 + pd.Timedelta(hours=window_h)
                  and e["id"] not in seen_events]
        # a 24h Saturday window can reach into a later day's slate; never pay
        # for the same event twice
        seen_events.update(e["id"] for e in events)
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
    # NEVER blind-overwrite: every props result in the project is built on this
    # file, and a mid-run API failure would otherwise truncate it. Re-pulls of
    # already-cached snapshots are free and reproduce the same rows, so the new
    # frame should be a SUPERSET — refuse to shrink it.
    if dest.exists():
        old = pd.read_parquet(dest)
        if len(df) < len(old):
            alt = dest.with_suffix(".partial.parquet")
            df.to_parquet(alt, index=False)
            raise RuntimeError(
                f"pull produced {len(df):,} rows but {dest.name} already has "
                f"{len(old):,}. Refusing to shrink it — wrote {alt.name} "
                "instead. Inspect before replacing.")
        print(f"  (superset check ok: {len(old):,} -> {len(df):,} rows)")
    df.to_parquet(dest, index=False)
    print(f"historical_prop_lines: {len(df):,} outcome rows -> {dest}")
    return df


if __name__ == "__main__":
    pull()
