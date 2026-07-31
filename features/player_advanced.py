"""PFF-adjacent player data from CFBD (free): situational usage + player PPA.

Two endpoints we hadn't tapped:
  /player/usage        usage share split by down and standard/passing downs —
                       tells us WHEN a player is used, not just how much. A
                       passing-down specialist sees more work when his team
                       trails (game script), which our flat share misses.
  /ppa/players/games   per-player per-game PPA (predicted points added), split
                       all/pass/rush — a per-game value metric, the closest
                       free analogue to a PFF grade.

Leakage rules: /player/usage is season-aggregate, so it is used ONLY as a
previous-season prior. /ppa/players/games is per-game, so trailing means are
built as-of (shifted one game).

  python -m features.player_advanced
"""
import numpy as np
import pandas as pd

from ingestion.cfbd_client import CFBDClient
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

SEASONS = (2022, 2023, 2024, 2025)
WEEKS = range(1, 16)
USAGE_DEST = PARQUET_DIR / "player_usage.parquet"
PPA_DEST = PARQUET_DIR / "player_ppa_games.parquet"


def pull_usage() -> pd.DataFrame:
    c = CFBDClient()
    rows = []
    for s in SEASONS:
        for r in c.get("/player/usage", {"year": s}):
            u = r.get("usage") or {}
            rows.append({"season": s, "player": r.get("name"),
                         "team": r.get("team"), "position": r.get("position"),
                         "usage_overall": u.get("overall"),
                         "usage_pass": u.get("pass"), "usage_rush": u.get("rush"),
                         "usage_third": u.get("thirdDown"),
                         "usage_standard": u.get("standardDowns"),
                         "usage_passing_downs": u.get("passingDowns")})
    df = pd.DataFrame(rows)
    df.to_parquet(USAGE_DEST, index=False)
    print(f"player_usage: {len(df):,} player-seasons -> {USAGE_DEST.name}")
    return df


def pull_ppa_games() -> pd.DataFrame:
    c = CFBDClient()
    rows = []
    for s in SEASONS:
        for w in WEEKS:
            try:
                data = c.get("/ppa/players/games", {"year": s, "week": w})
            except Exception:
                continue
            for r in data:
                a = r.get("averagePPA") or {}
                rows.append({"season": s, "week": w, "player": r.get("name"),
                             "team": r.get("team"), "position": r.get("position"),
                             "ppa_all": a.get("all"), "ppa_pass": a.get("pass"),
                             "ppa_rush": a.get("rush")})
        print(f"  {s}: {sum(1 for x in rows if x['season'] == s):,} rows "
              f"(calls {c.calls_used()})")
    df = pd.DataFrame(rows)
    df.to_parquet(PPA_DEST, index=False)
    print(f"player_ppa_games: {len(df):,} player-games -> {PPA_DEST.name}")
    return df


def usage_prior() -> pd.DataFrame:
    """Previous-season situational usage, keyed to the UPCOMING season."""
    u = pd.read_parquet(USAGE_DEST).copy()
    u["season"] += 1  # prior for the next season
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    s2i = dict(zip(teams.school, teams.id))
    u["team_id"] = u.team.map(s2i)
    u = u.dropna(subset=["team_id"]).astype({"team_id": int})
    # how pass-heavy is this player's role? (passing-down vs standard usage)
    u["pd_lean"] = (u.usage_passing_downs.fillna(0)
                    - u.usage_standard.fillna(0))
    return u[["season", "team_id", "player", "usage_third", "pd_lean",
              "usage_pass", "usage_rush"]]


def ppa_trailing() -> pd.DataFrame:
    """As-of trailing player PPA (shifted one game, EWMA), keyed by
    (season, week, player, team_id).

    Player NAME alone is not unique — 4,263 name-weeks collide across teams,
    which silently duplicated projection rows when joined. Always join on
    team_id too, and de-duplicate before returning.
    """
    p = pd.read_parquet(PPA_DEST)
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    p["team_id"] = p.team.map(dict(zip(teams.school, teams.id)))
    p = p.dropna(subset=["team_id"]).astype({"team_id": int})
    p = p.sort_values(["season", "team_id", "player", "week"])
    g = p.groupby(["season", "team_id", "player"])
    for c in ("ppa_all", "ppa_pass", "ppa_rush"):
        p[f"trail_{c}"] = g[c].transform(
            lambda s: s.shift(1).ewm(halflife=3, min_periods=1).mean())
    out = p[["season", "week", "team_id", "player", "trail_ppa_all",
             "trail_ppa_pass", "trail_ppa_rush"]]
    return out.drop_duplicates(subset=["season", "week", "team_id", "player"])


if __name__ == "__main__":
    pull_usage()
    pull_ppa_games()
    print(f"usage_prior rows: {len(usage_prior()):,} | "
          f"ppa_trailing rows: {len(ppa_trailing()):,}")
