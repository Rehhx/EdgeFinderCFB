"""Player-level defensive signal from free CFBD box-score stats.

CFBD season player stats (1 API call/season) give per-player PD (passes
defended), INT, SACKS, QB HUR, TFL, tackles. That is NOT who-covered-whom
(no coverage charting — that needs PFF/SIS), but it DOES let us:
  1. rank each team's key coverage defenders (DBs by PD+2*INT) and
     pass-rushers (edges by sacks+hurries+TFL), and
  2. flag — via the live injury feed — when a team's top coverage defender
     is OUT, which should boost opposing WR/QB production (a market-slow
     spot). This is the realistic FREE approximation of PFF's matchup value.

  python -m features.defense_players
"""
import pandas as pd

from ingestion.cfbd_client import CFBDClient
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR, PBP_DIR

DB_POS = {"CB", "S", "DB", "SAF", "FS", "SS"}
EDGE_POS = {"DE", "EDGE", "OLB", "DL", "LB", "DT"}


def pull_defensive_stats() -> pd.DataFrame:
    """Season defensive box stats per player, all downloaded seasons.
    Cheap: 1 CFBD call per season (cached)."""
    seasons = sorted(int(p.stem.split("_")[-1])
                     for p in PBP_DIR.glob("play_by_play_*.parquet"))
    c = CFBDClient()
    frames = []
    for s in seasons:
        rows = c.get("/stats/player/season", {"year": s, "category": "defensive"})
        df = pd.DataFrame(rows)
        df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
        wide = df.pivot_table(
            index=["season", "playerId", "player", "position", "team"],
            columns="statType", values="stat", aggfunc="first").reset_index()
        frames.append(wide)
    out = pd.concat(frames, ignore_index=True).fillna(
        {"PD": 0, "INT": 0, "SACKS": 0, "QB HUR": 0, "TFL": 0, "TOT": 0})
    print(f"defensive player stats: {len(out)} player-seasons "
          f"({out.season.min()}-{out.season.max()}), calls used {c.calls_used()}")
    return out


def key_defenders(stats: pd.DataFrame) -> pd.DataFrame:
    """Per (season, team): coverage & pass-rush scores + the top-name players."""
    s = stats.copy()
    s["cover_score"] = s.get("PD", 0) + 2 * s.get("INT", 0)
    s["rush_score"] = s.get("SACKS", 0) + 0.5 * s.get("QB HUR", 0) + 0.5 * s.get("TFL", 0)
    dbs = s[s.position.isin(DB_POS)]
    rush = s[s.position.isin(EDGE_POS)]

    def top_name(g, col):
        return g.loc[g[col].idxmax(), "player"] if len(g) and g[col].max() > 0 else None

    rows = []
    for (season, team), g in s.groupby(["season", "team"]):
        gdb = dbs[(dbs.season == season) & (dbs.team == team)]
        gr = rush[(rush.season == season) & (rush.team == team)]
        rows.append({
            "season": season, "team": team,
            "secondary_score": gdb.cover_score.sum(),
            "top_cb": top_name(gdb, "cover_score"),
            "top_cb_pd": gdb.cover_score.max() if len(gdb) else 0,
            "pass_rush_score": gr.rush_score.sum(),
            "top_rusher": top_name(gr, "rush_score"),
        })
    return pd.DataFrame(rows)


def coverage_injury_flags() -> pd.DataFrame:
    """Live signal: teams whose TOP coverage defender is currently OUT/quest.
    Joins the newest key-defender list to the availability layer. Opposing
    WR/QB overs are the market-slow spot when a shutdown DB is missing."""
    kd = pd.read_parquet(PARQUET_DIR / "defense_players_key.parquet")
    kd = kd[kd.season == kd.season.max()]
    try:
        from features.availability import availability_table
        avail = availability_table()
        out = set(avail[avail.status == "out"].player.dropna())
        quest = set(avail[avail.status == "questionable"].player.dropna())
    except Exception:
        out, quest = set(), set()
    kd = kd[kd.top_cb.notna()].copy()
    kd["top_cb_status"] = kd.top_cb.map(
        lambda p: "out" if p in out else "questionable" if p in quest else "active")
    return kd[kd.top_cb_status != "active"][
        ["team", "top_cb", "top_cb_status", "top_cb_pd"]]


def build() -> pd.DataFrame:
    stats = pull_defensive_stats()
    kd = key_defenders(stats)
    dest = PARQUET_DIR / "defense_players_key.parquet"
    kd.to_parquet(dest, index=False)
    print(f"key defenders: {len(kd)} team-seasons -> {dest.name}")
    last = kd[kd.season == kd.season.max()]
    print(f"\n{kd.season.max()} best secondaries (PD+2*INT):")
    print(last.nlargest(5, "secondary_score")[
        ["team", "secondary_score", "top_cb", "top_cb_pd"]].to_string(index=False))
    return kd


if __name__ == "__main__":
    build()
