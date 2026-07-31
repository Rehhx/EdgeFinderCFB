"""Offense-vs-defense matchup features (pass & rush splits), walk-forward.

CFBD /stats/season/advanced gives, per team, BOTH sides split by play type:
  offense.passingPlays / rushingPlays -> {ppa, successRate, explosiveness}
  defense.passingPlays / rushingPlays -> what the defense ALLOWS
plus havoc, stuffRate, lineYards. Queried with endWeek so each week's row uses
only games played BEFORE that week (no leakage).

Matchup feature = the offense's pass profile vs the opponent defense's pass
profile allowed (and same for rush) — i.e. "top-10 pass offense vs 90th-ranked
pass defense", expressed as percentile ranks and raw differentials.

  python -m features.matchups          # pull + build (≈45 CFBD calls)
"""
import numpy as np
import pandas as pd

from ingestion.cfbd_client import CFBDClient
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

SEASONS = (2023, 2024, 2025)
WEEKS = range(1, 16)
DEST = PARQUET_DIR / "matchup_stats.parquet"


def _flat(row: dict, side: str) -> dict:
    s = row.get(side) or {}
    out = {f"{side}_ppa": s.get("ppa"),
           f"{side}_success": s.get("successRate"),
           f"{side}_explosive": s.get("explosiveness"),
           f"{side}_havoc": (s.get("havoc") or {}).get("total")
           if isinstance(s.get("havoc"), dict) else s.get("havoc"),
           f"{side}_stuff": s.get("stuffRate"),
           f"{side}_line_yds": s.get("lineYards")}
    for pt in ("passingPlays", "rushingPlays"):
        p = s.get(pt) or {}
        tag = "pass" if pt == "passingPlays" else "rush"
        out[f"{side}_{tag}_ppa"] = p.get("ppa")
        out[f"{side}_{tag}_success"] = p.get("successRate")
        out[f"{side}_{tag}_explosive"] = p.get("explosiveness")
        out[f"{side}_{tag}_rate"] = p.get("rate")
    return out


def pull() -> pd.DataFrame:
    """As-of advanced stats: for (season, week) use data through week-1."""
    c = CFBDClient()
    rows = []
    for season in SEASONS:
        for week in WEEKS:
            if week == 1:
                continue  # nothing played yet this season
            try:
                data = c.get("/stats/season/advanced",
                             {"year": season, "endWeek": week - 1})
            except Exception as e:
                print(f"{season} wk{week}: {e}")
                continue
            for r in data:
                if not r.get("team"):
                    continue
                rows.append({"season": season, "week": week,
                             "team": r["team"],
                             **_flat(r, "offense"), **_flat(r, "defense")})
        print(f"{season}: {sum(1 for x in rows if x['season'] == season)} rows "
              f"| calls {c.calls_used()}")
    df = pd.DataFrame(rows)
    df.to_parquet(DEST, index=False)
    print(f"matchup_stats: {len(df):,} team-weeks -> {DEST.name}")
    return df


def matchup_features(stat: str) -> pd.DataFrame:
    """Per (season, week, team_id, opp_id): the offense-vs-defense edge for
    the play type this prop depends on, as percentile ranks + differentials."""
    ms = pd.read_parquet(DEST)
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    s2i = dict(zip(teams.school, teams.id))
    ms["team_id"] = ms.team.map(s2i)
    ms = ms.dropna(subset=["team_id"]).astype({"team_id": int})

    side = "rush" if stat == "rush_yds" else "pass"
    off = [f"offense_{side}_ppa", f"offense_{side}_success",
           f"offense_{side}_explosive"]
    dfn = [f"defense_{side}_ppa", f"defense_{side}_success",
           f"defense_{side}_explosive"]

    # percentile rank within each (season, week) so scales are comparable
    g = ms.groupby(["season", "week"])
    for c in off + dfn:
        ms[f"{c}_pct"] = g[c].rank(pct=True)

    o = ms[["season", "week", "team_id"] + [f"{c}_pct" for c in off]].copy()
    d = ms[["season", "week", "team_id"] + [f"{c}_pct" for c in dfn]].copy()
    d = d.rename(columns={"team_id": "opp_id"})
    m = o.merge(d, on=["season", "week"])
    m = m[m.team_id != m.opp_id]

    # the matchup edge: offense percentile MINUS defense-strength percentile.
    # NOTE defense_*_ppa_pct high = ALLOWS more = weak defense, so a high
    # value is GOOD for the offense; both point the same way.
    m["mu_ppa"] = m[f"offense_{side}_ppa_pct"] + m[f"defense_{side}_ppa_pct"]
    m["mu_success"] = (m[f"offense_{side}_success_pct"]
                       + m[f"defense_{side}_success_pct"])
    m["mu_explosive"] = (m[f"offense_{side}_explosive_pct"]
                         + m[f"defense_{side}_explosive_pct"])
    m["mu_total"] = (m.mu_ppa + m.mu_success + m.mu_explosive) / 6  # 0..1
    return m[["season", "week", "team_id", "opp_id", "mu_ppa", "mu_success",
              "mu_explosive", "mu_total"]]


if __name__ == "__main__":
    pull()
    for s in ("rush_yds", "rec_yds"):
        f = matchup_features(s)
        print(f"{s}: {len(f):,} matchup rows, mu_total "
              f"{f.mu_total.min():.2f}-{f.mu_total.max():.2f}")
