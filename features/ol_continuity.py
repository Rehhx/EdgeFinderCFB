"""O-line continuity from bulk rosters + OL performance from play-by-play.

Continuity (preseason-known, feeds the roster prior):
  ret_ol_share : share of last season's OL room back this season
  ol_exp       : mean class year (1=FR..4+=SR) of the current OL room
Performance (in-season, for later phases / props):
  per team-season stuff rate and sack rate allowed, from PBP.
"""
import pandas as pd

from ingestion.config import PARQUET_DIR, PBP_DIR

ROSTER_DIR = PARQUET_DIR / "rosters"
OL_POS = {"OL", "OT", "OG", "C", "G", "T"}


def _ol_room(season: int) -> pd.DataFrame:
    r = pd.read_parquet(ROSTER_DIR / f"rosters_{season}.parquet",
                        columns=["athlete_id", "team", "position", "year"])
    return r[r.position.isin(OL_POS)]


def ol_continuity(season: int) -> pd.DataFrame:
    """Per team (school name): ret_ol_share, ol_exp for `season`."""
    cur, prev = _ol_room(season), _ol_room(season - 1)
    prev_ids = prev.groupby("team").athlete_id.apply(set)
    cur_ids = cur.groupby("team").athlete_id.apply(set)

    teams = cur_ids.index.intersection(prev_ids.index)
    share = pd.Series(
        {t: len(cur_ids[t] & prev_ids[t]) / max(len(prev_ids[t]), 1)
         for t in teams}, name="ret_ol_share")
    exp = cur.groupby("team").year.mean().rename("ol_exp")
    return pd.concat([share, exp], axis=1)


def ol_performance(seasons: list[int]) -> pd.DataFrame:
    """Per team-season OL proxies from PBP: stuff rate, sack rate allowed."""
    out = []
    for season in seasons:
        df = pd.read_parquet(
            PBP_DIR / f"play_by_play_{season}.parquet",
            columns=["pos_team", "rush", "pass", "sack", "yds_rushed",
                     "wp_before"])
        df = df[df.wp_before.between(0.04, 0.96)]
        rush = df[df.rush]
        pas = df[df["pass"]]
        agg = pd.DataFrame({
            "stuff_rate": rush.groupby("pos_team")
                .yds_rushed.apply(lambda y: (y.dropna() <= 0).mean()),
            "sack_rate": pas.groupby("pos_team").sack.mean(),
            "rush_plays": rush.groupby("pos_team").size(),
        }).assign(season=season)
        out.append(agg)
    res = pd.concat(out)
    res = res[res.rush_plays >= 150]  # FBS-sized samples only
    dest = PARQUET_DIR / "ol_performance.parquet"
    res.to_parquet(dest)
    return res


if __name__ == "__main__":
    print(ol_continuity(2025).describe().round(3).to_string())
    perf = ol_performance([2021, 2022, 2023, 2024, 2025])
    print(perf.groupby("season")[["stuff_rate", "sack_rate"]]
          .mean().round(3).to_string())
