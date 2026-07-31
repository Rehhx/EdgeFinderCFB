"""First-half team ratings + official 1H scores, for 1H spread/total/ML.

Thesis: books derive 1H lines mechanically from the full game (~half the
spread, ~44% of the total), but teams have persistent first-half tendencies
(fast/slow starters, scripted openings). 1H-specific opponent-adjusted EPA
ratings + coach scripted-opening data may beat the derived line.

  load_1h_obs(seasons)  -> offense-side 1H EPA observations (fit_ratings-ready)
  one_half_scores(seas) -> official 1H home/away points from line scores
"""
import numpy as np
import pandas as pd

from features.epa_ratings import FCS_ID, SEASON_STRIDE
from ingestion.config import CFBD_PARQUET_DIR, PBP_DIR

PBP_COLS = ["game_id", "season", "seasonType", "week", "period", "pos_team",
            "def_pos_team", "EPA", "wp_before", "pass", "rush", "homeTeamId"]


def load_1h_obs(seasons: list[int]) -> pd.DataFrame:
    """One row per (game, offense side): mean EPA/play on FIRST-HALF plays."""
    frames = []
    for season in seasons:
        df = pd.read_parquet(PBP_DIR / f"play_by_play_{season}.parquet",
                             columns=PBP_COLS)
        for c in ("pass", "rush"):
            df[c] = df[c].fillna(False).astype(bool)
        df = df[(df["pass"] | df["rush"]) & df.EPA.notna()
                & df.period.isin([1, 2])            # first half only
                & df.wp_before.between(0.05, 0.95)  # light garbage filter
                & df.pos_team.notna() & df.def_pos_team.notna()].copy()
        df["season"] = season
        df["pos_team"] = df.pos_team.astype(int)
        df["def_pos_team"] = df.def_pos_team.astype(int)
        gp = pd.concat([df.groupby("pos_team").game_id.nunique(),
                        df.groupby("def_pos_team").game_id.nunique()]
                       ).groupby(level=0).max()
        fbs = set(gp[gp >= 5].index)
        df["off_id"] = df.pos_team.where(df.pos_team.isin(fbs), FCS_ID)
        df["def_id"] = df.def_pos_team.where(df.def_pos_team.isin(fbs), FCS_ID)
        df["off_home"] = df.pos_team == df.homeTeamId
        obs = df.groupby(
            ["game_id", "season", "seasonType", "week", "off_id", "def_id",
             "off_home"], as_index=False).agg(
            epa_pp=("EPA", "mean"), n_plays=("EPA", "size"))
        frames.append(obs)
    out = pd.concat(frames, ignore_index=True)
    out["week_idx"] = out.season * SEASON_STRIDE + out.week
    return out


def one_half_scores(seasons: list[int]) -> pd.DataFrame:
    """Official first-half points per game from quarter line scores."""
    rows = []
    for s in seasons:
        g = pd.read_parquet(CFBD_PARQUET_DIR / f"games_{s}.parquet")

        def h1(ls):
            try:
                return int(ls[0]) + int(ls[1])
            except (TypeError, IndexError, ValueError):
                return np.nan
        g = g[g.homeLineScores.notna() & g.awayLineScores.notna()].copy()
        g["h1_home"] = g.homeLineScores.apply(h1)
        g["h1_away"] = g.awayLineScores.apply(h1)
        rows.append(g[["id", "week", "neutralSite", "homeTeam", "awayTeam",
                       "h1_home", "h1_away"]].assign(season=s))
    out = pd.concat(rows, ignore_index=True).dropna(subset=["h1_home"])
    out["h1_margin"] = out.h1_home - out.h1_away
    out["h1_total"] = out.h1_home + out.h1_away
    return out


def home_ids(obs_1h: pd.DataFrame) -> pd.DataFrame:
    """(game_id -> home_id, away_id) from the 1H observations."""
    home = obs_1h[obs_1h.off_home].groupby("game_id").off_id.first().rename("home_id")
    away = obs_1h[~obs_1h.off_home].groupby("game_id").off_id.first().rename("away_id")
    return pd.concat([home, away], axis=1).reset_index()
