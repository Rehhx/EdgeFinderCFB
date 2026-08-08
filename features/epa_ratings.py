"""Opponent-adjusted EPA ratings via ridge regression (SP+-style baseline).

Pipeline: play-by-play -> per-game per-offense EPA/play observations ->
ridge fit of offense/defense ratings with recency weighting. Ratings are
in EPA/play units; the backtester calibrates them to point margins.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from ingestion.config import PBP_DIR

# Weeks between consecutive season week-indices; the ~15-week offseason gap
# this creates is deliberate — it decays last season's evidence for roster churn.
SEASON_STRIDE = 30
FCS_ID = -1  # all non-FBS opponents pooled into one pseudo-team
POSTSEASON = 3       # CFBD seasonType code; regular season is 2
POSTSEASON_WEEK = 20  # sorts bowls after week 16, still inside the stride

PBP_COLS = [
    "game_id", "season", "seasonType", "week", "pos_team", "def_pos_team",
    "EPA", "wp_before", "pass", "rush", "homeTeamId", "homeScore", "awayScore",
]


def load_game_obs(seasons: list[int]) -> pd.DataFrame:
    """One row per (game, offense side): mean EPA/play, plays, home flag."""
    frames = []
    for season in seasons:
        df = pd.read_parquet(
            PBP_DIR / f"play_by_play_{season}.parquet", columns=PBP_COLS
        )
        df = df[
            (df["pass"] | df["rush"])
            & df.EPA.notna()
            & df.wp_before.between(0.04, 0.96)  # garbage-time filter
            & (df.pos_team > 0) & (df.def_pos_team > 0)
        ].copy()
        df["season"] = season

        # Teams with <5 games in a season's PBP are FCS guests -> pooled bucket.
        games_per_team = (
            pd.concat([
                df.groupby("pos_team").game_id.nunique(),
                df.groupby("def_pos_team").game_id.nunique(),
            ]).groupby(level=0).max()
        )
        fbs_ids = set(games_per_team[games_per_team >= 5].index)
        df["off_id"] = df.pos_team.where(df.pos_team.isin(fbs_ids), FCS_ID)
        df["def_id"] = df.def_pos_team.where(df.def_pos_team.isin(fbs_ids), FCS_ID)
        df["off_home"] = df.pos_team == df.homeTeamId

        obs = (
            df.groupby(
                ["game_id", "season", "seasonType", "week", "off_id", "def_id", "off_home"],
                as_index=False,
            )
            .agg(epa_pp=("EPA", "mean"), n_plays=("EPA", "size"))
        )
        # Final score per game for margin labels (score columns are cumulative).
        finals = df.groupby("game_id", as_index=False).agg(
            home_id=("homeTeamId", "first"),
            home_score=("homeScore", "max"),
            away_score=("awayScore", "max"),
        )
        obs = obs.merge(finals, on="game_id")
        frames.append(obs)

    out = pd.concat(frames, ignore_index=True)
    # CFBD labels every POSTSEASON game `week=1` (verified: 2024 has 48 such
    # games, all week 1). A naive season*STRIDE + week therefore sorts a
    # season's bowls BEFORE its own week 2, and `fit_ratings` filters on
    # `week_idx < asof` — so December bowls were the most heavily recency-
    # weighted observations in every early-season fit (19-22% of week-2 fit
    # weight). Postseason must sort after the regular season.
    post = out.seasonType == POSTSEASON
    out["week_idx"] = out.season * SEASON_STRIDE + np.where(
        post, POSTSEASON_WEEK, out.week)
    return out


def asof_seasons(base: list[int], season: int) -> list[int]:
    """`base` plus the live season, once its play-by-play has actually landed.

    Live pick modules hard-coded a season range ending the PRIOR year, so they
    could not have used current-season form even if they asked for it.
    """
    if season in base or not (PBP_DIR / f"play_by_play_{season}.parquet").exists():
        return list(base)
    return list(base) + [season]


def live_asof(season: int, week: int | None) -> int:
    """Week index to fit ratings AS OF for a live pick.

    The live modules passed `season * SEASON_STRIDE + 1` — frozen at preseason
    all year — while every backtest refits weekly (corr between the two is only
    0.80, and 0.72 in weeks 6-15). `fit_ratings` filters strictly
    `week_idx < asof`, so week W sees everything before W and nothing from it.
    """
    return season * SEASON_STRIDE + max(int(week or 1), 1)


@dataclass
class RatingsModel:
    ratings: pd.DataFrame  # index: team id; cols: off, def, net
    hfa_epa: float

    def net(self, team_id: int) -> float:
        if team_id in self.ratings.index:
            return self.ratings.loc[team_id, "net"]
        return self.ratings.net.min()  # unseen team: assume worst


def fit_ratings(
    obs: pd.DataFrame,
    asof_week_idx: int,
    half_life_weeks: float = 8.0,
    alpha: float = 40.0,
) -> RatingsModel:
    """Fit offense/defense ratings on observations strictly before asof_week_idx."""
    d = obs[obs.week_idx < asof_week_idx].copy()
    if d.empty:
        raise ValueError("no training observations before as-of week")

    teams = np.union1d(d.off_id.unique(), d.def_id.unique())
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(d), len(teams)

    rows = np.arange(n)
    x_off = sparse.csr_matrix(
        (np.ones(n), (rows, d.off_id.map(idx))), shape=(n, k))
    x_def = sparse.csr_matrix(
        (np.ones(n), (rows, d.def_id.map(idx))), shape=(n, k))
    x_home = sparse.csr_matrix(d.off_home.astype(float).values.reshape(-1, 1))
    X = sparse.hstack([x_off, -x_def, x_home], format="csr")

    recency = 0.5 ** ((asof_week_idx - d.week_idx) / half_life_weeks)
    w = recency * np.sqrt(d.n_plays)

    model = Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg")
    model.fit(X, d.epa_pp.values, sample_weight=w)

    off = model.coef_[:k]
    deff = model.coef_[k:2 * k]
    ratings = pd.DataFrame({"off": off, "def": deff}, index=teams)
    ratings["net"] = ratings.off + ratings["def"]
    return RatingsModel(ratings=ratings, hfa_epa=model.coef_[-1])
