"""ML early-season margin model vs the linear EPA model, on the wks 1-5 edge.

The one place ML has a real shot: early season, EPA ratings are noisy (few
games), so roster priors + nonlinear interactions may beat linear EPA. Tests
ONLY where we have a validated edge (weeks 1-5 ATS). Features per game (home
minus away): as-of EPA net, previous-season net, returning production %,
recruiting z, portal net. Gradient boosting, walk-forward, compared to the
linear EPA spread on the identical games.

  python -m models.ml_spread
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from backtest.spread_baseline import consensus_lines, game_table
from features.epa_ratings import SEASON_STRIDE, fit_ratings, load_game_obs
from features.roster_priors import (portal_net, recruiting_talent, season_net,
                                    team_name_to_id)
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

PBP_SEASONS = list(range(2016, 2026))
FEATURE_SEASONS = [2021, 2022, 2023, 2024, 2025]
TEST_SEASONS = [2022, 2023, 2024, 2025]
WK_LO, WK_HI = 1, 5  # the validated edge window


def returning_by_id(season: int) -> pd.Series:
    try:
        ret = pd.read_parquet(CFBD_PARQUET_DIR / f"returning_{season}.parquet")
    except FileNotFoundError:
        return pd.Series(dtype=float)
    if not len(ret):
        return pd.Series(dtype=float)
    n2i = team_name_to_id()
    ret["team_id"] = ret.team.map(n2i)
    return ret.dropna(subset=["team_id"]).set_index(
        ret.dropna(subset=["team_id"]).team_id.astype(int)).percentPPA


def _sp_prev_by_id(season: int) -> pd.Series:
    """Previous-season SP+ rating (a strong, persistent power rating), by id."""
    n2i = team_name_to_id()
    sp = pd.read_parquet(CFBD_PARQUET_DIR / "sp_ratings.parquet")
    prev = sp[sp.season == season - 1].copy()
    prev["team_id"] = prev.team.map(n2i)
    return prev.dropna(subset=["team_id"]).set_index(
        prev.dropna(subset=["team_id"]).team_id.astype(int)).sp


def load_priors(season: int, obs: pd.DataFrame) -> dict:
    """Preseason per-team-id prior lookups for a season."""
    n2i = team_name_to_id()
    return {
        "prev_net": season_net(obs, season - 1),
        "ret": returning_by_id(season),
        "portal": portal_net(season).rename(index=lambda nm: n2i.get(nm, -99)),
        "recruit": recruiting_talent(season).rename(
            index=lambda nm: n2i.get(nm, -99)),
        "sp": _sp_prev_by_id(season),
    }


def feature_row(model, home_id, away_id, neutral, pr) -> dict:
    pn, sp = pr["prev_net"], pr["sp"]

    def f(tid):
        return (model.net(tid), pn.get(tid, pn.min()),
                pr["ret"].get(tid, np.nan), pr["portal"].get(tid, 0.0),
                pr["recruit"].get(tid, 0.0), sp.get(tid, sp.min()))
    h, a = f(home_id), f(away_id)
    return {"epa_diff": h[0] - a[0], "prev_diff": h[1] - a[1],
            "ret_diff": (h[2] or 0) - (a[2] or 0),
            "portal_diff": h[3] - a[3], "recruit_diff": h[4] - a[4],
            "sp_diff": h[5] - a[5], "home": 0.0 if neutral else 1.0}


def build_features() -> pd.DataFrame:
    obs = load_game_obs(PBP_SEASONS)
    games = game_table(obs)
    games = games[games.season.isin(FEATURE_SEASONS) & (games.season_type == 2)
                  & games.week.between(WK_LO, WK_HI)]
    lines = pd.concat([consensus_lines(s).assign(season=s)
                       for s in TEST_SEASONS])
    games = games.merge(lines[["id", "spread", "neutralSite", "homePoints",
                               "awayPoints"]], left_on="game_id",
                        right_on="id", how="left")
    games["margin"] = games.homePoints - games.awayPoints

    rows = []
    for season in TEST_SEASONS:
        pr = load_priors(season, obs)
        for week in range(WK_LO, WK_HI + 1):
            model = fit_ratings(obs, asof_week_idx=season * SEASON_STRIDE + week)
            wk = games[(games.season == season) & (games.week == week)
                       & games.spread.notna() & games.margin.notna()]
            for g in wk.itertuples():
                row = feature_row(model, g.home_id, g.away_id,
                                  bool(g.neutralSite), pr)
                rows.append(row | {
                    "season": season, "week": week, "game_id": g.game_id,
                    "margin": g.margin, "book_spread": g.spread})
    return pd.DataFrame(rows)


def train_live_model():
    """Fit the ML margin model on all early-season history; return (model,
    ratings, priors) ready to predict a target-season board."""
    df = build_features()
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                      max_iter=250, min_samples_leaf=25,
                                      l2_regularization=1.0, random_state=0)
    m.fit(df[FEATS].values, df.margin.values)
    return m


FEATS = ["epa_diff", "prev_diff", "ret_diff", "portal_diff", "recruit_diff",
         "sp_diff", "home"]


def run() -> None:
    df = build_features()
    print(f"early-season games (wks 1-5, {df.season.min()}-{df.season.max()}): "
          f"{len(df):,}")
    # linear EPA baseline (self-calibrated) vs ML, walk-forward by season
    lin_pred, ml_pred = np.full(len(df), np.nan), np.full(len(df), np.nan)
    idx = df.index.to_numpy()
    for season in TEST_SEASONS:
        tr = df[df.season < season]
        te = df[df.season == season]
        if len(tr) < 300 or te.empty:
            continue
        # linear: margin ~ epa_diff + home
        A = np.column_stack([tr.epa_diff, tr.home, np.ones(len(tr))])
        c, *_ = np.linalg.lstsq(A, tr.margin, rcond=None)
        lp = c[0] * te.epa_diff + c[1] * te.home + c[2]
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                          max_iter=250, min_samples_leaf=25,
                                          l2_regularization=1.0, random_state=0)
        m.fit(tr[FEATS].values, tr.margin.values)
        mp = m.predict(te[FEATS].values)
        pos = np.searchsorted(idx, te.index.to_numpy())
        lin_pred[pos], ml_pred[pos] = lp.values, mp
    df["lin_pred"], df["ml_pred"] = lin_pred, ml_pred
    d = df.dropna(subset=["lin_pred", "ml_pred"])
    print(f"\ngraded (2023+ walk-forward): {len(d):,}\n")
    print(f"{'model':10s} {'MAE':>7} {'ATS n':>7} {'ATS win':>8} {'ROI':>7}")
    for name, col in [("linear", "lin_pred"), ("ML", "ml_pred")]:
        mae = (d[col] - d.margin).abs().mean()
        edge = d[col] - (-d.book_spread)
        b = d[edge.abs() >= 4]
        cover = np.sign(b.margin + b.book_spread)
        won = ((np.sign(edge[edge.abs() >= 4]) == cover) & (cover != 0))
        g = cover != 0
        wr = won[g].mean()
        roi = (wr * 100 / 110 - (1 - wr)) * 100
        print(f"{name:10s} {mae:>7.2f} {int(g.sum()):>7} {wr:>7.1%} {roi:>+6.1f}%")


if __name__ == "__main__":
    run()
