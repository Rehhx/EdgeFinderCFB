"""Preseason roster priors: returning production + portal delta + recruiting.

For a target season, fits (on strictly earlier seasons) a model mapping
    [last season's net EPA rating, returning PPA%, recruiting talent,
     net portal talent] -> this season's full-season net EPA rating
and predicts each team's preseason prior. Everything is keyed by ESPN
team id and walk-forward safe: no data from the target season is used
except its own offseason roster inputs (known before kickoff).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from features.epa_ratings import SEASON_STRIDE, fit_ratings, load_game_obs
from features.ol_continuity import ol_continuity
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

# 247 composite ratings run ~0.70-1.00; value above a replacement walk-on.
REPLACEMENT_RATING = 0.79
STAR_TO_RATING = {5: 0.985, 4: 0.93, 3: 0.87, 2: 0.83, 1: 0.80}


def team_name_to_id() -> dict[str, int]:
    t = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    return dict(zip(t.school, t.id))


def portal_net(season: int) -> pd.Series:
    """Net incoming-minus-outgoing portal talent per team name."""
    p = pd.read_parquet(CFBD_PARQUET_DIR / f"portal_{season}.parquet")
    rating = p.rating.fillna(p.stars.map(STAR_TO_RATING)).fillna(REPLACEMENT_RATING)
    p = p.assign(value=(rating - REPLACEMENT_RATING).clip(lower=0))
    incoming = p.groupby("destination").value.sum()
    outgoing = p.groupby("origin").value.sum()
    return incoming.sub(outgoing, fill_value=0.0).rename("portal_net")


def recruiting_talent(season: int) -> pd.Series:
    """Mean class points over the 4 classes on the roster, z-scored."""
    r = pd.read_parquet(CFBD_PARQUET_DIR / "recruiting_teams.parquet")
    r = r[r.year.between(season - 3, season)]
    pts = r.groupby("team").points.mean()
    return ((pts - pts.mean()) / pts.std()).rename("recruit_z")


def staff_flags(season: int) -> pd.DataFrame:
    """new_hc / oc_change per school from the coach database."""
    cdb = pd.read_parquet(PARQUET_DIR / "coach_db.parquet")
    cdb = cdb[cdb.season == season].set_index("school")
    return cdb[["new_hc", "oc_change"]]


def roster_features(season: int) -> pd.DataFrame:
    try:
        ret = pd.read_parquet(CFBD_PARQUET_DIR / f"returning_{season}.parquet")
        assert len(ret)
    except (FileNotFoundError, AssertionError):
        # preseason before CFBD publishes returning production: all FBS
        # teams with league-median returning numbers from the prior year
        teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
        prev = pd.read_parquet(CFBD_PARQUET_DIR / f"returning_{season - 1}.parquet")
        ret = pd.DataFrame({
            "team": teams.school,
            "percentPPA": prev.percentPPA.median(),
            "usage": prev.usage.median(),
        })
    df = ret[["team", "percentPPA", "usage"]].set_index("team")
    df = df.join(portal_net(season), how="left").join(
        recruiting_talent(season), how="left")
    df = df.join(staff_flags(season), how="left")
    try:
        df = df.join(ol_continuity(season), how="left")
    except FileNotFoundError:  # rosters for `season` not published yet
        df["ret_ol_share"], df["ol_exp"] = np.nan, np.nan
    df["portal_net"] = df.portal_net.fillna(0.0)
    df["recruit_z"] = df.recruit_z.fillna(df.recruit_z.median())
    df["new_hc"] = df.new_hc.fillna(0)
    df["oc_change"] = df.oc_change.fillna(0)
    df["ret_ol_share"] = df.ret_ol_share.fillna(
        df.ret_ol_share.median() if df.ret_ol_share.notna().any() else 0.5)
    df["ol_exp"] = df.ol_exp.fillna(
        df.ol_exp.median() if df.ol_exp.notna().any() else 2.3)
    name_to_id = team_name_to_id()
    df["team_id"] = df.index.map(name_to_id)
    unmatched = df.team_id.isna().sum()
    if unmatched:
        print(f"  roster_features({season}): {unmatched} team names unmatched, dropped")
    return df.dropna(subset=["team_id"]).set_index(df.dropna(subset=["team_id"]).team_id.astype(int)).drop(columns="team_id")


def season_net(obs: pd.DataFrame, season: int) -> pd.Series:
    """Full-season net EPA rating (evenly weighted within the season)."""
    d = obs[obs.season == season]
    model = fit_ratings(
        d, asof_week_idx=season * SEASON_STRIDE + 25, half_life_weeks=999)
    return model.ratings.net


FEATURES = ["prev_net", "percentPPA", "usage", "recruit_z", "portal_net",
            "new_hc", "oc_change", "ret_ol_share", "ol_exp"]


def preseason_priors(target_season: int, obs: pd.DataFrame) -> pd.Series:
    """Prior net rating per team id for target_season (walk-forward safe)."""
    rows, ys = [], []
    for s in range(2022, target_season):
        feats = roster_features(s)
        prev = season_net(obs, s - 1)
        outcome = season_net(obs, s)
        common = feats.index.intersection(prev.index).intersection(outcome.index)
        f = feats.loc[common].assign(prev_net=prev.loc[common])
        rows.append(f[FEATURES])
        ys.append(outcome.loc[common])

    X = pd.concat(rows)
    y = pd.concat(ys)
    mu, sd = X.mean(), X.std().replace(0, 1.0)
    model = Ridge(alpha=1.0)
    model.fit(((X - mu) / sd).values, y.values)

    feats_t = roster_features(target_season)
    prev_t = season_net(obs, target_season - 1)
    common = feats_t.index.intersection(prev_t.index)
    Xt = feats_t.loc[common].assign(prev_net=prev_t.loc[common])[FEATURES]
    pred = pd.Series(
        model.predict(((Xt - mu) / sd).values), index=common, name="prior_net")

    coefs = dict(zip(FEATURES, model.coef_.round(4)))
    print(f"  prior model for {target_season} (trained on 2022-{target_season-1}): {coefs}")
    return pred
