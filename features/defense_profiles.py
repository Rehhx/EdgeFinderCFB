"""Opponent-adjusted, facet-split defensive profiles from play-by-play.

The props model previously used a single RAW allowed-yards number per facet
(ypc_allowed / ypa_allowed) — which flatters defenses that faced weak
offenses. This module gives, per team-season (and as-of-week for in-season):
  pass_def_epa : opponent-adjusted EPA/play allowed on pass plays (lower=better D)
  rush_def_epa : same for rush plays
  sack_rate    : sacks / dropbacks (pass-rush strength → hurts QB/pass props)
  pass_havoc   : havoc plays / pass plays (coverage+rush disruption)
  expl_pass_allowed : 20+ air/ YAC completions allowed / pass plays

HONEST LIMITATION: CFBD play-by-play has no coverage charting (who covered
whom), so we CANNOT model a shutdown corner shadowing a specific WR. This is
UNIT-level pass defense, not player-vs-player matchup. Player-level coverage
would need PFF charting data (paid, not owned).

  python -m features.defense_profiles
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from features.epa_ratings import FCS_ID, SEASON_STRIDE
from ingestion.config import PARQUET_DIR, PBP_DIR

COLS = ["game_id", "season", "seasonType", "week", "pos_team", "def_pos_team",
        "pass", "rush", "EPA", "sack", "havoc", "completion", "yds_receiving",
        "wp_before"]


def _load(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        df = pd.read_parquet(PBP_DIR / f"play_by_play_{s}.parquet", columns=COLS)
        for c in ("pass", "rush", "sack", "havoc", "completion"):
            df[c] = df[c].fillna(False).astype(bool)
        df = df[(df["pass"] | df["rush"]) & df.EPA.notna()
                & df.pos_team.notna() & df.def_pos_team.notna()
                & df.wp_before.between(0.05, 0.95)].copy()
        df["season"] = s
        df["pos_team"] = df.pos_team.astype(int)
        df["def_pos_team"] = df.def_pos_team.astype(int)
        # pool non-FBS (teams with tiny play counts) into one bucket
        gp = pd.concat([df.groupby("pos_team").game_id.nunique(),
                        df.groupby("def_pos_team").game_id.nunique()]
                       ).groupby(level=0).max()
        fbs = set(gp[gp >= 5].index)
        df["off_id"] = df.pos_team.where(df.pos_team.isin(fbs), FCS_ID)
        df["def_id"] = df.def_pos_team.where(df.def_pos_team.isin(fbs), FCS_ID)
        df["week_idx"] = df.season * SEASON_STRIDE + df.week
        df["expl_pass"] = df.completion & (df.yds_receiving.fillna(0) >= 20)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _adj_def_epa(plays: pd.DataFrame, alpha: float = 50.0) -> pd.Series:
    """Ridge: EPA ~ offense dummies + defense dummies. Returns the defense
    coefficient per team (lower = suppresses EPA = better defense)."""
    teams = np.union1d(plays.off_id.unique(), plays.def_id.unique())
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(plays), len(teams)
    rows = np.arange(n)
    x_off = sparse.csr_matrix((np.ones(n), (rows, plays.off_id.map(idx))),
                              shape=(n, k))
    x_def = sparse.csr_matrix((np.ones(n), (rows, plays.def_id.map(idx))),
                              shape=(n, k))
    X = sparse.hstack([x_off, x_def], format="csr")
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X, plays.EPA.values)
    return pd.Series(m.coef_[k:2 * k], index=teams)


def _profiles_from(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for season, s_df in df.groupby("season"):
        pas = s_df[s_df["pass"]]
        rush = s_df[s_df.rush]
        pass_def = _adj_def_epa(pas).rename("pass_def_epa")
        rush_def = _adj_def_epa(rush).rename("rush_def_epa")
        rates = pd.DataFrame({
            "sack_rate": pas.groupby("def_id").sack.mean(),
            "pass_havoc": pas.groupby("def_id").havoc.mean(),
            "expl_pass_allowed": pas.groupby("def_id").expl_pass.mean(),
            "pass_plays": pas.groupby("def_id").size(),
        })
        prof = pd.concat([pass_def, rush_def, rates], axis=1)
        prof["season"] = season
        out.append(prof)
    res = pd.concat(out).reset_index().rename(columns={"index": "team_id"})
    return res[res.pass_plays >= 100]


def defense_profiles(seasons: list[int], asof_week_idx: int | None = None,
                     df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (season, def team) profile. If asof_week_idx given, only uses
    plays strictly before it (walk-forward safe). Pass a preloaded `df`
    (from _load) to avoid re-reading parquet in walk-forward loops."""
    if df is None:
        df = _load(seasons)
    if asof_week_idx is not None:
        df = df[df.week_idx < asof_week_idx]
    return _profiles_from(df)


def build_asof() -> pd.DataFrame:
    """Walk-forward defense: profile for each (season, week, team) using only
    plays strictly before that week. For props' opponent adjustment."""
    seasons = sorted(int(p.stem.split("_")[-1])
                     for p in PBP_DIR.glob("play_by_play_*.parquet"))
    df = _load(seasons)
    out = []
    for season in seasons:
        for week in range(1, 16):
            asof = season * SEASON_STRIDE + week
            sub = df[df.week_idx < asof]
            if sub.empty:
                continue
            prof = _profiles_from(sub)
            prof = prof[prof.season == season]
            if prof.empty:  # early week: no current-season plays yet
                continue
            prof["week"] = week
            out.append(prof)
    res = pd.concat(out, ignore_index=True)
    dest = PARQUET_DIR / "defense_asof.parquet"
    res.to_parquet(dest, index=False)
    print(f"defense_asof: {len(res)} team-weeks -> {dest.name}")
    return res


def build() -> pd.DataFrame:
    seasons = sorted(int(p.stem.split("_")[-1])
                     for p in PBP_DIR.glob("play_by_play_*.parquet"))
    prof = defense_profiles(seasons)
    dest = PARQUET_DIR / "defense_profiles.parquet"
    prof.to_parquet(dest, index=False)
    print(f"defense_profiles: {len(prof)} team-seasons -> {dest.name}")
    build_asof()

    # face validity: best/worst pass defenses in the latest full season
    from ingestion.config import CFBD_PARQUET_DIR
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")[["id", "school"]]
    last = prof[prof.season == prof.season.max()].merge(
        teams, left_on="team_id", right_on="id")
    print(f"\n{prof.season.max()} best pass defenses (lowest EPA allowed):")
    print(last.nsmallest(5, "pass_def_epa")[
        ["school", "pass_def_epa", "sack_rate", "pass_havoc"]].round(3).to_string(index=False))
    print(f"\n{prof.season.max()} top pass-rush (sack rate):")
    print(last.nlargest(5, "sack_rate")[
        ["school", "sack_rate", "pass_def_epa"]].round(3).to_string(index=False))

    # is opponent-adjusted pass defense a stable team trait year to year?
    p = prof.sort_values(["team_id", "season"])
    p["next"] = p.season + 1
    pair = p.merge(p, left_on=["team_id", "next"],
                   right_on=["team_id", "season"], suffixes=("", "_y"))
    for m in ("pass_def_epa", "rush_def_epa", "sack_rate", "pass_havoc"):
        print(f"y/y stability {m:18s}: r={pair[m].corr(pair[f'{m}_y']):.2f} "
              f"({len(pair)} pairs)")
    return prof


if __name__ == "__main__":
    build()
