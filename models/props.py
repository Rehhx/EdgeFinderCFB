"""Player prop projections: usage share x team volume x efficiency.

For every player-game in the test window (2024-25, weeks 4-15) project
pregame: rush yards, receptions, receiving yards, pass yards.
All inputs are strictly pregame: trailing EWMAs are shifted one game,
team volume is adjusted by the closing spread (public pregame info),
opponent defense factors come from trailing allowed-efficiency.

Validation: no free historical prop lines exist, so the test is
(a) does the stack beat a naive trailing-average of the stat, and
(b) are the residual distributions calibrated enough to price overs/unders.

  python -m models.props
"""
import numpy as np
import pandas as pd

from backtest.spread_baseline import consensus_lines
from features.epa_ratings import load_game_obs
from ingestion.config import PARQUET_DIR

SEASONS = [2022, 2023, 2024, 2025]
TEST_SEASONS = [2023, 2024, 2025]  # prop lines exist from May 2023
HALFLIFE = 2.5
MIN_PRIOR_GAMES = 3
PRIOR_W_RETURN = 3.0   # pseudo-games for a returning player's last-season line
PRIOR_W_TRANSFER = 2.0  # portal arrival: old-school usage, discounted
TRANSFER_SHARE_DISCOUNT = 0.8
TEAM_PRIOR_W = 3.0

STATS = {
    #  stat        share col     team vol        eff (per unit)     shrink k
    "rush_yds": ("rush_share", "team_rush_att", "ypc", 40),
    "rec_yds":  ("tgt_share",  "team_pass_att", "ypt", 25),
    "receptions": ("tgt_share", "team_pass_att", "catch", 15),
    "pass_yds": ("qb_share",   "team_pass_att", "ypa", 80),
}
THRESH = {"rush_yds": 40, "rec_yds": 30, "receptions": 2.5, "pass_yds": 150}


def _trail(g: pd.Series) -> pd.Series:
    return g.shift(1).ewm(halflife=HALFLIFE, min_periods=1).mean()


PRIOR_COLS = ["rush_share", "tgt_share", "qb_share",
              "ypc", "ypt", "catch", "ypa"]


def player_priors(logs: pd.DataFrame) -> pd.DataFrame:
    """Season-opening priors per (season, team_id, player).

    Returning players: last season's shares/efficiency at the same team.
    Portal transfers: shares from the ORIGIN school (discounted), efficiency
    travels with the player. Weight = pseudo-games seeding the EWMA.
    """
    from ingestion.config import CFBD_PARQUET_DIR
    key = ["season", "week", "game_id", "team_id"]
    team = logs.groupby(key, as_index=False).agg(
        team_rush_att=("rush_att", "sum"), team_pass_att=("pass_att", "sum"))
    d = logs.merge(team, on=key)

    agg = d.groupby(["season", "team_id", "player"]).agg(
        rush_att=("rush_att", "sum"), rush_yds=("rush_yds", "sum"),
        targets=("targets", "sum"), receptions=("receptions", "sum"),
        rec_yds=("rec_yds", "sum"), pass_att=("pass_att", "sum"),
        pass_yds=("pass_yds", "sum"), games=("game_id", "nunique"),
        t_rush=("team_rush_att", "sum"), t_pass=("team_pass_att", "sum"),
    ).reset_index()
    agg["rush_share"] = agg.rush_att / agg.t_rush.clip(lower=1)
    agg["tgt_share"] = agg.targets / agg.t_pass.clip(lower=1)
    agg["qb_share"] = agg.pass_att / agg.t_pass.clip(lower=1)
    agg["ypc"] = agg.rush_yds / agg.rush_att.replace(0, np.nan)
    agg["ypt"] = agg.rec_yds / agg.targets.replace(0, np.nan)
    agg["catch"] = agg.receptions / agg.targets.replace(0, np.nan)
    agg["ypa"] = agg.pass_yds / agg.pass_att.replace(0, np.nan)
    agg["rush_att_pg"] = agg.rush_att / agg.games
    agg["targets_pg"] = agg.targets / agg.games
    agg["pass_att_pg"] = agg.pass_att / agg.games

    carry = ["rush_att_pg", "targets_pg", "pass_att_pg"] + PRIOR_COLS

    # returning players: same team, next season
    ret = agg.copy()
    ret["season"] += 1
    ret = ret[["season", "team_id", "player"] + carry]
    ret["prior_w"] = np.where(ret[["rush_att_pg", "targets_pg",
                                   "pass_att_pg"]].sum(axis=1) > 1,
                              PRIOR_W_RETURN, 0.0)

    # transfers: portal destination inherits origin-school profile
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    school_to_id = dict(zip(teams.school, teams.id))
    moves = []
    seasons = sorted(logs.season.unique())
    # +1: the upcoming season's portal class (e.g. 2026 arrivals from 2025 logs)
    for season in seasons[1:] + [int(seasons[-1]) + 1]:
        try:
            p = pd.read_parquet(CFBD_PARQUET_DIR / f"portal_{season}.parquet")
        except FileNotFoundError:
            continue
        p = p.dropna(subset=["destination"])
        p["player"] = p.firstName.str.strip() + " " + p.lastName.str.strip()
        p["from_id"] = p.origin.map(school_to_id)
        p["team_id"] = p.destination.map(school_to_id)
        p = p.dropna(subset=["from_id", "team_id"]).astype(
            {"from_id": int, "team_id": int})
        prev = agg[agg.season == season - 1].rename(
            columns={"team_id": "from_id"})
        t = p[["player", "from_id", "team_id"]].merge(
            prev[["from_id", "player"] + carry], on=["player", "from_id"])
        t["season"] = season
        # tier-translation multipliers fitted by features/transfer_elo.py
        # (e.g. G5->P4 share 0.77, eff 0.87; P4->G5 share 1.0, eff 1.04)
        from features.transfer_elo import tier_map
        from ingestion.config import load_coefs
        tt = load_coefs("transfer_translation", {})
        tiers = tier_map()
        move = (t.from_id.map(tiers).fillna("G5") + "->"
                + t.team_id.map(tiers).fillna("G5"))
        share_m = move.map(lambda mv: tt.get(mv, {}).get(
            "share", TRANSFER_SHARE_DISCOUNT)).values
        eff_m = move.map(lambda mv: tt.get(mv, {}).get("eff", 1.0)).values
        for c in ("rush_share", "tgt_share", "qb_share"):
            t[c] *= share_m
        for c in ("ypc", "ypt", "ypa"):
            t[c] *= eff_m
        t["prior_w"] = PRIOR_W_TRANSFER
        moves.append(t[["season", "team_id", "player", "prior_w", "from_id"]
                       + carry])

    both = pd.concat([ret] + moves, ignore_index=True)
    both = both[both.prior_w > 0]
    if "from_id" in both.columns:
        # a portal player must not keep a stale "returning" row at the
        # school he left
        left = both[both.from_id.notna()][["season", "player", "from_id"]] \
            .rename(columns={"from_id": "team_id"}).assign(_gone=True)
        both = both.merge(left, on=["season", "player", "team_id"], how="left")
        both = both[both._gone.isna() | (both.prior_w == PRIOR_W_TRANSFER)]
        both = both.drop(columns=["from_id", "_gone"])
    # transfer row wins over a stale returning row for the same player
    both = both.sort_values("prior_w").drop_duplicates(
        ["season", "team_id", "player"], keep="first")

    # star-RB bump (fitted in features/transfer_elo.py): proven lead backs
    # resist share regression — add the residual extra share
    from ingestion.config import load_coefs
    star = load_coefs("star_rb")
    if star:
        ypg = (both.ypc.fillna(0) * both.rush_att_pg.fillna(0))
        is_star = (both.rush_att_pg >= 4) & (ypg >= star["ypg_thresh"])
        both.loc[is_star, "rush_share"] = (
            both.loc[is_star, "rush_share"] + star["bump"]).clip(upper=0.8)

    return both.rename(columns={c: f"prior_{c}" for c in carry})


def build_table() -> pd.DataFrame:
    logs = pd.read_parquet(PARQUET_DIR / "player_game_logs.parquet")
    logs = logs[logs.season.isin(SEASONS)]

    key = ["season", "week", "game_id", "team_id"]
    team = logs.groupby(key, as_index=False).agg(
        team_rush_att=("rush_att", "sum"), team_rush_yds=("rush_yds", "sum"),
        team_pass_att=("pass_att", "sum"), team_pass_yds=("pass_yds", "sum"))

    # opponent per (game, team) + defensive allowed-efficiency
    pair = team.merge(team, on=["season", "week", "game_id"],
                      suffixes=("", "_opp"))
    pair = pair[pair.team_id != pair.team_id_opp]
    pair["ypc_allowed"] = pair.team_rush_yds_opp / pair.team_rush_att_opp
    pair["ypa_allowed"] = pair.team_pass_yds_opp / pair.team_pass_att_opp
    dfn = pair[key + ["ypc_allowed", "ypa_allowed", "team_id_opp"]].copy()
    dfn = dfn.sort_values(["season", "team_id", "week"])
    for c in ("ypc_allowed", "ypa_allowed"):
        dfn[f"trail_{c}"] = dfn.groupby(["season", "team_id"])[c].transform(_trail)

    # team trailing volumes, seeded with last season's per-game averages
    team = team.sort_values(["season", "team_id", "week"])
    team["team_game_no"] = team.groupby(["season", "team_id"]).cumcount()
    prev_team = team.groupby(["season", "team_id"], as_index=False)[
        ["team_rush_att", "team_pass_att"]].mean()
    prev_team["season"] += 1
    prev_team = prev_team.rename(columns={
        "team_rush_att": "pr_team_rush_att", "team_pass_att": "pr_team_pass_att"})
    team = team.merge(prev_team, on=["season", "team_id"], how="left")
    for c in ("team_rush_att", "team_pass_att"):
        t = team.groupby(["season", "team_id"])[c].transform(_trail)
        p, g = team[f"pr_{c}"], team.team_game_no
        team[f"trail_{c}"] = np.where(
            p.notna(), (t.fillna(0) * g + p * TEAM_PRIOR_W) / (g + TEAM_PRIOR_W),
            t)

    df = logs.merge(team, on=key)
    df["rush_share"] = df.rush_att / df.team_rush_att.clip(lower=1)
    df["tgt_share"] = df.targets / df.team_pass_att.clip(lower=1)
    df["qb_share"] = df.pass_att / df.team_pass_att.clip(lower=1)
    df["ypc"] = df.rush_yds / df.rush_att.replace(0, np.nan)
    df["ypt"] = df.rec_yds / df.targets.replace(0, np.nan)
    df["catch"] = df.receptions / df.targets.replace(0, np.nan)
    df["ypa"] = df.pass_yds / df.pass_att.replace(0, np.nan)

    df = df.sort_values(["season", "player", "team_id", "week"])
    pg = df.groupby(["season", "team_id", "player"])
    df["prior_games"] = pg.cumcount()
    for c in ("rush_share", "tgt_share", "qb_share", "ypc", "ypt", "catch",
              "ypa", "rush_yds", "rec_yds", "receptions", "pass_yds",
              "rush_att", "targets", "pass_att"):
        df[f"trail_{c}"] = pg[c].transform(_trail)
    df["cum_rush_att"] = pg.rush_att.transform(lambda s: s.shift(1).cumsum())
    df["cum_targets"] = pg.targets.transform(lambda s: s.shift(1).cumsum())
    df["cum_pass_att"] = pg.pass_att.transform(lambda s: s.shift(1).cumsum())

    # blend in season-opening player priors (returning + portal transfers):
    # prior acts as prior_w pseudo-games seeding each trailing feature
    df = df.merge(player_priors(logs), on=["season", "team_id", "player"],
                  how="left")
    df["prior_w"] = df.prior_w.fillna(0.0)
    g, w = df.prior_games, df.prior_w
    for c in PRIOR_COLS:
        t, p = df[f"trail_{c}"], df[f"prior_{c}"]
        blended = (t.fillna(0) * g + p.fillna(0) * np.where(p.notna(), w, 0)) \
            / (g + np.where(p.notna(), w, 0))
        df[f"trail_{c}"] = np.where(p.notna() | t.notna(), blended, np.nan)
    df["cum_rush_att"] = df.cum_rush_att.fillna(0) + w * df.prior_rush_att_pg.fillna(0)
    df["cum_targets"] = df.cum_targets.fillna(0) + w * df.prior_targets_pg.fillna(0)
    df["cum_pass_att"] = df.cum_pass_att.fillna(0) + w * df.prior_pass_att_pg.fillna(0)

    # opponent defense trailing factors joined by (game, own team)
    df = df.merge(
        dfn.rename(columns={"team_id_opp": "opp_id"})[
            key + ["opp_id", "trail_ypc_allowed", "trail_ypa_allowed"]],
        on=key, how="left")

    # opponent-adjusted, walk-forward pass/rush defense (features/defense_
    # profiles.py). Residual test showed raw allowed under-adjusts for D
    # quality (resid corr +0.10..+0.20). Joined by opponent + (season, week).
    try:
        dadj = pd.read_parquet(PARQUET_DIR / "defense_asof.parquet")[
            ["season", "week", "team_id", "pass_def_epa", "rush_def_epa"]
        ].rename(columns={"team_id": "opp_id",
                          "pass_def_epa": "opp_pass_def_epa",
                          "rush_def_epa": "opp_rush_def_epa"})
        df = df.merge(dadj, on=["season", "week", "opp_id"], how="left")
    except FileNotFoundError:
        df["opp_pass_def_epa"] = np.nan
        df["opp_rush_def_epa"] = np.nan

    # offense-vs-defense matchup quality (pass/rush splits from CFBD advanced
    # stats, walk-forward via endWeek). Validated: adds +4.4% ROI at EV>=3%
    # out-of-sample, positive in both 2024 and 2025 (features/matchups.py).
    try:
        from features.matchups import matchup_features
        for stat in ("rush_yds", "rec_yds"):
            mf = matchup_features(stat)[
                ["season", "week", "team_id", "opp_id", "mu_total"]]
            tag = "rush" if stat == "rush_yds" else "pass"
            df = df.merge(mf.rename(columns={"mu_total": f"mu_{tag}"}),
                          on=["season", "week", "team_id", "opp_id"], how="left")
    except FileNotFoundError:
        df["mu_rush"], df["mu_pass"] = np.nan, np.nan

    # per-player trailing PPA (CFBD /ppa/players/games) — a per-game value
    # metric, the closest free analogue to a PFF grade. Validated: +6.0% ->
    # +6.9% ROI, better in both OOS seasons. NOTE situational usage
    # (/player/usage third-down & passing-down lean) was tested here too and
    # HURT (+3.5%), so it is deliberately NOT used.
    try:
        from features.player_advanced import ppa_trailing
        df = df.merge(ppa_trailing()[["season", "week", "team_id", "player",
                                      "trail_ppa_all"]],
                      on=["season", "week", "team_id", "player"], how="left")
    except FileNotFoundError:
        df["trail_ppa_all"] = np.nan

    # closing spread as game-script input (positive = this team is a dog)
    obs = load_game_obs(SEASONS)
    home = obs.groupby("game_id", as_index=False).home_id.first()
    lines = pd.concat([consensus_lines(s).assign(season=s)
                       for s in SEASONS])[["id", "spread"]]
    df = df.merge(home, on="game_id", how="left").merge(
        lines, left_on="game_id", right_on="id", how="left")
    df["team_spread"] = np.where(df.team_id == df.home_id, df.spread, -df.spread)
    return df


def project(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df.season == 2022]  # earliest; keeps 2023+ fully out-of-sample
    lg = {c: train[c].mean() for c in ("ypc", "ypt", "catch", "ypa")}

    # volume models fitted on 2023: actual att ~ trailing att + spread
    vol_coefs = {}
    for c in ("team_rush_att", "team_pass_att"):
        t = train.groupby("game_id").first()
        t = t.dropna(subset=[f"trail_{c}", "team_spread"])
        X = np.column_stack([t[f"trail_{c}"], t.team_spread, np.ones(len(t))])
        vol_coefs[c], *_ = np.linalg.lstsq(X, t[c], rcond=None)
    print("volume coefs [trail, spread, const]:",
          {k: v.round(2).tolist() for k, v in vol_coefs.items()})
    from ingestion.config import save_coefs
    save_coefs("prop_volume", {
        "rush": [float(x) for x in vol_coefs["team_rush_att"]],
        "pass": [float(x) for x in vol_coefs["team_pass_att"]]})

    d = df[df.season.isin(TEST_SEASONS)
           & df.week.between(1, 15)
           & ((df.prior_games >= MIN_PRIOR_GAMES) | (df.prior_w > 0))
           & df.team_spread.notna()].copy()

    for c in ("team_rush_att", "team_pass_att"):
        b = vol_coefs[c]
        d[f"exp_{c}"] = b[0] * d[f"trail_{c}"] + b[1] * d.team_spread + b[2]

    def blend(trail, n, league, k):
        n = n.fillna(0)
        return (trail.fillna(league) * n + league * k) / (n + k)

    # Opponent factor from raw trailing allowed-yards (sqrt-damped).
    # NOTE: opponent-ADJUSTED defense EPA (features/defense_profiles.py) was
    # tested here and HURT the validated betting edge (EV>=5 +5.2% -> -5.3%):
    # the market already prices unit defense, so a more defense-aware
    # projection just agrees with the book more. Kept raw; opp-adj columns
    # remain in the table for research but are not used in the price.
    opp_rush = (d.trail_ypc_allowed / lg["ypc"]).clip(0.8, 1.25) ** 0.5
    opp_pass = (d.trail_ypa_allowed / lg["ypa"]).clip(0.8, 1.25) ** 0.5

    d["proj_rush_yds"] = (d.trail_rush_share * d.exp_team_rush_att
                          * blend(d.trail_ypc, d.cum_rush_att, lg["ypc"], 40)
                          * opp_rush.fillna(1))
    d["proj_targets"] = d.trail_tgt_share * d.exp_team_pass_att
    d["proj_receptions"] = d.proj_targets * blend(
        d.trail_catch, d.cum_targets, lg["catch"], 15)
    d["proj_rec_yds"] = (d.proj_targets
                         * blend(d.trail_ypt, d.cum_targets, lg["ypt"], 25)
                         * opp_pass.fillna(1))
    d["proj_pass_yds"] = (d.trail_qb_share * d.exp_team_pass_att
                          * blend(d.trail_ypa, d.cum_pass_att, lg["ypa"], 80)
                          * opp_pass.fillna(1))
    return d


def report(d: pd.DataFrame) -> None:
    print(f"\nprojection rows: {len(d):,} (2024-25, weeks 1-15; "
          f"{(d.week <= 3).sum():,} in weeks 1-3 via priors)")
    print(f"{'stat':12s} {'n':>6} {'stack MAE':>10} {'naive MAE':>10} "
          f"{'corr':>6} {'resid sd':>9}")
    for stat in STATS:
        proj, naive = d[f"proj_{stat}"], d[f"trail_{stat}"]
        mask = proj >= THRESH[stat]
        s = d[mask]
        resid = s[stat] - s[f"proj_{stat}"]
        print(f"{stat:12s} {mask.sum():>6} "
              f"{(s[stat] - s[f'proj_{stat}']).abs().mean():>10.2f} "
              f"{(s[stat] - s[f'trail_{stat}']).abs().mean():>10.2f} "
              f"{s[stat].corr(s[f'proj_{stat}']):>6.3f} {resid.std():>9.1f}")

    # calibration: fit sigma on 2024, check central-80% coverage on 2025
    print("\ncalibration (sigma from 2024 residuals, coverage on 2025):")
    for stat in STATS:
        s24 = d[(d.season == 2024) & (d[f"proj_{stat}"] >= THRESH[stat])]
        s25 = d[(d.season == 2025) & (d[f"proj_{stat}"] >= THRESH[stat])]
        if len(s24) < 50 or len(s25) < 50:
            continue
        sigma = (s24[stat] - s24[f"proj_{stat}"]).std()
        z = (s25[stat] - s25[f"proj_{stat}"]) / sigma
        print(f"  {stat:12s} sigma={sigma:5.1f}  "
              f"within +-1.28sd: {(z.abs() <= 1.28).mean():.0%} (target 80%)  "
              f"median bias: {z.median():+.2f}sd")

    # linear recalibration (fit actual ~ a + b*proj on 2024, apply to 2025)
    print("\n2025 MAE after 2024-fitted recalibration:")
    for stat in STATS:
        s24 = d[(d.season == 2024) & (d[f"proj_{stat}"] >= THRESH[stat])]
        s25 = d[(d.season == 2025) & (d[f"proj_{stat}"] >= THRESH[stat])]
        if len(s24) < 50 or len(s25) < 50:
            continue
        X = np.column_stack([s24[f"proj_{stat}"], np.ones(len(s24))])
        (b, a), *_ = np.linalg.lstsq(X, s24[stat], rcond=None)
        cal = a + b * s25[f"proj_{stat}"]
        print(f"  {stat:12s} recal (b={b:.2f}, a={a:+.1f}): "
              f"MAE {(s25[stat] - cal).abs().mean():.2f}  vs naive "
              f"{(s25[stat] - s25[f'trail_{stat}']).abs().mean():.2f}  "
              f"raw {(s25[stat] - s25[f'proj_{stat}']).abs().mean():.2f}")


if __name__ == "__main__":
    table = build_table()
    out = project(table)
    dest = PARQUET_DIR / "prop_projections.parquet"
    out.to_parquet(dest, index=False)
    report(out)
    print(f"\nsaved: {dest}")
