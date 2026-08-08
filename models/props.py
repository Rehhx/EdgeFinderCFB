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

# Game-script handling for usage shares. See script_factors().
#   "off"      shares used exactly as observed (the pre-2026-08 behaviour)
#   "descript" divide each PAST game's share by its realised script factor
#              before the EWMA -- removes known contamination from the input
#   "full"     also multiply the projection by E[factor | pregame spread]
# REJECTED 2026-08-04 and left at "off". The script effect is real but small
# once measured honestly: a player-fixed-effects fit puts the RB band at ~5%
# (0.950x at a 30-point win), not the ~20% a raw share/trail_share ratio
# suggests -- that ratio compares DIFFERENT players, and good teams both blow
# opponents out and run deeper committees, so most of the apparent effect is
# roster composition, not script. Head-to-head out-of-sample (backtest/
# script_shares.py): off +39.6u, full +34.5u, descript +26.1u over 2024-25.
SCRIPT_MODE = "off"
SCRIPT_SHARES = {"rush_share": "rush", "tgt_share": "tgt", "qb_share": "qb"}
SCRIPT_TRAIN_SEASON = 2022


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

    out = both.rename(columns={c: f"prior_{c}" for c in carry})

    # Vacated share: a returning player whose team-mates' usage LEFT should
    # carry a bigger prior. Departures previously just deleted the leaver's row
    # and the share evaporated. Roster-based, so it is known pregame.
    if VACATED_MODE == "on":
        try:
            vac = pd.read_parquet(PARQUET_DIR / "vacated_share.parquet")
        except FileNotFoundError:
            print("WARNING: vacated_share.parquet missing — run "
                  "features.vacated_share; priors left unadjusted.")
            return out
        coefs = fit_vacated(out, logs, vac)
        if coefs:
            print("vacated-share slopes (mean-1 multiplier, fit "
                  f"{VAC_FIT_SEASON}):",
                  {k: round(v["slope"], 3) for k, v in coefs.items()})
        out = out.merge(vac, on=["season", "team_id"], how="left")
        for tag, (pcol, _, vcol) in VAC_SPEC.items():
            if tag not in coefs or pcol not in out.columns:
                continue
            out[pcol] = out[pcol] * _vac_mult(out[vcol], coefs[tag])
        out = out.drop(columns=[c for c in vac.columns
                                if c.startswith("vac_") and c in out.columns])
    return out


def game_margins() -> pd.DataFrame:
    """(game_id, team_id) -> own-team final margin, plus the true home_id.

    Sourced from the CFBD games table, NOT from play-by-play. 4.9% of
    player-game rows have no PBP row, and home_id was previously taken from
    PBP; when it came back NaN the `team_id == home_id` test was False for
    BOTH sides, so both were handed the away-side spread. 144 games had their
    two teams recorded as the same-size underdog, which then fed the volume
    model with the favourite's sign flipped.
    """
    from ingestion.config import CFBD_PARQUET_DIR
    gs = []
    for s in SEASONS:
        try:
            g = pd.read_parquet(CFBD_PARQUET_DIR / f"games_{s}.parquet")
        except FileNotFoundError:
            continue
        gs.append(g[["id", "homeId", "awayId", "homePoints", "awayPoints"]])
    g = pd.concat(gs).dropna(subset=["homeId", "awayId"])
    pts = g.homePoints.notna() & g.awayPoints.notna()
    long = pd.concat([
        g.assign(team_id=g.homeId, is_home=1,
                 margin=np.where(pts, g.homePoints - g.awayPoints, np.nan)),
        g.assign(team_id=g.awayId, is_home=0,
                 margin=np.where(pts, g.awayPoints - g.homePoints, np.nan))])
    long = long.rename(columns={"id": "game_id", "homeId": "home_id"})
    return long[["game_id", "team_id", "home_id", "is_home", "margin"]].astype(
        {"game_id": "int64", "team_id": "int64", "home_id": "int64"})


def script_factors(df: pd.DataFrame) -> dict:
    """How a player's SHARE of his team's work responds to game script.

    A starter's share is not a constant he carries between games. It collapses
    when his team is blowing the opponent out -- he is on the bench in the
    fourth quarter -- and swells when the team is behind, because the QB throws
    all game and the lead back keeps whatever carries remain. Measured band on
    realised margin (observed share / trailing share):

        RB carries    1.154 (lost 14-25)  ->  0.944 (won 25+)
        WR/TE targets 1.141 (won 4-14)    ->  1.033 (won 25+)
        QB attempts   1.322 (lost 14-25)  ->  1.053 (won 25+)

    The trailing EWMA averages over whatever scripts a player happened to draw,
    so that contamination rides into every projection. It is also why the bias
    is worst early: September blowout wins depress the very shares that the
    EWMA then carries into the competitive games we bet.

    Fitted as a player-fixed-effects regression of log(share) on a quadratic in
    margin -- holding the player constant, so the league-wide level bias in
    shares cannot leak into the script term. Normalised to mean 1 on the train
    season, so de-scripting changes the SHAPE of a usage history, never its
    level. Fit on SCRIPT_TRAIN_SEASON only; every tested season stays
    out-of-sample.
    """
    out = {}
    tr = df[(df.season == SCRIPT_TRAIN_SEASON) & df.margin.notna()]
    for share, tag in SCRIPT_SHARES.items():
        t = tr[(tr[share] > 0) & tr[share].notna()].copy()
        t["ly"] = np.log(t[share])
        # within-player demeaning: the coefficient is identified off the same
        # player across different scripts, not off who plays in blowouts
        t["dy"] = t.ly - t.groupby(["season", "team_id", "player"]).ly.transform("mean")
        keep = t.groupby(["season", "team_id", "player"]).ly.transform("size") >= 3
        t = t[keep]
        m = t.margin.values / 10.0          # scale so the quadratic is tame
        X = np.column_stack([m, m ** 2])
        beta, *_ = np.linalg.lstsq(X, t.dy.values, rcond=None)

        f_tr = np.exp(X @ beta)
        norm = f_tr.mean()
        out[tag] = {"beta": [float(b) for b in beta], "norm": float(norm)}

        # forward factor: E[f | pregame spread]. Fitted by regressing the
        # realised factor on a quadratic in the spread, which integrates over
        # margin uncertainty automatically -- and that uncertainty is large
        # (residual sd of margin given the spread is 21.1 points), which is
        # exactly why the forward term is far weaker than the historical one.
        s = tr.dropna(subset=["team_spread"]).drop_duplicates(
            ["game_id", "team_id"])
        sm = s.margin.values / 10.0
        fs = np.exp(np.column_stack([sm, sm ** 2]) @ beta) / norm
        sp = s.team_spread.values / 10.0
        Xs = np.column_stack([sp, sp ** 2, np.ones(len(sp))])
        gam, *_ = np.linalg.lstsq(Xs, fs, rcond=None)
        out[tag]["gamma"] = [float(b) for b in gam]
    return out


def _script_f(margin, coef: dict):
    m = np.asarray(margin, float) / 10.0
    f = np.exp(coef["beta"][0] * m + coef["beta"][1] * m ** 2) / coef["norm"]
    return np.where(np.isnan(m), 1.0, f)


def _script_fwd(spread, coef: dict):
    s = np.asarray(spread, float) / 10.0
    g = coef["gamma"]
    f = g[0] * s + g[1] * s ** 2 + g[2]
    return np.where(np.isnan(s), 1.0, f)


# Vacated share: boost a returning player's season prior when his team-mates'
# usage left. Fitted, never assumed — see fit_vacated().
#   "off"  season priors used as-is
#   "on"   priors scaled by the fitted, mean-1 vacancy multiplier
#
# REJECTED 2026-08-05, left "off". backtest/vacated_ab.py, OOS 2024-25:
#     off  393 bets  +14.4% ROI  +40.5 u/szn   (2024 +42.0, 2025 +39.0)
#     on   398 bets  +10.0% ROI  +24.2 u/szn   (2024 +37.7, 2025 +10.7)
# The underlying effect IS real — a 38% monotone swing in returning players'
# usage across vacancy quartiles, roster-measured, slopes +0.63/+0.76/+0.76
# consistent across targets/carries/attempts. It just does not survive the
# pipeline. MAE is flat (rush 30.88->30.75, pass 75.54->76.75 WORSE), i.e. the
# adjustment MOVES projections without making them more accurate, and props bet
# selection is exquisitely sensitive to that (cf. backtest/props_blend_sweep.py,
# where one grid step of the market-blend weight swung the season 30-54 units).
# Early season, where it should help most, does not rescue it: pooled ROI rises
# +4.1%->+4.7% but the season split WORSENS (2024 +0.8% -> -2.7%).
# Worth revisiting when a 4th season of roster data exists; the mechanism is
# sound, the estimate is too noisy at n=272-832 to pay for itself.
VACATED_MODE = "off"
VAC_FIT_SEASON = 2023          # 2022 has no roster, so it cannot be the fit yr
VAC_SPEC = {"tgt": ("prior_tgt_share", "targets", "vac_tgt"),
            "rush": ("prior_rush_share", "rush_att", "vac_rush"),
            "qb": ("prior_qb_share", "pass_att", "vac_pass")}


def fit_vacated(pri: pd.DataFrame, logs: pd.DataFrame,
                vac: pd.DataFrame) -> dict:
    """Fit log(actual season share / prior share) ~ vacated, on ONE season.

    ⚠️ Only the SLOPE is kept, and the multiplier is normalised to mean 1 on
    the fit season. The raw intercept is large and negative (multipliers of
    0.53-0.74 at average vacancy), i.e. season priors systematically
    over-project share — but that is a LEVEL correction, not a vacancy one, and
    backtest/props_vs_book.py already refits `actual ~ a + b*proj` downstream.
    Shipping the intercept here would double-count it and then have it partly
    re-absorbed, changing far more than the effect that was actually validated.
    Same discipline as the game-script factors: change the SHAPE, not the level.

    Fitted on VAC_FIT_SEASON (2023), which is already the pricing calibration
    season, so 2024 and 2025 stay fully out-of-sample.
    """
    tot = logs.groupby(["season", "team_id"], as_index=False).agg(
        **{f"T_{k}": (c, "sum") for k, (_, c, _) in VAC_SPEC.items()})
    act = logs.groupby(["season", "team_id", "player"], as_index=False).agg(
        **{k: (c, "sum") for k, (_, c, _) in VAC_SPEC.items()})
    act = act.merge(tot, on=["season", "team_id"])
    m = pri.merge(act, on=["season", "team_id", "player"]).merge(
        vac, on=["season", "team_id"], how="inner")

    out = {}
    for tag, (pcol, _, vcol) in VAC_SPEC.items():
        m[f"a_{tag}"] = m[tag] / m[f"T_{tag}"].clip(lower=1)
        s = m[(m.season == VAC_FIT_SEASON) & (m[pcol] > 0.05)
              & (m[f"a_{tag}"] > 0)]
        if len(s) < 80:
            continue
        y = np.log(s[f"a_{tag}"] / s[pcol]).values
        X = np.column_stack([s[vcol].values, np.ones(len(s))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[tag] = {"slope": float(beta[0]), "vbar": float(s[vcol].mean())}
    return out


def _vac_mult(vac_col, coef: dict):
    """Mean-1 multiplier: exp(slope * (vacated - mean vacated))."""
    v = np.asarray(vac_col, float)
    mult = np.exp(coef["slope"] * (v - coef["vbar"]))
    return np.where(np.isnan(v), 1.0, mult)


def build_table(future: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (game, player) features. `future` appends synthetic rows for games
    that have NOT been played, so the SAME trailing-feature machinery serves
    live picks.

    This exists because `picks/prop_picks.py` used to build its own
    preseason-only projection — no trailing EWMA, no opponent factor — while
    `model_coefs.json` shipped recalibration slopes fitted on THIS table. The
    live book was pricing with a different, weaker model than the one every
    backtest measured (~+11.6 u/szn vs +56.9).

    It works because `_trail` is `shift(1).ewm(...)`: a row appended for an
    unplayed game receives the EWMA of the games BEFORE it and contributes
    nothing to its own features. In week 1 there are no prior games, so every
    trailing feature is NaN and the prior blend below falls back to the
    season-opening priors — exactly what the old opening-only path did. One
    code path is now correct in week 1 AND week 9.

    `future` must carry the logs schema with NaN stats, plus `team_spread`
    (positive = this team is the underdog) and `opp_id`.
    """
    logs = pd.read_parquet(PARQUET_DIR / "player_game_logs.parquet")
    seasons = list(SEASONS)
    if future is not None and len(future):
        seasons = sorted(set(seasons) | set(future.season.unique().tolist()))
    logs = logs[logs.season.isin(seasons)]
    log_cols = logs.columns.tolist()
    if future is not None and len(future):
        # Derived, not hard-coded: any log column that is not part of the key
        # is a stat, so adding a stat (e.g. touchdowns) cannot silently break
        # the `fut[keep]` selection below.
        key_cols = {"season", "week", "game_id", "team_id", "player"}
        stat_cols = [c for c in log_cols if c not in key_cols]
        fut = future.copy()
        for c in stat_cols:                       # unplayed: no stats, ever
            fut[c] = np.nan
        # `fut_spread`/`fut_opp` ride along so the known market number and the
        # known opponent survive the merges below; the CFBD line table has no
        # row for a game that has not been played.
        keep = log_cols + ["fut_spread", "fut_opp"]
        fut = fut.rename(columns={"team_spread": "fut_spread",
                                  "opp_id": "fut_opp"})[keep]
        # AS-OF GUARD. Projecting week W must not see week W or later. Drop
        # any logged row at or after the earliest future week in that season —
        # otherwise a re-run after kickoff would quietly grade itself on the
        # game it is pricing, and the acceptance test could not simulate a
        # past week honestly.
        for s, w in fut.groupby("season").week.min().items():
            logs = logs[~((logs.season == s) & (logs.week >= w))]
        logs = pd.concat([logs, fut], ignore_index=True)
    logs["fut_spread"] = logs.get("fut_spread", np.nan)
    logs["fut_opp"] = logs.get("fut_opp", np.nan)
    logs["_future"] = logs.fut_spread.notna()

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

    # Realised margin + closing spread are needed BEFORE the trailing EWMAs,
    # because the de-scripting has to happen on each past game's share while
    # it is still a single observation.
    df = df.merge(game_margins(), on=["game_id", "team_id"], how="left")
    obs_lines = pd.concat([consensus_lines(s).assign(season=s)
                           for s in SEASONS])[["id", "spread"]]
    df = df.merge(obs_lines, left_on="game_id", right_on="id", how="left")
    # a failed merge must yield NaN, never a silently flipped sign -- rows with
    # no team_spread are dropped in project()
    df["team_spread"] = np.where(
        df.is_home == 1, df.spread,
        np.where(df.is_home == 0, -df.spread, np.nan))
    # An unplayed game has no CFBD line row and no realised margin, so the
    # lookup above yields NaN. Use the market number the caller passed in.
    df["team_spread"] = df.team_spread.fillna(df.fut_spread)

    script = script_factors(df) if SCRIPT_MODE != "off" else {}
    if script:
        print("script factors (log-share response to margin, per 10 pts):",
              {k: [round(b, 3) for b in v["beta"]] for k, v in script.items()})

    # de-script each observed share against the game it was recorded in, so the
    # EWMA below runs over a cleaned series
    for c, tag in SCRIPT_SHARES.items():
        df[f"f_{tag}"] = _script_f(df.margin, script[tag]) if script else 1.0
        df[f"{c}_ds"] = df[c] / df[f"f_{tag}"]

    df = df.sort_values(["season", "player", "team_id", "week"])
    pg = df.groupby(["season", "team_id", "player"])
    df["prior_games"] = pg.cumcount()
    # WEEKS SINCE THIS PLAYER'S PREVIOUS APPEARANCE. Leak-free: shift(1) only
    # looks backwards. This is the availability signal the depth charts were
    # supposed to supply — `depth_charts.parquet` turned out to hold a single
    # 2026-07 snapshot with no history, and `news_extractions.parquet` has 5
    # rows, so neither can inform a 2023-25 fit without anachronism.
    #
    # A missed game leaves NO log row, so the trailing EWMA silently carries a
    # stale pre-absence share into the return game as if it were current. The
    # error is monotone in the gap and stable in every season (rush share bias
    # at gap>=3: 2023 -0.0704, 2024 -0.0411, 2025 -0.0237) — a returning player
    # is OVER-projected because his old role is not the one he comes back to.
    df["gap"] = df.week - pg.week.shift(1)
    for c in ("rush_share", "tgt_share", "qb_share", "ypc", "ypt", "catch",
              "ypa", "rush_yds", "rec_yds", "receptions", "pass_yds",
              "rush_att", "targets", "pass_att", "pass_comp"):
        src = f"{c}_ds" if f"{c}_ds" in df.columns else c
        df[f"trail_{c}"] = pg[src].transform(_trail)
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
    # Safety net: `dfn` pairs the two sides of a game, so it only resolves the
    # opponent for a future game when the caller supplied BOTH teams' rows.
    df["opp_id"] = df.opp_id.fillna(df.fut_opp)

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

    # team_spread (the game-script input, positive = this team is a dog) is
    # built near the top of this function, before the trailing EWMAs.
    return df


def fit_volume_coefs(train: pd.DataFrame) -> dict:
    """Team-volume model: actual attempts ~ trailing attempts + spread.

    One row per TEAM-game. This was previously groupby("game_id").first() over
    PLAYER rows, which kept a single arbitrary side of each game, so the fit
    depended on row order: a 3.5% change in upstream inputs reshuffled which
    teams trained the model, moved the coefficients in the third decimal, and
    swung the graded book by ~40 units/season. Half the sample, and unstable.
    """
    out = {}
    t0 = train.drop_duplicates(["game_id", "team_id"])
    for c in ("team_rush_att", "team_pass_att"):
        t = t0.dropna(subset=[f"trail_{c}", "team_spread"])
        X = np.column_stack([t[f"trail_{c}"], t.team_spread, np.ones(len(t))])
        out[c], *_ = np.linalg.lstsq(X, t[c], rcond=None)
    return out


# ⚠️ REJECTED 2026-08-06, left "off". The BIAS IS REAL and season-stable — a
# player returning from missed games is over-projected because the trailing
# EWMA carries his stale pre-absence share (rush-share error at gap>=3: 2023
# -0.0704, 2024 -0.0411, 2025 -0.0237; target share -0.0158/-0.0158/-0.0083).
# Correcting it still made the BOOK WORSE: -11.6 u/szn (+87.0 -> +75.4), with
# PROPS PRIME ROI falling 15.4% -> 10.9% on the same 120 bets/szn. Tested both
# raw and normalised so only the RELATIVE staleness penalty applied (gap1
# forced to 1.000, gap2 0.970, gap3 0.920) — identical result, so the
# double-correction was not the cause.
#
# THE LESSON: a real bias in an input does not mean correcting it improves
# betting. The recalibration and the market blend were fitted AROUND the
# uncorrected projection; changing the input shifts the whole EV surface and
# the selection with it. Judge a projection change by the BOOK, never by the
# projection's own accuracy.
GAP_MODE = "off"


def fit_gap_mult(train: pd.DataFrame) -> dict:
    """Share multiplier by weeks-since-last-appearance, fitted on `train`.

    Bucket 1 = played last week, 2 = missed one, 3 = missed two or more.
    Multiplier = mean(actual share) / mean(projected share) in that bucket, so
    it corrects the LEVEL of a stale share without touching its ranking.
    """
    out = {}
    for sh, num, team in (("rush_share", "rush_att", "team_rush_att"),
                          ("tgt_share", "targets", "team_pass_att"),
                          ("qb_share", "pass_att", "team_pass_att")):
        t = train[train[team].gt(0) & train[num].notna()
                  & train[f"trail_{sh}"].gt(0) & train.gap.notna()]
        if len(t) < 200:
            continue
        b = np.where(t.gap <= 1, 1, np.where(t.gap == 2, 2, 3))
        m = {}
        for k in (1, 2, 3):
            s = t[b == k]
            if len(s) < 60:
                m[k] = 1.0
                continue
            act = (s[num] / s[team]).mean()
            prj = s[f"trail_{sh}"].mean()
            m[k] = float(np.clip(act / prj, 0.6, 1.4)) if prj > 0 else 1.0
        # NORMALISE so the "played last week" bucket is exactly 1.0. Only the
        # RELATIVE staleness penalty is ours to apply — the overall level bias
        # is already absorbed by the recalibration slope in props_vs_book, and
        # correcting it twice measurably hurts (it cost -11.6 u/szn: PROPS
        # PRIME ROI fell 15.4% -> 10.9% before this normalisation).
        base = m.get(1, 1.0) or 1.0
        out[sh] = {k: v / base for k, v in m.items()}
    return out


def project(df: pd.DataFrame, vol_coefs: dict | None = None,
            write_coefs: bool = True,
            seasons: list[int] | None = None) -> pd.DataFrame:
    """Project every stat. `vol_coefs` overrides the fitted team-volume model
    so backtest/props_stability.py can resample it; `write_coefs=False` keeps
    an experiment from overwriting production model_coefs.json."""
    train = df[df.season == 2022]  # earliest; keeps 2023+ fully out-of-sample
    lg = {c: train[c].mean() for c in ("ypc", "ypt", "catch", "ypa")}

    # Volume models: actual att ~ trailing att + spread, fitted on the train
    # season. One row per TEAM-game. This used to be groupby("game_id").first()
    # over PLAYER rows, which kept a single arbitrary side of each game and so
    # depended on row order -- a 3.5% change in upstream inputs reshuffled which
    # teams trained the model, moved the coefficients in the third decimal, and
    # swung the graded book by ~40 units/season. Half the sample, and unstable.
    if vol_coefs is None:
        vol_coefs = fit_volume_coefs(train)
        print("volume coefs [trail, spread, const]:",
              {k: np.round(v, 3).tolist() for k, v in vol_coefs.items()})
    if write_coefs:
        from ingestion.config import save_coefs
        save_coefs("prop_volume", {
            "rush": [float(x) for x in vol_coefs["team_rush_att"]],
            "pass": [float(x) for x in vol_coefs["team_pass_att"]]})

    d = df[df.season.isin(TEST_SEASONS if seasons is None else seasons)
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

    # Forward re-scripting: the trailing shares are now script-neutral, so
    # optionally push them back toward the script this game is expected to
    # produce. Deliberately separable from de-scripting -- the historical term
    # uses a margin we KNOW, this one uses a spread that leaves a 21.1-point
    # residual sd on margin, and it is the half that risks merely agreeing
    # with the book more (cf. the opponent-adjusted defense EPA rejection).
    if SCRIPT_MODE == "full":
        script = script_factors(df)
        rush_f = _script_fwd(d.team_spread, script["rush"])
        tgt_f = _script_fwd(d.team_spread, script["tgt"])
        qb_f = _script_fwd(d.team_spread, script["qb"])
    else:
        rush_f = tgt_f = qb_f = 1.0
    # RETURN-FROM-ABSENCE adjustment, fitted on the TRAIN season only.
    # A player back after missed games carries a stale share (see the `gap`
    # note in build_table). The multiplier is mean(actual share)/mean(projected
    # share) per gap bucket on `train`, so it is never fitted on the OOS years.
    gap_m = fit_gap_mult(train) if GAP_MODE == "on" else {}
    def _gm(col_share, col_num, col_team):
        if not gap_m:
            return 1.0
        b = np.where(d.gap.isna(), 1, np.where(d.gap <= 1, 1,
                     np.where(d.gap == 2, 2, 3)))
        return pd.Series(b, index=d.index).map(
            gap_m.get(col_share, {})).fillna(1.0).values

    d["share_rush"] = d.trail_rush_share * rush_f * _gm(
        "rush_share", "rush_att", "team_rush_att")
    d["share_tgt"] = d.trail_tgt_share * tgt_f * _gm(
        "tgt_share", "targets", "team_pass_att")
    d["share_qb"] = d.trail_qb_share * qb_f * _gm(
        "qb_share", "pass_att", "team_pass_att")

    d["proj_rush_yds"] = (d.share_rush * d.exp_team_rush_att
                          * blend(d.trail_ypc, d.cum_rush_att, lg["ypc"], 40)
                          * opp_rush.fillna(1))
    d["proj_targets"] = d.share_tgt * d.exp_team_pass_att
    d["proj_receptions"] = d.proj_targets * blend(
        d.trail_catch, d.cum_targets, lg["catch"], 15)
    d["proj_rec_yds"] = (d.proj_targets
                         * blend(d.trail_ypt, d.cum_targets, lg["ypt"], 25)
                         * opp_pass.fillna(1))
    d["proj_pass_yds"] = (d.share_qb * d.exp_team_pass_att
                          * blend(d.trail_ypa, d.cum_pass_att, lg["ypa"], 80)
                          * opp_pass.fillna(1))

    # VOLUME projections. Yards = attempts x efficiency, and efficiency is the
    # noisy factor we bolt on — so we predict ATTEMPTS strictly better than the
    # yards derived from them (2024-25: rush att corr 0.541 vs 0.448 for rush
    # yards; pass att 0.454 vs 0.358). These are the numbers behind every
    # yardage projection above; they were simply never exposed as columns.
    #
    # Traditional US books rarely post NCAAF attempts/completions and The Odds
    # API's historical archive has none, so these CANNOT be backtested the way
    # the yardage markets were. DFS pick'em books do post them (typically near
    # the median at ~even money, e.g. "23.5 completions"). Treat as PAPER until
    # forward-graded — `player_game_logs` now carries the actuals to grade with.
    d["proj_rush_att"] = d.share_rush * d.exp_team_rush_att
    d["proj_pass_att"] = d.share_qb * d.exp_team_pass_att
    comp_rate = (d.trail_pass_comp / d.trail_pass_att.replace(0, np.nan))
    lg_comp = float((train.pass_comp.sum() / train.pass_att.sum())
                    if train.pass_att.sum() else 0.62)
    d["proj_pass_comp"] = d.proj_pass_att * blend(
        comp_rate, d.cum_pass_att, lg_comp, 80).clip(0.30, 0.85)
    return d


def project_upcoming(games: pd.DataFrame,
                     vol_coefs: dict | None = None) -> pd.DataFrame:
    """Project every stat for games that have NOT been played yet.

    THE live entry point. It runs the identical pipeline the backtests
    measure — `build_table` for trailing form and the opponent factor, then
    `project` for the volume model and the yardage blends — so a backtested
    ROI is a claim about the model that actually prices the board.

    `games`: one row per TEAM (so two per game) with
        season, week, game_id, team_id, opp_id, team_spread
    where `team_spread` is positive when THIS team is the underdog.

    The roster for each team is the union of players who have logged a game
    for it this season and players carrying a season-opening prior. In week 1
    only the second set exists, which is precisely the old behaviour.
    """
    logs = pd.read_parquet(PARQUET_DIR / "player_game_logs.parquet")
    priors = player_priors(logs)

    rows = []
    for g in games.itertuples():
        played = logs[(logs.season == g.season) & (logs.team_id == g.team_id)]
        pri = priors[(priors.season == g.season) & (priors.team_id == g.team_id)]
        names = sorted(set(played.player.unique()) | set(pri.player.unique()))
        for p in names:
            rows.append({
                "season": int(g.season), "week": float(g.week),
                "game_id": float(g.game_id), "team_id": int(g.team_id),
                "player": p, "team_spread": float(g.team_spread),
                "opp_id": float(g.opp_id),
            })
    if not rows:
        return pd.DataFrame()

    future = pd.DataFrame(rows)
    table = build_table(future=future)
    live = int(games.season.iloc[0])
    out = project(table, vol_coefs=vol_coefs, write_coefs=False,
                  seasons=sorted(set(TEST_SEASONS) | {live}))
    return out[out._future].copy()


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
