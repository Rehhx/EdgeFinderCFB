"""Anytime-TD model: P(player scores >=1 TD).

⚠️ **DO NOT BET THIS MARKET.** The model is well calibrated out-of-sample
(predicted 0.140 vs actual 0.141; overall 0.259 vs 0.258) but it does NOT
beat the price. Controls (backtest/td_vs_book.py):
  - blend weight on the model fits at 0.05 — it adds almost nothing
  - a market-only control with NO model does as well or better
    (EV>=10%: market-only +31.7% vs model +28.9%), 95% same bets
  - at MEDIAN (consensus) prices the model finds ZERO +5% EV bets
  - the "winning" bets sit at 1.67x the median price when the typical
    cross-book gap is 1.02x -> stale lines / data mismatches, not edge
  - blind control: betting every quoted player is -16.8% ROI (heavy vig:
    35% mean implied vs 27% actual)
Kept because the calibrated probabilities are useful context (e.g. sanity-
checking rushing/receiving projections), not as a bet source.

Structure mirrors the yardage props stack but targets scoring:
  lambda = (player's share of the team's scoring touches)
           x (team's expected TDs, from the spread/total)
  P(>=1 TD) = 1 - exp(-lambda)          [Poisson]

Player share blends two trailing signals (both shifted one game, EWMA):
  td_share : share of the team's actual TDs the player scored
  rz_share : share of the team's red-zone touches (carries + targets inside
             the 20) — a higher-volume, less noisy proxy for scoring role
Red-zone share is weighted more heavily early when TD counts are tiny.

Team expected TDs come from the market: team_total = total/2 - spread/2, then
TDs ~ (team_total - expected FG points) / 7.

  python -m models.td_model
"""
import numpy as np
import pandas as pd

from ingestion.config import PARQUET_DIR, PBP_DIR

SEASONS = (2023, 2024, 2025)
HALFLIFE = 3.0
RZ_WEIGHT = 0.6          # blend weight on red-zone share vs realised TD share
FG_POINTS = 4.5          # typical points per game from FGs/XP noise
POINTS_PER_TD = 7.0

COLS = ["game_id", "season", "week", "pos_team", "rush", "pass", "completion",
        "rush_td", "pass_td", "rusher_player_name", "receiver_player_name",
        "start.yardsToEndzone"]


def player_td_games() -> pd.DataFrame:
    """Per (season, week, game, team, player): TDs scored and RZ touches."""
    out = []
    for s in SEASONS:
        df = pd.read_parquet(PBP_DIR / f"play_by_play_{s}.parquet", columns=COLS)
        for c in ("rush", "pass", "completion", "rush_td", "pass_td"):
            df[c] = df[c].fillna(False).astype(bool)
        df = df[df.pos_team.notna()].copy()
        df["pos_team"] = df.pos_team.astype(int)
        df["season"] = s
        df["rz"] = df["start.yardsToEndzone"].fillna(99) <= 20

        rush = df[df.rush & df.rusher_player_name.notna()].assign(
            player=lambda d: d.rusher_player_name)
        rec = df[df["pass"] & df.receiver_player_name.notna()].assign(
            player=lambda d: d.receiver_player_name)
        key = ["season", "week", "game_id", "pos_team", "player"]
        r = rush.groupby(key).agg(td_rush=("rush_td", "sum"),
                                  rz_touch_r=("rz", "sum")).reset_index()
        c = rec.groupby(key).agg(td_rec=("pass_td", "sum"),
                                 rz_touch_c=("rz", "sum")).reset_index()
        g = r.merge(c, on=key, how="outer").fillna(0)
        g["tds"] = g.td_rush + g.td_rec
        g["rz_touches"] = g.rz_touch_r + g.rz_touch_c
        out.append(g)
    d = pd.concat(out, ignore_index=True).rename(columns={"pos_team": "team_id"})
    team = d.groupby(["season", "week", "game_id", "team_id"], as_index=False).agg(
        team_tds=("tds", "sum"), team_rz=("rz_touches", "sum"))
    d = d.merge(team, on=["season", "week", "game_id", "team_id"])
    d["td_share"] = d.tds / d.team_tds.clip(lower=1)
    d["rz_share"] = d.rz_touches / d.team_rz.clip(lower=1)
    return d


def build_features() -> pd.DataFrame:
    """Trailing (as-of) shares plus the market's expected team TDs."""
    d = player_td_games().sort_values(["season", "team_id", "player", "week"])
    g = d.groupby(["season", "team_id", "player"])
    for c in ("td_share", "rz_share"):
        d[f"trail_{c}"] = g[c].transform(
            lambda s: s.shift(1).ewm(halflife=HALFLIFE, min_periods=1).mean())
    d["games_prior"] = g.cumcount()

    # market-implied team TDs from the closing spread/total
    from backtest.spread_baseline import consensus_lines
    lines = pd.concat([consensus_lines(s).assign(season=s) for s in SEASONS])
    from ingestion.config import CFBD_PARQUET_DIR
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    s2i = dict(zip(teams.school, teams.id))
    lines["home_id"] = lines.homeTeam.map(s2i)
    lines["away_id"] = lines.awayTeam.map(s2i)
    home = lines.rename(columns={"home_id": "team_id"})[
        ["id", "season", "team_id", "spread", "total"]].assign(is_home=True)
    away = lines.rename(columns={"away_id": "team_id"})[
        ["id", "season", "team_id", "spread", "total"]].assign(is_home=False)
    tl = pd.concat([home, away]).dropna(subset=["team_id", "spread", "total"])
    tl["team_id"] = tl.team_id.astype(int)
    # team total = total/2 -/+ spread/2 (spread is the HOME number)
    tl["team_total"] = tl.total / 2 - np.where(tl.is_home, tl.spread, -tl.spread) / 2
    tl["exp_team_tds"] = ((tl.team_total - FG_POINTS) / POINTS_PER_TD).clip(lower=0.3)

    d = d.merge(tl[["id", "season", "team_id", "exp_team_tds", "team_total"]],
                left_on=["game_id", "season", "team_id"],
                right_on=["id", "season", "team_id"], how="left")
    # blended player share -> lambda
    share = (RZ_WEIGHT * d.trail_rz_share.fillna(0)
             + (1 - RZ_WEIGHT) * d.trail_td_share.fillna(0))
    d["lam"] = (share * d.exp_team_tds).clip(lower=0.01, upper=3.0)
    d["p_raw"] = 1 - np.exp(-d.lam)
    d["scored"] = (d.tds > 0).astype(float)

    # The raw Poisson probability is badly over-dispersed (predicts 0.018 where
    # reality is 0.146, and 0.633 where reality is 0.511) because the trailing
    # share is noisy and ignores baseline scoring chance. Fit a logistic
    # recalibration  logit(p) = a + b*logit(p_raw)  on the FIRST season only
    # and apply it out-of-sample.
    d["p_td"] = _recalibrate(d)
    return d


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _recalibrate(d: pd.DataFrame) -> pd.Series:
    from sklearn.linear_model import LogisticRegression
    fit = d[(d.season == SEASONS[0]) & d.exp_team_tds.notna()
            & (d.games_prior >= 2)]
    lr = LogisticRegression()
    lr.fit(_logit(fit.p_raw).values.reshape(-1, 1), fit.scored.values)
    a, b = float(lr.intercept_[0]), float(lr.coef_[0][0])
    print(f"TD recalibration fitted on {SEASONS[0]}: logit(p) = "
          f"{a:+.2f} + {b:.2f}*logit(p_raw)")
    return pd.Series(1 / (1 + np.exp(-(a + b * _logit(d.p_raw)))),
                     index=d.index)


def main() -> None:
    d = build_features()
    dest = PARQUET_DIR / "td_projections.parquet"
    d.to_parquet(dest, index=False)
    # calibration must be judged OUT-OF-SAMPLE (fit was on SEASONS[0])
    ok = d[d.exp_team_tds.notna() & (d.games_prior >= 2)
           & (d.season != SEASONS[0])]
    print(f"td_projections: {len(d):,} player-games -> {dest.name}")
    print(f"OOS subset: {len(ok):,} | actual TD rate {ok.scored.mean():.3f} "
          f"| mean predicted {ok.p_td.mean():.3f}")
    print("\ncalibration OUT-OF-SAMPLE (predicted vs actual):")
    ok = ok.copy()
    ok["bucket"] = pd.cut(ok.p_td, [0, .05, .1, .2, .3, .5, 1])
    print(ok.groupby("bucket", observed=True).agg(
        n=("scored", "size"), pred=("p_td", "mean"),
        actual=("scored", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
