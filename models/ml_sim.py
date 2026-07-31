"""ML sim experiment: gradient-boosted margin/total vs the linear game_sim.

Discipline (per the plan): ML only earns its place if it beats the linear
baseline in a walk-forward backtest on identical games. Features per game:
  EPA off/def ratings (as-of), coach pace/PROE priors, and the NEW
  opponent-adjusted facet-split DEFENSE profiles (pass/rush def EPA, sack
  rate, pass havoc) for both teams — all computed strictly as-of each week.

Targets: home margin and game total. Model: HistGradientBoosting (sklearn).
Compared to models/game_sim.py's saved linear predictions on the same games.

  python -m models.ml_sim
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features.defense_profiles import _load as load_def_pbp, defense_profiles
from features.epa_ratings import SEASON_STRIDE, fit_ratings, load_game_obs
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR
from models.game_sim import coach_priors, game_table

PBP_SEASONS = [2021, 2022, 2023, 2024, 2025]
FEATURE_SEASONS = [2023, 2024, 2025]   # games we build features for
TEST_SEASONS = [2024, 2025]
WEEKS = range(1, 16)
DEF_COLS = ["pass_def_epa", "rush_def_epa", "sack_rate", "pass_havoc",
            "expl_pass_allowed"]


def _team_ratings(model, tid, quantile_off, quantile_def):
    if tid in model.ratings.index:
        r = model.ratings.loc[tid]
        return r.off, r["def"]
    return quantile_off, quantile_def


def build_feature_table() -> pd.DataFrame:
    obs = load_game_obs(PBP_SEASONS)
    games = game_table(obs)
    games = games[games.season.isin(FEATURE_SEASONS) & (games.season_type == 2)]

    lines = []
    for s in TEST_SEASONS + [2023]:
        try:
            from backtest.spread_baseline import consensus_lines
            lines.append(consensus_lines(s).assign(season=s))
        except FileNotFoundError:
            pass
    lines = pd.concat(lines)[["id", "spread", "total", "neutralSite"]]
    games = games.merge(lines, left_on="game_id", right_on="id", how="left")

    # official finals (gotcha #1: never grade on PBP garbage-time scores)
    finals = pd.concat([
        pd.read_parquet(CFBD_PARQUET_DIR / f"games_{s}.parquet")[
            ["id", "homePoints", "awayPoints"]] for s in FEATURE_SEASONS])
    games = games.merge(finals, on="id", how="left")
    games["margin"] = games.homePoints - games.awayPoints

    cpri = coach_priors()
    pace_med, proe_med = cpri.pace_prior.median(), cpri.proe_prior.median()
    def_pbp = load_def_pbp(PBP_SEASONS)

    rows = []
    for season in FEATURE_SEASONS:
        for week in WEEKS:
            asof = season * SEASON_STRIDE + week
            wk = games[(games.season == season) & (games.week == week)]
            if wk.empty:
                continue
            rmodel = fit_ratings(obs, asof_week_idx=asof)
            q_off = rmodel.ratings.off.quantile(0.1)
            q_def = rmodel.ratings["def"].quantile(0.1)
            dprof = defense_profiles(PBP_SEASONS, asof_week_idx=asof, df=def_pbp)
            dmean = dprof[DEF_COLS].mean()  # league baseline (all seasons)
            dmap = dprof[dprof.season == season].set_index("team_id")

            def dfeat(tid):
                if tid in dmap.index:
                    return dmap.loc[tid, DEF_COLS]
                return dmean

            for g in wk.itertuples():
                if (pd.isna(g.total) or pd.isna(g.spread)
                        or pd.isna(g.margin) or pd.isna(g.homePoints)):
                    continue
                ho, hd = _team_ratings(rmodel, g.home_id, q_off, q_def)
                ao, ad = _team_ratings(rmodel, g.away_id, q_off, q_def)
                hp = cpri.pace_prior.get((g.home_id, season), pace_med)
                ap = cpri.pace_prior.get((g.away_id, season), pace_med)
                hpr = cpri.proe_prior.get((g.home_id, season), proe_med)
                apr = cpri.proe_prior.get((g.away_id, season), proe_med)
                hdf, adf = dfeat(g.home_id), dfeat(g.away_id)
                rows.append({
                    "game_id": g.game_id, "season": season, "week": week,
                    "week_idx": asof, "neutral": float(bool(g.neutralSite)),
                    "home_off": ho, "home_def": hd, "away_off": ao,
                    "away_def": ad, "home_pace": hp, "away_pace": ap,
                    "home_proe": hpr, "away_proe": apr,
                    # each team's OWN defense faces the OTHER team's offense
                    **{f"home_{c}": hdf[c] for c in DEF_COLS},
                    **{f"away_{c}": adf[c] for c in DEF_COLS},
                    "margin": g.margin, "total": g.homePoints + g.awayPoints,
                    "book_spread": g.spread, "book_total": g.total,
                })
    return pd.DataFrame(rows)


FEATURES = (["neutral", "home_off", "home_def", "away_off", "away_def",
             "home_pace", "away_pace", "home_proe", "away_proe"]
            + [f"home_{c}" for c in DEF_COLS] + [f"away_{c}" for c in DEF_COLS])


def walk_forward(feat: pd.DataFrame, features: list[str] = FEATURES
                 ) -> pd.DataFrame:
    preds = []
    for season in TEST_SEASONS:
        for week in WEEKS:
            asof = season * SEASON_STRIDE + week
            train = feat[feat.week_idx < asof]
            test = feat[(feat.season == season) & (feat.week == week)]
            if len(train) < 300 or test.empty:
                continue
            Xtr = train[features].values
            out = test[["game_id", "season", "week", "margin", "total",
                        "book_spread", "book_total"]].copy()
            for tgt in ("margin", "total"):
                m = HistGradientBoostingRegressor(
                    max_depth=3, learning_rate=0.05, max_iter=300,
                    l2_regularization=1.0, min_samples_leaf=30,
                    random_state=0)
                m.fit(Xtr, train[tgt].values)
                out[f"ml_{tgt}"] = m.predict(test[features].values)
            preds.append(out)
    return pd.concat(preds, ignore_index=True)


CORE_FEATURES = ["neutral", "home_off", "home_def", "away_off", "away_def",
                 "home_pace", "away_pace", "home_proe", "away_proe"]


def report(ml: pd.DataFrame) -> None:
    lin = pd.read_parquet(PARQUET_DIR / "backtest_totals_sim.parquet")[
        ["game_id", "pred_total", "pred_margin"]]
    d = ml.merge(lin, on="game_id", how="inner")
    print(f"\nhead-to-head on {len(d):,} identical games (2024-25):\n")
    print(f"{'metric':16s} {'ML':>9} {'linear':>9}")
    print(f"{'total MAE':16s} {(d.ml_total-d.total).abs().mean():>9.2f} "
          f"{(d.pred_total-d.total).abs().mean():>9.2f}")
    print(f"{'margin MAE':16s} {(d.ml_margin-d.margin).abs().mean():>9.2f} "
          f"{(d.pred_margin-d.margin).abs().mean():>9.2f}")

    # OU betting at edge>=5 (the linear sim's validated threshold)
    print("\nO/U betting (edge>=5 pts, flat -110, break-even 52.4%):")
    for name, col in [("ML", "ml_total"), ("linear", "pred_total")]:
        edge = d[col] - d.book_total
        bets = d[edge.abs() >= 5]
        won = np.where(bets[col] > bets.book_total,
                       bets.total > bets.book_total, bets.total < bets.book_total)
        graded = bets.total != bets.book_total
        wr = won[graded].mean()
        roi = (wr * 100 / 110 - (1 - wr)) * 100
        print(f"  {name:8s}: {graded.sum():4d} bets, {wr:.1%} win, {roi:+.1f}% ROI")

    # ATS at edge>=4
    print("\nATS betting (edge>=4 pts vs number):")
    for name, col in [("ML", "ml_margin"), ("linear", "pred_margin")]:
        edge = d[col] - (-d.book_spread)
        bets = d[edge.abs() >= 4]
        cover = np.sign(bets.margin + bets.book_spread)
        won = (np.sign(edge[edge.abs() >= 4]) == cover) & (cover != 0)
        graded = cover != 0
        wr = won[graded].mean()
        roi = (wr * 100 / 110 - (1 - wr)) * 100
        print(f"  {name:8s}: {int(graded.sum()):4d} bets, {wr:.1%} win, {roi:+.1f}% ROI")

    # which features the ML leaned on (permutation-free: refit on all, gain)
    print("\n(feature list:", ", ".join(FEATURES[:6]), "... +defense)")


if __name__ == "__main__":
    feat = build_feature_table()
    dest = PARQUET_DIR / "ml_sim_features.parquet"
    feat.to_parquet(dest, index=False)
    print(f"feature table: {len(feat):,} games -> {dest.name}")
    ml = walk_forward(feat)
    ml.to_parquet(PARQUET_DIR / "backtest_ml_sim.parquet", index=False)
    report(ml)
