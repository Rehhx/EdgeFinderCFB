"""Walk-forward baseline: EPA ratings vs closing spreads, 2024-2025.

For each week W: fit ratings on everything before W, predict home margin,
compare to the consensus closing spread, and simulate flat bets at -110
above edge thresholds. EPA->points calibration is fit only on weeks that
were already walked through (no lookahead).

  python -m backtest.spread_baseline
"""
import numpy as np
import pandas as pd

from features.epa_ratings import SEASON_STRIDE, fit_ratings, load_game_obs
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

TRAIN_SEASONS = [2023, 2024, 2025]
TEST_SEASONS = [2024, 2025]
FIRST_WEEK, LAST_WEEK = 4, 15
B_INIT, HFA_INIT = 65.0, 2.3  # EPA/play -> points slope; home-field points
VIG_BREAK_EVEN = 0.5238  # win rate needed at -110


def consensus_lines(season: int) -> pd.DataFrame:
    lines = pd.read_parquet(CFBD_PARQUET_DIR / f"lines_{season}.parquet")
    lines = lines[lines.seasonType == "regular"]
    cons = lines.groupby("id", as_index=False).agg(
        spread=("spread", "median"), total=("overUnder", "median"))
    games = pd.read_parquet(CFBD_PARQUET_DIR / f"games_{season}.parquet")
    games = games[["id", "week", "neutralSite", "homeTeam", "awayTeam",
                   "homePoints", "awayPoints"]]
    return cons.merge(games, on="id")


def game_table(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per game with home/away ids and final margin."""
    g = obs.groupby("game_id", as_index=False).agg(
        season=("season", "first"), week=("week", "first"),
        season_type=("seasonType", "first"),
        home_id=("home_id", "first"),
        home_score=("home_score", "first"), away_score=("away_score", "first"))
    ids = obs.groupby("game_id").off_id.unique()
    g["away_id"] = [
        next((t for t in ids[gid] if t != h), None)
        for gid, h in zip(g.game_id, g.home_id)
    ]
    # NOTE: home_score/away_score here come from garbage-time-filtered plays
    # (truncated in blowouts) — never grade bets against them. Official final
    # margins are merged from the CFBD games table in run().
    return g.dropna(subset=["away_id"])


def run() -> pd.DataFrame:
    obs = load_game_obs(TRAIN_SEASONS)
    games = game_table(obs)

    lines = pd.concat(
        [consensus_lines(s).assign(season=s) for s in TEST_SEASONS])
    games = games.merge(
        lines[["id", "spread", "total", "neutralSite", "homeTeam", "awayTeam",
               "homePoints", "awayPoints"]],
        left_on="game_id", right_on="id", how="left")
    games["margin"] = games.homePoints - games.awayPoints  # official finals

    b, hfa = B_INIT, HFA_INIT
    calib: list[tuple[float, float, bool]] = []  # (net_diff, margin, neutral)
    preds = []

    for season in TEST_SEASONS:
        for week in range(FIRST_WEEK, LAST_WEEK + 1):
            asof = season * SEASON_STRIDE + week
            model = fit_ratings(obs, asof_week_idx=asof)

            wk = games[
                (games.season == season) & (games.week == week)
                & (games.season_type == 2)  # 2 = regular
                & games.spread.notna() & games.margin.notna()
            ]
            if wk.empty:
                continue

            for g in wk.itertuples():
                nd = model.net(g.home_id) - model.net(g.away_id)
                neutral = bool(g.neutralSite)
                pred = b * nd + (0.0 if neutral else hfa)
                preds.append({
                    "season": season, "week": week, "game_id": g.game_id,
                    "home": g.homeTeam, "away": g.awayTeam,
                    "net_diff": nd, "neutral": neutral,
                    "pred_margin": pred, "book_spread": g.spread,
                    "book_margin": -g.spread, "margin": g.margin,
                })
                calib.append((nd, g.margin, neutral))

            # Re-fit EPA->points calibration on walked-through weeks only.
            if len(calib) >= 150:
                cd = pd.DataFrame(calib, columns=["nd", "margin", "neutral"])
                X = np.column_stack([cd.nd, (~cd.neutral).astype(float)])
                coef, *_ = np.linalg.lstsq(
                    np.column_stack([X, np.ones(len(cd))]), cd.margin, rcond=None)
                b, hfa = float(coef[0]), float(coef[1])

    df = pd.DataFrame(preds)
    df["edge"] = df.pred_margin - df.book_margin  # >0: model likes home side
    df["home_covered"] = np.sign(df.margin + df.book_spread)  # 0 = push
    df["bet_won"] = np.where(
        df.home_covered == 0, np.nan,
        (np.sign(df.edge) == df.home_covered).astype(float))
    return df


def report(df: pd.DataFrame) -> None:
    print(f"games: {len(df)}  seasons: {sorted(df.season.unique())}")
    print(f"model MAE vs margin: {(df.pred_margin - df.margin).abs().mean():.2f}")
    print(f"book  MAE vs margin: {(df.book_margin - df.margin).abs().mean():.2f}")
    print(f"corr(pred, book): {df.pred_margin.corr(df.book_margin):.3f}\n")

    print("ATS by edge threshold (flat bets @ -110, break-even 52.38%):")
    print(f"{'edge>=':>7} {'bets':>6} {'wins':>6} {'win%':>7} {'ROI%':>7}")
    for t in (0, 1, 2, 3, 4, 5, 6):
        bets = df[(df.edge.abs() >= t) & df.bet_won.notna()]
        if len(bets) < 10:
            continue
        wr = bets.bet_won.mean()
        roi = (wr * (100 / 110) - (1 - wr)) * 100
        print(f"{t:>7} {len(bets):>6} {int(bets.bet_won.sum()):>6} "
              f"{wr:>6.1%} {roi:>6.1f}%")

    print("\ncover rate of model-favored side by edge bucket:")
    buckets = pd.cut(df.edge.abs(), [0, 2, 4, 6, 9, 30])
    ok = df.bet_won.notna()
    print(df[ok].groupby(buckets[ok], observed=True)
          .bet_won.agg(["count", "mean"]).round(3).to_string())


if __name__ == "__main__":
    out = run()
    dest = PARQUET_DIR / "backtest_spread_baseline.parquet"
    out.to_parquet(dest, index=False)
    report(out)
    print(f"\nsaved: {dest}")
