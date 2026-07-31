"""Game simulator: per-team points model -> totals, team totals, spreads.

Each side of a game gets a predicted points distribution from:
  net_off   its offense rating minus opponent defense rating (EPA/play)
  pace_avg  expected sec/play, averaged from both play-callers' pace priors
  proe      its play-caller's pass-over-expected prior
  home      home-field
Coach priors come from coach_tendencies but only seasons BEFORE the game's
season (a new OC carries his profile from his old job — that's the edge).
Coefficients self-calibrate on walked-through games; walk-forward safe.

  python -m models.game_sim          # runs the totals backtest 2024-25
"""
import numpy as np
import pandas as pd

from backtest.spread_baseline import consensus_lines, game_table
from features.epa_ratings import FCS_ID, SEASON_STRIDE, fit_ratings, load_game_obs
from ingestion.config import PARQUET_DIR

PBP_SEASONS = [2021, 2022, 2023, 2024, 2025]
WARMUP_SEASONS = [2023]  # calibrate coefs here; no bets recorded
TEST_SEASONS = [2024, 2025]
FIRST_WEEK, LAST_WEEK = 1, 15
MIN_CALIB = 300  # side-rows (2 per game)
# net_off, pace, proe, mismatch, home, const — mismatch is |rating gap|:
# lopsided games run clock and cap scoring, so it should fit negative
INIT = np.array([65.0, -0.8, 0.0, -1.0, 1.5, 48.0])
VIG_BREAK_EVEN = 0.5238


def coach_priors() -> pd.DataFrame:
    """(team_id, season) -> pace/proe of its play-caller from PRIOR seasons."""
    prof = pd.read_parquet(PARQUET_DIR / "coach_tendencies.parquet")
    cdb = pd.read_parquet(PARQUET_DIR / "coach_db.parquet")
    cdb = cdb[cdb.team_id.notna()].astype({"team_id": int})
    cdb["playcaller"] = cdb.oc.fillna(cdb.hc)

    rows = []
    for r in cdb.itertuples():
        hist = prof[(prof.playcaller == r.playcaller) & (prof.season < r.season)]
        if hist.empty:
            continue
        latest = hist.sort_values("season").iloc[-1]
        rows.append({
            "team_id": r.team_id, "season": r.season,
            "pace_prior": latest.pace_neutral, "proe_prior": latest.proe,
        })
    return pd.DataFrame(rows).set_index(["team_id", "season"])


class GameSim:
    """Wraps ratings + coach priors; predicts (home_pts, away_pts)."""

    def __init__(self, ratings, priors: pd.DataFrame, coefs: np.ndarray,
                 pace_med: float, proe_med: float):
        self.r = ratings.ratings
        self.priors = priors
        self.coefs = coefs
        self.pace_med, self.proe_med = pace_med, proe_med

    def _off_def(self, tid: int) -> tuple[float, float]:
        if tid in self.r.index:
            return self.r.loc[tid, "off"], self.r.loc[tid, "def"]
        if FCS_ID in self.r.index:
            return self.r.loc[FCS_ID, "off"], self.r.loc[FCS_ID, "def"]
        return self.r.off.quantile(0.05), self.r["def"].quantile(0.05)

    def _coach(self, tid: int, season: int) -> tuple[float, float]:
        try:
            row = self.priors.loc[(tid, season)]
            return float(row.pace_prior), float(row.proe_prior)
        except KeyError:
            return self.pace_med, self.proe_med

    def _net(self, off_id, def_id) -> float:
        off_o, _ = self._off_def(off_id)
        _, def_d = self._off_def(def_id)
        return off_o - def_d

    def side_features(self, off_id, def_id, season, home: bool) -> np.ndarray:
        net_own = self._net(off_id, def_id)
        net_opp = self._net(def_id, off_id)
        pace_o, proe_o = self._coach(off_id, season)
        pace_d, _ = self._coach(def_id, season)
        return np.array([
            net_own, (pace_o + pace_d) / 2, proe_o,
            abs(net_own - net_opp), float(home), 1.0])

    def predict(self, home_id, away_id, season, neutral=False):
        xh = self.side_features(home_id, away_id, season, home=not neutral)
        xa = self.side_features(away_id, home_id, season, home=False)
        return float(xh @ self.coefs), float(xa @ self.coefs)


def run(use_coach_priors: bool = True) -> pd.DataFrame:
    obs = load_game_obs(PBP_SEASONS)
    games = game_table(obs)
    lines = pd.concat(
        [consensus_lines(s).assign(season=s)
         for s in WARMUP_SEASONS + TEST_SEASONS])
    games = games.merge(
        lines[["id", "spread", "total", "neutralSite", "homeTeam", "awayTeam",
               "homePoints", "awayPoints"]],
        left_on="game_id", right_on="id", how="left")

    priors = coach_priors()
    pace_med = priors.pace_prior.median()
    proe_med = priors.proe_prior.median()
    if not use_coach_priors:  # ablation: every team gets the league median
        priors = priors.iloc[0:0]
    print(f"coach priors: {len(priors)} team-seasons "
          f"(median pace {pace_med:.1f}s, median proe {proe_med:+.3f})")

    coefs = INIT.copy()
    walked_X, walked_y = [], []
    preds = []

    for season in WARMUP_SEASONS + TEST_SEASONS:
        warmup = season in WARMUP_SEASONS
        for week in range(FIRST_WEEK, LAST_WEEK + 1):
            ratings = fit_ratings(obs, asof_week_idx=season * SEASON_STRIDE + week)
            sim = GameSim(ratings, priors, coefs, pace_med, proe_med)
            wk = games[
                (games.season == season) & (games.week == week)
                & (games.season_type == 2)
                & games.total.notna() & games.homePoints.notna()
            ]
            if wk.empty:
                continue

            for g in wk.itertuples():
                neutral = bool(g.neutralSite)
                ph, pa = sim.predict(g.home_id, g.away_id, season, neutral)
                if not warmup:
                    preds.append({
                        "season": season, "week": week, "game_id": g.game_id,
                        "home": g.homeTeam, "away": g.awayTeam,
                        "pred_home": ph, "pred_away": pa,
                        "pred_total": ph + pa, "pred_margin": ph - pa,
                        "book_total": g.total, "book_spread": g.spread,
                        "actual_total": g.homePoints + g.awayPoints,
                        "margin": g.homePoints - g.awayPoints,
                    })
                xh = sim.side_features(g.home_id, g.away_id, season, not neutral)
                xa = sim.side_features(g.away_id, g.home_id, season, False)
                walked_X += [xh, xa]
                walked_y += [g.homePoints, g.awayPoints]

            if len(walked_y) >= MIN_CALIB:
                coefs, *_ = np.linalg.lstsq(
                    np.vstack(walked_X), np.array(walked_y), rcond=None)

    print("final points-model coefs "
          "[net_off, pace, proe, mismatch, home, const]:", coefs.round(2))
    from ingestion.config import save_coefs
    save_coefs("total", [float(x) for x in coefs])

    df = pd.DataFrame(preds)
    df["edge_total"] = df.pred_total - df.book_total
    df["ou_result"] = np.sign(df.actual_total - df.book_total)  # 0 = push
    df["ou_won"] = np.where(
        df.ou_result == 0, np.nan,
        (np.sign(df.edge_total) == df.ou_result).astype(float))
    return df


def report(df: pd.DataFrame) -> None:
    print(f"\ngames: {len(df)}")
    print(f"model total MAE: {(df.pred_total - df.actual_total).abs().mean():.2f} | "
          f"book total MAE: {(df.book_total - df.actual_total).abs().mean():.2f} | "
          f"corr(pred, book): {df.pred_total.corr(df.book_total):.3f}")
    sigma = (df.pred_total - df.actual_total).std()
    print(f"total residual sigma: {sigma:.1f}")

    for label, mask in [("ALL", np.ones(len(df), bool)),
                        ("weeks 1-5", df.week.between(1, 5).values),
                        ("weeks 6-15", df.week.between(6, 15).values)]:
        s = df[mask]
        print(f"\n--- O/U {label} ({len(s)} games) ---")
        for t in (2, 3, 4, 5):
            bets = s[(s.edge_total.abs() >= t) & s.ou_won.notna()]
            if len(bets) < 15:
                continue
            wr = bets.ou_won.mean()
            roi = (wr * (100 / 110) - (1 - wr)) * 100
            print(f"  edge>={t}: {len(bets):4d} bets, {wr:.1%} win, {roi:+.1f}% ROI")

    # sanity: the same sim's spread vs the book
    d = df[df.book_spread.notna()]
    print(f"\nsim spread MAE vs margin: "
          f"{(d.pred_margin - d.margin).abs().mean():.2f} "
          f"(book {( -d.book_spread - d.margin).abs().mean():.2f})")


if __name__ == "__main__":
    out = run()
    dest = PARQUET_DIR / "backtest_totals_sim.parquet"
    out.to_parquet(dest, index=False)
    report(out)
    print(f"\nsaved: {dest}")
