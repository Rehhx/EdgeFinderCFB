"""Phase 1 walk-forward: EPA ratings + preseason roster priors vs closing spreads.

Runs two variants over identical games (weeks 1-15, 2024-2025):
  epa_only : the Phase 0 baseline extended to the full season
  roster   : adds prior_net_diff * early-season weight (fades to 0 by week 7)
Both self-calibrate on walked-through games only; the roster prior model is
trained only on seasons before the one being predicted.

  python -m backtest.spread_phase1
"""
import numpy as np
import pandas as pd

from backtest.spread_baseline import consensus_lines, game_table, report
from features.epa_ratings import SEASON_STRIDE, fit_ratings, load_game_obs
from features.roster_priors import preseason_priors
from ingestion.config import PARQUET_DIR

PBP_SEASONS = [2021, 2022, 2023, 2024, 2025]
TEST_SEASONS = [2024, 2025]
FIRST_WEEK, LAST_WEEK = 1, 15
MIN_CALIB = 150

INIT_COEFS = {
    "epa_only": np.array([65.0, 2.3, 0.0]),        # [epa_diff, home, 1]
    "roster":   np.array([65.0, 30.0, 2.3, 0.0]),  # [epa_diff, prior_term, home, 1]
}


def early_weight(week: int) -> float:
    return max(0.0, (7 - week) / 6)


def design(variant: str, rows: pd.DataFrame) -> np.ndarray:
    home = (~rows.neutral).astype(float)
    if variant == "epa_only":
        return np.column_stack([rows.epa_diff, home, np.ones(len(rows))])
    return np.column_stack(
        [rows.epa_diff, rows.prior_diff * rows.early_w, home, np.ones(len(rows))])


def run() -> pd.DataFrame:
    obs = load_game_obs(PBP_SEASONS)
    games = game_table(obs)
    lines = pd.concat(
        [consensus_lines(s).assign(season=s) for s in TEST_SEASONS])
    games = games.merge(
        lines[["id", "spread", "total", "neutralSite", "homeTeam", "awayTeam",
               "homePoints", "awayPoints"]],
        left_on="game_id", right_on="id", how="left")
    games["margin"] = games.homePoints - games.awayPoints

    priors = {s: preseason_priors(s, obs) for s in TEST_SEASONS}

    coefs = {k: v.copy() for k, v in INIT_COEFS.items()}
    walked: list[dict] = []
    preds = []

    for season in TEST_SEASONS:
        pri = priors[season]
        pri_floor = pri.quantile(0.10)  # unseen/new FBS teams
        for week in range(FIRST_WEEK, LAST_WEEK + 1):
            model = fit_ratings(obs, asof_week_idx=season * SEASON_STRIDE + week)
            wk = games[
                (games.season == season) & (games.week == week)
                & (games.season_type == 2)
                & games.spread.notna() & games.margin.notna()
            ]
            if wk.empty:
                continue

            rows = pd.DataFrame({
                "epa_diff": [model.net(g.home_id) - model.net(g.away_id)
                             for g in wk.itertuples()],
                "prior_diff": [pri.get(g.home_id, pri_floor)
                               - pri.get(g.away_id, pri_floor)
                               for g in wk.itertuples()],
                "neutral": wk.neutralSite.astype(bool).values,
                "early_w": early_weight(week),
                "margin": wk.margin.values,
            })

            rec = wk[["season", "week", "game_id", "homeTeam", "awayTeam"]] \
                .reset_index(drop=True).rename(
                    columns={"homeTeam": "home", "awayTeam": "away"})
            rec["book_margin"] = -wk.spread.values
            rec["book_spread"] = wk.spread.values
            rec["margin"] = wk.margin.values
            for v in coefs:
                rec[f"pred_{v}"] = design(v, rows) @ coefs[v]
            preds.append(pd.concat([rec, rows.drop(columns="margin")], axis=1))
            walked.extend(rows.to_dict("records"))

            if len(walked) >= MIN_CALIB:
                wdf = pd.DataFrame(walked)
                for v in coefs:
                    coefs[v], *_ = np.linalg.lstsq(
                        design(v, wdf), wdf.margin.values, rcond=None)

    from ingestion.config import save_coefs
    c = coefs["roster"]
    save_coefs("spread", {"epa": float(c[0]), "prior": float(c[1]),
                          "hfa": float(c[2]), "const": float(c[3])})

    df = pd.concat(preds, ignore_index=True)
    df["home_covered"] = np.sign(df.margin + df.book_spread)
    for v in INIT_COEFS:
        df[f"edge_{v}"] = df[f"pred_{v}"] - df.book_margin
        df[f"won_{v}"] = np.where(
            df.home_covered == 0, np.nan,
            (np.sign(df[f"edge_{v}"]) == df.home_covered).astype(float))
    print("final combiner coefs:",
          {v: coefs[v].round(2).tolist() for v in coefs})
    return df


def report_variant(df: pd.DataFrame, v: str) -> None:
    d = df.rename(columns={
        f"pred_{v}": "pred_margin", f"edge_{v}": "edge", f"won_{v}": "bet_won"})
    print(f"\n================ {v} ================")
    for label, mask in [
        ("ALL weeks 1-15", np.ones(len(d), bool)),
        ("weeks 1-5", d.week.between(1, 5).values),
        ("weeks 6-15", d.week.between(6, 15).values),
    ]:
        s = d[mask]
        print(f"\n--- {label} ({len(s)} games) ---")
        print(f"model MAE {(s.pred_margin - s.margin).abs().mean():.2f} | "
              f"book MAE {(s.book_margin - s.margin).abs().mean():.2f} | "
              f"corr(pred, book) {s.pred_margin.corr(s.book_margin):.3f}")
        for t in (2, 3, 4):
            bets = s[(s.edge.abs() >= t) & s.bet_won.notna()]
            if len(bets) < 15:
                continue
            wr = bets.bet_won.mean()
            roi = (wr * (100 / 110) - (1 - wr)) * 100
            print(f"  edge>={t}: {len(bets):4d} bets, {wr:.1%} win, {roi:+.1f}% ROI")


if __name__ == "__main__":
    out = run()
    dest = PARQUET_DIR / "backtest_spread_phase1.parquet"
    out.to_parquet(dest, index=False)
    for v in INIT_COEFS:
        report_variant(out, v)
    print(f"\nsaved: {dest}")
