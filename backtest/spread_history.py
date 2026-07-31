"""Long-history walk-forward: EPA spread + moneyline over 8 seasons (2018-25).

Extends the 2-season backtests to the full CFBD-lines era we have PBP for,
so the headline claims (weeks 1-5 ATS edge; ML calibration) get real sample.
EPA-only (roster/coach priors need 2021+ data, tested separately). Ratings
history from 2016; self-calibrating EPA->points on walked-through games.

  python -m backtest.spread_history
"""
import numpy as np
import pandas as pd

from backtest.spread_baseline import consensus_lines, game_table
from features.epa_ratings import SEASON_STRIDE, fit_ratings, load_game_obs
from ingestion.config import PARQUET_DIR

PBP_SEASONS = list(range(2016, 2026))
TEST_SEASONS = list(range(2018, 2026))
FIRST_WEEK, LAST_WEEK = 1, 15
MIN_CALIB = 200


def run() -> pd.DataFrame:
    obs = load_game_obs(PBP_SEASONS)
    games = game_table(obs)
    lines = pd.concat([consensus_lines(s).assign(season=s)
                       for s in TEST_SEASONS])
    games = games.merge(
        lines[["id", "spread", "total", "neutralSite", "homeTeam",
               "awayTeam", "homePoints", "awayPoints"]],
        left_on="game_id", right_on="id", how="left")
    games["margin"] = games.homePoints - games.awayPoints

    b_margin, hfa = 65.0, 2.3
    tot_coef = np.array([40.0, 52.0])  # [k*combined_strength, intercept]
    calib = []       # (net_diff, home_flag, margin)
    tcalib = []      # (combined_strength, total)
    preds = []
    for season in TEST_SEASONS:
        for week in range(FIRST_WEEK, LAST_WEEK + 1):
            asof = season * SEASON_STRIDE + week
            model = fit_ratings(obs, asof_week_idx=asof)
            wk = games[(games.season == season) & (games.week == week)
                       & (games.season_type == 2)
                       & games.spread.notna() & games.margin.notna()]
            for g in wk.itertuples():
                nd = model.net(g.home_id) - model.net(g.away_id)
                # combined offensive strength vs both defenses (for totals)
                ho, hd = (model.ratings.loc[g.home_id, ["off", "def"]]
                          if g.home_id in model.ratings.index else (0.0, 0.0))
                ao, ad = (model.ratings.loc[g.away_id, ["off", "def"]]
                          if g.away_id in model.ratings.index else (0.0, 0.0))
                combined = (ho - ad) + (ao - hd)
                neutral = bool(g.neutralSite)
                pred_margin = b_margin * nd + (0.0 if neutral else hfa)
                actual_total = g.homePoints + g.awayPoints
                preds.append({
                    "season": season, "week": week, "game_id": g.game_id,
                    "net_diff": nd, "pred_margin": pred_margin,
                    "book_margin": -g.spread, "book_spread": g.spread,
                    "margin": g.margin, "neutral": neutral,
                    "pred_total": tot_coef[0] * combined + tot_coef[1],
                    "book_total": g.total, "actual_total": actual_total})
                calib.append((nd, 0.0 if neutral else 1.0, g.margin))
                if pd.notna(g.total):
                    tcalib.append((combined, actual_total))
            if len(calib) >= MIN_CALIB:
                cd = np.array(calib)
                A = np.column_stack([cd[:, 0], cd[:, 1], np.ones(len(cd))])
                coef, *_ = np.linalg.lstsq(A, cd[:, 2], rcond=None)
                b_margin, hfa = float(coef[0]), float(coef[1])
                td = np.array(tcalib)
                At = np.column_stack([td[:, 0], np.ones(len(td))])
                tot_coef, *_ = np.linalg.lstsq(At, td[:, 1], rcond=None)

    df = pd.DataFrame(preds)
    df["edge"] = df.pred_margin - df.book_margin
    df["cover"] = np.sign(df.margin + df.book_spread)
    df["won"] = np.where(df.cover == 0, np.nan,
                         (np.sign(df.edge) == df.cover).astype(float))
    sigma = (df.pred_margin - df.margin).std()
    df["p_home"] = 1 / (1 + np.exp(-df.pred_margin / (sigma * 0.55)))
    df["ml_pick_home"] = df.pred_margin > 0
    df["ml_won"] = np.where(df.margin == 0, np.nan,
                            (df.ml_pick_home == (df.margin > 0)).astype(float))
    return df


def report(df: pd.DataFrame) -> None:
    print(f"games: {len(df):,} across {df.season.nunique()} seasons "
          f"({df.season.min()}-{df.season.max()})")
    print(f"model margin MAE {(df.pred_margin-df.margin).abs().mean():.2f} | "
          f"book {(df.book_margin-df.margin).abs().mean():.2f} | "
          f"corr {df.pred_margin.corr(df.book_margin):.3f}\n")

    print("=== ATS by week bucket x edge threshold (flat -110, BE 52.4%) ===")
    for label, mask in [("weeks 1-5", df.week.between(1, 5)),
                        ("weeks 6-15", df.week.between(6, 15))]:
        print(f"\n{label}:")
        for t in (2, 3, 4, 6):
            b = df[mask & (df.edge.abs() >= t) & df.won.notna()]
            if len(b) < 50:
                continue
            wr = b.won.mean()
            se = (wr * (1 - wr) / len(b)) ** 0.5
            roi = (wr * 100 / 110 - (1 - wr)) * 100
            print(f"  edge>={t}: {len(b):5d} bets  {wr:.1%} (+-{se:.1%})  "
                  f"{roi:+.1f}% ROI")

    print("\n=== weeks 1-5 edge>=4 by season (is it every year?) ===")
    e = df[df.week.between(1, 5) & (df.edge.abs() >= 4) & df.won.notna()]
    per = e.groupby("season").won.agg(["size", "mean"]).round(3)
    print(per.to_string())

    print("\n=== O/U by week bucket (EPA-only totals, edge>=5 pts) ===")
    t = df[df.book_total.notna()].copy()
    t["tedge"] = t.pred_total - t.book_total
    t["ou"] = np.sign(t.actual_total - t.book_total)
    t["ou_won"] = np.where(t.ou == 0, np.nan,
                           (np.sign(t.tedge) == t.ou).astype(float))
    print(f"  totals MAE {(t.pred_total-t.actual_total).abs().mean():.2f} | "
          f"book {(t.book_total-t.actual_total).abs().mean():.2f}")
    for label, mask in [("weeks 1-5", t.week.between(1, 5)),
                        ("weeks 6-15", t.week.between(6, 15))]:
        b = t[mask & (t.tedge.abs() >= 5) & t.ou_won.notna()]
        if len(b) < 50:
            continue
        wr = b.ou_won.mean()
        print(f"  {label}: {len(b):5d} bets  {wr:.1%}  "
              f"{(wr*100/110-(1-wr))*100:+.1f}% ROI")

    print("\n=== ML calibration (all weeks, 8 seasons) ===")
    d = df[df.ml_won.notna()].copy()
    d["conf"] = np.maximum(d.p_home, 1 - d.p_home)
    d["pick_won"] = np.where(d.ml_pick_home, d.margin > 0, d.margin < 0)
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        s = d[(d.conf >= lo) & (d.conf < hi)]
        if len(s) < 50:
            continue
        print(f"  conf {lo:.0%}-{hi:.0%}: n={len(s):5d}  stated {s.conf.mean():.1%}  "
              f"actual {s.pick_won.mean():.1%}")


if __name__ == "__main__":
    out = run()
    out.to_parquet(PARQUET_DIR / "backtest_spread_history.parquet", index=False)
    report(out)
