"""Does moneyline VALUE betting actually profit? 8-season test at real prices.

We verified our ML win-probabilities are well calibrated (spread_history),
but never graded actual ML bets against real historical prices. CFBD lines
carry homeMoneyline/awayMoneyline per book, so we can price EV properly AND
line-shop (take the best of the quoted books, as we would live).

Tests three probability sources:
  model  - raw model win prob (calibrated but can't out-pick the market)
  mkt    - the no-vig market probability (pure line-shopping edge)
  blend  - logit blend of the two (the production ml_value approach)

  python -m backtest.ml_value_history
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

SEASONS = range(2018, 2026)
BLEND_W = 0.30  # weight on model prob (production ml_value uses 0.5 live)


def dec(american: pd.Series) -> pd.Series:
    return np.where(american < 0, 1 + 100 / american.abs(),
                    1 + american / 100)


def load_prices() -> pd.DataFrame:
    """Best available ML price per game per side, across books."""
    frames = []
    for s in SEASONS:
        try:
            l = pd.read_parquet(CFBD_PARQUET_DIR / f"lines_{s}.parquet")
        except FileNotFoundError:
            continue
        l = l.dropna(subset=["homeMoneyline", "awayMoneyline"])
        l["dec_home"] = dec(l.homeMoneyline)
        l["dec_away"] = dec(l.awayMoneyline)
        g = l.groupby("id", as_index=False).agg(
            dec_home_best=("dec_home", "max"), dec_away_best=("dec_away", "max"),
            dec_home_med=("dec_home", "median"),
            dec_away_med=("dec_away", "median"), n_books=("provider", "nunique"))
        g["season"] = s
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def run() -> pd.DataFrame:
    pred = pd.read_parquet(PARQUET_DIR / "backtest_spread_history.parquet")
    px = load_prices()
    m = pred.merge(px, left_on=["game_id", "season"],
                   right_on=["id", "season"], how="inner")
    m = m[m.margin != 0].copy()
    print(f"games with model preds + real ML prices: {len(m):,}")

    for c in ("dec_home_best", "dec_away_best", "dec_home_med", "dec_away_med"):
        m[c] = m[c].astype(float)  # price cols arrive as object dtype
    sigma = (m.pred_margin - m.margin).std()
    m["p_model_home"] = norm.cdf(m.pred_margin / (sigma * 0.55))
    imp_h, imp_a = 1 / m.dec_home_med, 1 / m.dec_away_med
    m["p_mkt_home"] = (imp_h / (imp_h + imp_a)).astype(float)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6)
                          / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m["p_blend_home"] = 1 / (1 + np.exp(
        -(BLEND_W * lg(m.p_model_home) + (1 - BLEND_W) * lg(m.p_mkt_home))))
    m["home_won"] = (m.margin > 0).astype(float)
    return m


def grade(m: pd.DataFrame, pcol: str, tag: str) -> None:
    d = m.copy()
    d["ev_home"] = d[pcol] * d.dec_home_best - 1
    d["ev_away"] = (1 - d[pcol]) * d.dec_away_best - 1
    d["bet_home"] = d.ev_home >= d.ev_away
    d["ev"] = d[["ev_home", "ev_away"]].max(axis=1)
    d["won"] = np.where(d.bet_home, d.home_won == 1, d.home_won == 0)
    d["dec"] = np.where(d.bet_home, d.dec_home_best, d.dec_away_best)
    d["pnl"] = np.where(d.won, d.dec - 1, -1.0)
    print(f"\n--- {tag} (best price, {d.season.nunique()} seasons) ---")
    print(f"{'EV>=':>6} {'bets':>6} {'win%':>7} {'ROI%':>8} {'seasons+':>9}")
    for t in (0.0, 0.03, 0.05, 0.10):
        b = d[d.ev >= t]
        if len(b) < 60:
            continue
        per = b.groupby("season").pnl.mean()
        print(f"{t:>6.0%} {len(b):>6} {b.won.mean():>6.1%} "
              f"{b.pnl.mean()*100:>+7.1f}% {(per > 0).sum():>4}/{len(per)}")
    # dog vs favourite split at a usable threshold
    b = d[d.ev >= 0.05].copy()
    if len(b) > 60:
        b["on_dog"] = np.where(b.bet_home, b.dec_home_best > 2, b.dec_away_best > 2)
        print("  by side:", b.groupby("on_dog").pnl.agg(["size", "mean"])
              .round(3).to_dict("index"))


if __name__ == "__main__":
    m = run()
    m.to_parquet(PARQUET_DIR / "backtest_ml_value_history.parquet", index=False)
    for col, tag in [("p_model_home", "MODEL prob"),
                     ("p_mkt_home", "MARKET prob (pure line shopping)"),
                     ("p_blend_home", f"BLEND {BLEND_W:.0%} model")]:
        grade(m, col, tag)
