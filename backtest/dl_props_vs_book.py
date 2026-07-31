"""Does the DL distributional prop model beat the book (and our production
model)? The only test that matters: P(over) -> EV -> graded bets at real prices.

Uses the same discipline as backtest/props_vs_book.py: calibrate on 2024,
evaluate out-of-sample on 2025, blend toward the no-vig market price, settle
at the best quoted price.

  python -m backtest.dl_props_vs_book
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import consensus_lines
from ingestion.config import PARQUET_DIR
from models.dl_props import STATS, sf


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def run() -> pd.DataFrame:
    dl = pd.read_parquet(PARQUET_DIR / "dl_prop_preds.parquet")
    lines = consensus_lines()
    m = lines.merge(dl, on=["season", "week", "stat", "player"], how="inner")
    print(f"matched DL rows to book lines: {len(m):,}")

    # DL distributional P(over)
    p_over = np.empty(len(m))
    for stat, kind in STATS.items():
        s = m.stat == stat
        if not s.any():
            continue
        p_over[s.values] = sf(kind, m.loc[s, "line"].values,
                              m.loc[s, "dl_mu"].values,
                              m.loc[s, "dl_disp"].values)
    m["p_dl"] = p_over

    imp_o = 1 / (1 + m.pay_over)
    imp_u = 1 / (1 + m.pay_under)
    m["p_mkt"] = imp_o / (imp_o + imp_u)

    # blend weight fitted on 2024 by log-loss (same protocol as production)
    t = m[(m.season == 2024) & (m.y != m.line)]
    y = (t.y > t.line).astype(float)
    w_best, ll_best = 0.0, -np.inf
    for w in np.arange(0.0, 0.65, 0.05):
        pb = 1 / (1 + np.exp(-(w * logit(t.p_dl) + (1 - w) * logit(t.p_mkt))))
        ll = (y * np.log(pb) + (1 - y) * np.log(1 - pb)).sum()
        if ll > ll_best:
            w_best, ll_best = w, ll
    print(f"DL blend weight (2024 log-loss fit): {w_best:.2f}")
    m["p_bl"] = 1 / (1 + np.exp(
        -(w_best * logit(m.p_dl) + (1 - w_best) * logit(m.p_mkt))))

    m["ev_o"] = m.p_bl * m.pay_over_best - (1 - m.p_bl)
    m["ev_u"] = (1 - m.p_bl) * m.pay_under_best - m.p_bl
    m["side"] = np.where(m.ev_o >= m.ev_u, "over", "under")
    m["ev"] = m[["ev_o", "ev_u"]].max(axis=1)
    over_won = m.y > m.line
    push = m.y == m.line
    m["won"] = np.where(push, np.nan, np.where(
        m.side == "over", over_won, ~over_won).astype(float))
    pay = np.where(m.side == "over", m.pay_over_best, m.pay_under_best)
    m["pnl"] = np.where(m.won.isna(), 0.0, np.where(m.won == 1, pay, -1.0))
    return m


def report(m: pd.DataFrame) -> None:
    print("\n=== DL distributional model vs book (blended, best price) ===")
    for season, label in [(2024, "2024 (blend fit in-sample)"),
                          (2025, "2025 OUT-OF-SAMPLE")]:
        s = m[(m.season == season) & (m.n_books >= 2)]
        print(f"\n{label}: {len(s):,} lines")
        for t in (0.02, 0.03, 0.05):
            b = s[(s.ev >= t) & s.won.notna()]
            if len(b) < 30:
                continue
            print(f"  EV>={t:.0%}: {len(b):4d} bets  {b.won.mean():.1%} win  "
                  f"{b.pnl.mean()*100:+.1f}% ROI")

    print("\n--- production model (same 2025 rows, from props_vs_book) ---")
    prod = pd.read_parquet(PARQUET_DIR / "backtest_props_vs_book.parquet")
    p = prod[(prod.season == 2025) & (prod.n_books >= 2)]
    for t in (0.02, 0.03, 0.05):
        b = p[(p.bet_ev_bl >= t) & p.bet_won_bl.notna()]
        if len(b) < 30:
            continue
        print(f"  EV>={t:.0%}: {len(b):4d} bets  {b.bet_won_bl.mean():.1%} win  "
              f"{b.pnl_bl.mean()*100:+.1f}% ROI")


if __name__ == "__main__":
    out = run()
    out.to_parquet(PARQUET_DIR / "backtest_dl_props.parquet", index=False)
    report(out)
