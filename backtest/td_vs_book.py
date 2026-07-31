"""Anytime-TD model vs real book prices.

Same discipline as the yardage props: model probability -> blend toward the
no-vig market -> EV at the best quoted price -> grade. Recalibration is fitted
on 2023 (inside models/td_model), so 2024+2025 are out-of-sample.

Anytime TD is a heavy-vig market (books often hold 8-15% across the board),
so the bar is higher than for two-way yardage props.

  python -m backtest.td_vs_book
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import payout
from ingestion.config import PARQUET_DIR

BLEND_W = 0.25  # weight on model prob; refit below by log-loss on 2023


def consensus() -> pd.DataFrame:
    """Best and median TD price per (season, week, player)."""
    raw = pd.read_parquet(PARQUET_DIR / "historical_td_lines.parquet")
    raw = raw.dropna(subset=["player", "price"])
    # 'Yes'/player-name outcomes only; drop any 'No' side if present
    raw = raw[~raw.name.astype(str).str.lower().eq("no")]
    raw["pay"] = payout(raw.price)
    g = raw.groupby(["season", "week", "player"], as_index=False).agg(
        pay_med=("pay", "median"), pay_best=("pay", "max"),
        n_books=("book", "nunique"))
    return g


def run() -> pd.DataFrame:
    proj = pd.read_parquet(PARQUET_DIR / "td_projections.parquet")
    proj = proj[proj.exp_team_tds.notna() & (proj.games_prior >= 2)]
    lines = consensus()
    m = lines.merge(proj[["season", "week", "player", "p_td", "scored"]],
                    on=["season", "week", "player"], how="inner")
    m = m.drop_duplicates(subset=["season", "week", "player"])
    print(f"matched TD lines to projections: {len(m):,}")

    # no-vig market probability. Anytime TD is one-sided in our pull, so use
    # the implied prob directly and strip a flat overround estimate.
    m["p_mkt_raw"] = 1 / (1 + m.pay_med)
    # calibrate the market's implied prob to reality on 2023 (removes vig)
    from sklearn.linear_model import LogisticRegression
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4)
                          / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    f = m[m.season == 2023]
    lr = LogisticRegression().fit(lg(f.p_mkt_raw).values.reshape(-1, 1),
                                  f.scored.values)
    a, b = float(lr.intercept_[0]), float(lr.coef_[0][0])
    m["p_mkt"] = 1 / (1 + np.exp(-(a + b * lg(m.p_mkt_raw))))
    print(f"market de-vig fit (2023): logit(p) = {a:+.2f} + {b:.2f}*logit(imp) "
          f"| mean implied {m.p_mkt_raw.mean():.3f} -> de-vigged "
          f"{m.p_mkt.mean():.3f} | actual {m.scored.mean():.3f}")

    # blend weight by log-loss on 2023
    best_w, best_ll = 0.0, -np.inf
    for w in np.arange(0, 0.65, 0.05):
        pb = 1 / (1 + np.exp(-(w * lg(f.p_td) + (1 - w) * lg(f.p_mkt_raw))))
        pb = 1 / (1 + np.exp(-(a + b * lg(pb))))
        ll = (f.scored * np.log(pb) + (1 - f.scored) * np.log(1 - pb)).sum()
        if ll > best_ll:
            best_w, best_ll = w, ll
    print(f"blend weight on model prob (2023 log-loss): {best_w:.2f}")
    blend = 1 / (1 + np.exp(-(best_w * lg(m.p_td)
                              + (1 - best_w) * lg(m.p_mkt_raw))))
    m["p_bl"] = 1 / (1 + np.exp(-(a + b * lg(blend))))

    m["ev"] = m.p_bl * m.pay_best - (1 - m.p_bl)
    m["pnl"] = np.where(m.scored == 1, m.pay_best, -1.0)
    return m


def report(m: pd.DataFrame) -> None:
    oos = m[(m.season != 2023) & (m.n_books >= 2)]
    print(f"\n=== ANYTIME TD betting, OOS 2024+2025 ({len(oos):,} lines) ===")
    print(f"{'EV>=':>6} {'bets':>6} {'/szn':>5} {'hit%':>7} {'ROI':>8} {'seasons':>22}")
    for t in (0.0, 0.03, 0.05, 0.10):
        b = oos[oos.ev >= t]
        if len(b) < 50:
            continue
        per = b.groupby("season").pnl.mean().round(3).to_dict()
        print(f"{t:6.0%} {len(b):6d} {len(b)/2:5.0f} {b.scored.mean():6.1%} "
              f"{b.pnl.mean()*100:+7.1f}% {str(per):>22s}")
    # blind control: bet everything at the best price
    print(f"\nBLIND control (bet every quoted player): "
          f"{oos.pnl.mean()*100:+.1f}% ROI over {len(oos):,} bets")


if __name__ == "__main__":
    out = run()
    out.to_parquet(PARQUET_DIR / "backtest_td.parquet", index=False)
    report(out)
