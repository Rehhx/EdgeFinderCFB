"""Share-trend correction — ⚠️ REJECTED for production (one partial candidate).

VERDICT (2026-08-03): correcting the projection for share momentum makes the
book WORSE, 56.3%/+7.0% -> 55.7%/+6.1% (units 51.7 -> 50.7).

Per stat, OOS 2024+2025 at EV>=5%:
    rush_yds   +6.0% -> +4.7%   (2025 +5.1% -> +1.4%)   WORSE
    pass_yds   +6.5% -> +3.7%   (2024 +4.5% -> -1.7%)   WORSE
    receptions +11.8% -> +16.4% (both seasons up)        better
The two that got worse are the two with POSITIVE momentum (rising rush/QB share
keeps rising). That is exactly the information a book can read off a box score,
so correcting for it just moves our projection toward the line and destroys the
disagreement that generates the edge. Same failure as opponent-adjusted defence,
player PPA and situational usage: a better projection is not a better bet when
the market already has the input.

The receptions case is the opposite sign — target share MEAN-REVERTS (b=-0.09)
— and it improved in both seasons (+16.5%, +16.4%). But it is ONE cell out of
three tested, on n=116, where the 1-SE band is ~9pp. **Not shipped.** Logged as
a paper-trial candidate alongside ml_spread; revisit with 2026 data.

---
Original hypothesis and method below.

Can we PREDICT our prop losses and beat them? — share-trend correction.

The loss review (backtest/loss_review.py) found losses are driven by the
player's realised share exceeding the trailing EWMA, and that the bias decays
as data accumulates. An EWMA lags a trend by construction, so a player whose
role is EXPANDING should be systematically under-projected.

Mechanism check (walk-forward, prior_games>=3, n=30,485) — confirmed:
    resid = a + b*trend     rush b=+0.177 t=+14.6
                            qb   b=+0.481 t=+38.1
                            tgt  b=-0.060 t= -4.8   <- targets MEAN-REVERT
Trend autocorrelation 0.41-0.58, so trends persist; this is not noise.

But predicting our own residual is NOT the same as beating the price — the book
can see an expanding role too. This runs the corrected projections through the
IDENTICAL pricing pipeline (props_vs_book.run) for a fair head-to-head, with
the mandatory controls: median price, per-season, and the week arc.

Correction is fitted on the CALIBRATION season only (2023) and applied
out-of-sample, exactly like the recalibration it feeds.

  python -m backtest.props_trend
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import EXCLUDE_STATS, run as price_run
from ingestion.config import PARQUET_DIR

KEY = ["season", "team_id", "player"]
SHARES = {"rush_share": "trail_rush_share",
          "qb_share": "trail_qb_share",
          "tgt_share": "trail_tgt_share"}
CAL_SEASON = 2023
ADJ_PATH = PARQUET_DIR / "prop_projections_trend.parquet"


def add_trend(p: pd.DataFrame) -> pd.DataFrame:
    """Short-window minus long-window share, both strictly shifted."""
    p = p.sort_values(KEY + ["week"]).copy()
    g = p.groupby(KEY, sort=False)
    for s in SHARES:
        p[f"{s}_s2"] = g[s].transform(
            lambda x: x.shift(1).rolling(2, min_periods=1).mean())
        p[f"{s}_l5"] = g[s].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        p[f"{s}_trend"] = (p[f"{s}_s2"] - p[f"{s}_l5"]).fillna(0.0)
    return p


def build_adjusted() -> pd.DataFrame:
    p = pd.read_parquet(PARQUET_DIR / "prop_projections.parquet")
    n0 = len(p)
    p = add_trend(p)
    assert len(p) == n0 == 46487, f"row count drifted: {n0} -> {len(p)}"

    # fit beta per share on the CALIBRATION season only
    betas = {}
    for s, trail in SHARES.items():
        f = p[(p.season == CAL_SEASON) & p[s].notna() & p[trail].notna()
              & (p.prior_games >= 3)]
        resid = (f[s] - f[trail]).values
        x = f[f"{s}_trend"].values
        betas[s] = float(np.polyfit(x, resid, 1)[0])
    print(f"trend betas fitted on {CAL_SEASON}: "
          + ", ".join(f"{k} {v:+.3f}" for k, v in betas.items()))

    # corrected shares -> rebuild the projections that use them.
    # Mirrors models/props.project() exactly; efficiency terms are unchanged,
    # so we scale each projection by the share ratio.
    for s, trail in SHARES.items():
        adj = p[trail] + betas[s] * p[f"{s}_trend"]
        # a share cannot go negative or above 1
        p[f"{trail}_adj"] = adj.clip(lower=0.0, upper=1.0)
        p[f"{s}_ratio"] = (p[f"{trail}_adj"]
                           / p[trail].replace(0, np.nan)).fillna(1.0)

    p["proj_rush_yds"] = p.proj_rush_yds * p.rush_share_ratio
    p["proj_pass_yds"] = p.proj_pass_yds * p.qb_share_ratio
    p["proj_targets"] = p.proj_targets * p.tgt_share_ratio
    p["proj_receptions"] = p.proj_receptions * p.tgt_share_ratio
    p["proj_rec_yds"] = p.proj_rec_yds * p.tgt_share_ratio
    p.to_parquet(ADJ_PATH, index=False)
    print(f"adjusted projections -> {ADJ_PATH.name}")
    return p


def summarise(m: pd.DataFrame, label: str) -> dict:
    b = m[m.season.isin([2024, 2025]) & (~m.stat.isin(EXCLUDE_STATS))
          & (m.n_books >= 2) & m.bet_won_bl.notna() & (m.bet_ev_bl >= 0.05)]
    per = b.groupby("season").pnl_bl.mean()
    out = {"label": label, "bets": len(b), "win": b.bet_won_bl.mean(),
           "roi": b.pnl_bl.mean(), "2024": per.get(2024, np.nan),
           "2025": per.get(2025, np.nan)}
    # production stake plan: PRIME (5-8% EV) 2u, STANDARD 1u, wk1-4 STANDARD off
    prime = b.bet_ev_bl.between(0.05, 0.08, inclusive="left")
    keep = (b.week >= 5) | prime
    st = np.where(prime, 2.0, 1.0)[keep.values]
    out["units"] = (b.pnl_bl.values[keep.values] * st).sum() / 2
    return out


def main() -> None:
    build_adjusted()
    base = price_run(write_coefs=False)
    trend = price_run(proj_path=ADJ_PATH, write_coefs=False)

    rows = [summarise(base, "baseline (production)"),
            summarise(trend, "share-trend corrected")]
    print("\n=== HEAD-TO-HEAD, OOS 2024+2025, EV>=5% ===")
    print(f"{'model':26s} {'bets':>6} {'win%':>7} {'ROI':>8} {'2024':>8} "
          f"{'2025':>8} {'units/szn':>10}")
    for r in rows:
        print(f"{r['label']:26s} {r['bets']:6d} {r['win']:6.1%} "
              f"{r['roi']*100:+7.1f}% {r['2024']*100:+7.1f}% "
              f"{r['2025']*100:+7.1f}% {r['units']:+10.1f}")

    print("\n=== by week arc (does it fix the weak early season?) ===")
    for lbl, m in [("baseline", base), ("trend", trend)]:
        b = m[m.season.isin([2024, 2025]) & (~m.stat.isin(EXCLUDE_STATS))
              & (m.n_books >= 2) & m.bet_won_bl.notna()
              & (m.bet_ev_bl >= 0.05)].copy()
        b["wk"] = pd.cut(b.week, [0, 4, 8, 15], labels=["1-4", "5-8", "9-15"])
        g = b.groupby("wk", observed=True).agg(
            bets=("pnl_bl", "size"), win=("bet_won_bl", "mean"),
            roi=("pnl_bl", "mean"))
        print(f"\n{lbl}:")
        print(g.round(3).to_string())

    print("\n=== by stat (targets mean-revert — did that hurt receptions?) ===")
    for lbl, m in [("baseline", base), ("trend", trend)]:
        b = m[m.season.isin([2024, 2025]) & (~m.stat.isin(EXCLUDE_STATS))
              & (m.n_books >= 2) & m.bet_won_bl.notna()
              & (m.bet_ev_bl >= 0.05)]
        g = b.groupby("stat").agg(bets=("pnl_bl", "size"),
                                  roi=("pnl_bl", "mean"))
        print(f"{lbl:9s} " + " | ".join(
            f"{s} {r.roi*100:+.1f}% (n={int(r.bets)})"
            for s, r in g.iterrows()))


if __name__ == "__main__":
    main()
