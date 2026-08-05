"""Should the market-blend weight be estimated at all?

backtest/props_blend_sweep.py showed the book is violently sensitive to w
(+0.4 -> +30.0 -> +54.2 units across 0.25/0.30/0.35). The natural conclusion
was "the 0.05 grid is too coarse". Measuring the objective says otherwise:

    continuous argmax w = 0.310  (the 0.05 grid picks 0.30 — no real difference)
    moving +-0.05 from the optimum costs 0.523 log-loss out of ~2,781,
    i.e. 0.000129 PER ROW on n=4,048

**The surface is nearly flat, so w is barely identified — while the book that
falls out of it is not.** A finer grid gives a sharper estimate of noise. The
real choice is between three positions:

    grid     estimate w from each calibration sample (status quo)
    fixed    do not estimate it; the data cannot support the estimate
    average  average the blended PROBABILITY over the band the data cannot
             distinguish, so the result moves smoothly with the sample

Judged on TWO axes, because they can disagree:
  1. units/season out-of-sample — does it bet better?
  2. bootstrap sd of units — does it bet more STABLY? (the whole point)

  python -m backtest.blend_mode_ab
"""
import numpy as np
import pandas as pd

import backtest.props_vs_book as P
from backtest.props_vs_book import EXCLUDE_STATS, run

OOS = (2024, 2025)
N_BOOT = 40


def book(m: pd.DataFrame, season=None):
    d = m[(~m.stat.isin(EXCLUDE_STATS)) & (m.n_books >= 2)
          & m.bet_won_bl.notna() & (m.bet_ev_bl >= 0.04)].copy()
    below = d.bet_ev_bl < 0.08
    d = d[(d.week >= 5) & ((d.week >= 9) | below)]
    d = d[d.season.isin(OOS)] if season is None else d[d.season == season]
    if d.empty:
        return 0, 0.0, 0.0
    probe = d.bet_ev_bl.between(0.04, 0.05, inclusive="left")
    prime = d.bet_ev_bl.between(0.05, 0.08, inclusive="left")
    u = np.where(probe, 1.0, np.where(prime, 2.0, 1.0))
    n = len(OOS) if season is None else 1
    return len(d), float((u * d.pnl_bl).sum() / n), float(d.pnl_bl.mean())


def main() -> None:
    rows = []
    for mode in ("grid", "fixed", "average"):
        P.BLEND_MODE = mode
        m = run(write_coefs=False)
        n, u, roi = book(m)
        _, u24, _ = book(m, 2024)
        _, u25, _ = book(m, 2025)

        rng = np.random.default_rng(2026)
        boots = []
        for _ in range(N_BOOT):
            try:
                bm = run(write_coefs=False, cal_rng=rng)
            except Exception:
                continue
            boots.append(book(bm)[1])
        sd = float(np.std(boots)) if boots else float("nan")
        p5 = float(np.percentile(boots, 5)) if boots else float("nan")
        rows.append({"mode": mode, "bets": n, "roi": roi, "u": u,
                     "u24": u24, "u25": u25, "sd": sd, "p5": p5,
                     "pneg": float(np.mean(np.array(boots) <= 0)) if boots else np.nan})

    r = pd.DataFrame(rows)
    print(f"\n\n{'='*72}\n=== BLEND MODE HEAD-TO-HEAD\n{'='*72}")
    print(f"{'mode':>9} {'bets':>6} {'ROI':>8} {'u/szn':>8} {'2024':>8} "
          f"{'2025':>8} | {'boot sd':>8} {'5th %':>8} {'P(<=0)':>7}")
    for x in rows:
        print(f"{x['mode']:>9} {x['bets']:6d} {x['roi']:+8.1%} {x['u']:+8.1f} "
              f"{x['u24']:+8.1f} {x['u25']:+8.1f} | {x['sd']:8.1f} "
              f"{x['p5']:+8.1f} {x['pneg']:7.1%}")

    print("\nREAD: 'grid' is the status quo. If 'fixed' or 'average' cuts the")
    print("bootstrap sd materially WITHOUT giving up units, estimating w was")
    print("costing us stability for nothing — the data never identified it.")
    print("If units fall a lot, the adaptivity was real and worth its variance.")


if __name__ == "__main__":
    main()
