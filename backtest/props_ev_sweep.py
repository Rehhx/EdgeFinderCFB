"""Are the props EV thresholds plateaus, or more fitted spikes?

Discipline rule 10 says sweep every grid-searched hyperparameter the way you
sweep a threshold. The props book has three that have never had that test:

    EV >= 0.05        minimum edge to bet at all
    PRIME band 0.08   the 5-8% band is staked 2u, 8%+ only 1u
    n_books >= 2      how many books must quote the player

This matters more than usual right now. backtest/props_blend_sweep.py showed the
market-blend weight is NOT a plateau — one grid step swings the season +0.4 ->
+30.0 -> +54.2 units. If the EV cuts are ALSO spikes, props are even shakier
than that implied and the extra haircut in bankroll_sim is too small. If they
are plateaus, that partially offsets the w-sensitivity: it would mean the
selection RULE is sound and only the probability SCALE is uncertain.

Reads the graded backtest parquet directly, so this is exact and instant — no
refitting, and the 2023-only calibration discipline is inherited from whatever
wrote the file.

  python -m backtest.props_ev_sweep
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import EXCLUDE_STATS
from ingestion.config import PARQUET_DIR

OOS = (2024, 2025)
PRIME_LO, PRIME_HI = 0.05, 0.08
MIN_BOOKS = 2


def load() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_props_vs_book.parquet")
    return d[(~d.stat.isin(EXCLUDE_STATS)) & d.bet_won_bl.notna()
             & d.season.isin(OOS)].copy()


def grade(d, ev_min, prime_hi, min_books=MIN_BOOKS):
    """Shipped gate + staking: nothing before wk5, STANDARD only from wk9."""
    s = d[(d.n_books >= min_books) & (d.bet_ev_bl >= ev_min)].copy()
    prime = s.bet_ev_bl.between(ev_min, prime_hi, inclusive="left")
    s = s[(s.week >= 5) & ((s.week >= 9) | prime)]
    if s.empty:
        return 0, 0.0, 0.0, 0.0, 0.0
    u = np.where(s.bet_ev_bl.between(ev_min, prime_hi, inclusive="left"), 2.0, 1.0)
    per = []
    for season in OOS:
        m = s.season == season
        per.append(float((u[m.values] * s[m].pnl_bl).sum()))
    return (len(s) // len(OOS), float(s.pnl_bl.mean()),
            float((u * s.pnl_bl).sum() / len(OOS)), per[0], per[1])


def main() -> None:
    d = load()
    print(f"graded OOS rows: {len(d):,} (2024-25, rec_yds excluded)")

    print("\n=== 1. MINIMUM EV to bet (PRIME boundary held at 0.08) ===")
    print(f"{'ev_min':>7} {'bets/szn':>9} {'ROI':>8} {'units/szn':>10} "
          f"{'2024':>8} {'2025':>8}")
    for ev in (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12):
        n, roi, u, u24, u25 = grade(d, ev, PRIME_HI)
        mark = "  <- SHIPPED" if abs(ev - PRIME_LO) < 1e-9 else ""
        print(f"{ev:7.2f} {n:9d} {roi:+8.1%} {u:+10.1f} {u24:+8.1f} "
              f"{u25:+8.1f}{mark}")

    print("\n=== 2. PRIME/STANDARD boundary (ev_min held at 0.05) ===")
    print("   below the boundary = 2u, above = 1u (deliberately staked DOWN)")
    print(f"{'bound':>7} {'bets/szn':>9} {'ROI':>8} {'units/szn':>10} "
          f"{'2024':>8} {'2025':>8}")
    for hi in (0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 9.0):
        n, roi, u, u24, u25 = grade(d, PRIME_LO, hi)
        lab = "flat 2u" if hi >= 9 else f"{hi:.2f}"
        mark = "  <- SHIPPED" if abs(hi - PRIME_HI) < 1e-9 else ""
        print(f"{lab:>7} {n:9d} {roi:+8.1%} {u:+10.1f} {u24:+8.1f} "
              f"{u25:+8.1f}{mark}")

    print("\n=== 3. is the band really non-monotone? (raw ROI by EV band) ===")
    s = d[(d.n_books >= MIN_BOOKS) & (d.week >= 5)]
    for lo, hi in [(0.03, 0.05), (0.05, 0.08), (0.08, 0.12), (0.12, 9.0)]:
        b = s[s.bet_ev_bl.between(lo, hi, inclusive="left")]
        if len(b) < 30:
            continue
        per = b.groupby("season").pnl_bl.mean()
        t = b.pnl_bl.mean() / (b.pnl_bl.std() / np.sqrt(len(b)))
        print(f"  EV {lo:.2f}-{hi if hi < 9 else 9:.2f}: {len(b):4d} bets  "
              f"{b.pnl_bl.mean():+6.1%} ROI  t={t:+5.2f}   "
              + "  ".join(f"{k}:{v:+.1%}" for k, v in per.items()))

    print("\n=== 4. minimum books quoting ===")
    for mb in (1, 2, 3, 4):
        n, roi, u, u24, u25 = grade(d, PRIME_LO, PRIME_HI, mb)
        mark = "  <- SHIPPED" if mb == MIN_BOOKS else ""
        print(f"  n_books>={mb}: {n:4d} bets/szn  {roi:+6.1%} ROI  "
              f"{u:+6.1f} u/szn   2024 {u24:+6.1f}  2025 {u25:+6.1f}{mark}")

    print("\nREAD: a broad plateau in sections 1-2 means the SELECTION rule is "
          "sound and\nonly the probability SCALE is uncertain (the blend-weight "
          "problem). Spikes\nhere would mean props are fitted end-to-end and "
          "the extra haircut is too small.")


if __name__ == "__main__":
    main()
