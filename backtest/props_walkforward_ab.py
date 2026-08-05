"""Does a SECOND calibration season help? Fixed-2023 vs walk-forward.

backtest/props_pricing_stability.py found the pricing calibration contributes
sd 15.6 u/szn — 4.5x the team-volume model — and that a 3-season calibration
cut that only 10%. But that test was IN-SAMPLE (the graded seasons sat inside
the calibration set), so it could only speak to the spread, never the level.

This is the honest version. Each season is priced on the seasons that actually
preceded it, so nothing leaks:

    fixed         2024 and 2025 both priced on 2023
    walk-forward  2024 on 2023, 2025 on 2023+2024

Only 2025 differs between the two, and only 2025 gets the extra season — so
**the 2025 column is the whole experiment.** If a second calibration season is
worth having, it shows up there and nowhere else. Comparing pooled totals would
hide it, because 2024 is identical by construction.

  python -m backtest.props_walkforward_ab
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import EXCLUDE_STATS, run, walkforward

OOS = (2024, 2025)


def book(m: pd.DataFrame, season=None):
    """Shipped rules: EV>=4%, no rec_yds, 2+ books, nothing before wk5,
    STANDARD only from wk9, PROBE 1u / PRIME 2u / STANDARD 1u."""
    d = m[(~m.stat.isin(EXCLUDE_STATS)) & (m.n_books >= 2)
          & m.bet_won_bl.notna() & (m.bet_ev_bl >= 0.04)].copy()
    below = d.bet_ev_bl < 0.08
    d = d[(d.week >= 5) & ((d.week >= 9) | below)]
    if season is not None:
        d = d[d.season == season]
    if d.empty:
        return 0, 0.0, 0.0
    probe = d.bet_ev_bl.between(0.04, 0.05, inclusive="left")
    prime = d.bet_ev_bl.between(0.05, 0.08, inclusive="left")
    u = np.where(probe, 1.0, np.where(prime, 2.0, 1.0))
    n_szn = 1 if season is not None else d.season.nunique()
    return len(d), float((u * d.pnl_bl).sum() / max(n_szn, 1)), float(d.pnl_bl.mean())


def main() -> None:
    print("=== FIXED calibration (everything on 2023) ===")
    fixed = run(write_coefs=False)
    print("\n=== WALK-FORWARD calibration ===")
    wf = walkforward()

    print(f"\n\n{'='*62}\n=== HEAD-TO-HEAD — read the 2025 row, it is the "
          f"experiment\n{'='*62}")
    print(f"{'season':>8} {'variant':>14} {'bets':>6} {'ROI':>8} {'units':>8}")
    for s in OOS:
        for lab, m in (("fixed", fixed), ("walk-forward", wf)):
            d = m[m.season == s]
            if d.empty:
                continue
            n, u, roi = book(d, s)
            note = ""
            if s == 2024:
                note = "  (identical by construction)"
            print(f"{s:>8} {lab:>14} {n:6d} {roi:+8.1%} {u:+8.1f}{note}")

    print("\n--- 2025 only, side by side (the one season that gains a year) ---")
    a = fixed[fixed.season == 2025]
    b = wf[wf.season == 2025]
    for lab, d in (("1 cal season ", a), ("2 cal seasons", b)):
        n, u, roi = book(d, 2025)
        won = d[(d.bet_ev_bl >= 0.04) & d.bet_won_bl.notna()]
        print(f"  {lab}: {n:4d} bets  {roi:+6.1%} ROI  {u:+6.1f} u  "
              f"| matched rows {len(d):,}")

    print("\n--- projection-independent check: does the extra season improve")
    print("    the CALIBRATION itself? (MAE of recalibrated proj vs actual) ---")
    for lab, d in (("fixed       ", a), ("walk-forward", b)):
        s = d[d.stat != "rec_yds"]
        print(f"  {lab}: MAE {(s.actual - s.proj_cal).abs().mean():7.3f}  "
              f"on {len(s):,} rows")

    print("\nREAD: if walk-forward 2025 beats fixed 2025 on BOTH units and")
    print("calibration MAE, a second season is worth having and the fixed")
    print("scheme is leaving information on the table. If it does not, the")
    print("pricing fragility is not a sample-size problem — consistent with")
    print("the blend weight being a coarse grid search.")


if __name__ == "__main__":
    main()
