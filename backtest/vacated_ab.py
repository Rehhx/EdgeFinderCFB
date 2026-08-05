"""Head-to-head: does the vacated-share prior adjustment beat production?

Same harness discipline as backtest/script_shares.py — both variants run
through the IDENTICAL pricing pipeline with write_coefs=False, so only the
projection differs.

The adjustment is fitted on 2023 (the pricing calibration season, already
burned), so 2024 and 2025 stay fully out-of-sample. 2022 cannot be the fit
season because vacated share needs a roster and rosters start in 2022, making
2023 the first computable season.

  python -m backtest.vacated_ab
"""
import numpy as np
import pandas as pd

import models.props as P
from backtest.props_vs_book import EXCLUDE_STATS, run
from ingestion.config import PARQUET_DIR

OOS = (2024, 2025)


def units(m: pd.DataFrame, season=None) -> tuple[int, float, float]:
    """Shipped book: EV>=4%, no rec_yds, 2+ books, nothing before wk5,
    STANDARD only from wk9, PROBE 1u / PRIME 2u / STANDARD 1u."""
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
    res = {}
    for mode in ("off", "on"):
        print(f"\n{'='*66}\n=== VACATED_MODE = {mode}\n{'='*66}")
        P.VACATED_MODE = mode
        dest = PARQUET_DIR / f"prop_projections_vac_{mode}.parquet"
        P.project(P.build_table(), write_coefs=False).to_parquet(dest, index=False)
        m = run(proj_path=dest, write_coefs=False)
        res[mode] = m

    print(f"\n\n{'='*66}\n=== HEAD-TO-HEAD (identical pricing, projection only)"
          f"\n{'='*66}")
    print(f"{'variant':>8} {'bets':>6} {'ROI':>8} {'u/szn':>8}   "
          f"{'2024 u':>8} {'2025 u':>8}")
    for mode in ("off", "on"):
        n, u, roi = units(res[mode])
        _, u24, _ = units(res[mode], 2024)
        _, u25, _ = units(res[mode], 2025)
        print(f"{mode:>8} {n:6d} {roi:+8.1%} {u:+8.1f}   {u24:+8.1f} {u25:+8.1f}")

    print("\n--- projection accuracy on matched rows (MAE) ---")
    for stat in ("rush_yds", "receptions", "pass_yds"):
        line = f"  {stat:12s}"
        for mode in ("off", "on"):
            s = res[mode]
            s = s[s.stat == stat]
            line += f"  {mode}={(s.actual - s.proj).abs().mean():7.3f}"
        print(line)

    print("\n--- EARLY SEASON (weeks 1-4, currently NOT bet) ---")
    print("   the adjustment should help most here: the season prior does all")
    print("   the work and no in-season data exists yet")
    for mode in ("off", "on"):
        e = res[mode]
        e = e[(e.week <= 4) & (~e.stat.isin(EXCLUDE_STATS))
              & e.bet_won_bl.notna() & (e.bet_ev_bl >= 0.04)
              & e.season.isin(OOS)]
        if len(e) < 30:
            continue
        per = e.groupby("season").pnl_bl.mean()
        print(f"  {mode:>4}: {len(e):4d} bets  {e.pnl_bl.mean():+6.1%} ROI   "
              + "  ".join(f"{k}:{v:+.1%}" for k, v in per.items()))


if __name__ == "__main__":
    main()
