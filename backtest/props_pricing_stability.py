"""How much of the props edge depends on the ONE season the pricing was fitted on?

backtest/props_stability.py resampled the team-VOLUME model's training data and
found it contributes only +-3.5 u/szn. But that is not the only fitted thing in
the props chain, and arguably not the important one. Three more parameters are
fitted on the 2023 matched rows alone:

  recal (a, b, sigma, c_matchup)  per stat — sets the projection LEVEL and the
                                   width of the distribution we price against
  NB dispersion r                  the receptions tail
  market-blend weight w            how much of the final probability is ours
                                   rather than the book's

`sigma` and `w` are the two that most directly control which bets clear EV>=5%,
so a single-season draw for them is a real exposure. This resamples whole WEEKS
of the calibration season, refits all three together (they are coupled — a
wider sigma changes p_over, which changes the fitted blend weight), and regrades
2024-25 through the unchanged pipeline.

  python -m backtest.props_pricing_stability
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import EXCLUDE_STATS, run

N_BOOT = 60
OOS_SEASONS = (2024, 2025)


def units(m: pd.DataFrame, season=None) -> tuple[int, float]:
    """Whole book under the shipped season-arc gate: EV>=5%, no rec_yds,
    2+ books, nothing before wk5, STANDARD only from wk9, PRIME 2u / STD 1u."""
    d = m[(~m.stat.isin(EXCLUDE_STATS)) & (m.n_books >= 2)
          & m.bet_won_bl.notna() & (m.bet_ev_bl >= 0.05)].copy()
    prime = d.bet_ev_bl.between(0.05, 0.08, inclusive="left")
    d = d[(d.week >= 5) & ((d.week >= 9) | prime)]
    d = d[d.season.isin(OOS_SEASONS)] if season is None else d[d.season == season]
    u = np.where(d.bet_ev_bl.between(0.05, 0.08, inclusive="left"), 2.0, 1.0)
    n_szn = len(OOS_SEASONS) if season is None else 1
    return len(d), float((u * d.pnl_bl).sum() / n_szn)


def bootstrap(cal_seasons) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    for b in range(N_BOOT):
        try:
            m = run(write_coefs=False, cal_rng=rng, cal_seasons=cal_seasons)
        except Exception as exc:              # a degenerate draw is a result
            print(f"  boot {b}: failed ({exc})")
            continue
        n, u = units(m)
        rows.append({"boot": b, "bets": n, "u": u,
                     "u24": units(m, 2024)[1], "u25": units(m, 2025)[1]})
    return pd.DataFrame(rows)


def main() -> None:
    n0, u0 = units(run(write_coefs=False))
    print(f"\nbaseline: {n0} bets, {u0:+.1f} units/season (2024-25 OOS)\n")
    r = bootstrap((2023,))
    print(f"\n=== {len(r)} bootstrap refits of the 2023 PRICING calibration ===")
    print(f"units/season   median {r.u.median():+.1f}   mean {r.u.mean():+.1f}   "
          f"sd {r.u.std():.1f}")
    print(f"               5th %ile {np.percentile(r.u, 5):+.1f}   "
          f"95th %ile {np.percentile(r.u, 95):+.1f}")
    print(f"               P(unprofitable season) {(r.u <= 0).mean():.1%}")
    print(f"bet count      median {r.bets.median():.0f}  "
          f"range {r.bets.min()}-{r.bets.max()}")
    print(f"\nper-season stability (both should stay positive):")
    print(f"  2024  median {r.u24.median():+.1f}  P(<=0) {(r.u24 <= 0).mean():.1%}")
    print(f"  2025  median {r.u25.median():+.1f}  P(<=0) {(r.u25 <= 0).mean():.1%}")
    print(f"\nbaseline sits at the {(r.u < u0).mean():.0%} percentile of its own "
          "resampling distribution.")
    print("\nCOMPARE: the VOLUME model contributes sd 3.5 u/szn "
          "(backtest/props_stability.py).\nIf the sd here is much larger, the "
          "pricing calibration — not the projection —\nis the fragile part of "
          "the props chain, and one extra calibration season would\nbe worth "
          "more than any modelling change.")

    # Does calibrating on all three seasons actually shrink that spread? This
    # is IN-SAMPLE (the graded seasons are inside the calibration set) so the
    # LEVEL is meaningless and is not reported — only the SPREAD is, because
    # spread is what we are trying to reduce and is what production inherits.
    print(f"\n{'='*66}\n=== does a 3-season calibration shrink the spread?\n"
          f"{'='*66}")
    r3 = bootstrap((2023, 2024, 2025))
    print(f"\n  1-season calibration (backtest, honest OOS): sd {r.u.std():5.1f} "
          f"u/szn, bets {r.bets.min()}-{r.bets.max()}")
    print(f"  3-season calibration (what PRODUCTION uses):  sd {r3.u.std():5.1f} "
          f"u/szn, bets {r3.bets.min()}-{r3.bets.max()}")
    print(f"\n  variance reduction: {1 - r3.u.std()/r.u.std():.0%}  "
          f"(sqrt(3) would predict ~42%)")
    print("  ⚠️ Only the SPREAD is comparable here — the 3-season row is "
          "in-sample,\n  so ignore its level entirely.")


if __name__ == "__main__":
    main()
