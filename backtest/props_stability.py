"""How much of the props edge survives resampling its own training data?

Discovered 2026-08-04: the team-volume model was fitted on ONE arbitrary side
of each training game (groupby("game_id").first() over player rows), so it
depended on row order. A 3.5% change in upstream inputs moved its coefficients
in the third decimal -- and moved the graded book by ~40 units/season.

That is a statement about fragility, not about the fix. If a coefficient
perturbation that small can swing the season by 40 units, then the headline
props ROI is not a measurement of an edge, it is one draw from a wide
distribution, and the bankroll projection built on it is overconfident.

So: resample the TRAIN season's team-games with replacement, refit the volume
coefficients, re-project, and grade the whole book through the identical
pricing pipeline. The spread of the resulting units/season is the honest error
bar on the props edge.

The table is built ONCE and only the projection step is repeated, so nothing
downstream of the coefficients is held fixed artificially.

  python -m backtest.props_stability
"""
import numpy as np
import pandas as pd

import models.props as P
from backtest.props_vs_book import EXCLUDE_STATS, run
from ingestion.config import PARQUET_DIR

N_BOOT = 40
TMP = PARQUET_DIR / "_stability_proj.parquet"
OOS_SEASONS = (2024, 2025)   # 2023 is the pricing calibration season


def units(m: pd.DataFrame, season=None) -> tuple[int, float]:
    """Whole book under the shipped rules: EV>=5%, no rec_yds, 2+ books,
    week-5 gate on STANDARD only, PRIME at 2u and STANDARD at 1u."""
    d = m[(~m.stat.isin(EXCLUDE_STATS)) & (m.n_books >= 2)
          & m.bet_won_bl.notna() & (m.bet_ev_bl >= 0.05)].copy()
    prime = d.bet_ev_bl.between(0.05, 0.08, inclusive="left")
    d = d[(d.week >= 5) | prime]
    d = d[d.season.isin(OOS_SEASONS)] if season is None else d[d.season == season]
    u = np.where(d.bet_ev_bl.between(0.05, 0.08, inclusive="left"), 2.0, 1.0)
    n_szn = len(OOS_SEASONS) if season is None else 1
    return len(d), float((u * d.pnl_bl).sum() / n_szn)


def main() -> None:
    print("building the projection table once...")
    P.SCRIPT_MODE = "off"
    tbl = P.build_table()
    train = tbl[tbl.season == 2022]
    base_coefs = P.fit_volume_coefs(train)
    print("baseline volume coefs:",
          {k: np.round(v, 3).tolist() for k, v in base_coefs.items()})

    def grade_with(coefs) -> tuple[int, float]:
        P.project(tbl, vol_coefs=coefs, write_coefs=False).to_parquet(
            TMP, index=False)
        return units(run(proj_path=TMP, write_coefs=False))

    n0, u0 = grade_with(base_coefs)
    print(f"\nbaseline: {n0} bets, {u0:+.1f} units/season (2024-25 OOS)\n")

    keys = tbl[tbl.season == 2022].drop_duplicates(["game_id", "team_id"])[
        ["game_id", "team_id"]].reset_index(drop=True)
    rng = np.random.default_rng(2026)
    rows = []
    for b in range(N_BOOT):
        pick = keys.iloc[rng.integers(0, len(keys), len(keys))]
        res = train.merge(pick, on=["game_id", "team_id"], how="inner")
        try:
            coefs = P.fit_volume_coefs(res)
            n, u = grade_with(coefs)
        except Exception as exc:            # a degenerate resample is a result
            print(f"  boot {b}: failed ({exc})")
            continue
        rows.append({"boot": b, "bets": n, "u": u,
                     "rush_trail": coefs["team_rush_att"][0],
                     "rush_spread": coefs["team_rush_att"][1],
                     "pass_spread": coefs["team_pass_att"][1]})
        print(f"  boot {b:2d}: {n:4d} bets  {u:+7.1f} u/szn   "
              f"rush=[{coefs['team_rush_att'][0]:.3f}, "
              f"{coefs['team_rush_att'][1]:.3f}]")

    r = pd.DataFrame(rows)
    print(f"\n=== {len(r)} bootstrap refits of the volume model ===")
    print(f"units/season   median {r.u.median():+.1f}   "
          f"mean {r.u.mean():+.1f}   sd {r.u.std():.1f}")
    print(f"               5th %ile {np.percentile(r.u, 5):+.1f}   "
          f"95th %ile {np.percentile(r.u, 95):+.1f}")
    print(f"               P(unprofitable season) {(r.u <= 0).mean():.1%}")
    print(f"bet count      median {r.bets.median():.0f}  "
          f"range {r.bets.min()}-{r.bets.max()}")
    print(f"\ncoefficient spread (sd): rush_trail {r.rush_trail.std():.4f}  "
          f"rush_spread {r.rush_spread.std():.4f}  "
          f"pass_spread {r.pass_spread.std():.4f}")
    print(f"\nbaseline sits at the {(r.u < u0).mean():.0%} percentile of its "
          "own resampling distribution.")
    print("\nREAD: sd here is uncertainty from the TRAINING sample alone, with "
          "the\nbet outcomes held fixed. Real forward uncertainty is strictly "
          "larger.")
    TMP.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
