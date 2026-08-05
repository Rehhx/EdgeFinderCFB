"""Head-to-head: does making usage shares game-script-aware beat production?

A starter's share of his team's work is not a constant. He sits in the fourth
quarter of a blowout win and gets fed when the team is behind. The trailing
EWMA averages over whatever scripts he happened to draw, so that contamination
rides into every projection -- and it is worst early, because September
cupcake blowouts depress the very shares the EWMA carries into the competitive
games we actually bet.

Two separable interventions, tested separately on purpose:

  descript  divide each PAST game's share by its realised script factor before
            the EWMA. Uses a margin we KNOW. Pure input cleaning.
  full      also multiply the projection by E[factor | pregame spread]. Uses a
            spread that leaves a 21.1-point residual sd on margin, and is the
            half that risks merely agreeing with the book more -- the same way
            opponent-adjusted defense EPA did when it was tested and rejected.

Every variant runs through backtest.props_vs_book.run() with write_coefs=False,
so pricing, recalibration, tails, market blend and line shopping are identical
and only the projection differs.

  python -m backtest.script_shares
"""
import numpy as np
import pandas as pd

import models.props as P
from backtest.props_vs_book import EXCLUDE_STATS, run
from ingestion.config import PARQUET_DIR

MODES = ("off", "descript", "full")
EV_BANDS = [(0.05, 0.08, "PRIME (2u)"), (0.08, 9.0, "STANDARD (1u)")]


def build(mode: str) -> pd.DataFrame:
    P.SCRIPT_MODE = mode
    dest = PARQUET_DIR / f"prop_projections_script_{mode}.parquet"
    out = P.project(P.build_table())
    out.to_parquet(dest, index=False)
    return dest


def grade(m: pd.DataFrame, label: str) -> list[dict]:
    """Bet book under the shipped rules: EV>=5%, no rec_yds, 2+ books,
    and the week-5 gate on the STANDARD band only."""
    d = m[(~m.stat.isin(EXCLUDE_STATS)) & (m.n_books >= 2)
          & m.bet_won_bl.notna() & (m.bet_ev_bl >= 0.05)].copy()
    prime = d.bet_ev_bl.between(0.05, 0.08, inclusive="left")
    d = d[(d.week >= 5) | prime].copy()
    rows = []
    for lo, hi, band in EV_BANDS:
        s = d[d.bet_ev_bl.between(lo, hi, inclusive="left")]
        if s.empty:
            continue
        u = 2.0 if band.startswith("PRIME") else 1.0
        for season, ss in [("both", s)] + list(s.groupby("season")):
            if len(ss) < 30:
                continue
            rows.append({
                "variant": label, "band": band, "season": season,
                "bets": len(ss), "win": (ss.pnl_bl > 0).mean(),
                "roi": ss.pnl_bl.mean(),
                "u_per_szn": u * ss.pnl_bl.sum() / (1 if season != "both" else 2),
            })
    return rows


def main() -> None:
    results, books = [], {}
    for mode in MODES:
        print(f"\n{'='*66}\n=== SCRIPT_MODE = {mode}\n{'='*66}")
        path = build(mode)
        m = run(proj_path=path, write_coefs=False)
        books[mode] = m
        results += grade(m, mode)

    r = pd.DataFrame(results)
    print(f"\n\n{'='*66}\n=== HEAD-TO-HEAD (identical pricing, only the "
          f"projection differs)\n{'='*66}")
    for band in r.band.unique():
        print(f"\n--- {band} ---")
        piv = r[r.band == band].pivot_table(
            index="season", columns="variant",
            values=["bets", "win", "roi", "u_per_szn"])
        print(piv.round(3).to_string())

    print("\n--- whole book, per season (both bands, shipped stake plan) ---")
    tot = r[r.season != "both"].groupby(["variant", "season"]).agg(
        bets=("bets", "sum"), u=("u_per_szn", "sum"))
    print(tot.round(1).to_string())

    print("\n--- projection accuracy (MAE on matched prop rows) ---")
    for stat in ("rush_yds", "receptions", "pass_yds"):
        line = f"  {stat:12s}"
        for mode in MODES:
            s = books[mode]
            s = s[s.stat == stat]
            line += f"  {mode}={(s.actual - s.proj).abs().mean():6.2f}"
        print(line)

    print("\nREAD: 'off' is production. 'descript' cleans the input using a "
          "margin we know.\n'full' adds a forward term off the spread, which "
          "is where the\n'just agrees with the book more' failure mode lives.")


if __name__ == "__main__":
    main()
