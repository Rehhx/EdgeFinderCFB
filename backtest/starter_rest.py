"""Are the FAVOURITE's starters mispriced in big mismatches?

Today's script measurement says a starter's share of his team's work collapses
when his side blows the opponent out -- RB carries fall to 0.944x his trailing
share, QB attempts to 1.053x from 1.322x when behind. He is on the bench in the
fourth quarter.

That is a fact about exactly the games we already bet: the Q1 and 1H big-dog
plays live at spreads of 17+ and 25+. So the question is whether the book marks
the favourite's player props down enough for the rest that is coming.

Test order matters. Section 1 is BLIND -- no model, just "did the favourite's
starters go under the posted line more often than the dog's?" If the book is
adequately pricing rest, that is 50/50 at every spread and there is nothing
here regardless of what any model says. Only if the blind test moves do the
model-selected sections mean anything.

Note this analysis was not trustworthy before 2026-08-04: team_spread was
sign-flipped on the 4.9% of rows with no play-by-play, and those rows averaged
a 39.7-point spread -- i.e. precisely the mismatch games this file is about.

  python -m backtest.starter_rest
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import EXCLUDE_STATS, run
from ingestion.config import PARQUET_DIR

SPREAD_BUCKETS = [(0, 7), (7, 14), (14, 21), (21, 28), (28, 99)]
MIN_N = 40


def book() -> pd.DataFrame:
    m = run(write_coefs=False)
    proj = pd.read_parquet(PARQUET_DIR / "prop_projections.parquet")
    key = ["season", "week", "player"]
    sp = proj[key + ["team_spread"]].dropna().drop_duplicates(key)
    m = m.merge(sp, on=key, how="inner")
    m = m[~m.stat.isin(EXCLUDE_STATS) & (m.n_books >= 2)].copy()
    # positive team_spread = this player's team is the DOG
    m["fav"] = np.where(m.team_spread < 0, "favourite", "dog")
    m["gap"] = m.team_spread.abs()
    return m


def blind(m: pd.DataFrame) -> None:
    """No model. Did the stat land under the posted line, by side and spread?"""
    d = m[m.actual.notna() & m.line.notna()].copy()
    d["under"] = (d.actual < d.line).astype(float)
    d = d[d.actual != d.line]
    print("\n=== 1. BLIND: P(stat lands UNDER the posted line) ===")
    print("   if the book prices starter rest correctly this is ~50% everywhere")
    print(f"{'spread':>12} {'favourite':>22} {'dog':>22}")
    for lo, hi in SPREAD_BUCKETS:
        row = f"  {lo:2d}-{hi:2d} pts  "
        for side in ("favourite", "dog"):
            s = d[(d.fav == side) & d.gap.between(lo, hi, inclusive="left")]
            if len(s) < MIN_N:
                row += f"{'n/a':>22}"
                continue
            p = s.under.mean()
            se = np.sqrt(p * (1 - p) / len(s))
            row += f"  {p:6.1%} (n={len(s):5d}, {(p-.5)/se:+5.2f}sd)"
        print(row)

    print("\n   by stat, favourites only, spread >= 21:")
    s = d[(d.fav == "favourite") & (d.gap >= 21)]
    for stat, ss in s.groupby("stat"):
        if len(ss) < MIN_N:
            continue
        p = ss.under.mean()
        se = np.sqrt(p * (1 - p) / len(ss))
        print(f"     {stat:12s} {p:6.1%} under  (n={len(ss):4d}, "
              f"{(p-.5)/se:+.2f}sd)")


def priced(m: pd.DataFrame) -> None:
    """Flat-bet every favourite-starter UNDER at the best available price."""
    d = m[m.actual.notna() & m.line.notna() & (m.fav == "favourite")].copy()
    d = d[d.actual != d.line]
    d["won"] = (d.actual < d.line).astype(float)
    d["pnl"] = np.where(d.won == 1, d.pay_under_best, -1.0)
    print("\n=== 2. PRICED: flat-bet the favourite's starters UNDER ===")
    print(f"{'spread':>12} {'bets':>7} {'win':>7} {'ROI':>8} {'t':>7}")
    for lo, hi in SPREAD_BUCKETS:
        s = d[d.gap.between(lo, hi, inclusive="left")]
        if len(s) < MIN_N:
            continue
        t = s.pnl.mean() / (s.pnl.std() / np.sqrt(len(s)))
        print(f"  {lo:2d}-{hi:2d} pts  {len(s):7d} {s.won.mean():7.1%} "
              f"{s.pnl.mean():+7.1%} {t:+7.2f}")

    print("\n   per season, spread >= 21 (must hold in BOTH):")
    for season, s in d[d.gap >= 21].groupby("season"):
        if len(s) < MIN_N:
            continue
        print(f"     {season}: {len(s):4d} bets  {s.won.mean():6.1%}  "
              f"{s.pnl.mean():+6.1%} ROI")


def model_agreement(m: pd.DataFrame) -> None:
    """Does our own model already say 'under' on these, and does it help?"""
    d = m[m.bet_won_bl.notna() & (m.bet_ev_bl >= 0.05) & (m.fav == "favourite")]
    print("\n=== 3. MODEL-SELECTED: our EV>=5% bets on favourites ===")
    for lo, hi in SPREAD_BUCKETS:
        s = d[d.gap.between(lo, hi, inclusive="left")]
        if len(s) < 25:
            continue
        print(f"  {lo:2d}-{hi:2d} pts  {len(s):5d} bets  "
              f"{(s.bet_side_bl == 'under').mean():5.1%} of them UNDER  "
              f"{s.pnl_bl.mean():+6.1%} ROI")


def main() -> None:
    m = book()
    print(f"\nmatched rows with a signed spread: {len(m):,}")
    blind(m)
    priced(m)
    model_agreement(m)
    print("\nREAD: section 1 is the one that matters. A model-selected edge "
          "with a\nflat 50% blind rate is the model working; a blind skew is "
          "the BOOK being lazy.")


if __name__ == "__main__":
    main()
