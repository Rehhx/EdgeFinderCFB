"""Can $1,000 reach $5,000 in one season? Which levers actually get there.

bankroll_sim.py answers "what is the honest expectation" with FLAT stakes off
the starting bankroll. That is the right way to state an edge, but it is not
how you grow a bankroll 5x, and it understates what is reachable. This file
tests the three real levers, on the same block-bootstrapped book:

  1. COMPOUNDING   stake a % of the CURRENT bankroll, not the starting one.
                   Free to implement, no new bets, no new risk of ruin per bet
                   — the single most underused lever we have.
  2. STAKE SIZE    2% -> 4% -> 6%. Linear in return, superlinear in drawdown.
  3. PARLAYS       parlaying INDEPENDENT +EV legs MULTIPLIES the edge:
                   EV_parlay = (1+EV1)(1+EV2) - 1. Two +45.7% Q1 legs price at
                   +112% EV. The cost is the hit rate collapses (0.755^2 = 57%)
                   and variance explodes. For a 5x GOAL that trade can be
                   worth taking; for capital preservation it is not.

Every path is graded on REAL historical outcomes. Parlay legs are always drawn
from DIFFERENT GAMES in the same week, and section 0 measures whether that is
enough to treat them as independent — if big dogs win and lose together within
a week, the parlay maths above is wrong and this file says so.

  python -m backtest.growth_paths
"""
import numpy as np
import pandas as pd

from backtest.bankroll_sim import (BANKROLL, PROPS_EXTRA_HAIRCUT, Q1_MAX_STAKE,
                                   build_book)

N_SIMS = 20000
TARGET = 5000.0
RUIN = 250.0          # practical ruin: 75% drawdown, most people stop here
SEASONS = 3


def leg_correlation(book: pd.DataFrame) -> float:
    """Do derived-line legs win and lose TOGETHER within a week?

    This is the assumption the whole parlay case rests on. If a week where
    favourites cover is a week where ALL our big dogs fail, then legs are
    positively correlated, joint win probability is BELOW p1*p2, and parlays
    are worse than the multiplication says — not better.
    """
    d = book[(book.game_id > 0)].copy()
    d["won"] = (d.pnl > 0).astype(float)
    wk = d.groupby(["season", "week"]).won.agg(["mean", "size"])
    wk = wk[wk["size"] >= 3]
    # variance of weekly win rate vs what independence would give
    p = d.won.mean()
    obs = wk["mean"].var()
    exp = (p * (1 - p) / wk["size"]).mean()
    return float(obs / exp) if exp > 0 else np.nan


def build_parlays(book: pd.DataFrame, legs: int, rng) -> pd.DataFrame:
    """Pair up same-week derived-line bets from DIFFERENT games.

    Priced the way a book actually prices a parlay: multiply the decimal odds.
    Graded the way it actually settles: every leg must win.
    """
    d = book[(book.game_id > 0) & book.play.isin(
        ["Q1 PREMIUM", "BIG-DOG 1H", "SPREAD PREMIUM"])].copy()
    # decimal odds implied by the realised payout on a win
    d["dec"] = np.where(d.pnl > 0, 1 + d.pnl, 1 + 100 / 110)
    d["won"] = (d.pnl > 0).astype(float)
    out = []
    for (season, week), g in d.groupby(["season", "week"]):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        for i in range(0, len(g) - legs + 1, legs):
            chunk = g.iloc[i:i + legs]
            if chunk.game_id.nunique() < legs:      # never two legs, one game
                continue
            won = float(chunk.won.min())            # all must win
            dec = float(chunk.dec.prod())
            out.append({"season": season, "week": week, "game_id": -99,
                        "play": f"PARLAY{legs}", "units": 1.0,
                        "pnl": (dec - 1) if won else -1.0,
                        "dog_full_line": np.nan})
    return pd.DataFrame(out)


def _weekly(book, haircut):
    """Per-week list of (stake_units, pnl) with the play-specific haircut."""
    b = book.copy()
    roi = b.groupby("play").pnl.transform("mean")
    h = np.where(b.play.str.startswith("PROPS"),
                 min(haircut + PROPS_EXTRA_HAIRCUT, 0.95), haircut)
    b["pnl_adj"] = b.pnl - h * roi
    return [g[["units", "pnl_adj", "is_q1"]].assign(
                is_q1=g.play.str.startswith("Q1")).values
            for _, g in b.assign(is_q1=b.play.str.startswith("Q1"))
            .groupby(["season", "week"])]


def simulate(book, unit_pct, haircut, rng, compound: bool):
    """Week-block bootstrap. compound=True stakes off the CURRENT bankroll."""
    blocks = _weekly(book, haircut)
    n_weeks = int(round(len(blocks) / SEASONS))
    finals, peaks_hit, ruined = np.empty(N_SIMS), 0, 0
    for i in range(N_SIMS):
        bank = BANKROLL
        hit = False
        for j in rng.integers(0, len(blocks), n_weeks):
            blk = blocks[j]
            base = bank if compound else BANKROLL
            stake = blk[:, 0] * base * unit_pct
            # Q1's low limits bite in dollars, and they bite harder as the
            # bankroll grows — this is what caps compounding in practice
            stake = np.where(blk[:, 2] > 0, np.minimum(stake, Q1_MAX_STAKE),
                             stake)
            bank += float((stake * blk[:, 1]).sum())
            if bank >= TARGET:
                hit = True
            if bank <= RUIN:
                bank = RUIN
                break
        finals[i] = bank
        peaks_hit += hit
        ruined += bank <= RUIN
    return finals, peaks_hit / N_SIMS, ruined / N_SIMS


def row(label, fin, hit, ruin):
    print(f"{label:<34} {np.median(fin):>8,.0f} {np.percentile(fin, 5):>8,.0f} "
          f"{np.percentile(fin, 95):>9,.0f} {hit:>8.1%} {ruin:>7.1%}")


def main() -> None:
    book = build_book()
    rng = np.random.default_rng(2026)

    print(f"\n=== 0. ARE LEGS INDEPENDENT WITHIN A WEEK? ===")
    vr = leg_correlation(book)
    print(f"  weekly win-rate variance / independence expectation: {vr:.2f}")
    print("  1.0 = independent. >1 means our dogs win and lose TOGETHER, which")
    print("  makes parlays WORSE than the multiplication implies, not better.")

    print(f"\n=== 1. LEVERS, $1,000 -> ${TARGET:,.0f}, 50% haircut "
          f"(70% on props) ===")
    print(f"{'strategy':<34} {'median':>8} {'5th %':>8} {'95th %':>9} "
          f"{'P($5k)':>8} {'P(ruin)':>7}")
    for pct in (0.02, 0.04, 0.06):
        f, h, r = simulate(book, pct, 0.50, rng, compound=False)
        row(f"flat {pct:.0%} of START", f, h, r)
    for pct in (0.02, 0.04, 0.06):
        f, h, r = simulate(book, pct, 0.50, rng, compound=True)
        row(f"COMPOUND {pct:.0%} of current", f, h, r)

    print(f"\n=== 2. ADDING PARLAYS (2- and 3-leg, different games) ===")
    for legs in (2, 3):
        p = build_parlays(book, legs, rng)
        if p.empty:
            continue
        n = len(p) / SEASONS
        roi = p.pnl.mean()
        winr = (p.pnl > 0).mean()
        print(f"  {legs}-leg: {n:.0f}/szn  {winr:.1%} win  {roi:+.1%} ROI  "
              f"t={roi/(p.pnl.std()/np.sqrt(len(p))):+.2f}")
        combo = pd.concat([book, p], ignore_index=True)
        for pct in (0.02, 0.04):
            f, h, r = simulate(combo, pct, 0.50, rng, compound=True)
            row(f"  COMPOUND {pct:.0%} + {legs}-leg parlays", f, h, r)

    print(f"\n=== 3. WHAT WOULD IT TAKE? (0% haircut = backtest at face value) ===")
    for pct in (0.04, 0.06, 0.08):
        f, h, r = simulate(book, pct, 0.0, rng, compound=True)
        row(f"COMPOUND {pct:.0%}, NO haircut", f, h, r)
    print("\n  The 0% rows are NOT a forecast — they assume every measured edge")
    print("  is exactly real. They are the ceiling, not the plan.")


if __name__ == "__main__":
    main()
