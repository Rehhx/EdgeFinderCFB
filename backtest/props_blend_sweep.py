"""Is the props edge a plateau in the blend weight, or a spike at w=0.30?

backtest/props_pricing_stability.py found the pricing calibration contributes
sd 15.6 u/szn — 4.5x the team-volume model — and that tripling the calibration
data only shrinks it 10%, far short of the ~42% that sqrt(3) would predict if
this were ordinary sampling noise.

So the variance is not "not enough data". The suspect is that two of the three
fitted parameters come from COARSE GRID SEARCHES:

    blend weight w   np.arange(0.0, 0.65, 0.05)
    NB dispersion r  (1.5, 2, 3, 4, 5, 6, 8, 10, 15, 25, 50)

A small shift in the calibration data flips w from 0.30 to 0.25 or r from 15 to
25, and w in particular controls how far our probability sits from the book's —
therefore how many bets clear EV>=5% at all. That is a discrete jump no amount
of extra data will smooth out.

Same test that validated the Q1 thresholds: sweep the parameter and look at the
shape. A plateau means the edge is robust to the choice. A spike at exactly the
fitted value means the props edge is a hyperparameter artifact.

Re-blends the already-priced book rather than refitting, so the sweep is exact
and costs one run().

  python -m backtest.props_blend_sweep
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import EXCLUDE_STATS, run

OOS_SEASONS = (2024, 2025)
FITTED_W = 0.30          # what the 2023 log-loss fit currently chooses


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def regrade(m: pd.DataFrame, w: float) -> pd.DataFrame:
    """Re-blend at weight w and re-derive side / EV / pnl exactly as run() does."""
    d = m.copy()
    d["p_bl"] = 1 / (1 + np.exp(-(w * _logit(d.p_over)
                                  + (1 - w) * _logit(d.p_mkt))))
    ev_o = d.p_bl * d.pay_over_best - (1 - d.p_bl)
    ev_u = (1 - d.p_bl) * d.pay_under_best - d.p_bl
    d["side"] = np.where(ev_o >= ev_u, "over", "under")
    d["ev"] = np.maximum(ev_o, ev_u)
    over_won = d.actual > d.line
    push = d.actual == d.line
    d["won"] = np.where(push, np.nan,
                        np.where(d.side == "over", over_won, ~over_won))
    pay = np.where(d.side == "over", d.pay_over_best, d.pay_under_best)
    d["pnl"] = np.where(pd.isna(d.won), 0.0, np.where(d.won == 1, pay, -1.0))
    return d


def book(d: pd.DataFrame) -> tuple[int, float, float]:
    """Shipped gate: EV>=5%, no rec_yds, 2+ books, nothing before wk5,
    STANDARD only from wk9. Returns (bets/szn, units/szn, ROI)."""
    s = d[(~d.stat.isin(EXCLUDE_STATS)) & (d.n_books >= 2) & d.won.notna()
          & (d.ev >= 0.05) & d.season.isin(OOS_SEASONS)].copy()
    prime = s.ev.between(0.05, 0.08, inclusive="left")
    s = s[(s.week >= 5) & ((s.week >= 9) | prime)]
    if s.empty:
        return 0, 0.0, 0.0
    u = np.where(s.ev.between(0.05, 0.08, inclusive="left"), 2.0, 1.0)
    n = len(OOS_SEASONS)
    return len(s) // n, float((u * s.pnl).sum() / n), float(s.pnl.mean())


def main() -> None:
    m = run(write_coefs=False)
    print("\n=== units/season vs the market-blend weight ===")
    print("w = 0 means 'bet the book's own probability' (pure line shopping);")
    print("higher w leans harder on our model.\n")
    print(f"{'w':>6} {'bets/szn':>9} {'ROI':>8} {'units/szn':>10}   "
          f"{'2024':>7} {'2025':>7}")
    rows = []
    for w in np.arange(0.0, 0.65, 0.05):
        d = regrade(m, w)
        n, u, roi = book(d)
        per = []
        for s in OOS_SEASONS:
            _, us, _ = book(d[d.season == s])
            per.append(us)
        mark = "  <- FITTED" if abs(w - FITTED_W) < 1e-9 else ""
        print(f"{w:6.2f} {n:9d} {roi:+8.1%} {u:+10.1f}   "
              f"{per[0]:+7.1f} {per[1]:+7.1f}{mark}")
        rows.append({"w": w, "bets": n, "u": u, "roi": roi,
                     "u24": per[0], "u25": per[1]})

    r = pd.DataFrame(rows)
    best = r.loc[r.u.idxmax()]
    fitted = r[np.isclose(r.w, FITTED_W)].iloc[0]
    print(f"\nfitted w={FITTED_W:.2f} -> {fitted.u:+.1f} u/szn;"
          f"  best on this grid w={best.w:.2f} -> {best.u:+.1f} u/szn")
    span = r[r.w.between(0.15, 0.45)]
    print(f"plateau check, w in [0.15,0.45]: units range "
          f"{span.u.min():+.1f} to {span.u.max():+.1f} "
          f"(spread {span.u.max()-span.u.min():.1f} u)")
    both_pos = (r.u24 > 0) & (r.u25 > 0)
    print(f"w values positive in BOTH seasons: "
          f"{', '.join(f'{x:.2f}' for x in r.w[both_pos])}")
    print("\nREAD: a broad plateau means the edge survives the grid flipping "
          "between\nneighbouring w values, and the sd 15.6 is mostly harmless "
          "churn in WHICH\nbets get taken. A spike at the fitted value means "
          "the props edge is a\nhyperparameter artifact and should be sized "
          "far more cautiously.")
    robust(m)


ROBUST_W = (0.25, 0.30, 0.35, 0.40)


def robust(m: pd.DataFrame) -> None:
    """Bet only what survives the hyperparameter choice.

    Picking a better w is just fitting the grid harder. The defensible move is
    to require a bet to clear EV>=5% at EVERY w in a band around the fitted
    value, AND to want the same side at each — so the bets we take are the ones
    whose edge does not depend on which grid point the log-loss search landed
    on. Graded at the shipped w so the prices are the ones we would really get.
    """
    print(f"\n=== w-ROBUST selection: must clear EV>=5% at every w in "
          f"{ROBUST_W} ===")
    graded = {w: regrade(m, w) for w in ROBUST_W}
    base = graded[FITTED_W]
    ok = np.ones(len(base), bool)
    for w in ROBUST_W:
        g = graded[w]
        ok &= (g.ev.values >= 0.05) & (g.side.values == base.side.values)
    d = base[ok].copy()
    d["ev"] = np.minimum.reduce([graded[w].ev.values[ok] for w in ROBUST_W])

    n, u, roi = book(d)
    print(f"  all seasons pooled: {n} bets/szn, {roi:+.1%} ROI, {u:+.1f} u/szn")
    for s in OOS_SEASONS:
        ns, us, rs = book(d[d.season == s])
        print(f"    {s}: {ns:4d} bets  {rs:+6.1%} ROI  {us:+6.1f} u")

    print("\n  sensitivity of the ROBUST book to w (grade it at each w):")
    for w in ROBUST_W:
        g = graded[w][ok].copy()
        g["ev"] = d.ev.values
        _, uw, rw = book(g)
        print(f"    graded at w={w:.2f}: {rw:+6.1%} ROI  {uw:+6.1f} u/szn")
    print("\n  COMPARE the naive book, which swings +0.4 -> +54.2 across the "
          "same w band.\n  If this row is flat, the selection is robust and "
          "the level is trustworthy.")


if __name__ == "__main__":
    main()
