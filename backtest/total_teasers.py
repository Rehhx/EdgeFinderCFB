"""Teasers on sub-period TOTALS — hunting the next bounded market.

backtest/teasers.py established the rule: a teaser pays when the distribution
you are buying points against is COMPRESSED, and sub-period margins are
compressed because possessions are capped. Full-game teasers pay ~3x less than
1H teasers for exactly that reason.

Totals are the obvious next market, and they are the SHARPEST possible test of
the thesis. We already PROVED the straight Q1 total has no edge — books price
the ratio at 0.2152 against a true 0.2250, i.e. conservative, not lazy
(backtest/q1_totals.py). So if a Q1 total TEASER is +EV, it cannot be coming
from a pricing error in the straight number. It can only be the distribution.

The mechanism in one number: **the tease measured in standard deviations.**
A 10-point tease against a full-game total (sd ~14) buys 0.7 sd. The same 10
points against a Q1 total (sd ~7) buys 1.4 sd — twice the distribution for the
same price. That is the whole thesis, and this file measures it.

RESULT (2026-08-04): the thesis is CONFIRMED and quantified — residual sd runs
Q1 6.99 < 1H 10.74 < full-game 15.94, so the same +6 buys 0.86sd / 0.56sd /
0.38sd and the leg hit rate falls 81.1% / 70.0% / 63.1% in lockstep. Full-game
total teasers are strongly NEGATIVE (-19.5% EV) where Q1 is strongly positive.

BUT THE PRICE IS THE CONSTRAINT, and this is the refinement that matters:
**boundedness tells you where the points are worth MOST; it does not tell you
the book is SELLING them cheaply.** Applying full-game-spread teaser prices to
a Q1 total is not a real quote. Break-evens on the UNDER side are -194 (+6),
-240 (+7), -472 (+10) for a 2-teamer, and -429 / -975 as single alt-lines.
Those are the numbers to shop against; we do not currently hold alt Q1 total
prices to test them (our pull has exactly one point per game/book/side).
At realistic prices the 1H TOTAL teaser lands ~65% joint vs a 64.3% break-even
— essentially a coin flip, not a play. Only the 1H SPREAD teaser survives real
pricing, because books apply standard full-game teaser prices to it.

⚠️ ZERO-FLOOR ARTIFACT: teasing an OVER down on a small total quickly becomes
unloseable — at +10 the teased line is <=0 on 33.8% of Q1 games, at +14 on
97.7%. Those rows are arithmetic, not edge. Only the UNDER side is meaningful.

  python -m backtest.total_teasers
"""
import numpy as np
import pandas as pd

from ingestion.config import PARQUET_DIR

TEASE = (6.0, 7.0, 10.0, 14.0)
PRICE_2TEAM = {6.0: -110, 7.0: -130, 10.0: -180, 14.0: -250}
SEASONS = 3


def payout(american: float) -> float:
    return 100 / abs(american) if american < 0 else american / 100


def markets() -> dict:
    """(actual, line) pairs for each period, plus how the line is sourced."""
    q = pd.read_parquet(PARQUET_DIR / "backtest_q1_totals.parquet")
    q = q[q.q1_points.notna() & q.q1_total.notna() & (q.n_books >= 2)]
    out = {"Q1 total": (q.q1_points.values, q.q1_total.values,
                        q.season.values, q.week.values, q.game_id.values)}

    # 1H total: actual from the graded 1H parquet, line from the raw pull.
    # Odds-API team names differ from CFBD's, so the join MUST go through
    # team_matcher() — joining on raw name strings silently drops everything.
    from picks.edge_report import team_matcher
    match = team_matcher()
    h = pd.read_parquet(PARQUET_DIR / "backtest_h1_saturation.parquet")
    h = h[h.h_h1.notna() & h.a_h1.notna()].drop_duplicates("game_id").copy()
    h["h1_points"] = h.h_h1 + h.a_h1
    raw = pd.read_parquet(PARQUET_DIR / "historical_1h_lines.parquet")
    raw = raw[(raw.market == "totals_h1") & raw.point.notna()
              & (raw.name == "Under")].copy()
    raw["home_id"] = raw.home.map(match)
    raw["away_id"] = raw.away.map(match)
    raw = raw.dropna(subset=["home_id", "away_id"])
    key = ["season", "week", "home_id", "away_id"]
    line = raw.groupby(key, as_index=False).point.median()
    h = h.merge(line, on=key, how="inner")
    if len(h) >= 100:
        out["1H total"] = (h.h1_points.values, h.point.values,
                           h.season.values, h.week.values, h.game_id.values)

    # full game — the CONTROL. Unbounded, so the tease should buy less.
    f = pd.read_parquet(PARQUET_DIR / "backtest_q1_totals.parquet")
    f = f[f.actual_total.notna() & f.book_total.notna()]
    out["FULL total (control)"] = (f.actual_total.values, f.book_total.values,
                                   f.season.values, f.week.values,
                                   f.game_id.values)
    return out


def describe(actual, line, label) -> float:
    resid = actual - line
    sd = float(np.std(resid))
    print(f"\n=== {label}  (n={len(actual)}) ===")
    print(f"  mean line {np.mean(line):5.1f}   mean actual {np.mean(actual):5.1f}"
          f"   residual sd {sd:5.2f}   bias {np.mean(resid):+5.2f}")
    return sd


def leg_rates(actual, line, sd, label) -> dict:
    print(f"  tease   in sd    UNDER hit   OVER hit    best")
    out = {}
    for pts in TEASE:
        u = float((actual < line + pts).mean())
        o = float((actual > line - pts).mean())
        best = max(u, o)
        out[pts] = (u, o)
        print(f"  +{pts:4.1f}  {pts/sd:5.2f}sd   {u:8.1%}   {o:8.1%}   "
              f"{'UNDER' if u >= o else 'OVER':>5}")
    return out


def two_team(actual, line, season, week, game_id, pts, price, side, rng):
    """Real same-week, different-game pairs — no independence assumption."""
    won = (actual < line + pts) if side == "under" else (actual > line - pts)
    d = pd.DataFrame({"season": season, "week": week, "game_id": game_id,
                      "won": won.astype(float)})
    rows = []
    for _, g in d.groupby(["season", "week"]):
        g = g.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        for i in range(0, len(g) - 1, 2):
            a, b = g.iloc[i], g.iloc[i + 1]
            if a.game_id == b.game_id:
                continue
            rows.append({"season": a.season, "won": min(a.won, b.won)})
    r = pd.DataFrame(rows)
    if len(r) < 20:
        return None
    pay = payout(price)
    pnl = r.won * pay - (1 - r.won)
    per = r.groupby("season").won.mean()
    return {"n": len(r), "true": float(r.won.mean()), "ev": float(pnl.mean()),
            "t": float(pnl.mean() / (pnl.std() / np.sqrt(len(r)))),
            "allpos": bool(((per * pay - (1 - per)) > 0).all()),
            "per": per.round(3).to_dict()}


def main() -> None:
    rng = np.random.default_rng(7)
    mk = markets()
    print("THESIS: a teaser buys a fixed number of POINTS. What matters is how")
    print("much of the DISTRIBUTION those points cover — i.e. points/sd. Fewer")
    print("possessions = tighter distribution = the same 10 points buys more.")

    for label, (actual, line, season, week, gid) in mk.items():
        sd = describe(actual, line, label)
        rates = leg_rates(actual, line, sd, label)
        print(f"\n  2-team teasers, REAL same-week pairs:")
        for pts in TEASE:
            price = PRICE_2TEAM[pts]
            u, o = rates[pts]
            side = "under" if u >= o else "over"
            res = two_team(actual, line, season, week, gid, pts, price, side, rng)
            if res is None:
                continue
            per = " ".join(f"{v:.0%}" for v in res["per"].values())
            flag = "ALL+" if res["allpos"] else "MIXED"
            print(f"    +{pts:4.1f} {side:<5} @{price:>5}  n={res['n']:3d}  "
                  f"TRUE {res['true']:5.1%}  EV {res['ev']:+6.1%}  "
                  f"t={res['t']:+5.2f}  {flag}  [{per}]")

    print("\nREAD: compare points/sd across the three. If the Q1/1H rows buy")
    print("more sd per point AND clear +EV where the full-game control does")
    print("not, the bounded-market teaser thesis generalises beyond spreads.")
    print("\n⚠️ AVAILABILITY: sub-period TOTAL teasers are rarer than spread")
    print("teasers. Confirm the market exists before treating any of this as")
    print("placeable — the edge is worthless if you cannot get the ticket down.")


if __name__ == "__main__":
    main()
