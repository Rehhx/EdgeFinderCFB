"""First-quarter TOTAL — ⚠️ REJECTED. Books price this one correctly.

VERDICT (2026-08-03): no edge. **Books use a ratio of 0.2152** (sd 0.014) while
the true Q1 share of regulation points is 0.2250 — they are already slightly
CONSERVATIVE, not lazy. The naive 0.25 this test was built to exploit is not
what they use. Over-grant flips sign across total buckets (-0.18, -0.75, +0.77,
+0.66): no systematic bias.
  BLIND Q1 UNDER: 1,664 bets, 53.7%, **+1.3% ROI**, t=+0.58,
  bootstrap 95% CI [-3.3, +5.9] -> includes zero.
  seasons -3.4% / +5.7% / +1.8% (inconsistent); by week +6.1/-0.8/-1.4.
The OVER control at -12.9% shows Q1 scoring really does run under — but the
PRICE already absorbs it. That is a correctly-priced market, not an edge.

WHY THIS FAILS WHERE THE Q1 SPREAD PAYS — the sharpened rule:
a derived constant only breaks when the sub-period quantity is BOUNDED in a way
the parent quantity is not.
  Q1 SPREAD: full-game margin is unbounded, but Q1 margin saturates near 4 pts
    (a quarter holds ~3-4 possessions a side). A linear rule must over-grant at
    the extremes, and the error GROWS with the spread -> big edge.
  Q1 TOTAL: Q1 points and full-game points scale together — no ceiling. The
    real ratio is stable (0.219 at low totals, 0.233 at high), so a flat ratio
    is the RIGHT model and books use a good one -> no edge.
So don't hunt constants generally; hunt constants applied to a quantity with a
CEILING its parent lacks. See backtest/q1_spreads.py for the paying case.

---
Original hypothesis and method below.

First-quarter TOTAL: is the derived Q1 number set too high?

Same playbook as the Q1 spread edge (backtest/q1_spreads.py) — hunt a CONSTANT
that reality varies. Free mechanism test over 20,555 games with quarter scores:

    share of regulation points   Q1 22.50%   Q2 30.44%   Q3 22.45%   Q4 24.65%

Q1 is the LOWEST-scoring quarter (scripted openers, fresh defences, no
two-minute drill) while Q2 is the highest. A naive 25% derivation therefore
overstates the Q1 total by ~1.3 points at every total level (40-50: 10.03
actual vs 11.36; 70+: 18.90 vs 20.31). Books shaded the 1H high (0.568 vs a
true 0.504) and the Q1 spread high (0.289 linear), so the prior is that they
shade this high too -> bet the UNDER.

The decisive question, answered first: what ratio do books ACTUALLY use? If
they already price ~0.225 there is no edge and the rest is noise.

Controls: blind (no model at all — this is a pure market-structure test),
median AND best price, per-season, total bucket, week arc.

  python -m backtest.q1_totals
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import payout
from backtest.q1_spreads import quarter_scores
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR
from picks.edge_report import team_matcher

SEASONS = (2023, 2024, 2025)
DEST = PARQUET_DIR / "backtest_q1_totals.parquet"


def consensus() -> pd.DataFrame:
    """Modal Q1 total, with Over/Under prices taken only from books at that
    exact number (gotcha #3 — never pair a line with another line's price)."""
    raw = pd.read_parquet(PARQUET_DIR / "historical_q1_totals.parquet")
    raw = raw.dropna(subset=["point", "price"])
    match = team_matcher()
    raw["home_id"] = raw.home.map(match)
    raw["away_id"] = raw.away.map(match)
    raw = raw.dropna(subset=["home_id", "away_id"]).astype(
        {"home_id": int, "away_id": int})
    raw["pay"] = payout(raw.price)
    key = ["season", "week", "home_id", "away_id"]

    piv = raw.pivot_table(index=key + ["book", "point"], columns="name",
                          values="pay", aggfunc="first").reset_index()
    if not {"Over", "Under"} <= set(piv.columns):
        raise SystemExit("Q1 totals pull has no Over/Under outcomes")
    piv = piv.dropna(subset=["Over", "Under"])
    modal = piv.groupby(key).point.agg(
        lambda s: s.mode().iloc[0]).rename("q1_total")
    piv = piv.merge(modal, on=key)
    at = piv[piv.point == piv.q1_total]
    return at.groupby(key + ["q1_total"], as_index=False).agg(
        over_med=("Over", "median"), under_med=("Under", "median"),
        over_best=("Over", "max"), under_best=("Under", "max"),
        n_books=("book", "nunique"))


def build() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_spread_history.parquet")
    d = d[d.season.isin(SEASONS) & d.book_total.notna()].copy()
    d["game_id"] = d.game_id.astype("int64")
    games = pd.concat([pd.read_parquet(CFBD_PARQUET_DIR / f"games_{s}.parquet")
                       for s in SEASONS])
    t = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    s2i = dict(zip(t.school, t.id))
    games["home_id"] = games.homeTeam.map(s2i)
    games["away_id"] = games.awayTeam.map(s2i)
    d = d.merge(games[["id", "home_id", "away_id"]].astype({"id": "int64"}),
                left_on="game_id", right_on="id", how="inner")
    d = d.dropna(subset=["home_id", "away_id"]).astype(
        {"home_id": int, "away_id": int})
    q = quarter_scores()
    d = d.merge(q, left_on="game_id", right_on="id", how="inner",
                suffixes=("", "_q"))
    d["q1_points"] = d.h_q1 + d.a_q1

    c = consensus()
    m = d.merge(c, on=["season", "week", "home_id", "away_id"], how="inner")
    print(f"games with a Q1 total: {len(m):,}")
    m["ratio"] = m.q1_total / m.book_total
    m["over_won"] = np.where(m.q1_points > m.q1_total, 1.0,
                             np.where(m.q1_points < m.q1_total, 0.0, np.nan))
    m["pnl_under_med"] = np.where(m.over_won.isna(), 0.0,
                                  np.where(m.over_won == 0, m.under_med, -1.0))
    m["pnl_under_best"] = np.where(m.over_won.isna(), 0.0,
                                   np.where(m.over_won == 0, m.under_best, -1.0))
    m["pnl_over_med"] = np.where(m.over_won.isna(), 0.0,
                                 np.where(m.over_won == 1, m.over_med, -1.0))
    return m


def report(m: pd.DataFrame) -> None:
    print("\n=== WHAT RATIO DO BOOKS ACTUALLY USE? (the thesis) ===")
    print(f"mean Q1_total / full_total = {m.ratio.mean():.4f} "
          f"(sd {m.ratio.std():.4f})")
    print("reality from 20,555 games: Q1 takes 22.50% of regulation points\n")
    v = m.copy()
    v["tb"] = pd.cut(v.book_total, [0, 45, 52, 59, 200])
    g = v.groupby("tb", observed=True).agg(
        n=("ratio", "size"), full=("book_total", "mean"),
        book_q1=("q1_total", "mean"), actual_q1=("q1_points", "mean"))
    g["book_ratio"] = g.book_q1 / g.full
    g["over_grant"] = g.book_q1 - g.actual_q1
    print(g.round(3).to_string())
    print("over_grant = points the book's Q1 total sits ABOVE actual Q1 scoring")

    ok = m[m.over_won.notna() & (m.n_books >= 2)]
    print(f"\n=== BLIND Q1 UNDER, every game ({len(ok):,}) ===")
    print(f"{'':16} {'bets':>6} {'win%':>7} {'ROI med':>9} {'ROI best':>10}")
    u = ok.pnl_under_med
    print(f"{'UNDER':16} {len(ok):6d} {(ok.over_won == 0).mean():6.1%} "
          f"{u.mean()*100:+8.1f}% {ok.pnl_under_best.mean()*100:+9.1f}%")
    print(f"{'OVER (control)':16} {len(ok):6d} {ok.over_won.mean():6.1%} "
          f"{ok.pnl_over_med.mean()*100:+8.1f}%")
    print(f"push rate: {m.over_won.isna().mean():.1%}")

    if len(ok):
        p = ok.pnl_under_med.values
        t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
        rng = np.random.default_rng(5)
        bs = np.array([rng.choice(p, len(p), replace=True).mean()
                       for _ in range(5000)])
        print(f"\nUNDER t={t:+.2f}  bootstrap 95% CI "
              f"[{np.percentile(bs, 2.5)*100:+.1f}, "
              f"{np.percentile(bs, 97.5)*100:+.1f}]")
        print("\nby season:")
        print(ok.groupby("season").agg(
            bets=("pnl_under_med", "size"),
            win=("over_won", lambda s: (s == 0).mean()),
            roi=("pnl_under_med", "mean")).round(3).to_string())
        ok = ok.copy()
        ok["tb"] = pd.cut(ok.book_total, [0, 45, 52, 59, 200])
        print("\nby full-game total:")
        print(ok.groupby("tb", observed=True).agg(
            bets=("pnl_under_med", "size"),
            win=("over_won", lambda s: (s == 0).mean()),
            roi=("pnl_under_med", "mean")).round(3).to_string())
        ok["wk"] = pd.cut(ok.week, [0, 5, 10, 15])
        print("\nby week:")
        print(ok.groupby("wk", observed=True).agg(
            bets=("pnl_under_med", "size"),
            win=("over_won", lambda s: (s == 0).mean()),
            roi=("pnl_under_med", "mean")).round(3).to_string())


if __name__ == "__main__":
    out = build()
    out.to_parquet(DEST, index=False)
    report(out)
    print(f"\nsaved: {DEST}")
