"""First-quarter spread on big underdogs — ✅ VALIDATED, the strongest edge found.

RESULT (2026-08-03): take the BIG UNDERDOG on the Q1 spread.
  |spread|>=17: 102 bets/szn, 61.9%, +20.0% ROI, t=+3.70, CI [+9.2,+30.4]
  |spread|>=25:  31 bets/szn, 75.5%, +45.7% ROI, t=+5.27, CI [+28.8,+62.2]
Every threshold positive in all 3 seasons; every bootstrap CI excludes zero;
monotone in spread size exactly as the mechanism predicts. Worst 10-bet run at
>=25 is -0.6u; longest losing streak 4.

⚠️ This is a MARKET-STRUCTURE edge, NOT our model's. Blind (bet every big dog)
beats model-selected (75.5%/+45.7% vs 73.3%/+42.1%) — the ratings add nothing
and slightly hurt. Do not dress this up as model skill.
⚠️ PRACTICAL LIMIT: Q1 spreads on 25+ point mismatches are low-limit markets.
Expect small maximums. That is likely WHY it is mispriced — the handle does not
justify careful modelling, which is exactly "where the market is lazy".

WHY IT WORKS — a linear rule against a saturating reality:
books derive the Q1 line at ~0.289 x the full spread (linear, R^2 0.85). But
the dog's ACTUAL Q1 deficit plateaus around 4 points no matter how large the
spread — a quarter holds ~3-4 possessions per side, so you cannot lose by 30 in
it. Full-game margin is unbounded; Q1 margin is not. So the linear rule
over-grants the dog +1.9 pts at a 19-pt spread, +3.9 at 28, +4.3 at 37.
This is the same class of error as the 1H constant (56.8% vs a true 50.4%),
found by the rule that came out of the alt-spread rejection: hunt CONSTANTS —
places where a derivation holds something fixed that reality varies.

---
Original hypothesis and method below.

First-quarter spread on big-dog games — the sharper version of the 1H edge.

Thesis chain:
  1. The 1H edge is a BROKEN CONSTANT: books derive the 1H line at 56.8% of the
     full spread while the big dog's actual 1H share is ~50.4%.
  2. Quarter scores say the dog's outperformance is concentrated in Q1 (+1.62
     pts vs an even-split line) and Q4 (+1.81), with ~nothing in Q2 (+0.04) or
     Q3 (+0.09). The 1H edge is really a Q1 edge diluted by a dead Q2.
  3. So the Q1 line should carry the same mispricing in concentrated form.

The decisive question this backtest answers FIRST: is the book's Q1/full ratio
actually a CONSTANT? If books vary it sensibly by matchup, there is no formula
error to exploit and the rest is noise (see backtest/alt_spreads.py, where the
ladder turned out to be correctly priced and the "edge" moved between subsets).

Controls, all mandatory: blind (no model selection), median AND best price,
per-season, and the week arc.

  python -m backtest.q1_spreads
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import payout
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR
from picks.edge_report import team_matcher

SEASONS = (2023, 2024, 2025)
BIG_SPREAD = 17.0
MIN_EDGE = 6.0
DEST = PARQUET_DIR / "backtest_q1_spreads.parquet"


def consensus_q1() -> pd.DataFrame:
    """Consensus Q1 line + prices quoted AT THAT LINE.

    GOTCHA (#3, new guise): books do NOT all quote the same Q1 point. 79% of
    game-teams have 2+ distinct points, mean spread 1.23 pts, e.g. +3.5 at -110
    from three books and +5.5 at -130 from five. Taking median(point) and
    median(price) — let alone max(price) — independently pairs a long line with
    a short line's price and grades a combination NO BOOK OFFERED. That is what
    inflated the first run's "best price" ROI.

    Correct construction: pick the modal point across books, then take the
    median and best price only among books quoting exactly that point.
    """
    raw = pd.read_parquet(PARQUET_DIR / "historical_q1_lines.parquet")
    raw = raw.dropna(subset=["point", "price"])
    match = team_matcher()
    raw["home_id"] = raw.home.map(match)
    raw["away_id"] = raw.away.map(match)
    raw["team_id"] = raw.name.map(match)
    raw = raw.dropna(subset=["home_id", "away_id", "team_id"]).astype(
        {"home_id": int, "away_id": int, "team_id": int})
    raw["pay"] = payout(raw.price)
    key = ["season", "week", "home_id", "away_id", "team_id"]
    # one row per book (a book should quote one point; keep its first)
    raw = raw.drop_duplicates(subset=key + ["book"])
    modal = raw.groupby(key).point.agg(
        lambda s: s.mode().iloc[0]).rename("q1_line")
    raw = raw.merge(modal, on=key)
    at = raw[raw.point == raw.q1_line]
    return at.groupby(key + ["q1_line"], as_index=False).agg(
        pay_med=("pay", "median"), pay_best=("pay", "max"),
        n_books=("book", "nunique"))


def quarter_scores() -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(CFBD_PARQUET_DIR / f"games_{s}.parquet")
                   for s in SEASONS])
    g = g[g.homeLineScores.notna() & g.awayLineScores.notna()].copy()

    def q(ls, i):
        try:
            return float(ls[i])
        except (TypeError, IndexError, ValueError):
            return np.nan

    g["h_q1"] = g.homeLineScores.apply(lambda x: q(x, 0))
    g["a_q1"] = g.awayLineScores.apply(lambda x: q(x, 0))
    return g[["id", "h_q1", "a_q1"]].astype({"id": "int64"}).dropna()


def build() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_spread_history.parquet")
    d = d[d.season.isin(SEASONS) & d.margin.notna()
          & d.book_spread.notna()].copy()
    d["game_id"] = d.game_id.astype("int64")

    games = pd.concat([pd.read_parquet(CFBD_PARQUET_DIR / f"games_{s}.parquet")
                       for s in SEASONS])
    t = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    s2i = dict(zip(t.school, t.id))
    games["home_id"] = games.homeTeam.map(s2i)
    games["away_id"] = games.awayTeam.map(s2i)
    d = d.merge(games[["id", "home_id", "away_id"]].astype({"id": "int64"}),
                left_on="game_id", right_on="id", how="inner")
    # FBS-vs-FCS games have no odds-side id (see alt_spreads coverage note)
    d = d.dropna(subset=["home_id", "away_id"]).astype(
        {"home_id": int, "away_id": int})
    d = d.merge(quarter_scores(), left_on="game_id", right_on="id",
                how="inner", suffixes=("", "_q"))

    d["home_is_dog"] = d.book_spread > 0
    d["model_on_dog"] = np.where(d.home_is_dog, d.edge, -d.edge)
    d["dog_id"] = np.where(d.home_is_dog, d.home_id, d.away_id)
    d["dog_full_line"] = d.book_spread.abs()
    sgn = np.where(d.home_is_dog, 1.0, -1.0)
    d["dog_q1_margin"] = (d.h_q1 - d.a_q1) * sgn

    q1 = consensus_q1()
    m = d.merge(q1, left_on=["season", "week", "home_id", "away_id", "dog_id"],
                right_on=["season", "week", "home_id", "away_id", "team_id"],
                how="inner")
    print(f"games with a Q1 line for the dog: {len(m):,}")

    # the smoking-gun measurement
    m["q1_ratio"] = m.q1_line.abs() / m.dog_full_line.replace(0, np.nan)
    m["covered"] = np.where(m.dog_q1_margin + m.q1_line > 0, 1.0,
                            np.where(m.dog_q1_margin + m.q1_line < 0, 0.0,
                                     np.nan))
    m["pnl_med"] = np.where(m.covered.isna(), 0.0,
                            np.where(m.covered == 1, m.pay_med, -1.0))
    m["pnl_best"] = np.where(m.covered.isna(), 0.0,
                             np.where(m.covered == 1, m.pay_best, -1.0))
    m["big"] = m.dog_full_line >= BIG_SPREAD
    m["picked"] = m.big & (m.model_on_dog >= MIN_EDGE) & m.week.between(1, 5)
    return m


def report(m: pd.DataFrame) -> None:
    print("\n=== THE MECHANISM: a LINEAR rule against a SATURATING reality ===")
    v = m[m.dog_full_line >= 3].copy()
    fit = np.polyfit(v.dog_full_line, v.q1_line, 1)
    r2 = np.corrcoef(v.dog_full_line, v.q1_line)[0, 1] ** 2
    print(f"book's rule:  q1_line = {fit[0]:.4f} * full {fit[1]:+.3f} "
          f"(linear R^2 {r2:.3f})")
    g = v.groupby(pd.cut(v.dog_full_line, [3, 10, 17, 21, 25, 31, 60]),
                  observed=True).agg(
        n=("q1_line", "size"), full=("dog_full_line", "mean"),
        book_q1=("q1_line", "mean"), actual_q1=("dog_q1_margin", "mean"))
    g["fair_line"] = -g.actual_q1
    g["over_grant"] = g.book_q1 - g.fair_line
    print(g.round(2).to_string())
    print("The dog's ACTUAL Q1 deficit plateaus near 4 pts however big the")
    print("spread gets — one quarter has ~3-4 possessions, so you cannot lose")
    print("by 30 in it. The book keeps extending the line linearly, so it")
    print("over-grants points to the dog, and the error GROWS with the spread.")

    print("\n=== EDGE BY SPREAD THRESHOLD (blind: no model selection) ===")
    rng = np.random.default_rng(11)
    ok = m[m.covered.notna() & (m.n_books >= 2)]
    print(f"{'|spread|>=':>10} {'bets':>5} {'/szn':>5} {'win%':>7} {'ROI':>8} "
          f"{'t':>6} {'boot 95% CI':>18} {'by season':>26}")
    for thr in (14, 17, 21, 25, 28):
        d = ok[ok.dog_full_line >= thr]
        if len(d) < 25:
            continue
        p = d.pnl_med.values
        t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
        bs = np.array([rng.choice(p, len(p), replace=True).mean()
                       for _ in range(5000)])
        per = d.groupby("season").pnl_med.mean()
        print(f"{thr:10d} {len(d):5d} {len(d)/3:5.0f} {d.covered.mean():6.1%} "
              f"{p.mean()*100:+7.1f}% {t:+6.2f} "
              f"[{np.percentile(bs, 2.5)*100:+6.1f},"
              f"{np.percentile(bs, 97.5)*100:+6.1f}] "
              + " ".join(f"{per.get(s, np.nan)*100:+7.1f}%" for s in SEASONS))

    sel = m[m.picked & m.covered.notna() & (m.n_books >= 2)]
    blind = m[m.big & m.week.between(1, 5) & m.covered.notna()
              & (m.n_books >= 2)]
    print(f"\n=== BIG-DOG Q1 (wks1-5, |spread|>=17, model on dog >=6) ===")
    print(f"{'':22} {'bets':>6} {'win%':>7} {'ROI med':>9} {'ROI best':>10}")
    for lbl, b in [("MODEL-SELECTED", sel), ("BLIND (no model)", blind)]:
        if len(b):
            print(f"{lbl:22} {len(b):6d} {b.covered.mean():6.1%} "
                  f"{b.pnl_med.mean()*100:+8.1f}% {b.pnl_best.mean()*100:+9.1f}%")
    if len(sel) and len(blind):
        print(f"{'selection value':22} {'':6} {'':7} "
              f"{(sel.pnl_med.mean()-blind.pnl_med.mean())*100:+8.1f}pp")

    print("\nby season (model-selected):")
    print(sel.groupby("season").agg(
        bets=("pnl_med", "size"), win=("covered", "mean"),
        roi_med=("pnl_med", "mean"), roi_best=("pnl_best", "mean")
    ).round(3).to_string())

    print("\n=== week arc (big dogs, model-selected, all weeks) ===")
    a = m[m.big & (m.model_on_dog >= MIN_EDGE) & m.covered.notna()
          & (m.n_books >= 2)].copy()
    a["wk"] = pd.cut(a.week, [0, 5, 10, 15], labels=["1-5", "6-10", "11-15"])
    print(a.groupby("wk", observed=True).agg(
        bets=("pnl_med", "size"), win=("covered", "mean"),
        roi=("pnl_med", "mean")).round(3).to_string())

    print("\n=== generic Q1 (any game, no big-dog filter, wks1-5) ===")
    for thr in (2, 4, 6):
        b = m[(m.model_on_dog >= thr) & m.week.between(1, 5)
              & m.covered.notna() & (m.n_books >= 2)]
        if len(b) >= 40:
            print(f"  dog-edge>={thr}: {len(b):5d} bets {b.covered.mean():6.1%} "
                  f"{b.pnl_med.mean()*100:+6.1f}%")
    print(f"\npush rate (Q1 tie): {m.covered.isna().mean():.1%}")


if __name__ == "__main__":
    out = build()
    out.to_parquet(DEST, index=False)
    report(out)
    print(f"\nsaved: {DEST}")
