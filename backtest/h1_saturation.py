"""Does the 1H spread saturate like the Q1 spread? (free — data already on disk)

The Q1 edge came from a bounded quantity priced with a linear rule: the dog's
Q1 deficit plateaus near 4 pts however big the spread, while books derive the
Q1 line at ~0.289 x full. The 1H margin is bounded too — a half holds ~7-8
possessions a side, so a dog cannot realistically trail by 25 at the break —
yet books derive the 1H line at a flat 0.568 of the full spread.

Our shipped BIG-DOG 1H play requires the MODEL to like the dog by >=6 and is
capped at weeks 1-5. Q1 showed the model filter was not only unnecessary but
slightly harmful, and that the effect ran all season. If 1H has the same
structure we are leaving both volume and ROI on the table.

Screen (from the derived-line rule): compare the book's implied ratio to the
REALISED fair ratio across full-spread buckets. If the realised ratio declines
with spread size while the book's stays flat, the constant is broken.

  python -m backtest.h1_saturation
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import payout
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR
from picks.edge_report import team_matcher

SEASONS = (2023, 2024, 2025)


def consensus_1h() -> pd.DataFrame:
    """Modal 1H line for the DOG, priced only at that number (gotcha #3)."""
    raw = pd.read_parquet(PARQUET_DIR / "historical_1h_lines.parquet")
    raw = raw[(raw.market == "spreads_h1")].dropna(subset=["point", "price"])
    match = team_matcher()
    raw["home_id"] = raw.home.map(match)
    raw["away_id"] = raw.away.map(match)
    raw["team_id"] = raw.name.map(match)
    raw = raw.dropna(subset=["home_id", "away_id", "team_id"]).astype(
        {"home_id": int, "away_id": int, "team_id": int})
    raw["pay"] = payout(raw.price)
    key = ["season", "week", "home_id", "away_id", "team_id"]
    raw = raw.drop_duplicates(subset=key + ["book"])
    modal = raw.groupby(key).point.agg(
        lambda s: s.mode().iloc[0]).rename("h1_line")
    raw = raw.merge(modal, on=key)
    at = raw[raw.point == raw.h1_line]
    return at.groupby(key + ["h1_line"], as_index=False).agg(
        pay_med=("pay", "median"), pay_best=("pay", "max"),
        n_books=("book", "nunique"))


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
    d = d.dropna(subset=["home_id", "away_id"]).astype(
        {"home_id": int, "away_id": int})

    # official 1H margins from the quarter line scores
    g = games[games.homeLineScores.notna() & games.awayLineScores.notna()].copy()

    def h1(ls):
        try:
            return float(ls[0]) + float(ls[1])
        except (TypeError, IndexError, ValueError):
            return np.nan

    g["h_h1"] = g.homeLineScores.apply(h1)
    g["a_h1"] = g.awayLineScores.apply(h1)
    d = d.merge(g[["id", "h_h1", "a_h1"]].astype({"id": "int64"}).dropna(),
                left_on="game_id", right_on="id", how="inner",
                suffixes=("", "_h"))

    d["home_is_dog"] = d.book_spread > 0
    d["model_on_dog"] = np.where(d.home_is_dog, d.edge, -d.edge)
    d["dog_id"] = np.where(d.home_is_dog, d.home_id, d.away_id)
    d["dog_full_line"] = d.book_spread.abs()
    sgn = np.where(d.home_is_dog, 1.0, -1.0)
    d["dog_h1_margin"] = (d.h_h1 - d.a_h1) * sgn

    c = consensus_1h()
    m = d.merge(c, left_on=["season", "week", "home_id", "away_id", "dog_id"],
                right_on=["season", "week", "home_id", "away_id", "team_id"],
                how="inner")
    print(f"games with a 1H line for the dog: {len(m):,}")
    m["covered"] = np.where(m.dog_h1_margin + m.h1_line > 0, 1.0,
                            np.where(m.dog_h1_margin + m.h1_line < 0, 0.0,
                                     np.nan))
    m["pnl_med"] = np.where(m.covered.isna(), 0.0,
                            np.where(m.covered == 1, m.pay_med, -1.0))
    m["pnl_best"] = np.where(m.covered.isna(), 0.0,
                             np.where(m.covered == 1, m.pay_best, -1.0))
    return m


def report(m: pd.DataFrame) -> None:
    print("\n=== THE SCREEN: book's ratio vs the REALISED fair ratio ===")
    v = m[m.dog_full_line >= 3].copy()
    fit = np.polyfit(v.dog_full_line, v.h1_line, 1)
    print(f"book's rule: h1_line = {fit[0]:.4f} * full {fit[1]:+.3f} "
          f"(R^2 {np.corrcoef(v.dog_full_line, v.h1_line)[0,1]**2:.3f})")
    g = v.groupby(pd.cut(v.dog_full_line, [3, 10, 17, 21, 25, 31, 60]),
                  observed=True).agg(
        n=("h1_line", "size"), full=("dog_full_line", "mean"),
        book_h1=("h1_line", "mean"), actual_h1=("dog_h1_margin", "mean"))
    g["fair_line"] = -g.actual_h1
    g["over_grant"] = g.book_h1 - g.fair_line
    g["book_ratio"] = g.book_h1 / g.full
    g["true_ratio"] = g.fair_line / g.full
    print(g.round(3).to_string())
    print("If true_ratio FALLS while book_ratio stays flat, the constant is")
    print("broken and the error grows with spread size — same as Q1.")

    ok = m[m.covered.notna() & (m.n_books >= 2)]
    print(f"\n=== BLIND 1H DOG by spread threshold ({len(ok):,} games) ===")
    rng = np.random.default_rng(3)
    print(f"{'|spread|>=':>10} {'bets':>5} {'/szn':>5} {'win%':>7} {'ROI':>8} "
          f"{'t':>6} {'boot 95% CI':>18} {'by season':>26}")
    for thr in (14, 17, 21, 25, 28, 31):
        b = ok[ok.dog_full_line >= thr]
        if len(b) < 25:
            continue
        p = b.pnl_med.values
        t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
        bs = np.array([rng.choice(p, len(p), replace=True).mean()
                       for _ in range(5000)])
        per = b.groupby("season").pnl_med.mean()
        print(f"{thr:10d} {len(b):5d} {len(b)/3:5.0f} {b.covered.mean():6.1%} "
              f"{p.mean()*100:+7.1f}% {t:+6.2f} "
              f"[{np.percentile(bs, 2.5)*100:+6.1f},"
              f"{np.percentile(bs, 97.5)*100:+6.1f}] "
              + " ".join(f"{per.get(s, np.nan)*100:+7.1f}%" for s in SEASONS))

    print("\n=== does the MODEL filter help or hurt? (Q1 said hurt) ===")
    big = ok[ok.dog_full_line >= 17]
    for lbl, b in [("blind (all big dogs)", big),
                   ("model on dog >=6", big[big.model_on_dog >= 6]),
                   ("model on the FAVOURITE", big[big.model_on_dog < 0])]:
        if len(b) >= 20:
            print(f"  {lbl:24s} {len(b):4d} bets {b.covered.mean():6.1%} "
                  f"{b.pnl_med.mean()*100:+6.1f}%")

    print("\n=== week arc (shipped play is capped at wks 1-5 — justified?) ===")
    a = ok[ok.dog_full_line >= 17].copy()
    a["wk"] = pd.cut(a.week, [0, 5, 10, 15], labels=["1-5", "6-10", "11-15"])
    print(a.groupby("wk", observed=True).agg(
        bets=("pnl_med", "size"), win=("covered", "mean"),
        roi=("pnl_med", "mean")).round(3).to_string())
    a25 = ok[ok.dog_full_line >= 25].copy()
    a25["wk"] = pd.cut(a25.week, [0, 5, 10, 15], labels=["1-5", "6-10", "11-15"])
    print("\nsame for |spread|>=25:")
    print(a25.groupby("wk", observed=True).agg(
        bets=("pnl_med", "size"), win=("covered", "mean"),
        roi=("pnl_med", "mean")).round(3).to_string())


if __name__ == "__main__":
    out = build()
    out.to_parquet(PARQUET_DIR / "backtest_h1_saturation.parquet", index=False)
    report(out)
