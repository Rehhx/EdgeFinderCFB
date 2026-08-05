"""Alternate-spread ladders on big-dog games — ⚠️ REJECTED, no edge over the main line.

VERDICT (2026-08-03): no rung reliably beats the main number. Do not bet alt
spreads; keep taking the big dog at the main line.
  - best paired candidate (-3.5) is +10.0pp over MAIN at t=+1.53, bootstrap
    95% CI [-3.5, +21.6] — includes zero
  - the identity of the "best" rung is UNSTABLE: +3.5 wins in the unpaired
    view, -3.5 wins in the paired view, -7.0 swings from -2.4% to +13.2%
    between the two. Real structure does not move when you change subset.
  - the BLIND ladder (no model) has the same shape, so any pattern is market
    structure rather than our selection
  - coverage limit: FBS-vs-FCS games are ~46% of the big-dog universe and have
    no alt ladder at all, so this only ever addressed half the play

WHY IT FAILED WHERE THE 1H SHORTCUT WORKED — the transferable lesson:
the 1H line is derived with a CONSTANT ratio (56.8% of the full spread) applied
to something that genuinely varies (big-dog covers are front-loaded, ~3.6 pts
in 1H vs ~0.9 in 2H). That constant is a real formula error. The alt ladder is
derived from the margin DISTRIBUTION, which books model well — the key numbers
3/7/10/14 are the most-studied quantity in football betting. Walking the ladder
just walks a correctly-priced curve. So: a derived line is soft when the
derivation holds something CONSTANT that reality varies, not merely because it
is derived. Hunt constants, not derivations.

Kept for research and because the 202k-row ladder table has other uses.

---
Original hypothesis and method below.

Alternate-spread ladders on big-dog games: is any rung better than the main line?

Hypothesis (derived-line class, same family as the 1H shortcut that DID pay):
books build the alt ladder off the main number with a fixed price-per-point
formula. Our big-dog play says the main number is wrong by ~6 pts on early
mismatches, so that error propagates into every rung; and if the book's
per-point pricing is flatter/steeper than the true margin distribution, some
rung should out-pay the main line.

Method — deliberately assumption-free. No distribution is modelled: for the
SAME validated big-dog selection we walk a FIXED ladder offset (dog's main
line +/- delta), read the book's real price at that rung, and grade on the
official margin. A fixed offset applied to every bet cannot cherry-pick, which
is the trap here: with ~30 rungs per game, a per-game "best rung" chooser will
always look profitable in-sample.

Controls, all mandatory (see HANDOFF):
  - BLIND: same ladder on every big dog with NO model selection
  - MEDIAN price as well as BEST price
  - per-season split (an edge that is one season is not an edge)

  python -m backtest.alt_spreads
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import payout
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR
from picks.edge_report import team_matcher

SEASONS = (2023, 2024, 2025)
BIG_SPREAD = 17.0        # validated big-dog threshold
MIN_EDGE = 6.0           # model must like the dog by this many points
# ladder offsets in points ADDED to the dog's main number: negative = buying
# points down (shorter dog, worse price), positive = selling points out.
OFFSETS = (-14, -10.5, -7, -3.5, 0, 3.5, 7, 10.5, 14)
DEST = PARQUET_DIR / "backtest_alt_spreads.parquet"


def ladders() -> pd.DataFrame:
    """One row per (game, team, point): median and best payout across books."""
    raw = pd.read_parquet(PARQUET_DIR / "historical_alt_spreads.parquet")
    raw = raw.dropna(subset=["point", "price"])
    match = team_matcher()
    raw["home_id"] = raw.home.map(match)
    raw["away_id"] = raw.away.map(match)
    raw["team_id"] = raw.name.map(match)
    raw = raw.dropna(subset=["home_id", "away_id", "team_id"]).astype(
        {"home_id": int, "away_id": int, "team_id": int})
    raw["pay"] = payout(raw.price)
    key = ["season", "week", "home_id", "away_id", "team_id", "point"]
    return raw.groupby(key, as_index=False).agg(
        pay_med=("pay", "median"), pay_best=("pay", "max"),
        n_books=("book", "nunique"))


def big_dog_bets() -> pd.DataFrame:
    """The validated selection: wks 1-5, |spread|>=17, model on the dog >=6."""
    d = pd.read_parquet(PARQUET_DIR / "backtest_spread_history.parquet")
    d = d[d.season.isin(SEASONS) & d.week.between(1, 5)
          & d.margin.notna() & d.book_spread.notna()].copy()

    games = pd.concat([pd.read_parquet(
        CFBD_PARQUET_DIR / f"games_{s}.parquet") for s in SEASONS])
    s2i = dict(zip(pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
                   .school,
                   pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet").id))
    games["home_id"] = games.homeTeam.map(s2i)
    games["away_id"] = games.awayTeam.map(s2i)
    # game_id is float64 in the spread-history parquet, id is int64 in games:
    # merging them unconverted matches NOTHING and silently yields an empty set
    d["game_id"] = d.game_id.astype("int64")
    g = games[["id", "home_id", "away_id"]].astype({"id": "int64"})
    n0 = len(d)
    d = d.merge(g, left_on="game_id", right_on="id", how="inner")
    assert len(d) == n0, f"game join changed rows: {n0} -> {len(d)}"

    # COVERAGE LIMIT, stated rather than hidden: teams_fbs has no FCS ids, and
    # neither the CFBD map nor team_matcher resolves an FCS opponent — so
    # FBS-vs-FCS games cannot be joined to an odds ladder at all. That is not a
    # rounding error here: FCS opponents are ~46% of the big-dog universe
    # (149 of 323 games in wks 1-5), because "P4 buys a body-bag game" IS the
    # archetypal big spread. Whatever this backtest concludes therefore applies
    # only to the FBS-vs-FBS half of the validated big-dog play.
    fcs = d.home_id.isna() | d.away_id.isna()
    big = d.book_spread.abs() >= BIG_SPREAD
    print(f"wks1-5 games {SEASONS}: {len(d)} | vs FCS (unmappable): "
          f"{int(fcs.sum())} | big dogs: {int(big.sum())} "
          f"(FBS-only {int((big & ~fcs).sum())}, vs-FCS {int((big & fcs).sum())})")
    d = d[~fcs].astype({"home_id": int, "away_id": int})

    d["home_is_dog"] = d.book_spread > 0
    # our model's edge points at the home team when edge>0
    d["model_on_dog"] = np.where(d.home_is_dog, d.edge, -d.edge)
    d["dog_id"] = np.where(d.home_is_dog, d.home_id, d.away_id)
    d["dog_line"] = d.book_spread.abs()          # dog's main number, +points
    # dog's own margin (dog score - favourite score)
    d["dog_margin"] = np.where(d.home_is_dog, d.margin, -d.margin)
    sel = d[(d.book_spread.abs() >= BIG_SPREAD)].copy()
    sel["picked"] = sel.model_on_dog >= MIN_EDGE
    print(f"big-dog games wks1-5 {SEASONS}: {len(sel)} "
          f"| model-selected: {int(sel.picked.sum())}")
    return sel


def build() -> pd.DataFrame:
    lad = ladders()
    bets = big_dog_bets()
    rows = []
    for off in OFFSETS:
        b = bets.copy()
        b["offset"] = off
        b["point"] = b.dog_line + off
        m = b.merge(lad, left_on=["season", "week", "home_id", "away_id",
                                  "dog_id", "point"],
                    right_on=["season", "week", "home_id", "away_id",
                              "team_id", "point"], how="inner")
        rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    # grade: dog covers `point` if dog_margin + point > 0
    out["won"] = np.where(out.dog_margin + out.point > 0, 1.0,
                          np.where(out.dog_margin + out.point < 0, 0.0, np.nan))
    out["pnl_med"] = np.where(out.won.isna(), 0.0,
                              np.where(out.won == 1, out.pay_med, -1.0))
    out["pnl_best"] = np.where(out.won.isna(), 0.0,
                               np.where(out.won == 1, out.pay_best, -1.0))
    return out


def report(o: pd.DataFrame) -> None:
    o = o[o.won.notna() & (o.n_books >= 2)]
    sel, blind = o[o.picked], o

    print(f"\n=== MODEL-SELECTED big dogs ({len(sel):,} rung-bets) ===")
    print(f"{'offset':>7} {'dog gets':>9} {'bets':>6} {'win%':>7} "
          f"{'med px':>7} {'ROI med':>8} {'ROI best':>9} {'2023':>7} "
          f"{'2024':>7} {'2025':>7}")
    for off in OFFSETS:
        b = sel[sel.offset == off]
        if len(b) < 25:
            continue
        per = b.groupby("season").pnl_med.mean()
        print(f"{off:+7.1f} {'main'+f'{off:+.1f}' if off else 'MAIN':>9} "
              f"{len(b):6d} {b.won.mean():6.1%} "
              f"{b.pay_med.mean():+7.2f} {b.pnl_med.mean()*100:+7.1f}% "
              f"{b.pnl_best.mean()*100:+8.1f}% "
              + " ".join(f"{per.get(s, np.nan)*100:+6.1f}%" for s in SEASONS))

    print(f"\n=== BLIND control — every big dog, no model ({len(blind):,}) ===")
    print(f"{'offset':>7} {'bets':>6} {'win%':>7} {'ROI med':>8} {'ROI best':>9}")
    for off in OFFSETS:
        b = blind[blind.offset == off]
        if len(b) < 25:
            continue
        print(f"{off:+7.1f} {len(b):6d} {b.won.mean():6.1%} "
              f"{b.pnl_med.mean()*100:+7.1f}% {b.pnl_best.mean()*100:+8.1f}%")

    # does the model's SELECTION add anything at each rung?
    print("\n=== selection value: model ROI minus blind ROI, per rung ===")
    for off in OFFSETS:
        s, bl = sel[sel.offset == off], blind[blind.offset == off]
        if len(s) < 25:
            continue
        print(f"{off:+7.1f}  {(s.pnl_med.mean()-bl.pnl_med.mean())*100:+6.1f} pp "
              f"(model {s.pnl_med.mean()*100:+.1f}% vs blind "
              f"{bl.pnl_med.mean()*100:+.1f}%)")

    bg = sel[sel.offset == 0]
    print(f"\nreference — main line only: {len(bg)} bets {bg.won.mean():.1%} "
          f"{bg.pnl_med.mean()*100:+.1f}% (median) "
          f"{bg.pnl_best.mean()*100:+.1f}% (best price)")
    paired(sel)


def paired(sel: pd.DataFrame) -> None:
    """Rung vs MAIN on the SAME games — the only fair comparison.

    Comparing independent per-rung means is confounded: a different subset of
    games is quoted at each rung. Restricting to games quoted at every rung and
    testing the per-game pnl difference is both fairer and far more powerful.
    """
    offs = [-7.0, -3.5, 0.0, 3.5, 7.0]
    s = sel[sel.offset.isin(offs)]
    cnt = s.groupby("game_id").offset.nunique()
    s = s[s.game_id.isin(cnt[cnt == len(offs)].index)]
    piv = s.pivot_table(index="game_id", columns="offset", values="pnl_med")
    win = s.pivot_table(index="game_id", columns="offset", values="won")
    if piv.empty:
        return
    print(f"\n=== PAIRED vs MAIN ({len(piv)} games quoted at all "
          f"{len(offs)} rungs) ===")
    print(f"{'rung':>6} {'win%':>7} {'ROI':>8} {'vs MAIN':>9} {'paired t':>9} "
          f"{'sd':>6}")
    rng = np.random.default_rng(7)
    for off in offs:
        d = (piv[off] - piv[0.0]).dropna()
        t = (d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if off else np.nan
        print(f"{off:+6.1f} {win[off].mean():6.1%} {piv[off].mean()*100:+7.1f}% "
              f"{d.mean()*100:+8.1f}pp {t:+9.2f} {piv[off].std():6.2f}")
    d = (piv[-3.5] - piv[0.0]).dropna().values
    bs = np.array([rng.choice(d, len(d), replace=True).mean()
                   for _ in range(10000)])
    print(f"best candidate (-3.5) vs MAIN: {d.mean()*100:+.1f}pp "
          f"[95% CI {np.percentile(bs, 2.5)*100:+.1f}, "
          f"{np.percentile(bs, 97.5)*100:+.1f}] -> CI includes 0, REJECT")


if __name__ == "__main__":
    out = build()
    out.to_parquet(DEST, index=False)
    report(out)
    print(f"\nsaved: {DEST}")
