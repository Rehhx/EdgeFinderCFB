"""Stability audit for the derived-line plays (Q1, 1H, full-game spread).

The props audit (backtest/props_stability.py) resampled the model's TRAINING
data, because props have fitted coefficients. These plays mostly do not — Q1 is
deliberately model-free. What *was* fitted here is the discretionary choices:
the spread thresholds, the week splits, the model filter. Those were picked by
looking at the data, so they are where the overfitting risk lives.

Four tests per play:

  1. THRESHOLD SWEEP  — the decisive one. A real structural edge is a PLATEAU:
     neighbouring cutoffs all work, because nothing special happens at 25.0. A
     fitted one is a SPIKE that collapses either side. This is what killed
     alternate spreads ("the best rung moves depending on which subset you
     look at").
  2. LEAVE-ONE-SEASON-OUT — drop each season in turn. If one season carries the
     play, the remaining ones will say so.
  3. MEDIAN vs BEST PRICE — discipline rule 2. An edge that needs the single
     best quoted number is stale lines, not skill.
  4. WEEK-BLOCK BOOTSTRAP — resample whole weeks (preserving same-week and
     same-game correlation) for an honest CI on units/season.

  python -m backtest.play_stability
"""
import numpy as np
import pandas as pd

from ingestion.config import PARQUET_DIR

N_BOOT = 5000
SEASONS = (2023, 2024, 2025)
MIN_BETS = 25


# ---------------------------------------------------------------- loading
def q1() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_q1_spreads.parquet")
    d = d[d.covered.notna() & (d.n_books >= 2)].copy()
    return d


def h1() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_h1_saturation.parquet")
    d = d[d.covered.notna() & (d.n_books >= 2)].copy()
    return d


def spread() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_spread_history.parquet")
    d = d[d.season.isin(SEASONS) & d.won.notna()].copy()
    d["dog_full_line"] = d.book_spread.abs()
    d["model_on_dog"] = np.where(d.book_spread > 0, d.edge, -d.edge)
    d["pnl_med"] = np.where(d.won == 1, 100 / 110, -1.0)
    d["pnl_best"] = d.pnl_med          # flat -110, no shopping to test
    return d


def select(d, lo, hi=None, weeks=None, model_min=None) -> pd.DataFrame:
    s = d[d.dog_full_line >= lo]
    if hi is not None:
        s = s[s.dog_full_line < hi]
    if weeks is not None:
        s = s[s.week.between(*weeks)]
    if model_min is not None:
        s = s[s.model_on_dog >= model_min]
    return s


# ---------------------------------------------------------------- tests
def stat_line(s: pd.DataFrame, col="pnl_med") -> str:
    if len(s) < MIN_BETS:
        return f"{len(s):5d} bets   (below {MIN_BETS}, not reported)"
    roi, sd = s[col].mean(), s[col].std()
    t = roi / (sd / np.sqrt(len(s)))
    return (f"{len(s):5d} bets  {(s[col] > 0).mean():6.1%} win  "
            f"{roi:+7.1%} ROI  t={t:+5.2f}")


def sweep(d, name, cuts, weeks=None, model_min=None, chosen=None) -> None:
    print(f"\n  --- 1. THRESHOLD SWEEP: {name} "
          f"(plateau = real, spike = fitted) ---")
    for lo in cuts:
        s = select(d, lo, weeks=weeks, model_min=model_min)
        mark = "  <- SHIPPED" if lo == chosen else ""
        print(f"    spread >= {lo:4.1f}: {stat_line(s)}{mark}")


def loso(d, name, lo, weeks=None, model_min=None) -> None:
    print(f"\n  --- 2. LEAVE-ONE-SEASON-OUT: {name} ---")
    s = select(d, lo, weeks=weeks, model_min=model_min)
    for season in sorted(s.season.unique()):
        held = s[s.season != season]
        one = s[s.season == season]
        print(f"    without {season}: {stat_line(held)}")
        print(f"       ({season} alone: {stat_line(one)})")


def price_control(d, name, lo, weeks=None, model_min=None) -> None:
    s = select(d, lo, weeks=weeks, model_min=model_min)
    if len(s) < MIN_BETS:
        return
    print(f"\n  --- 3. MEDIAN vs BEST PRICE: {name} ---")
    print(f"    median price: {stat_line(s, 'pnl_med')}")
    print(f"    best price  : {stat_line(s, 'pnl_best')}")
    gap = s.pnl_best.mean() - s.pnl_med.mean()
    print(f"    line-shopping adds {gap:+.1%} — the edge must survive the "
          "MEDIAN row to be real.")


def boot(d, name, lo, units, weeks=None, model_min=None) -> None:
    s = select(d, lo, weeks=weeks, model_min=model_min)
    if len(s) < MIN_BETS:
        return
    blocks = [g.pnl_med.values for _, g in s.groupby(["season", "week"])]
    n_weeks = int(round(len(blocks) / s.season.nunique()))
    rng = np.random.default_rng(2026)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(blocks), n_weeks)
        out[i] = units * sum(blocks[j].sum() for j in idx)
    print(f"\n  --- 4. WEEK-BLOCK BOOTSTRAP: {name} ({units:.0f}u stake) ---")
    print(f"    units/season  median {np.median(out):+6.1f}   "
          f"90% CI [{np.percentile(out, 5):+.1f}, {np.percentile(out, 95):+.1f}]"
          f"   P(losing season) {(out < 0).mean():.1%}")


def audit(d, name, lo, units, cuts, weeks=None, model_min=None,
          blind_note=None) -> None:
    print(f"\n{'='*72}\n=== {name}\n{'='*72}")
    sweep(d, name, cuts, weeks, model_min, chosen=lo)
    loso(d, name, lo, weeks, model_min)
    price_control(d, name, lo, weeks, model_min)
    boot(d, name, lo, units, weeks, model_min)
    if model_min is not None:
        s_on = select(d, lo, weeks=weeks, model_min=model_min)
        s_off = select(d, lo, weeks=weeks)
        print(f"\n  --- 5. MODEL FILTER control ---")
        print(f"    with filter (>={model_min}): {stat_line(s_on)}")
        print(f"    BLIND (no model)          : {stat_line(s_off)}")
    if blind_note:
        print(f"    NOTE {blind_note}")


def main() -> None:
    print("Stability audit of the derived-line plays. The threshold sweep is "
          "the\ndecisive test: a structural edge should be a PLATEAU, not a "
          "spike at the\nexact number that was chosen.")

    Q, H, S = q1(), h1(), spread()

    audit(Q, "Q1 PREMIUM (dog on the Q1 line, spread >= 25)", 25.0, 2.0,
          cuts=[19, 21, 23, 25, 27, 29, 31],
          blind_note="Q1 is deliberately model-free — blind BEAT model-selected.")
    # NOTE: this is the whole Q1 play at >=17 (no upper bound), so it CONTAINS
    # the PREMIUM games above. It is not the 17-25 slice, and its units are not
    # additive with PREMIUM — bankroll_sim de-duplicates before staking.
    audit(Q, "Q1 ALL (spread >= 17, includes PREMIUM)", 17.0, 1.0,
          cuts=[13, 15, 17, 19, 21])
    audit(H, "BIG-DOG 1H, weeks 1-5 (spread >= 17)", 17.0, 2.0,
          cuts=[13, 15, 17, 19, 21, 23], weeks=(1, 5), model_min=6)
    audit(H, "BIG-DOG 1H, weeks 6-15 (spread >= 21)", 21.0, 2.0,
          cuts=[15, 17, 19, 21, 23, 25], weeks=(6, 15), model_min=6)
    audit(S, "SPREAD PREMIUM (spread >= 25, weeks 1-5)", 25.0, 2.0,
          cuts=[19, 21, 23, 25, 27, 29], weeks=(1, 5), model_min=6)

    print(f"\n{'='*72}")
    print("READ: a cut that only works at the exact shipped number is fitted.\n"
          "Neighbouring cutoffs should agree — nothing physical happens at 25.0.")


if __name__ == "__main__":
    main()
