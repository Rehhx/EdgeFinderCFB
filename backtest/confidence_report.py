"""Confidence calibration: when the model says X% — how often does it hit?

For every backtested market (props, spreads ATS, moneyline picks), buckets
bets by the model's stated probability of the chosen side and reports the
ACTUAL hit rate plus how many such bets exist per season.

  python -m backtest.confidence_report
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from ingestion.config import PARQUET_DIR

BUCKETS = [0.50, 0.55, 0.60, 0.70, 0.80, 1.001]
LABELS = ["50-55%", "55-60%", "60-70%", "70-80%", "80%+"]


def bucket_table(df: pd.DataFrame, p_col: str, won_col: str,
                 title: str) -> None:
    d = df[df[won_col].notna() & df[p_col].notna()].copy()
    d["bucket"] = pd.cut(d[p_col], BUCKETS, labels=LABELS, right=False)
    print(f"\n=== {title} ({len(d):,} graded bets) ===")
    print(f"{'confidence':>10} {'n total':>8} {'n/season':>9} "
          f"{'stated':>7} {'actual':>7} {'gap':>6}")
    n_seasons = d.season.nunique()
    for b in LABELS:
        s = d[d.bucket == b]
        if len(s) < 10:
            continue
        stated, actual = s[p_col].mean(), s[won_col].mean()
        print(f"{b:>10} {len(s):>8,} {len(s) / n_seasons:>9,.0f} "
              f"{stated:>6.1%} {actual:>6.1%} {actual - stated:>+6.1%}")
    per = d.groupby("season").size()
    print("bets per season:", dict(per))


def props() -> None:
    m = pd.read_parquet(PARQUET_DIR / "backtest_props_vs_book.parquet")
    bucket_table(m, "p_side", "bet_won",
                 "PLAYER PROPS (model prob of chosen O/U side vs main lines)")
    m25 = m[m.season == 2025]
    if len(m25):
        bucket_table(m25, "p_side", "bet_won",
                     "PLAYER PROPS — 2025 only (out-of-sample)")


def spreads_and_ml() -> None:
    df = pd.read_parquet(PARQUET_DIR / "backtest_spread_phase1.parquet")
    sigma = (df.pred_roster - df.margin).std()
    print(f"\n[spread model margin sigma: {sigma:.1f}]")

    # ATS: prob that the model-favored side covers the book number
    df["p_cover"] = norm.cdf(df.edge_roster.abs() / sigma)
    bucket_table(df, "p_cover", "won_roster",
                 "SPREADS ATS (prob model's side covers the closing number)")

    # Moneyline: prob the model-picked team wins outright
    p_home = norm.cdf(df.pred_roster / sigma)
    df["p_ml"] = np.maximum(p_home, 1 - p_home)
    picked_home = p_home >= 0.5
    won = np.sign(df.margin)
    df["ml_won"] = np.where(won == 0, np.nan, np.where(
        picked_home, won > 0, won < 0).astype(float))
    bucket_table(df, "p_ml", "ml_won",
                 "MONEYLINE (prob model's pick wins the game outright)")


if __name__ == "__main__":
    props()
    spreads_and_ml()
    print("\nNote: a +gap means the model UNDERSTATES its edge in that "
          "bucket; a -gap means overconfidence. Props graded vs real "
          "2024-25 lines; spread/ML vs closing consensus.")
