"""Do established power ratings (SP+) improve the early-season spread edge?

Tests previous-season SP+ (a market-independent power rating that persists
r=0.75 y/y and already lives in points) as an early-season margin prior vs
our as-of EPA model, and a blend. Focus: the validated big-dog play (wks
1-5, |spread|>=17, model on dog, edge>=6) — does SP+ find more/better ones?

  python -m backtest.sp_spread
"""
import numpy as np
import pandas as pd

from backtest.spread_baseline import consensus_lines
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

SEASONS = list(range(2016, 2026))
MIN_CALIB = 200


def sp_prev() -> dict:
    sp = pd.read_parquet(CFBD_PARQUET_DIR / "sp_ratings.parquet")
    return {(r.season + 1, r.team): r.sp for r in sp.itertuples()}  # prev-season


def ats(df, edge_col, tag):
    """Report ATS overall + the big-dog validated play."""
    d = df.dropna(subset=[edge_col]).copy()
    d["cover"] = np.sign(d.margin + d.spread)
    d = d[d.cover != 0]
    d["won"] = (np.sign(d[edge_col]) == d.cover).astype(float)
    fav_home = d.spread < 0
    picks_home = d[edge_col] > 0
    d["dog_play"] = ((picks_home != fav_home) & (d.spread.abs() >= 17)
                     & (d[edge_col].abs() >= 6) & d.week.between(1, 5))
    for label, sub in [(f"{tag} wks1-5 edge>=4", d[d.week.between(1, 5)
                        & (d[edge_col].abs() >= 4)]),
                       (f"{tag} BIG-DOG play", d[d.dog_play])]:
        if len(sub) < 50:
            continue
        wr = sub.won.mean()
        se = (wr * (1 - wr) / len(sub)) ** 0.5
        print(f"  {label:26s}: {len(sub):5d} bets {wr:.1%} (+-{se:.1%}) "
              f"{(wr*100/110-(1-wr))*100:+.1f}%")


def run() -> None:
    spp = sp_prev()
    rows = []
    for s in SEASONS:
        c = consensus_lines(s).assign(season=s).dropna(
            subset=["spread", "homePoints", "awayPoints"])
        c["margin"] = c.homePoints - c.awayPoints
        c["sp_diff"] = [spp.get((s, h), np.nan) - spp.get((s, a), np.nan)
                        for h, a in zip(c.homeTeam, c.awayTeam)]
        rows.append(c)
    df = pd.concat(rows, ignore_index=True).dropna(subset=["sp_diff"])

    # self-calibrated SP+ margin model (walk-forward by season)
    df = df.sort_values(["season", "week"])
    df["pred_sp"] = np.nan
    b, hfa = 1.0, 2.5
    cal = []
    for i, r in df.iterrows():
        neutral = bool(r.neutralSite) if pd.notna(r.neutralSite) else False
        df.at[i, "pred_sp"] = b * r.sp_diff + (0 if neutral else hfa)
        cal.append((r.sp_diff, 0.0 if neutral else 1.0, r.margin))
        if len(cal) % 400 == 0 and len(cal) >= MIN_CALIB:
            cd = np.array(cal)
            coef, *_ = np.linalg.lstsq(
                np.column_stack([cd[:, 0], cd[:, 1], np.ones(len(cd))]),
                cd[:, 2], rcond=None)
            b, hfa = float(coef[0]), float(coef[1])
    df["edge_sp"] = df.pred_sp - (-df.spread)
    print(f"SP+ margin MAE: {(df.pred_sp - df.margin).abs().mean():.2f} "
          f"| book: {(-df.spread - df.margin).abs().mean():.2f} "
          f"| final SP+ slope {b:.2f}, hfa {hfa:.1f}\n")

    print("=== SP+ (prev-season) early-season ATS, 10 seasons ===")
    ats(df, "edge_sp", "SP+")

    # blend with our EPA model where available (spread_history 2018-25)
    try:
        epa = pd.read_parquet(PARQUET_DIR / "backtest_spread_history.parquet")[
            ["game_id", "pred_margin"]]
        m = df.merge(epa, left_on="id", right_on="game_id", how="inner")
        m["pred_blend"] = 0.5 * m.pred_sp + 0.5 * m.pred_margin
        m["edge_blend"] = m.pred_blend - (-m.spread)
        m["edge_epa"] = m.pred_margin - (-m.spread)
        print(f"\n=== EPA vs SP+ vs BLEND on shared games (2018-25, "
              f"n={len(m)}) ===")
        print(f"MAE — EPA {(m.pred_margin-m.margin).abs().mean():.2f} | "
              f"SP+ {(m.pred_sp-m.margin).abs().mean():.2f} | "
              f"blend {(m.pred_blend-m.margin).abs().mean():.2f}")
        for col, tag in [("edge_epa", "EPA"), ("edge_sp", "SP+"),
                         ("edge_blend", "BLEND")]:
            ats(m, col, tag)
    except FileNotFoundError:
        print("(run spread_history first for the EPA comparison)")


if __name__ == "__main__":
    run()
