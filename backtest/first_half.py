"""Backtest first-half spread / total / moneyline vs derived book 1H lines.

Walk-forward 1H opponent-adjusted EPA ratings (period 1-2 only) → predicted
1H margin & total, self-calibrated on walked-through games; compared to the
book's 1H lines and graded on official 1H (quarter) scores.

  python -m backtest.first_half
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import payout  # vectorized (np.where)
from features.epa_ratings import SEASON_STRIDE, fit_ratings
from features.first_half import load_1h_obs, one_half_scores
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR
from picks.edge_report import team_matcher

PBP_SEASONS = list(range(2021, 2026))
TEST_SEASONS = [2023, 2024, 2025]
FIRST_WEEK, LAST_WEEK = 1, 15
MIN_CALIB = 200


def consensus_1h() -> pd.DataFrame:
    raw = pd.read_parquet(PARQUET_DIR / "historical_1h_lines.parquet")
    match = team_matcher()
    raw["home_id"] = raw.home.map(match)
    raw["away_id"] = raw.away.map(match)
    raw = raw.dropna(subset=["home_id", "away_id"]).astype(
        {"home_id": int, "away_id": int})
    key = ["season", "week", "home_id", "away_id"]

    # spreads_h1: home point; totals_h1: over point (median across books)
    sp = raw[(raw.market == "spreads_h1") & (raw.name == raw.home)]
    spread = sp.groupby(key).point.median().rename("book_1h_spread")
    tot = raw[(raw.market == "totals_h1") & (raw.name == "Over")]
    total = tot.groupby(key).point.median().rename("book_1h_total")
    # h2h_h1 best price each side (payout space)
    ml = raw[raw.market == "h2h_h1"].copy()
    ml["pay"] = payout(ml.price)
    mlh = ml[ml.name == ml.home].groupby(key).pay.max().rename("ml_home_pay")
    mla = ml[ml.name == ml.away].groupby(key).pay.max().rename("ml_away_pay")
    return pd.concat([spread, total, mlh, mla], axis=1).reset_index()


def run() -> pd.DataFrame:
    obs = load_1h_obs(PBP_SEASONS)
    scores = one_half_scores(TEST_SEASONS)
    s2id = dict(zip(
        pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet").school,
        pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet").id))
    scores["home_id"] = scores.homeTeam.map(s2id)
    scores["away_id"] = scores.awayTeam.map(s2id)
    scores = scores.dropna(subset=["home_id", "away_id"]).astype(
        {"home_id": int, "away_id": int})
    lines = consensus_1h()
    g = scores.merge(lines, on=["season", "week", "home_id", "away_id"],
                     how="inner")
    print(f"1H games with lines: {len(g):,}")

    b, hfa, tk, tc = 40.0, 1.2, 22.0, 24.0
    mcal, tcal, preds = [], [], []
    for season in TEST_SEASONS:
        for week in range(FIRST_WEEK, LAST_WEEK + 1):
            asof = season * SEASON_STRIDE + week
            model = fit_ratings(obs, asof_week_idx=asof)
            wk = g[(g.season == season) & (g.week == week)]
            for r in wk.itertuples():
                net = model.net(r.home_id) - model.net(r.away_id)
                ho, hd = (model.ratings.loc[r.home_id, ["off", "def"]]
                          if r.home_id in model.ratings.index else (0, 0))
                ao, ad = (model.ratings.loc[r.away_id, ["off", "def"]]
                          if r.away_id in model.ratings.index else (0, 0))
                combined = (ho - ad) + (ao - hd)
                neutral = bool(r.neutralSite)
                pm = b * net + (0 if neutral else hfa)
                preds.append({
                    "season": season, "week": week,
                    "pred_margin": pm, "pred_total": tk * combined + tc,
                    "book_1h_spread": r.book_1h_spread,
                    "book_1h_total": r.book_1h_total,
                    "ml_home_pay": r.ml_home_pay, "ml_away_pay": r.ml_away_pay,
                    "h1_margin": r.h1_margin, "h1_total": r.h1_total})
                mcal.append((net, 0 if neutral else 1, r.h1_margin))
                if pd.notna(r.book_1h_total):
                    tcal.append((combined, r.h1_total))
            if len(mcal) >= MIN_CALIB:
                cd = np.array(mcal)
                coef, *_ = np.linalg.lstsq(
                    np.column_stack([cd[:, 0], cd[:, 1], np.ones(len(cd))]),
                    cd[:, 2], rcond=None)
                b, hfa = float(coef[0]), float(coef[1])
                td = np.array(tcal)
                tcf, *_ = np.linalg.lstsq(
                    np.column_stack([td[:, 0], np.ones(len(td))]),
                    td[:, 1], rcond=None)
                tk, tc = float(tcf[0]), float(tcf[1])
    from ingestion.config import save_coefs
    save_coefs("first_half", {"b": b, "hfa": hfa, "tk": tk, "tc": tc})
    return pd.DataFrame(preds)


def report(df: pd.DataFrame) -> None:
    sigma = (df.pred_margin - df.h1_margin).std()
    print(f"\n1H margin MAE {(df.pred_margin-df.h1_margin).abs().mean():.2f} "
          f"| corr vs book {df.pred_margin.corr(-df.book_1h_spread):.3f} "
          f"| sigma {sigma:.1f}")

    # ATS
    d = df.dropna(subset=["book_1h_spread"]).copy()
    d["edge"] = d.pred_margin - (-d.book_1h_spread)
    d["cover"] = np.sign(d.h1_margin + d.book_1h_spread)
    d["won"] = np.where(d.cover == 0, np.nan,
                        (np.sign(d.edge) == d.cover).astype(float))
    print(f"\n1H SPREAD ATS (flat -110, BE 52.4%) — {d.season.nunique()} seasons:")
    for label, mask in [("wks 1-5", d.week.between(1, 5)),
                        ("wks 6-15", d.week.between(6, 15)),
                        ("ALL", d.week > 0)]:
        for t in (1.5, 2.5, 3.5):
            b = d[mask & (d.edge.abs() >= t) & d.won.notna()]
            if len(b) < 60:
                continue
            wr = b.won.mean()
            se = (wr * (1 - wr) / len(b)) ** 0.5
            print(f"  {label:8s} edge>={t}: {len(b):5d}  {wr:.1%} (+-{se:.1%})  "
                  f"{(wr*100/110-(1-wr))*100:+.1f}%")
    print("  wks 1-5 edge>=2 by season:")
    e = d[d.week.between(1, 5) & (d.edge.abs() >= 2) & d.won.notna()]
    print(e.groupby("season").won.agg(["size", "mean"]).round(3).to_string())

    # totals
    t = df.dropna(subset=["book_1h_total"]).copy()
    t["edge"] = t.pred_total - t.book_1h_total
    t["ou"] = np.sign(t.h1_total - t.book_1h_total)
    t["won"] = np.where(t.ou == 0, np.nan,
                        (np.sign(t.edge) == t.ou).astype(float))
    print("\n1H TOTAL O/U:")
    for t_thr in (2, 3, 4):
        b = t[(t.edge.abs() >= t_thr) & t.won.notna()]
        if len(b) < 60:
            continue
        wr = b.won.mean()
        print(f"  edge>={t_thr}: {len(b):5d}  {wr:.1%}  "
              f"{(wr*100/110-(1-wr))*100:+.1f}%")

    # moneyline value (blend model prob toward market, like ml_value)
    m = df.dropna(subset=["ml_home_pay", "ml_away_pay"]).copy()
    m["p_home"] = 1 / (1 + np.exp(-m.pred_margin / (sigma * 0.55)))
    imp_h = 1 / (1 + m.ml_home_pay)
    imp_a = 1 / (1 + m.ml_away_pay)
    m["p_mkt_h"] = imp_h / (imp_h + imp_a)
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    m["p_bl"] = 1 / (1 + np.exp(-(0.3 * lg(m.p_home) + 0.7 * lg(m.p_mkt_h))))
    m["ev_h"] = m.p_bl * m.ml_home_pay - (1 - m.p_bl)
    m["ev_a"] = (1 - m.p_bl) * m.ml_away_pay - m.p_bl
    m["bet_home"] = m.ev_h >= m.ev_a
    m["ev"] = m[["ev_h", "ev_a"]].max(axis=1)
    m["won"] = np.where(m.bet_home, m.h1_margin > 0, m.h1_margin < 0)
    m["pnl"] = np.where(m.won,
                        np.where(m.bet_home, m.ml_home_pay, m.ml_away_pay), -1.0)
    m = m[m.h1_margin != 0]
    print("\n1H MONEYLINE (blended prob, best price):")
    for t in (0.03, 0.05, 0.08):
        b = m[m.ev >= t]
        if len(b) < 40:
            continue
        print(f"  EV>={t:.0%}: {len(b):5d}  {b.won.mean():.1%} win  "
              f"{b.pnl.mean()*100:+.1f}% ROI")


if __name__ == "__main__":
    out = run()
    out.to_parquet(PARQUET_DIR / "backtest_first_half.parquet", index=False)
    report(out)
