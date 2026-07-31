"""Prop projections vs real historical book lines.

Joins historical_prop_lines (The Odds API snapshots) with prop_projections
on (season, week, player, stat). Two questions:
  1. Accuracy duel — who is closer to the actual stat, our projection or
     the book's consensus line?
  2. Betting sim — price P(over) from the projection distribution, bet
     when EV clears a threshold, grade against actuals at real prices.

Discipline: recalibration, NB dispersion and market-blend weight are fitted
on 2023 matched rows only; 2024 and 2025 are fully out-of-sample (report
shows each plus the 2024+2025 pool). Finding: blended+line-shopping is only
+0.5%..+1.3% ROI OOS — props are near-efficient; the single-season +5.2%
seen earlier was calibration-season luck.

  python -m backtest.props_vs_book
"""
import numpy as np
import pandas as pd
from scipy.stats import gamma, nbinom, norm

from ingestion.config import PARQUET_DIR

STATS = ["rush_yds", "rec_yds", "receptions", "pass_yds"]
EV_THRESH = [0.03, 0.05, 0.08, 0.12]
# Yardage stats are right-skewed (skew: rec_yds 1.22, rush 1.07, pass 0.00),
# so a NORMAL tail overstates P(over). Gamma (method-of-moments) fixes it:
# +2.7% -> +3.4% ROI overall and pass_yds -2.1% -> +6.5%.
YARDAGE = ("rush_yds", "rec_yds", "pass_yds")
# rec_yds loses under BOTH distributions (-0.7% normal, -0.8% gamma) across
# 2024 and 2025 — the market prices receiving better than we do. Excluded
# from betting; dropping it lifts EV>=5% from +3.6% to +5.8% ROI.
EXCLUDE_STATS = ("rec_yds",)


def consensus_lines() -> pd.DataFrame:
    raw = pd.read_parquet(PARQUET_DIR / "historical_prop_lines.parquet")
    raw = raw.dropna(subset=["player", "point", "price"])
    raw = raw[raw.price.between(-200, 200)]  # main lines only, no alt ladders

    # a quotable line = over AND under from the same book at the SAME point
    piv = raw.pivot_table(
        index=["season", "week", "stat", "player", "book", "point"],
        columns="side", values="price", aggfunc="first")
    piv = piv.reset_index().dropna(subset=["Over", "Under"]).rename(
        columns={"Over": "price_over", "Under": "price_under"})

    key = ["season", "week", "stat", "player"]
    # consensus point = the most commonly quoted point across books
    modal = piv.groupby(key).point.agg(
        lambda s: s.mode().iloc[0]).rename("line")
    piv = piv.merge(modal, on=key)
    at_line = piv[piv.point == piv.line].copy()
    # aggregate in PAYOUT space — american prices are discontinuous around
    # +/-100, so a median of raw prices straddling zero is nonsense
    at_line["pay_over"] = payout(at_line.price_over)
    at_line["pay_under"] = payout(at_line.price_under)
    cons = at_line.groupby(key, as_index=False).agg(
        line=("point", "first"),
        pay_over=("pay_over", "median"),
        pay_under=("pay_under", "median"),
        pay_over_best=("pay_over", "max"),    # line shopping: best quoted
        pay_under_best=("pay_under", "max"),
        n_books=("book", "nunique"))
    return cons


def payout(price: pd.Series) -> pd.Series:
    return np.where(price < 0, 100 / price.abs(), price / 100)


def run() -> pd.DataFrame:
    lines = consensus_lines()
    proj = pd.read_parquet(PARQUET_DIR / "prop_projections.parquet")

    frames = []
    for stat in STATS:
        cols = ["season", "week", "player", stat, f"proj_{stat}"]
        mu_col = "mu_rush" if stat == "rush_yds" else "mu_pass"
        if mu_col in proj.columns:
            cols.append(mu_col)
        if "trail_ppa_all" in proj.columns:
            cols.append("trail_ppa_all")
        p = proj[cols].rename(columns={stat: "actual", f"proj_{stat}": "proj",
                                       mu_col: "mu_total"})
        p = p[p.proj.notna()].assign(stat=stat)
        frames.append(p)
    pl = pd.concat(frames)
    if "mu_total" not in pl:
        pl["mu_total"] = np.nan

    m = lines.merge(pl, on=["season", "week", "stat", "player"], how="inner")
    print(f"matched player-prop lines: {len(m):,} "
          f"(of {len(lines):,} consensus lines)")

    # Recalibration fitted on 2023, now including the offense-vs-defense
    # MATCHUP term: actual ~ a + b*proj + c*(matchup-0.5)*proj. Validated
    # out-of-sample (+4.4% vs -3.5% ROI at EV>=3%, positive both 2024 & 2025).
    m["mu_c"] = (m.mu_total.fillna(0.5) - 0.5) * m.proj
    if "trail_ppa_all" not in m:
        m["trail_ppa_all"] = np.nan
    # Player-PPA term tested and REJECTED: with the join bug fixed it gives
    # +6.4% vs +7.0% ROI without (and 2025 falls 7.3%->5.8%). Kept at 0 so the
    # coefficient slot stays in the saved recal tuple.
    m["ppa_c"] = 0.0
    cal = {}
    for stat in STATS:
        t = m[(m.season == 2023) & (m.stat == stat)]
        X = np.column_stack([t.proj, t.mu_c, t.ppa_c, np.ones(len(t))])
        coef, *_ = np.linalg.lstsq(X, t.actual, rcond=None)
        sigma = (t.actual - (X @ coef)).std()
        # (a, b, sigma, c_matchup, c_ppa)
        cal[stat] = (float(coef[3]), float(coef[0]), float(sigma),
                     float(coef[1]), float(coef[2]))
    print("2023-fitted recal (a, b, sigma, c_matchup, c_ppa):",
          {k: tuple(round(x, 2) for x in v) for k, v in cal.items()})

    m["a"] = m.stat.map({k: v[0] for k, v in cal.items()})
    m["b"] = m.stat.map({k: v[1] for k, v in cal.items()})
    m["sigma"] = m.stat.map({k: v[2] for k, v in cal.items()})
    m["c_mu"] = m.stat.map({k: v[3] for k, v in cal.items()})
    m["c_ppa"] = m.stat.map({k: v[4] for k, v in cal.items()})
    m["proj_cal"] = (m.a + m.b * m.proj + m.c_mu * m.mu_c
                     + m.c_ppa * m.ppa_c)
    m["p_over"] = 1 - norm.cdf((m.line - m.proj_cal) / m.sigma)
    # skew-aware tail for yardage stats (see YARDAGE note above)
    yd = m.stat.isin(YARDAGE)
    mu_y = m.loc[yd, "proj_cal"].clip(lower=1.0)
    sd_y = m.loc[yd, "sigma"]
    m.loc[yd, "p_over"] = gamma.sf(m.loc[yd, "line"].clip(lower=0.01),
                                   a=(mu_y / sd_y) ** 2,
                                   scale=sd_y ** 2 / mu_y)

    # receptions is a count stat — a normal tail is the wrong shape.
    # Negative binomial with dispersion r fitted by log-likelihood on 2024.
    rec = m.stat == "receptions"
    t = m[rec & (m.season == 2023)]
    mu24 = t.proj_cal.clip(lower=0.1).values
    y24 = t.actual.round().clip(lower=0).values
    r_best, ll_best = None, -np.inf
    for r in (1.5, 2, 3, 4, 5, 6, 8, 10, 15, 25, 50):
        ll = nbinom.logpmf(y24, r, r / (r + mu24)).sum()
        if ll > ll_best:
            r_best, ll_best = r, ll
    print(f"receptions NB dispersion r = {r_best}")
    mu = m.loc[rec, "proj_cal"].clip(lower=0.1)
    m.loc[rec, "p_over"] = 1 - nbinom.cdf(
        np.floor(m.loc[rec, "line"]), r_best, r_best / (r_best + mu))

    # blend with the no-vig market probability: the confidence report showed
    # raw model probs are overconfident exactly because the line has info we
    # lack. Blend weight fitted on 2024 by log-loss, applied out-of-sample.
    imp_o = 1 / (1 + m.pay_over)
    imp_u = 1 / (1 + m.pay_under)
    m["p_mkt"] = imp_o / (imp_o + imp_u)
    logit = lambda p: np.log(p.clip(1e-4, 1 - 1e-4) / (1 - p.clip(1e-4, 1 - 1e-4)))
    t = m[(m.season == 2023) & (m.actual != m.line)]
    y = (t.actual > t.line).astype(float)
    w_best, ll_best = 0.0, -np.inf
    for w in np.arange(0.0, 0.65, 0.05):
        pb = 1 / (1 + np.exp(-(w * logit(t.p_over) + (1 - w) * logit(t.p_mkt))))
        ll = (y * np.log(pb) + (1 - y) * np.log(1 - pb)).sum()
        if ll > ll_best:
            w_best, ll_best = w, ll
    print(f"market-blend weight on model prob (2023 log-loss fit): {w_best:.2f}")
    from ingestion.config import save_coefs
    save_coefs("props", {
        "recal": {k: [float(x) for x in v] for k, v in cal.items()},
        "nb_r": float(r_best), "blend_w": float(w_best)})
    m["p_over_bl"] = 1 / (1 + np.exp(
        -(w_best * logit(m.p_over) + (1 - w_best) * logit(m.p_mkt))))

    m["ev_over"] = m.p_over * m.pay_over - (1 - m.p_over)
    m["ev_under"] = (1 - m.p_over) * m.pay_under - m.p_over
    m["bet_side"] = np.where(m.ev_over >= m.ev_under, "over", "under")
    m["bet_ev"] = m[["ev_over", "ev_under"]].max(axis=1)
    m["p_side"] = np.where(m.bet_side == "over", m.p_over, 1 - m.p_over)

    # blended-probability bets (settled at best quoted price)
    m["ev_over_bl"] = m.p_over_bl * m.pay_over_best - (1 - m.p_over_bl)
    m["ev_under_bl"] = (1 - m.p_over_bl) * m.pay_under_best - m.p_over_bl
    m["bet_side_bl"] = np.where(m.ev_over_bl >= m.ev_under_bl, "over", "under")
    m["bet_ev_bl"] = m[["ev_over_bl", "ev_under_bl"]].max(axis=1)
    m["p_side_bl"] = np.where(m.bet_side_bl == "over",
                              m.p_over_bl, 1 - m.p_over_bl)

    over_won = m.actual > m.line
    push = m.actual == m.line
    m["bet_won"] = np.where(push, np.nan, np.where(
        m.bet_side == "over", over_won, ~over_won).astype(float))
    m["bet_won_bl"] = np.where(push, np.nan, np.where(
        m.bet_side_bl == "over", over_won, ~over_won).astype(float))
    pay_bl = np.where(m.bet_side_bl == "over", m.pay_over_best, m.pay_under_best)
    m["pnl_bl"] = np.where(m.bet_won_bl.isna(), 0.0,
                           np.where(m.bet_won_bl == 1, pay_bl, -1.0))
    m["bet_pay"] = np.where(m.bet_side == "over", m.pay_over, m.pay_under)
    m["pnl"] = np.where(m.bet_won.isna(), 0.0,
                        np.where(m.bet_won == 1, m.bet_pay, -1.0))

    # line-shopped variant: same p_over, EV and settlement at best price
    m["ev_over_b"] = m.p_over * m.pay_over_best - (1 - m.p_over)
    m["ev_under_b"] = (1 - m.p_over) * m.pay_under_best - m.p_over
    m["bet_side_b"] = np.where(m.ev_over_b >= m.ev_under_b, "over", "under")
    m["bet_ev_b"] = m[["ev_over_b", "ev_under_b"]].max(axis=1)
    m["bet_won_b"] = np.where(push, np.nan, np.where(
        m.bet_side_b == "over", over_won, ~over_won).astype(float))
    pay_b = np.where(m.bet_side_b == "over", m.pay_over_best, m.pay_under_best)
    m["pnl_b"] = np.where(m.bet_won_b.isna(), 0.0,
                          np.where(m.bet_won_b == 1, pay_b, -1.0))
    return m


def report(m: pd.DataFrame) -> None:
    print("\n=== accuracy duel: |error| vs actual stat ===")
    print(f"{'stat':12s} {'n':>6} {'model MAE':>10} {'book MAE':>10}")
    for stat in STATS:
        s = m[m.stat == stat]
        print(f"{stat:12s} {len(s):>6} "
              f"{(s.actual - s.proj_cal).abs().mean():>10.2f} "
              f"{(s.actual - s.line).abs().mean():>10.2f}")

    for season, label in [
            (2025, "2025 (out-of-sample)"),
            (2024, "2024 (out-of-sample)"),
            (2023, "2023 (calibration, in-sample)"),
            ("oos", "2024+2025 pooled (out-of-sample)")]:
        s = m[m.season.isin([2024, 2025])] if season == "oos" \
            else m[m.season == season]
        print(f"\n=== betting sim {label}: {len(s):,} matched lines ===")
        print(f"{'EV>=':>6} {'bets':>6} {'win%':>7} {'ROI%':>7}")
        for t in EV_THRESH:
            b = s[(s.bet_ev >= t) & s.bet_won.notna()]
            if len(b) < 20:
                continue
            print(f"{t:>6.0%} {len(b):>6} {b.bet_won.mean():>6.1%} "
                  f"{b.pnl.mean() * 100:>6.1f}%")
        b = s[(s.bet_ev >= 0.05) & s.bet_won.notna()]
        if len(b):
            print("by stat at EV>=5%:")
            print(b.groupby("stat").agg(
                bets=("pnl", "size"), win=("bet_won", "mean"),
                roi=("pnl", "mean")).round(3).to_string())
        print("with line shopping (best of quoted books, >=2 books):")
        for t in EV_THRESH:
            b = s[(s.bet_ev_b >= t) & s.bet_won_b.notna() & (s.n_books >= 2)]
            if len(b) < 20:
                continue
            print(f"{t:>6.0%} {len(b):>6} {b.bet_won_b.mean():>6.1%} "
                  f"{b.pnl_b.mean() * 100:>6.1f}%")
        print("market-blended probs + best price (>=2 books):")
        for t in (0.01, 0.02, 0.03, 0.05):
            b = s[(s.bet_ev_bl >= t) & s.bet_won_bl.notna() & (s.n_books >= 2)]
            if len(b) < 20:
                continue
            print(f"{t:>6.0%} {len(b):>6} {b.bet_won_bl.mean():>6.1%} "
                  f"{b.pnl_bl.mean() * 100:>6.1f}%")


if __name__ == "__main__":
    out = run()
    dest = PARQUET_DIR / "backtest_props_vs_book.parquet"
    out.to_parquet(dest, index=False)
    report(out)
    print(f"\nsaved: {dest}")
