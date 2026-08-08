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

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

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

    # ONE EVENT PER PLAYER-WEEK. CFBD's week-0/week-1 overlap means a handful
    # of (season, week, stat, player) keys span two different events; without
    # this, the consensus below collapses them and the SAME line is graded
    # twice against two different games (117 of 14,183 graded rows, 0.83%).
    # Keep the event the player actually appears in most — quote count is the
    # tie-break, then the earliest kickoff.
    ev = (raw.groupby(["season", "week", "stat", "player", "event_id"])
             .agg(q=("price", "size"), c=("commence", "min")).reset_index()
             .sort_values(["q", "c"], ascending=[False, True])
             .drop_duplicates(["season", "week", "stat", "player"]))
    raw = raw.merge(ev[["season", "week", "stat", "player", "event_id"]],
                    on=["season", "week", "stat", "player", "event_id"],
                    how="inner")

    # a quotable line = over AND under from the same book at the SAME point
    kick = ev.set_index(["season", "week", "stat", "player"]).c
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
    # kickoff of the event this line was quoted for — lets the projection merge
    # pin the RIGHT game when CFBD labels two of a team's games the same week
    # (its week-0 kickoff weekend is tagged week 1, so 267 player-weeks have
    # two distinct game_ids and the line would otherwise grade against both).
    cons = cons.merge(kick.rename("kickoff").reset_index(),
                      on=["season", "week", "stat", "player"], how="left")
    return cons


def payout(price: pd.Series) -> pd.Series:
    return np.where(price < 0, 100 / price.abs(), price / 100)


# Backtest calibration: 2023 only, so 2024-25 stay strictly out-of-sample.
# PRODUCTION calibration uses every available season — see fit_production().
CAL_SEASONS = (2023,)

# How the market-blend weight is chosen.
#   "grid"     argmax of calibration log-loss (the historical behaviour)
#   "fixed"    a constant — no estimation, no estimation variance
#   "average"  average the blended PROBABILITY over BLEND_BAND
#
# ⚠️ THE LOG-LOSS SURFACE IN w IS NEARLY FLAT. Measured on the 2023 rows
# (n=4,048): the continuous argmax is 0.310, and moving +-0.05 away costs
# 0.523 log-loss out of ~2,781 — **0.000129 per row**. So the data cannot
# meaningfully distinguish w=0.25 from w=0.40, yet those produce books of
# +0.4u and +39u (backtest/props_blend_sweep.py). A FINER GRID IS NOT THE FIX:
# it buys precision on a nearly-flat objective, i.e. a sharper estimate of
# noise. The grid is refined to 0.01 anyway so the artefact is gone, but the
# real question is whether to estimate w at all.
# SHIPPED "fixed" 2026-08-05 (backtest/blend_mode_ab.py, 40 bootstrap refits):
#   mode      bets  ROI     u/szn   2024    2025  | boot sd  5th %ile
#   grid       534  +15.1%  +62.7   +66.0   +59.3 |    15.3     +11.3
#   fixed      504  +15.2%  +56.9   +57.9   +55.9 |     7.0     +32.8
#   average    563  +14.1%  +63.1   +84.2   +42.0 |     7.7     +40.1
# Estimating w cost MORE THAN HALF the stability for nothing: sd 15.3 -> 7.0,
# 5th %ile +11.3 -> +32.8. The clincher is what grid and fixed differ BY — the
# refined 0.01 grid picks 0.31, fixed uses 0.30, and that ONE HUNDREDTH moves
# the season by 5.8 units and 30 bets. A parameter the data cannot identify
# should not be estimated.
# "average" scores more units but its seasons split 84.2/42.0 versus fixed's
# 57.9/55.9, and it needs BLEND_BAND — a NEW discretionary parameter, which is
# the kind of choice that created this problem. Rejected on season balance, the
# same rule that set the props week gate.
BLEND_MODE = "fixed"
BLEND_FIXED = 0.30
BLEND_BAND = (0.20, 0.45)


def _cal_weeks(m: pd.DataFrame, rng, cal_seasons):
    """Week draw for the calibration season. None = use it as-is.

    With `rng`, whole WEEKS are drawn with replacement so
    backtest/props_pricing_stability.py can ask how much of the measured edge
    depends on this single season's draw. Weeks rather than rows, to preserve
    same-week and same-game correlation. The draw is made ONCE and reused for
    all three fits (recal, NB dispersion, blend weight) — they must see the
    same pseudo-season or the bootstrap understates the coupling between them.
    """
    if rng is None:
        return None
    weeks = m.loc[m.season.isin(cal_seasons), ["season", "week"]]
    weeks = list(weeks.drop_duplicates().itertuples(index=False, name=None))
    idx = rng.integers(0, len(weeks), len(weeks))
    return [weeks[i] for i in idx]


def _cal_subset(m: pd.DataFrame, weeks, cal_seasons) -> pd.DataFrame:
    """Calibration rows, honouring a bootstrap week draw. Only the FIT sees
    these; the graded book `m` is never resampled."""
    base = m[m.season.isin(cal_seasons)]
    if weeks is None:
        return base
    return pd.concat([base[(base.season == s) & (base.week == w)]
                      for s, w in weeks], ignore_index=True)


def run(proj_path=None, write_coefs: bool = True, cal_rng=None,
        cal_seasons=CAL_SEASONS) -> pd.DataFrame:
    """Price and grade the props book.

    `proj_path` swaps in an alternative projections table so a candidate model
    can be run through the IDENTICAL pricing pipeline (recalibration, gamma/NB
    tails, market blend, EV, line shopping) for a fair head-to-head.
    `write_coefs=False` stops an experiment from overwriting the production
    coefficients in model_coefs.json — always pass it when testing a candidate.
    """
    lines = consensus_lines()
    proj = pd.read_parquet(proj_path or
                           (PARQUET_DIR / "prop_projections.parquet"))

    frames = []
    # game_id + kickoff let a line be pinned to the game it was quoted for
    sched = {}
    try:
        import json
        for szn in sorted(proj.season.dropna().unique()):
            g = pd.read_parquet(CFBD_PARQUET_DIR / f"games_{int(szn)}.parquet",
                                columns=["id", "startDate"])
            sched.update(dict(zip(g.id.astype("int64"),
                                  pd.to_datetime(g.startDate, utc=True,
                                                 format="ISO8601"))))
    except Exception:
        sched = {}

    for stat in STATS:
        cols = ["season", "week", "player", "game_id", stat, f"proj_{stat}"]
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
    # PIN THE GAME. CFBD tags its week-0 kickoff weekend as week 1, so 267
    # player-weeks carry TWO distinct game_ids and a single quoted line was
    # being graded against BOTH (117 duplicated rows, 0.83%). Keep the game
    # whose kickoff is nearest the time the line was actually quoted for.
    if sched and "game_id" in m.columns:
        gt = m.game_id.map(sched)
        kt = pd.to_datetime(m.kickoff, utc=True, format="ISO8601",
                            errors="coerce")
        m["_gap"] = (gt - kt).abs()
        m = (m.sort_values("_gap")
               .drop_duplicates(["season", "week", "stat", "player"],
                                keep="first")
               .drop(columns="_gap"))
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
    cal_weeks = _cal_weeks(m, cal_rng, cal_seasons)
    cal_src = _cal_subset(m, cal_weeks, cal_seasons)
    cal = {}
    for stat in STATS:
        t = cal_src[cal_src.stat == stat]
        X = np.column_stack([t.proj, t.mu_c, t.ppa_c, np.ones(len(t))])
        coef, *_ = np.linalg.lstsq(X, t.actual, rcond=None)
        sigma = (t.actual - (X @ coef)).std()
        # (a, b, sigma, c_matchup, c_ppa)
        cal[stat] = (float(coef[3]), float(coef[0]), float(sigma),
                     float(coef[1]), float(coef[2]))
    print(f"{sorted(cal_seasons)}-fitted recal (a, b, sigma, c_matchup, c_ppa):",
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
    cal_now = _cal_subset(m, cal_weeks, cal_seasons)  # same draw, w/ proj_cal
    t = cal_now[cal_now.stat == "receptions"]
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
    cal_now = _cal_subset(m, cal_weeks, cal_seasons)  # same draw, w/ p_mkt
    t = cal_now[cal_now.actual != cal_now.line]
    y = (t.actual > t.line).astype(float)
    w_best, ll_best = 0.0, -np.inf
    for w in np.arange(0.0, 0.65, 0.01):
        pb = 1 / (1 + np.exp(-(w * logit(t.p_over) + (1 - w) * logit(t.p_mkt))))
        ll = (y * np.log(pb) + (1 - y) * np.log(1 - pb)).sum()
        if ll > ll_best:
            w_best, ll_best = w, ll
    if BLEND_MODE == "fixed":
        w_best = BLEND_FIXED
    print(f"market-blend weight: {w_best:.2f} "
          f"(mode={BLEND_MODE}, log-loss argmax kept for reference)")
    if write_coefs:
        from ingestion.config import save_coefs
        save_coefs("props", {
            "recal": {k: [float(x) for x in v] for k, v in cal.items()},
            "nb_r": float(r_best), "blend_w": float(w_best)})
    if BLEND_MODE == "average":
        # Model-average over the region the data cannot distinguish. Every w in
        # BLEND_BAND sits within ~2 log-loss of the optimum out of ~2,781, so
        # picking one is arbitrary; averaging the PROBABILITY across them is
        # the honest summary and moves smoothly with the data.
        ws = np.arange(BLEND_BAND[0], BLEND_BAND[1] + 1e-9, 0.01)
        acc = np.zeros(len(m))
        for w in ws:
            acc += 1 / (1 + np.exp(
                -(w * logit(m.p_over) + (1 - w) * logit(m.p_mkt))))
        m["p_over_bl"] = acc / len(ws)
    else:
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


def season_arc(m: pd.DataFrame) -> None:
    """Does the props edge change through the season? (It does — a lot.)

    Graded on the PRODUCTION selection: blended prob, best price, >=2 books,
    rec_yds excluded, EV>=5%. Both the week split and the stake plan below
    are what picks/prop_picks.py implements.
    """
    oos = m[m.season.isin([2024, 2025]) & (~m.stat.isin(EXCLUDE_STATS))
            & (m.n_books >= 2) & m.bet_won_bl.notna()
            & (m.bet_ev_bl >= 0.05)].copy()
    oos["prime"] = oos.bet_ev_bl.between(0.05, 0.08, inclusive="left")
    oos["wk"] = pd.cut(oos.week, [0, 4, 8, 15], labels=["1-4", "5-8", "9-15"])

    print(f"\n=== SEASON ARC (production selection, {len(oos)} OOS bets) ===")
    g = oos.groupby("wk", observed=True).agg(
        bets=("pnl_bl", "size"), win=("bet_won_bl", "mean"),
        roi=("pnl_bl", "mean"))
    g["per_szn"] = (g.bets / 2).round(0)
    for yr in (2024, 2025):
        g[f"roi{yr}"] = oos[oos.season == yr].groupby(
            "wk", observed=True).pnl_bl.mean()
    print(g.round(3).to_string())

    print("\nby EV band x week (the loser is wk1-4 STANDARD):")
    print(oos.groupby(["prime", "wk"], observed=True).agg(
        bets=("pnl_bl", "size"), win=("bet_won_bl", "mean"),
        roi=("pnl_bl", "mean")).round(3).to_string())

    # significance: is the week gradient real, or 780-bet noise?
    x, y = oos.week.values.astype(float), oos.pnl_bl.values
    b = np.polyfit(x, y, 1)[0]
    se = ((y - np.polyval(np.polyfit(x, y, 1), x)).std(ddof=2)
          / (x.std() * np.sqrt(len(x))))
    gap = oos[oos.week >= 5].pnl_bl.mean() - oos[oos.week <= 4].pnl_bl.mean()
    rng = np.random.default_rng(42)
    perm = np.array([(lambda p: y[p >= 5].mean() - y[p <= 4].mean())(
        rng.permutation(x)) for _ in range(5000)])
    print(f"\nslope {b*100:+.2f} pp ROI per week (t={b/se:+.2f}) | "
          f"wk>=5 minus wk<=4 gap {gap*100:+.1f} pp | "
          f"permutation p={np.mean(perm >= gap):.4f}")

    print(f"\n{'stake plan':34s} {'bets/szn':>9} {'win%':>7} {'ROI':>8} "
          f"{'units/szn':>10}")
    for label, keep in [
            ("all weeks (old)", oos.index),
            ("wk1-4 STANDARD off (NEW)",
             oos.index[(oos.week >= 5) | oos.prime]),
            ("all of wk1-4 off", oos.index[oos.week >= 5])]:
        d = oos.loc[keep]
        u = (d.pnl_bl * np.where(d.prime, 2.0, 1.0)).sum() / 2
        print(f"{label:34s} {len(d)/2:9.0f} {d.bet_won_bl.mean():6.1%} "
              f"{d.pnl_bl.mean()*100:+7.1f}% {u:+10.1f}")


def walkforward(proj_path=None) -> pd.DataFrame:
    """Price each test season on EVERY season that preceded it.

    The fixed scheme calibrates everything on 2023, which throws away real
    information: when we grade 2025, the 2024 season has already happened and
    using it is still strictly out-of-sample. This is the only way to get a
    "second calibration season" without new data — 2022 has no prop lines at
    all (The Odds API historical props start May 2023), so there is no earlier
    season to buy.

        2024 priced on 2023        (1 season, unchanged)
        2025 priced on 2023+2024   (2 seasons, DOUBLE the calibration data)

    2023 itself cannot be priced this way and is dropped — it has no
    predecessor. That is the honest cost: the pooled book loses the calibration
    season it was never allowed to bet anyway.

    Implemented by calling run() once per test season rather than refactoring
    its internals, so the pricing path is provably identical to production.

    RESULT (2026-08-05, backtest/props_walkforward_ab.py) — NOT adopted for the
    backtest. 2025 is the only season that differs, and it gains:
        1 cal season   206 bets  +12.1% ROI  +39.0 u  cal MAE 30.979
        2 cal seasons  132 bets  +13.5% ROI  +28.3 u  cal MAE 30.934
    A second season makes the model better PER BET and more conservative, but
    36% fewer bets clear EV>=4%, so total units FALL. The calibration MAE gain
    is 0.15% — which confirms out-of-sample what props_pricing_stability.py
    saw in-sample: the pricing fragility is a COARSE-GRID-SEARCH artifact, not
    a sample-size problem, so buying more calibration data barely moves it.
    Kept as a research tool. NOTE production already calibrates on every
    available season via fit_production(), so production is not affected.
    """
    seasons = sorted(TEST_SEASONS)
    frames = []
    for i, s in enumerate(seasons):
        prior = tuple(seasons[:i])
        if not prior:
            continue
        m = run(proj_path=proj_path, write_coefs=False, cal_seasons=prior)
        frames.append(m[m.season == s])
        print(f"  season {s} priced on {list(prior)}: {len(frames[-1]):,} rows")
    return pd.concat(frames, ignore_index=True)


def fit_production() -> None:
    """Refit the pricing calibration on EVERY available season, for live picks.

    The backtest must calibrate on 2023 alone to keep 2024-25 out-of-sample.
    Production has no such constraint and should not throw away two seasons of
    matched lines: `picks/prop_picks.py` reads these coefficients from
    model_coefs.json, so until 2026-08-04 the LIVE system was pricing every bet
    off a single season's calibration.

    That is not a cosmetic difference. backtest/props_pricing_stability.py
    bootstrapped the one-season fit and found **sd 15.6 units/season** — versus
    3.5 for the team-volume model — with the bet count swinging between 33 and
    642 depending on the draw, because sigma and the blend weight decide which
    bets clear EV>=5% at all. Three seasons instead of one is the single
    cheapest variance reduction available to us.

    ⚠️ The book returned here is IN-SAMPLE. Its ROI must never be quoted as a
    result; this function exists only to write coefficients.
    """
    seasons = tuple(sorted(TEST_SEASONS))
    print(f"\n{'='*66}\n=== PRODUCTION calibration refit on {list(seasons)}\n"
          f"{'='*66}")
    run(write_coefs=True, cal_seasons=seasons)
    print("\nwrote props recal/nb_r/blend_w to model_coefs.json for LIVE picks.")
    print("[!] in-sample by construction - do NOT quote the ROI above.")


TEST_SEASONS = (2023, 2024, 2025)

if __name__ == "__main__":
    # Backtest first, with the honest 2023-only calibration, and do NOT let it
    # write coefficients — production gets its own fit below.
    out = run(write_coefs=False)
    dest = PARQUET_DIR / "backtest_props_vs_book.parquet"
    out.to_parquet(dest, index=False)
    report(out)
    season_arc(out)
    print(f"\nsaved: {dest}")
    fit_production()
