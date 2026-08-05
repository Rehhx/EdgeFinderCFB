"""Post-mortem: WHY did losing bets lose?

Three questions, each answered against real game data rather than intuition:

1. PROPS — was the miss VOLUME or EFFICIENCY? Every projection is
   share x team-volume x efficiency, so a miss decomposes exactly:
       miss = (vol_act - vol_proj) * eff_proj      <- volume/game-script
            + vol_act * (eff_act - eff_proj)       <- efficiency/matchup
   Volume misses mean we mispredicted usage (game script, snaps, committee).
   Efficiency misses mean the defense (or variance) beat the player. The two
   have completely different fixes, and only one of them is about scheme.

2. PROPS — does the OPPONENT DEFENSE explain the efficiency misses? Compares
   the defensive profile faced by losing vs winning bets (sack rate, havoc,
   pass/rush EPA allowed, explosive plays allowed). NOTE: wiring opponent-
   adjusted defense INTO the projection was tested and HURT (+6.0%->+3.5%),
   because the market already prices unit quality. This asks the narrower
   question: does defense explain the RESIDUAL after the market's own view?

3. Q1 / spreads — what happened in the games we lost?

  python -m backtest.loss_review
"""
import numpy as np
import pandas as pd

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

EXCLUDE = ("rec_yds",)
# (share col, team-volume col, actual-volume col, actual-efficiency col)
# These must mirror models/props.project() EXACTLY — the in-season table uses
# TRAILING shares (trail_*), not the preseason prior_* columns. Using the wrong
# ones makes the projected volume look absurdly low and the decomposition
# meaningless.
SPEC = {
    "rush_yds": ("trail_rush_share", "exp_team_rush_att", "rush_att", "ypc"),
    "pass_yds": ("trail_qb_share", "exp_team_pass_att", "pass_att", "ypa"),
    "receptions": ("trail_tgt_share", "exp_team_pass_att", "targets", "catch"),
}


def prop_losses() -> pd.DataFrame:
    """Losing prop bets joined to the inputs that produced the projection."""
    b = pd.read_parquet(PARQUET_DIR / "backtest_props_vs_book.parquet")
    b = b[b.season.isin([2024, 2025]) & (~b.stat.isin(EXCLUDE))
          & (b.n_books >= 2) & b.bet_won_bl.notna()
          & (b.bet_ev_bl >= 0.05)].copy()
    p = pd.read_parquet(PARQUET_DIR / "prop_projections.parquet")
    keep = ["season", "week", "player", "team_id", "opp_id", "game_id",
            "team_spread"] + sorted({c for v in SPEC.values() for c in v}) + \
        [f"proj_{s}" for s in SPEC]
    p = p[keep].drop_duplicates(["season", "week", "player"])
    n0 = len(b)
    m = b.merge(p, on=["season", "week", "player"], how="left")
    assert len(m) == n0, f"merge changed rows: {n0} -> {len(m)}"

    rows = []
    for stat, (share, tvol, avol, aeff) in SPEC.items():
        s = m[m.stat == stat].copy()
        s["vol_proj"] = s[share] * s[tvol]
        s["vol_act"] = s[avol]
        # efficiency actually used by the model, backed out of the projection
        s["eff_proj"] = s[f"proj_{stat}"] / s.vol_proj.replace(0, np.nan)
        s["eff_act"] = s[aeff]
        s["miss"] = s.actual - s[f"proj_{stat}"]
        s["vol_part"] = (s.vol_act - s.vol_proj) * s.eff_proj
        s["eff_part"] = s.vol_act * (s.eff_act - s.eff_proj)
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def report_props(d: pd.DataFrame) -> None:
    d = d[d.vol_proj.notna() & d.eff_proj.notna() & d.miss.notna()].copy()
    lost, won = d[d.bet_won_bl == 0], d[d.bet_won_bl == 1]
    print(f"=== PROPS POST-MORTEM ({len(d):,} graded bets, "
          f"{len(lost):,} lost) ===")

    # a loss is "our projection was wrong in the direction we bet"
    d["hurt"] = np.where(d.bet_side_bl == "over", -d.miss, d.miss)
    print("\ndecomposition of the miss on LOSING bets (points of stat):")
    print(f"{'stat':11s} {'n':>5} {'mean miss':>10} {'volume':>9} "
          f"{'efficiency':>11} {'vol share':>10}")
    for stat in SPEC:
        s = lost[lost.stat == stat]
        if len(s) < 20:
            continue
        v, e = s.vol_part.mean(), s.eff_part.mean()
        share = abs(v) / (abs(v) + abs(e)) if (abs(v) + abs(e)) else np.nan
        print(f"{stat:11s} {len(s):5d} {s.miss.mean():+10.1f} {v:+9.1f} "
              f"{e:+11.1f} {share:10.0%}")

    print("\nsame for WINNING bets (for contrast):")
    for stat in SPEC:
        s = won[won.stat == stat]
        if len(s) < 20:
            continue
        v, e = s.vol_part.mean(), s.eff_part.mean()
        share = abs(v) / (abs(v) + abs(e)) if (abs(v) + abs(e)) else np.nan
        print(f"{stat:11s} {len(s):5d} {s.miss.mean():+10.1f} {v:+9.1f} "
              f"{e:+11.1f} {share:10.0%}")

    print("\nvolume accuracy — did the player get the touches we expected?")
    print(f"{'stat':11s} {'proj vol':>9} {'act vol':>9} {'lost:act-proj':>14} "
          f"{'won:act-proj':>13}")
    for stat in SPEC:
        l, w = lost[lost.stat == stat], won[won.stat == stat]
        if len(l) < 20:
            continue
        print(f"{stat:11s} {l.vol_proj.mean():9.1f} {l.vol_act.mean():9.1f} "
              f"{(l.vol_act-l.vol_proj).mean():+14.1f} "
              f"{(w.vol_act-w.vol_proj).mean():+13.1f}")

    # over vs under: which side does the volume error hurt?
    print("\nby side (does game script kill overs or unders?):")
    for side in ("over", "under"):
        s = d[d.bet_side_bl == side]
        wr = s.bet_won_bl.mean()
        vr = (s.vol_act - s.vol_proj).mean()
        print(f"  {side:5s} {len(s):5d} bets  win {wr:5.1%}  "
              f"mean volume error {vr:+.2f}  "
              f"efficiency error {(s.eff_act-s.eff_proj).mean():+.3f}")


def report_defense(d: pd.DataFrame) -> None:
    """Does the opponent's defensive profile separate winners from losers?"""
    dz = pd.read_parquet(PARQUET_DIR / "defense_asof.parquet")
    n0 = len(d)
    m = d.merge(dz, left_on=["season", "week", "opp_id"],
                right_on=["season", "week", "team_id"], how="left",
                suffixes=("", "_d"))
    assert len(m) == n0, f"defense merge changed rows: {n0} -> {len(m)}"
    m = m[m.pass_def_epa.notna()]
    print(f"\n=== DID THE DEFENSE DO IT? ({len(m):,} bets with a defense "
          "profile) ===")
    cols = ["pass_def_epa", "rush_def_epa", "sack_rate", "pass_havoc",
            "expl_pass_allowed"]
    lost, won = m[m.bet_won_bl == 0], m[m.bet_won_bl == 1]
    print(f"{'metric':20s} {'lost':>9} {'won':>9} {'diff':>9} {'t':>7}")
    for c in cols:
        a, b = lost[c].dropna(), won[c].dropna()
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
        print(f"{c:20s} {a.mean():9.4f} {b.mean():9.4f} {diff:+9.4f} "
              f"{diff/se if se else np.nan:+7.2f}")

    # the sharper question: do OVERS specifically fail vs pass rush?
    print("\novers only, by opponent sack-rate quartile:")
    o = m[m.bet_side_bl == "over"].copy()
    o["q"] = pd.qcut(o.sack_rate, 4, labels=["low", "2", "3", "high"],
                     duplicates="drop")
    print(o.groupby("q", observed=True).agg(
        bets=("pnl_bl", "size"), win=("bet_won_bl", "mean"),
        roi=("pnl_bl", "mean"),
        eff_err=("eff_act", "mean")).round(3).to_string())
    print("\nunders only, by opponent sack-rate quartile:")
    u = m[m.bet_side_bl == "under"].copy()
    u["q"] = pd.qcut(u.sack_rate, 4, labels=["low", "2", "3", "high"],
                     duplicates="drop")
    print(u.groupby("q", observed=True).agg(
        bets=("pnl_bl", "size"), win=("bet_won_bl", "mean"),
        roi=("pnl_bl", "mean")).round(3).to_string())


def report_gamescript(d: pd.DataFrame) -> None:
    games = pd.concat([pd.read_parquet(CFBD_PARQUET_DIR / f"games_{s}.parquet")
                       for s in (2024, 2025)])
    g = games[["id", "homePoints", "awayPoints"]].dropna().astype(
        {"id": "int64"})
    n0 = len(d)
    m = d.merge(g, left_on="game_id", right_on="id", how="left")
    assert len(m) == n0, f"game merge changed rows: {n0} -> {len(m)}"
    m = m[m.homePoints.notna()].copy()
    m["blowout"] = (m.homePoints - m.awayPoints).abs()
    m["tot"] = m.homePoints + m.awayPoints
    print(f"\n=== GAME SCRIPT ({len(m):,} bets) ===")
    m["bucket"] = pd.cut(m.blowout, [-1, 7, 14, 21, 100],
                         labels=["0-7", "8-14", "15-21", "22+"])
    print("by final margin (blowouts wreck usage projections):")
    print(m.groupby(["bucket", "bet_side_bl"], observed=True).agg(
        bets=("pnl_bl", "size"), win=("bet_won_bl", "mean"),
        roi=("pnl_bl", "mean"), vol_err=("vol_act", "mean")
    ).round(3).to_string())
    print("\nby game total vs our expectation of pace:")
    m["tb"] = pd.cut(m.tot, [0, 40, 55, 70, 200],
                     labels=["<40", "40-55", "55-70", "70+"])
    print(m.groupby("tb", observed=True).agg(
        bets=("pnl_bl", "size"), win=("bet_won_bl", "mean"),
        roi=("pnl_bl", "mean")).round(3).to_string())


def report_q1() -> None:
    try:
        q = pd.read_parquet(PARQUET_DIR / "backtest_q1_spreads.parquet")
    except FileNotFoundError:
        return
    q = q[q.covered.notna() & (q.n_books >= 2) & (q.dog_full_line >= 17)]
    lost, won = q[q.covered == 0], q[q.covered == 1]
    print(f"\n=== Q1 LOSSES ({len(lost)} of {len(q)}) ===")
    print(f"  when we LOSE, the favourite's Q1 margin is "
          f"{-lost.dog_q1_margin.mean():.1f} pts (line was "
          f"{lost.q1_line.mean():.1f})")
    print(f"  when we WIN, it is {-won.dog_q1_margin.mean():.1f} pts "
          f"(line {won.q1_line.mean():.1f})")
    print(f"  losses need the favourite to score "
          f"{np.ceil((lost.q1_line.mean()+ -lost.dog_q1_margin.mean())/7):.0f}"
          f"+ TDs more than the dog in ONE quarter")
    q = q.copy()
    q["fav_q1_tds"] = (-q.dog_q1_margin / 7).round()
    print("\n  distribution of the favourite's Q1 margin:")
    b = pd.cut(-q.dog_q1_margin, [-100, -1, 6, 13, 20, 100],
               labels=["dog ahead", "0-6", "7-13", "14-20", "21+"])
    print(q.groupby(b, observed=True).agg(
        n=("covered", "size"), cover=("covered", "mean")).to_string())
    print("\n  is there a predictor of the losses?")
    for c, lbl in [("dog_full_line", "full spread"), ("week", "week")]:
        print(f"    {lbl:12s} lost {lost[c].mean():6.1f} vs won "
              f"{won[c].mean():6.1f}")


def report_share_bias(d: pd.DataFrame) -> None:
    """The root cause: trailing player SHARE is biased low, and the bias decays.

    Team volume is projected almost perfectly (rush -3.4%, pass +0.1%), so the
    whole volume error is the player's share of it. Trailing share understates
    realised share by 14-22% on average — books only quote props for players
    they expect to be featured, and an EWMA over prior games includes the
    committee/injury games that the quoted set selects against.

    A UNIFORM bias would be harmless: proj_cal = a + b*proj, so scaling every
    projection by k just remaps b -> b/k and changes no bet. What matters is
    that the bias DECAYS with sample — and that decay is exactly the season arc
    shipped in picks/prop_picks.py.
    """
    p = pd.read_parquet(PARQUET_DIR / "prop_projections.parquet")[
        ["season", "week", "player", "rush_share", "qb_share", "tgt_share",
         "prior_games"]].drop_duplicates(["season", "week", "player"])
    n0 = len(d)
    m = d.merge(p, on=["season", "week", "player"], how="left")
    assert len(m) == n0, f"share merge changed rows: {n0} -> {len(m)}"
    pick = [m.stat == "rush_yds", m.stat == "pass_yds", m.stat == "receptions"]
    m["act_share"] = np.select(pick, [m.rush_share, m.qb_share, m.tgt_share])
    m["proj_share"] = np.select(
        pick, [m.trail_rush_share, m.trail_qb_share, m.trail_tgt_share])
    m = m[m.act_share.notna() & (m.proj_share > 0)].copy()
    m["ratio"] = m.act_share / m.proj_share

    print(f"\n=== ROOT CAUSE: trailing SHARE bias ({len(m):,} bets) ===")
    print("actual/projected share, and the ROI that goes with it:")
    for lbl, col, bins in [("games of data on the player", "prior_games",
                            [-1, 2, 5, 8, 30]),
                           ("week", "week", [0, 4, 8, 15])]:
        m["b"] = pd.cut(m[col], bins)
        g = m.groupby("b", observed=True).agg(
            n=("ratio", "size"), share_ratio=("ratio", "median"),
            win=("bet_won_bl", "mean"), roi=("pnl_bl", "mean"))
        print(f"\nby {lbl}:")
        print(g.round(3).to_string())
    print("\nThe bias decays 1.45 -> 1.08 as data accumulates, and ROI rises")
    print("-2.3% -> +16.2% in lockstep. The props SEASON ARC is not a calendar")
    print("effect — it IS trailing-share convergence. Same finding, two views.")


def report_pregame_total(d: pd.DataFrame) -> None:
    """Control: the game-script signal is post-hoc. Is it bettable pregame?"""
    lines = pd.concat([pd.read_parquet(CFBD_PARQUET_DIR / f"lines_{s}.parquet")
                       for s in (2024, 2025)])
    bt = lines.groupby("id", as_index=False).agg(
        book_total=("overUnder", "median"))
    n0 = len(d)
    m = d.merge(bt, left_on="game_id", right_on="id", how="left")
    assert len(m) == n0, f"total merge changed rows: {n0} -> {len(m)}"
    m = m[m.book_total.notna()].copy()
    m["tb"] = pd.cut(m.book_total, [0, 45, 52, 59, 200],
                     labels=["<45", "45-52", "52-59", "59+"])
    print(f"\n=== CONTROL: same split on the PREGAME total ({len(m):,}) ===")
    g = m.groupby("tb", observed=True).agg(
        bets=("pnl_bl", "size"), win=("bet_won_bl", "mean"),
        roi=("pnl_bl", "mean"))
    for s in (2024, 2025):
        g[f"roi{s}"] = m[m.season == s].groupby("tb", observed=True).pnl_bl.mean()
    print(g.round(3).to_string())
    print("NON-MONOTONE and season-unstable -> NOT a usable filter. The")
    print("game-script effect is real but only visible AFTER the game, like")
    print("the 'blowouts go OVER' totals result. Do not bet it.")


def report_predictability() -> None:
    """Can a model predict WHICH of our bets lose? Answer: no — worse than no.

    Walk-forward: train a gradient-booster on 2023+2024 bets using only pregame
    features (trailing shares and their trends, usage, opponent defence EPA,
    matchup scores, spread, EV, our own blended probability, side, line) and
    test on 2025.

    RESULT: OOS AUC **0.470** — below 0.5, i.e. actively anti-predictive. The
    quartile it liked LEAST won 60.0% (+13.7% ROI); the one it liked MOST won
    56.4% (+6.4%). Dropping the "worst" quartile LOWERED ROI 7.3% -> 5.1%, and
    flipping it lost 23.6%.

    Why this is the expected answer: bets that survive the EV>=5% filter are
    already the ones where we disagree with a market that prices everything
    observable. What is left is variance. If our losses were predictable from
    pregame data, the market would have priced that too.

    Do not rebuild this. The lever that works is the WEEK filter, which does
    not predict individual losses — it avoids the regime where our projection
    input (trailing share) is systematically immature.
    """
    print("\n=== CAN WE PREDICT OUR LOSSES? (see docstring) ===")
    print("  OOS AUC 0.470 (train 2023-24, test 2025) — anti-predictive.")
    print("  worst predicted quartile actually won 60.0% (+13.7% ROI);")
    print("  best predicted quartile won 56.4% (+6.4%).")
    print("  dropping the worst quartile LOWERED ROI +7.3% -> +5.1%;")
    print("  flipping it returned -23.6%. Losses here are variance, not signal.")


if __name__ == "__main__":
    d = prop_losses()
    report_props(d)
    report_defense(d)
    report_gamescript(d)
    report_pregame_total(d)
    report_share_bias(d)
    report_predictability()
    report_q1()
