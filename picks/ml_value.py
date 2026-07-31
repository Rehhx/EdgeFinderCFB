"""Moneyline value picks — ⚠️ INFORMATIONAL ONLY, DO NOT BET (see below).

**2026-07-30 verdict (backtest/ml_value_history.py, 8 seasons, REAL historical
moneyline prices with line shopping): moneyline betting has NO edge for us.**
  model probability   -5.0% ROI, positive in 1 of 5 seasons
  30% market blend    -6.1% to -8.6% ROI, 0-1 of 5 seasons
  pure line shopping  looks +12% at EV>=10% but bets +828 avg longshots and
                      swings +25%/+7.6%/-16.1% by season -> variance, not edge
Why: our win probabilities are well CALIBRATED (verified 8 seasons) but the
market is simply a better forecaster, and moneyline vig — especially on dogs —
eats the remainder. Calibration != profitability.

This module still prices the board (useful context, and the probabilities feed
nothing else), but its output is NOT a bet list. Bet the spread edges instead:
big-dog spreads (62.3% PREMIUM) and 1H spreads (57.8%).


Backtest calibration (backtest/confidence_report.py) showed the model's ML
probabilities are the best-calibrated output — even under-confident at
60-80% stated. EV is still computed two ways:
  ev_model : raw model probability (calibration-backed)
  ev_cons  : probability shrunk halfway (logit space) toward the no-vig
             market probability — the ranking number, keeps us honest on a
             July opening board.
Picks = conservative EV >= 3%, |spread| <= 21 (blowout guard), price not a
lottery ticket.

  python -m picks.ml_value
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ingestion.config import PROJECT_ROOT
from picks.edge_report import payout, run

# 0.3 on the July opening board (stale-ratings regime, mirrors the props-
# validated blend); move toward 0.5 in-season once ratings are live
MODEL_WEIGHT = 0.3
MIN_EV = 0.03
MAX_PRICE = 600  # skip lottery-ticket dogs


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def overhaul_table(season: int = 2026) -> pd.DataFrame:
    """Incoming portal talent per school — teams the model must not fade.

    The model's season-level prior underweights transfer classes (small
    fitted coefficient + returning production unpublished in July), so a
    faded team with a big portal haul (e.g. 2026 Oklahoma State: 55 in,
    0.98 QB) is a blind spot, not an edge.
    """
    from features.roster_priors import REPLACEMENT_RATING, STAR_TO_RATING
    from ingestion.config import CFBD_PARQUET_DIR
    p = pd.read_parquet(CFBD_PARQUET_DIR / f"portal_{season}.parquet")
    p = p.dropna(subset=["destination"])
    rating = p.rating.fillna(p.stars.map(STAR_TO_RATING)) \
        .fillna(REPLACEMENT_RATING)
    p = p.assign(value=(rating - REPLACEMENT_RATING).clip(lower=0),
                 r=rating)
    g = p.groupby("destination").agg(
        in_value=("value", "sum"),
        qb_max=("r", lambda s: s[p.loc[s.index, "position"] == "QB"].max()))
    g["in_pct"] = g.in_value.rank(pct=True)
    return g


def build_picks(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["ml_home_best", "ml_away_best"]).copy()
    d = d[d.book_spread.abs() <= 21]
    if "both_rated" in d.columns:
        d = d[d.both_rated]  # drop FCS-fallback ratings (e.g. NDSU games)

    imp_h = 1 / (1 + d.ml_home_best.map(payout))
    imp_a = 1 / (1 + d.ml_away_best.map(payout))
    d["p_mkt_home"] = imp_h / (imp_h + imp_a)
    d["p_cons_home"] = 1 / (1 + np.exp(
        -(MODEL_WEIGHT * logit(d.p_home)
          + (1 - MODEL_WEIGHT) * logit(d.p_mkt_home))))

    rows = []
    for g in d.itertuples():
        for side, p_m, p_c, price in [
            ("home", g.p_home, g.p_cons_home, g.ml_home_best),
            ("away", 1 - g.p_home, 1 - g.p_cons_home, g.ml_away_best),
        ]:
            if abs(price) > MAX_PRICE and price > 0:
                continue
            pay = payout(price)
            rows.append({
                "commence": g.commence[:10], "away": g.away, "home": g.home,
                "pick": g.home if side == "home" else g.away,
                "best_price": int(price),
                "p_model": round(p_m, 3), "p_market": round(
                    g.p_mkt_home if side == "home" else 1 - g.p_mkt_home, 3),
                "conf": round(p_c, 3),  # calibrated win prob of the pick
                "ev_model": round(p_m * pay - (1 - p_m), 3),
                "ev_cons": round(p_c * pay - (1 - p_c), 3),
            })
    picks = pd.DataFrame(rows)
    picks = picks[picks.ev_cons >= MIN_EV].sort_values(
        "ev_cons", ascending=False)

    # overhaul guard: the faded team (the opponent of our pick)
    from ingestion.config import CFBD_PARQUET_DIR
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    full_to_school = {f"{r.school} {r.mascot}": r.school
                      for r in teams.itertuples()}
    ov = overhaul_table()
    picks["faded"] = np.where(picks.pick == picks.home,
                              picks.away, picks.home)
    faded_school = picks.faded.map(full_to_school).fillna(picks.faded)
    picks["fade_portal_pct"] = faded_school.map(ov.in_pct).fillna(0).round(2)
    picks["fade_qb_in"] = faded_school.map(ov.qb_max).fillna(0).round(2)
    picks["overhaul_risk"] = ((picks.fade_portal_pct >= 0.75)
                              | (picks.fade_qb_in >= 0.92))
    return picks


def main() -> None:
    df = run()
    picks = build_picks(df)
    clean = picks[~picks.overhaul_risk].drop(columns=["faded"])
    risky = picks[picks.overhaul_risk].drop(columns=["faded"])
    day = datetime.now(timezone.utc).date().isoformat()
    path = PROJECT_ROOT / "reports" / f"ml_value_{day}.md"
    lines = [
        f"# Moneyline Value — {day} (2026 opening board)",
        "\n> ⚠️ **INFORMATIONAL ONLY — DO NOT BET.** 8-season backtest at real "
        "prices with line shopping: model −5.0% ROI (1/5 seasons positive), "
        "blend −6% to −9%. Our ML probabilities are calibrated but the market "
        "forecasts better and the vig eats the rest. Bet the spread edges "
        "instead (big-dog 62.3% PREMIUM, 1H 57.8%).\n",
        f"\n{len(clean)} playable picks at conservative EV ≥ {MIN_EV:.0%} "
        f"(model prob shrunk {1 - MODEL_WEIGHT:.0%} toward no-vig market, "
        "settled at best quoted price).",
        "\nBacktest context: model ML picks stated 60–70% hit 67.9%, stated "
        "70–80% hit 83.5% (2024–25, ~1,600 games). Opening-board caveat: "
        "2026 priors are partial until rosters publish in August.\n",
        (f"**Best edge:** {clean.iloc[0].pick} {clean.iloc[0].best_price:+d} "
         f"({clean.iloc[0].away} @ {clean.iloc[0].home}) — "
         f"{clean.iloc[0].ev_cons:+.0%} conservative EV  \n"
         f"**Best confidence:** "
         f"{clean.loc[clean.conf.idxmax()].pick} "
         f"{clean.loc[clean.conf.idxmax()].best_price:+d} — "
         f"{clean.conf.max():.0%} calibrated win prob\n"
         if len(clean) else ""),
        "## Playable\n",
        clean.to_markdown(index=False),
        f"\n## Excluded — overhaul risk ({len(risky)})",
        "\nThese require fading a team with a top-quartile incoming portal "
        "class or a star transfer QB — a known model blind spot in July "
        "(e.g. Oklahoma State: 55 transfers in, 0.98-rated QB). Do not bet; "
        "revisit when 2026 data publishes.\n",
        risky.to_markdown(index=False),
        "\nStake guidance: flat 1u or quarter-Kelly on ev_cons; cap total "
        "week-1 ML exposure. Log everything — CLV vs close is the test.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {path}\n")
    print("=== PLAYABLE ===")
    print(clean.head(12).to_string(index=False))
    print("\n=== EXCLUDED (overhaul risk) ===")
    print(risky[["commence", "pick", "best_price", "ev_cons",
                 "fade_portal_pct", "fade_qb_in"]].to_string(index=False))


if __name__ == "__main__":
    main()
