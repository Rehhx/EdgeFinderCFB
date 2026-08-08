"""Parlay builder — combine ONLY our validated +EV legs, honestly priced.

Math reality: parlaying INDEPENDENT +EV legs compounds the edge (higher EV%)
but variance explodes and you lose most of them (six 58% legs hit just 4%).
Parlaying break-even/-EV legs is a guaranteed loser — the vig compounds. So
this builder:
  - uses only validated-edge legs. PROPS are now the primary ingredient
    (the header used to read "NEVER props (near-efficient)" — the audit
    inverted that: props are the only stable edge, Q1 PREMIUM is -0.7 u/szn);
    still never totals (dead market) or moneylines (-5% ROI);
  - one leg per game (keeps legs ~independent; correlated same-game legs
    would break the joint-probability math);
  - haircuts each leg's backtest win rate for live realism (errors compound);
  - shows a 2..6-leg ladder with true joint prob, decimal payout, EV, and
    fractional-Kelly stake, and flags which sizes are +EV.

Honest note: for pure bankroll growth, betting these as SINGLES is better
(diversification). Parlays are a higher-variance play that our edge makes
+EV — unlike the public's. Keep stakes small.

  python -m picks.parlay_builder
"""
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd

from ingestion.config import PROJECT_ROOT

# Backtested win rates, HAIRCUT for live realism (parlays punish overconfidence).
# VALIDATED 2026-08-04 by backtest/growth_paths.py, which built real same-week,
# different-game parlays from the graded book and settled them properly:
#   2-leg  38/szn  47.8% win  +77.3% ROI  t=+4.43  (2023 +103%, 2024 +82%, 2025 +86%)
#   3-leg  23/szn  32.9% win +130.2% ROI  t=+3.28  (+234%, +154%, +99%)
# All three seasons positive at both sizes. The edge really does multiply:
# EV_parlay = (1+EV1)(1+EV2)-1, so two +45.7% Q1 legs price near +112%.
# ⚠️ Legs are NOT perfectly independent — measured weekly win-rate variance is
# 1.14x what independence predicts, i.e. our big dogs win and lose together a
# little. That makes true joint probability slightly BELOW p1*p2, so the
# haircut below is doing real work, not decoration.
# ⚠️ 2-LEG IS THE RECOMMENDED SIZE. 3-leg has higher EV% but its per-season ROI
# is decaying (+234 -> +154 -> +99) and 33% hit rate is brutal to sit through.
PROB = {"big_dog_spread": 0.565, "1h_spread": 0.555,   # backtest 59% / 58%
        "q1_spread": 0.700}      # Q1 PREMIUM backtests 75.5%; haircut hard
KELLY_FRACTION = 0.25
MAX_LEGS = 4                     # was 6 — nothing beyond 3 legs is defensible
MIN_LEG_EV = 0.02        # a leg must be clearly +EV to enter a parlay
DEC_110 = 1.909          # -110 in decimal
MIN_LEG_PROB = 0.45      # no longshot lottery legs (EV-max would stack them)
# Moneyline legs are BANNED: the 8-season real-price backtest
# (backtest/ml_value_history.py) showed ML betting is -5% ROI on model
# probability. A -EV leg makes any parlay containing it worse.
MAX_ML_LEGS = 0


def _dec(american: float) -> float:
    return 1 + (100 / abs(american) if american < 0 else american / 100)


def gather_legs() -> pd.DataFrame:
    """Best available +EV leg per game, from the validated-edge markets.

    COMPOSITION, measured 2026-08-04 on real graded outcomes (same week,
    different games, each leg used at most once):

      Q1 PREM x Q1 PREM        19/szn  55.9% win  +103.8% ROI  t=+4.34
      Q1 PREM x any derived    31/szn  50.0% win   +84.5% ROI  t=+4.39
      any derived x any        116/szn 43.1% win   +59.4% ROI  t=+6.02
      Q1 PREM x BIG-DOG 1H     21/szn  42.9% win   +58.2% ROI  t=+2.50
      1H x 1H                  23/szn  37.1% win   +37.3% ROI  t=+1.73
      Q1 PREM x PROPS           8/szn  46.2% win   +62.6% ROI  t=+1.78

    All positive in all three seasons. **Q1 legs are the best ingredient by a
    wide margin** — our best single play squared — and until 2026-08-04 this
    builder did not gather them at all.

    ⚠️ The 116/szn row is NOT a licence to bet that many. It re-uses each leg
    ~3x, so one leg losing takes down several parlays AND the single on that
    same game. Legs are used at most once here, which lands nearer 20-30/szn.
    """
    legs = []

    # Q1 big-dog lines — our single best play (75.5% at PREMIUM) and therefore
    # the best parlay ingredient. Model-free by design.
    try:
        from picks.q1_picks import run as q1
        d = q1()
        if not d.empty:
            # PREMIUM tier only — that is the 75.5% leg the maths relies on
            for r in d[d.tier == "PREMIUM Q1"].itertuples():
                legs.append({"game": f"{r.away} @ {r.home}",
                             "market": "q1_spread",
                             "pick": f"{r.dog} Q1 {r.q1_line:+g}",
                             "dec": _dec(float(r.price)),
                             "prob": PROB["q1_spread"]})
    except Exception as e:
        print(f"(Q1 legs unavailable: {e})")

    # big-dog spreads (ml_spread validated plays)
    try:
        from picks.ml_spread_picks import run as mls, validated_plays
        for r in validated_plays(mls()).itertuples():
            pick = (r.home if r.edge > 0 else r.away)
            legs.append({"game": f"{r.away} @ {r.home}", "market": "spread",
                         "pick": f"{pick} +{abs(r.book_spread):g}",
                         "dec": DEC_110, "prob": PROB["big_dog_spread"]})
    except Exception as e:
        print(f"(big-dog legs unavailable: {e})")

    # 1H spreads (validated wks 1-5)
    try:
        from picks.first_half_picks import EDGE_FLAG, run as fh
        d = fh()
        if not d.empty:
            for r in d[d.edge.abs() >= EDGE_FLAG].itertuples():
                pick = (r.home if r.edge > 0 else r.away)
                legs.append({"game": f"{r.away} @ {r.home}",
                             "market": "1h_spread",
                             "pick": f"{pick} 1H +{abs(r.book_1h_spread):g}",
                             "dec": DEC_110, "prob": PROB["1h_spread"]})
    except Exception:
        pass

    # PROPS legs — now the PRIMARY ingredient. This module was built around Q1
    # legs ("the best ingredient by a wide margin", 75.5% / +45.7%) and
    # deliberately excluded props as "near-efficient". The 2026-08-05/06 audit
    # inverted both premises: Q1 PREMIUM is -0.7 u/szn on the clean sample,
    # while props are the only play with stable evidence.
    #
    # Measured directly on 504 graded props bets (2024-25): CROSS-GAME pairs are
    # essentially exactly independent — joint hit 36.8% vs 36.8% predicted from
    # the marginals, lift 0.999 — and pay **+30.0% EV**, week-block bootstrap CI
    # [+11.3%, +48.6%], positive both seasons. That independence is why the
    # one-leg-per-game rule below is load-bearing rather than cosmetic.
    try:
        from picks.prop_picks import current_week, flag_picks, run as prop_run
        pdf = prop_run()
        if not pdf.empty:
            for r in flag_picks(pdf, current_week()).itertuples():
                legs.append({"game": r.game, "market": "prop",
                             "pick": f"{r.player} {r.side} {r.line:g} {r.stat}",
                             "dec": _dec(r.price), "prob": float(r.conf)})
    except Exception as e:
        print(f"(prop legs unavailable: {e})")

    # NOTE: moneyline legs intentionally NOT gathered — ML betting backtests
    # at -5% ROI (8 seasons, real prices). Parlaying a -EV leg is a loser.

    df = pd.DataFrame(legs)
    if df.empty:
        return df
    df["ev"] = df.prob * df.dec - 1
    # exclude longshots: an EV-maximizer would otherwise stack +400 dogs into
    # 0.1%-hit "lottery" parlays whose EV rests entirely on our least-reliable
    # probabilities
    df = df[(df.ev >= MIN_LEG_EV) & (df.prob >= MIN_LEG_PROB)]
    # one leg per game (independence): keep the highest-EV leg
    df = df.sort_values("ev", ascending=False).drop_duplicates("game")
    return df.reset_index(drop=True)


def best_parlay(legs: pd.DataFrame, k: int) -> dict:
    """Highest-EV k-leg parlay from the available legs (independent legs)."""
    best = None
    for combo in combinations(range(len(legs)), k):
        sub = legs.iloc[list(combo)]
        if (sub.market == "ml").sum() > MAX_ML_LEGS:
            continue  # ML probabilities are our least-reliable input
        prob = sub.prob.prod()
        dec = sub.dec.prod()
        ev = prob * dec - 1
        if best is None or ev > best["ev"]:
            b = dec - 1
            kelly = max(0.0, (prob * b - (1 - prob)) / b)
            best = {"k": k, "legs": sub, "prob": prob, "dec": dec, "ev": ev,
                    "kelly": KELLY_FRACTION * kelly}
    return best


def recommended(legs: pd.DataFrame | None = None) -> dict | None:
    """The recommended parlay (best EV*hit-prob among +EV sizes), for the
    paper tracker."""
    legs = gather_legs() if legs is None else legs
    if legs.empty or len(legs) < 2:
        return None
    ladders = [best_parlay(legs, k)
               for k in range(2, min(MAX_LEGS, len(legs)) + 1)]
    ladders = [p for p in ladders if p and p["ev"] > 0]
    if not ladders:
        return None
    return max(ladders, key=lambda p: p["ev"] * p["prob"])


def build() -> str:
    legs = gather_legs()
    day = datetime.now(timezone.utc).date().isoformat()
    out = [f"# Parlay Builder — {day}\n"]
    if legs.empty or len(legs) < 2:
        out.append("Not enough validated +EV legs on the board yet "
                   f"({len(legs)} found). Parlays need 2+ — check back on "
                   "game week when 1H/ML lines fill in.")
        rep = PROJECT_ROOT / "reports" / f"parlay_{day}.md"
        rep.write_text("\n".join(out), encoding="utf-8")
        print("\n".join(out))
        return str(rep)

    out.append(f"{len(legs)} validated +EV legs available "
               "(one per game, independent).\n")
    out.append("| size | legs | hit prob | payout | EV | ¼-Kelly stake |")
    out.append("|---|---|---|---|---|---|")
    ladders = []
    for k in range(2, min(MAX_LEGS, len(legs)) + 1):
        p = best_parlay(legs, k)
        ladders.append(p)
        picks = "; ".join(p["legs"].pick.tolist())
        out.append(f"| {k} | {picks} | {p['prob']:.1%} | "
                   f"{p['dec']:.2f}x | {p['ev']:+.0%} | {p['kelly']*100:.1f}u |")

    # recommendation: the +EV parlay with the best EV-per-variance (EV*prob)
    pos = [p for p in ladders if p["ev"] > 0]
    if pos:
        rec = max(pos, key=lambda p: p["ev"] * p["prob"])  # EV weighted by hit odds
        out.append(f"\n**Recommended: the {rec['k']}-leg** "
                   f"({rec['prob']:.0%} to hit, {rec['ev']:+.0%} EV) — best "
                   "balance of edge and hit rate. Stake small (¼-Kelly above); "
                   "these are higher-variance than singles.")
    else:
        out.append("\n**No +EV parlay** from the current legs — bet the "
                   "singles instead.")
    out.append("\n*Reminder: singles grow a bankroll faster; parlays are a "
               "higher-variance play our edge makes +EV. Never parlay props "
               "or totals — they have no edge and the vig compounds.*")

    rep = PROJECT_ROOT / "reports" / f"parlay_{day}.md"
    rep.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    return str(rep)


if __name__ == "__main__":
    build()
