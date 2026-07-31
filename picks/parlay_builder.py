"""Parlay builder — combine ONLY our validated +EV legs, honestly priced.

Math reality: parlaying INDEPENDENT +EV legs compounds the edge (higher EV%)
but variance explodes and you lose most of them (six 58% legs hit just 4%).
Parlaying break-even/-EV legs is a guaranteed loser — the vig compounds. So
this builder:
  - uses only validated-edge legs (big-dog spreads, 1H spreads, ML value);
    NEVER props (near-efficient) or totals (dead market);
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

# backtested win rates, HAIRCUT for live realism (parlays punish overconfidence)
PROB = {"big_dog_spread": 0.565, "1h_spread": 0.555}  # backtest 59% / 58%
KELLY_FRACTION = 0.25
MAX_LEGS = 6
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
    """Best available +EV leg per game, from the validated-edge markets."""
    legs = []

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
