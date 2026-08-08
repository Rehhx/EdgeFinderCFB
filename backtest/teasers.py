"""Teasers and same-game combos — the high-HIT-RATE way to use our edge.

Parlays raise EV by SPENDING hit rate (2-leg ~51%, 3-leg ~33%). Teasers do the
opposite: you buy points, so the hit rate goes UP and the price goes down. That
matters here because our edge is a SATURATION effect — the dog's sub-period
deficit plateaus near 4 points however big the full spread is. Buying points
against a distribution that is already compressed should be unusually cheap.

Two questions:

  1. TEASERS — move the dog's Q1/1H line N points in our favour and re-grade on
     real margins. A 2-team teaser needs both legs; at what N does the joint
     hit rate clear the price?
  2. SAME-GAME (Q1 + 1H on one game) — these agree 76% of the time. If a book
     prices that combination as INDEPENDENT, the true joint probability is far
     higher than p1*p2 and the combination is mispriced in our favour. This
     measures the true joint rate so the gap is known before betting a cent.

[!] Availability is the catch and it is not a modelling question: full-game and
1H teasers are widely offered, Q1 teasers usually are NOT. The Q1 rows below
are the ceiling of what this idea is worth, not a bet you can necessarily place.

  python -m backtest.teasers
"""
import numpy as np
import pandas as pd

from ingestion.config import PARQUET_DIR

# 2-team teaser prices seen in US books. CFB 2-team 6pt is typically -110/-120;
# bigger teases cost more. (price, decimal payout)
TEASER_PRICE = {6.0: -110, 6.5: -120, 7.0: -130, 10.0: -180}
SEASONS = 3


def payout(american: float) -> float:
    return 100 / abs(american) if american < 0 else american / 100


def load_q1() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_q1_spreads.parquet")
    d = d[d.covered.notna() & (d.n_books >= 2) & (d.dog_full_line >= 25)].copy()
    d["margin"], d["line"] = d.dog_q1_margin, d.q1_line
    return d


def load_h1() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_h1_saturation.parquet")
    d = d[d.covered.notna() & (d.n_books >= 2) & (d.model_on_dog >= 6)].copy()
    keep = (((d.dog_full_line >= 17) & d.week.between(1, 5))
            | ((d.dog_full_line >= 21) & (d.week >= 6)))
    d = d[keep].copy()
    d["margin"], d["line"] = d.dog_h1_margin, d.h1_line
    return d


def hit_rates(d: pd.DataFrame, label: str) -> pd.Series:
    """Single-leg cover rate as the line is teased in our favour."""
    print(f"\n=== {label}: single-leg cover rate by teaser points ===")
    out = {}
    for pts in (0.0, 3.0, 4.0, 6.0, 6.5, 7.0, 10.0):
        cov = d.margin + d.line + pts
        live = cov != 0
        p = float((cov[live] > 0).mean())
        out[pts] = p
        print(f"  +{pts:4.1f} pts: {p:6.1%} cover   (n={int(live.sum())})")
    return pd.Series(out)


def teaser_ev(p: pd.Series, label: str) -> None:
    """2-team teaser EV at real book prices, assuming independent legs."""
    print(f"\n=== {label}: 2-TEAM teaser EV at book prices ===")
    print(f"{'pts':>5} {'leg hit':>8} {'joint':>7} {'price':>7} {'EV':>8}")
    for pts, price in TEASER_PRICE.items():
        if pts not in p.index:
            continue
        leg = p[pts]
        joint = leg ** 2
        ev = joint * payout(price) - (1 - joint)
        flag = "  <-- +EV" if ev > 0 else ""
        print(f"{pts:5.1f} {leg:8.1%} {joint:7.1%} {price:7.0f} {ev:+8.1%}{flag}")
    print("  (independent legs; the same-game section below shows why that "
          "assumption\n   must NOT be used for two legs from the SAME game)")


def same_game(q1: pd.DataFrame, h1: pd.DataFrame) -> None:
    """True joint hit rate of Q1 dog + 1H dog on the SAME game."""
    a = q1[["season", "game_id", "covered"]].rename(columns={"covered": "q1"})
    b = h1[["season", "game_id", "covered"]].rename(columns={"covered": "h1"})
    m = a.merge(b, on=["season", "game_id"])
    m = m[(m.q1 != 0.5) & (m.h1 != 0.5)]
    if len(m) < 25:
        print("\n(too few games quoted in both markets to measure)")
        return
    pq, ph = m.q1.mean(), m.h1.mean()
    joint = float(((m.q1 > 0) & (m.h1 > 0)).mean())
    indep = pq * ph
    print(f"\n=== SAME-GAME: Q1 dog + 1H dog, {len(m)} games ===")
    print(f"  Q1 leg alone     {pq:.1%}")
    print(f"  1H leg alone     {ph:.1%}")
    print(f"  BOTH (true)      {joint:.1%}")
    print(f"  BOTH (if indep)  {indep:.1%}")
    print(f"  correlation lift {joint - indep:+.1%}")
    dec = 1 / indep if indep > 0 else np.nan
    print(f"\n  A book pricing this as independent offers ~{dec:.2f} decimal.")
    print(f"  True EV at that price: {joint * dec - 1:+.1%}")
    print("  [!] Most books apply a correlation adjustment to same-game combos,")
    print("  so treat this as the size of the prize IF you find one that does")
    print("  not — verify the actual quoted price before betting.")


def real_pairs(d: pd.DataFrame, pts: float, price: float, rng) -> dict | None:
    """TRUE 2-team teaser rate from real same-week, different-game pairs.

    The EV table above assumes independent legs. Our legs are NOT independent
    (weekly win-rate variance runs 1.14x independence — big dogs win and lose
    together), so the assumption has to be checked rather than trusted. This
    pairs actual games and settles both, which prices the correlation in.
    """
    d = d.copy()
    d["tc"] = d.margin + d.line + pts
    d = d[d.tc != 0].copy()
    d["won"] = (d.tc > 0).astype(float)
    rows = []
    for _, g in d.groupby(["season", "week"]):
        g = g.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        for i in range(0, len(g) - 1, 2):
            a, b = g.iloc[i], g.iloc[i + 1]
            if a.game_id == b.game_id:
                continue
            rows.append({"season": a.season, "won": min(a.won, b.won)})
    r = pd.DataFrame(rows)
    if len(r) < 20:
        return None
    pay = payout(price)
    pnl = r.won * pay - (1 - r.won)
    return {"n": len(r), "true": float(r.won.mean()),
            "indep": float(d.won.mean() ** 2), "ev": float(pnl.mean()),
            "t": float(pnl.mean() / (pnl.std() / np.sqrt(len(r)))),
            "per": r.groupby("season").won.mean().round(3).to_dict()}


def breakeven() -> None:
    """What hit rate each price needs — so any book's number can be checked."""
    print("\n=== BREAK-EVEN hit rate by 2-team teaser price ===")
    print("  Prices vary a lot by book. Compare YOUR price against these.")
    for price in (-110, -120, -130, -150, -180, -200, -250, -300):
        be = 1 / (1 + payout(price))
        print(f"  {price:>5}: need {be:6.1%}")


def main() -> None:
    q1, h1 = load_q1(), load_h1()
    print(f"Q1 PREMIUM legs: {len(q1)}   BIG-DOG 1H legs: {len(h1)}")

    p_h1 = hit_rates(h1, "BIG-DOG 1H (teasers ARE widely offered)")
    teaser_ev(p_h1, "BIG-DOG 1H")

    p_q1 = hit_rates(q1, "Q1 PREMIUM ([!] Q1 teasers rarely offered)")
    teaser_ev(p_q1, "Q1 PREMIUM")

    rng = np.random.default_rng(3)
    print("\n=== VALIDATION: TRUE 2-team rate from real same-week pairs ===")
    print("  (the EV tables above assume independence — this does not)")
    for d, nm in ((h1, "1H"), (q1, "Q1")):
        for pts, price in TEASER_PRICE.items():
            r = real_pairs(d, pts, price, rng)
            if r is None:
                continue
            per = " ".join(f"{v:.0%}" for v in r["per"].values())
            print(f"  {nm} +{pts:<4.1f} n={r['n']:3d}  TRUE {r['true']:5.1%} "
                  f"(indep said {r['indep']:5.1%})  EV {r['ev']:+6.1%}  "
                  f"t={r['t']:+5.2f}  [{per}]")

    breakeven()

    same_game(q1, h1)

    print("\nREAD: teasers are the HIGH-HIT-RATE tool — they trade EV% for win%,")
    print("the exact opposite of parlays. If a teaser clears +EV here it is the")
    print("only structure in the book that pays AND wins 70-80% of the time.")
    print("\n[!] BEFORE BETTING: confirm your book (a) offers 1H teasers at all,")
    print("(b) allows teasing a line this large, and (c) quotes a price at or")
    print("better than the break-even above. Availability is the binding")
    print("constraint here, not the edge.")


if __name__ == "__main__":
    main()
