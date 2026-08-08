"""2-team teasers on BIG-DOG 1H legs — the high-hit-rate play.

Parlays raise EV by SPENDING hit rate. Teasers do the opposite: you buy points,
the hit rate goes up and the price goes down. That is unusually cheap for US
because our whole edge is a SATURATION effect — the dog's first-half deficit
plateaus near 4 points however large the full spread — so the outcome
distribution we are buying points against is already compressed.

Validated in backtest/teasers.py on REAL same-week, different-game pairs
(not an independence assumption — our legs run 1.14x correlated):

    tease   n    TRUE hit   EV       t      by season
    +6.0    80    71.2%    +36.0%  +3.71   64 / 71 / 78%
    +7.0    80    75.0%    +32.7%  +3.79   72 / 75 / 78%
    +10.0   80    83.8%    +30.3%  +4.69   80 / 82 / 89%

+10 is the recommended size: highest hit rate, best t, and the most consistent
across seasons. At 83.8% a -180 price needs only 64.3%, so the play survives
down to about -400 — an unusually wide margin, and the reason it is worth
placing even at a mediocre quote.

⚠️ AVAILABILITY IS THE BINDING CONSTRAINT, NOT THE EDGE. Confirm your book
(a) offers 1H teasers at all, (b) permits teasing a line this large, and
(c) quotes at or better than the break-even printed below.

  python -m picks.teaser_picks
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ingestion.config import PROJECT_ROOT

TEASE_PTS = 10.0
TEASER_UNITS = 1.0        # 1u — a teaser is two legs of exposure in one ticket
ASSUMED_PRICE = -180.0
# backtest/teasers.py, real same-week pairs
VALIDATED = {6.0: (0.712, 0.360), 7.0: (0.750, 0.327), 10.0: (0.838, 0.303)}


def payout(american: float) -> float:
    return 100 / abs(american) if american < 0 else american / 100


def breakeven(american: float) -> float:
    return 1 / (1 + payout(american))


def build(df: pd.DataFrame | None = None, pts: float = TEASE_PTS) -> pd.DataFrame:
    """Pair BIG-DOG 1H plays into 2-team teasers, one leg per game.

    Legs come only from the validated big-dog population — teasing a leg we
    have no edge on just buys points at a fair price and loses the juice.
    """
    if df is None:
        from picks.first_half_picks import run as fh
        df = fh()
    if df.empty or "big_dog" not in df.columns:
        return pd.DataFrame()
    legs = df[df.big_dog.fillna(False)].copy()
    if len(legs) < 2:
        return pd.DataFrame()
    # dog side, and the line we would actually be teasing
    legs["dog_is_home"] = legs.fg_spread > 0
    legs["dog"] = np.where(legs.dog_is_home, legs.home, legs.away)
    legs["side"] = np.where(legs.dog_is_home, "home", "away")
    # book_1h_spread is home-relative; the dog's own number is +|line|
    legs["dog_line"] = legs.book_1h_spread.abs()
    legs["teased"] = legs.dog_line + pts
    legs = legs.sort_values("fg_spread", key=abs, ascending=False)

    rows = []
    for i in range(0, len(legs) - 1, 2):
        a, b = legs.iloc[i], legs.iloc[i + 1]
        rows.append({
            "commence": min(a.commence, b.commence),
            "tease_pts": pts,
            "leg1": f"{a.dog} 1H +{a.teased:g}", "leg2": f"{b.dog} 1H +{b.teased:g}",
            "game1": f"{a.away} @ {a.home}", "game2": f"{b.away} @ {b.home}",
            "home1": a.home, "away1": a.away, "side1": a.side,
            "line1": float(a.teased),
            "home2": b.home, "away2": b.away, "side2": b.side,
            "line2": float(b.teased),
            # Carry team IDs so the ledger can store CFBD SCHOOL names. `home`/
            # `away` above are Odds-API display names ("Alabama Crimson Tide"),
            # and settle() keys first-half margins by CFBD school ("Alabama").
            # Zero of 183 display names match, so every ticket stayed open
            # forever — never graded, never in P&L, never compounding.
            "home1_id": a.home_id, "away1_id": a.away_id,
            "home2_id": b.home_id, "away2_id": b.away_id,
            "price": ASSUMED_PRICE, "units": TEASER_UNITS,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = build()
    day = datetime.now(timezone.utc).date().isoformat()
    path = PROJECT_ROOT / "reports" / f"teasers_{day}.md"
    hit, ev = VALIDATED.get(TEASE_PTS, (np.nan, np.nan))
    be = breakeven(ASSUMED_PRICE)
    head = (f"# 2-Team Teasers — {day}\n\n"
            f"**{TEASE_PTS:g}-point teaser on BIG-DOG 1H legs.** Backtested "
            f"**{hit:.1%} hit rate, {ev:+.1%} EV** on real same-week pairs "
            f"(t=+4.69, 80/82/89% by season).\n\n"
            f"Assumed price **{ASSUMED_PRICE:+.0f}** needs **{be:.1%}** to "
            f"break even — you have room down to roughly **-400**.\n\n"
            "> ⚠️ **Check availability first.** Confirm the book offers 1H "
            "teasers, allows teasing a line this large, and quotes at or "
            "better than break-even. Availability is the binding constraint "
            "here, not the edge.\n\n"
            "> ⚠️ Each ticket is **two legs of exposure**. Do not also bet "
            "these legs as singles — that doubles the same position.\n\n")
    if df.empty:
        path.write_text(head + "No qualifying big-dog 1H legs on the board "
                        "(need at least 2).", encoding="utf-8")
        print("no teaser-eligible legs yet — run on game week.")
        return
    show = df[["commence", "leg1", "leg2", "game1", "game2", "units"]]
    path.write_text(head + show.to_markdown(index=False), encoding="utf-8")
    print(f"report -> {path}")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
