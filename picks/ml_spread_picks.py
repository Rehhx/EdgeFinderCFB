"""Live ML early-season spread picks — paper-trial vs the linear model.

Trains the gradient-boosted margin model on all early-season history, then
prices the live board's weeks 1-5 games. Logged to the paper tracker as
market 'ml_spread' so we compare it head-to-head with the linear 'spread'
picks all season. Backtest: +6.3% ROI vs linear +5.3% (2023-25, wks 1-5).

  python -m picks.ml_spread_picks
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from features.epa_ratings import SEASON_STRIDE, fit_ratings, load_game_obs
from ingestion.config import PROJECT_ROOT
from ingestion.odds_client import NCAAF, OddsClient
from models.ml_spread import (FEATS, PBP_SEASONS, feature_row, load_priors,
                              train_live_model)
from picks.edge_report import team_matcher

SEASON = 2026
# validated edge (8 seasons): dog side, big spread, strong disagreement
EDGE_FLAG = 6.0
MIN_ABS_SPREAD = 17.0
PREMIUM_SPREAD = 25.0   # 62.3% / +18.8% ROI vs 59.2% / +13.0% at 17+
# 8-season tier stats (wks 1-5, dog side, edge>=6), for sizing
TIERS = {"PREMIUM": (0.623, 302, 18.8), "STANDARD": (0.569, 259, 8.6)}


def validated_plays(df: pd.DataFrame) -> pd.DataFrame:
    """The 8-season-validated play: model on the UNDERDOG vs a big
    early-season spread, disagreeing by >= EDGE_FLAG points.

    Tiered by spread size — bigger spreads are where the public most
    overbets the favorite, and they backtest materially better:
      PREMIUM  |spread| >= 25 -> 62.3% ATS, +18.8% ROI (8/8 seasons)
      STANDARD |spread| 17-25 -> ~56.9% ATS, +8.6% ROI
    Size PREMIUM ~2x STANDARD.
    """
    if df.empty:
        return df
    fav_is_home = df.book_spread < 0
    picks_home = df.edge > 0
    picks_dog = picks_home != fav_is_home
    keep = (picks_dog & (df.book_spread.abs() >= MIN_ABS_SPREAD)
            & (df.edge.abs() >= EDGE_FLAG))
    out = df[keep].copy()
    out["tier"] = np.where(out.book_spread.abs() >= PREMIUM_SPREAD,
                           "PREMIUM", "STANDARD")
    out["units"] = np.where(out.tier == "PREMIUM", 2.0, 1.0)
    return out.sort_values(["tier", "edge"], key=lambda c: (
        c if c.name != "edge" else c.abs()), ascending=[True, False])


def run() -> pd.DataFrame:
    obs = load_game_obs(PBP_SEASONS)
    model = train_live_model()
    ratings = fit_ratings(obs, asof_week_idx=SEASON * SEASON_STRIDE + 1)
    pr = load_priors(SEASON, obs)
    match = team_matcher()

    client = OddsClient()
    events = client.get(f"/sports/{NCAAF}/odds",
                        {"markets": "spreads", "regions": "us",
                         "oddsFormat": "american"}, refresh=True)
    rows = []
    for e in events:
        hid, aid = match(e["home_team"]), match(e["away_team"])
        if not hid or not aid or hid not in ratings.ratings.index \
                or aid not in ratings.ratings.index:
            continue
        pts = [o["point"] for bk in e.get("bookmakers", [])
               for m in bk.get("markets", []) if m["key"] == "spreads"
               for o in m["outcomes"] if o["name"] == e["home_team"]]
        if not pts:
            continue
        book = float(np.median(pts))
        fr = feature_row(ratings, hid, aid, False, pr)
        pred = float(model.predict(np.array([[fr[c] for c in FEATS]]))[0])
        rows.append({"commence": e["commence_time"][:10],
                     "away": e["away_team"], "home": e["home_team"],
                     "home_id": hid, "away_id": aid,
                     "fair_spread": round(-pred, 1), "book_spread": book,
                     "edge": round(pred - (-book), 1)})
    df = pd.DataFrame(rows)
    print(f"ML-spread priced: {len(df)} games | credits {client.remaining()}")
    return df


def main() -> None:
    df = run()
    day = datetime.now(timezone.utc).date().isoformat()
    path = PROJECT_ROOT / "reports" / f"ml_spread_{day}.md"
    if df.empty:
        path.write_text(f"# ML Spread — {day}\n\nno board yet.", encoding="utf-8")
        print("no games priced.")
        return
    plays = validated_plays(df)
    plays = plays.assign(bet=np.where(plays.edge > 0, plays.home, plays.away)
                         + " +" + plays.book_spread.abs().astype(str))
    prem = (plays.tier == "PREMIUM").sum()
    path.write_text(
        f"# ML Early-Season Spread Picks — {day}\n\n"
        f"{len(plays)} validated plays ({prem} PREMIUM, {len(plays)-prem} "
        "STANDARD). Weeks 1-5 only; bet the UNDERDOG getting the points.\n\n"
        "⏱️ **BET EARLY — run this as soon as lines post (Sun/Mon).** Grading "
        "these plays against the OPENING line beats the closing line in every "
        "season tested: PREMIUM 65.9% vs 62.9% (+25.9% vs +20.2% ROI), "
        "STANDARD 60.0% vs 58.2%. The number barely moves (+0.04 pts) — the "
        "gain is in *which* games qualify, so price them off the opener.\n\n"
        "| tier | filter | 8-season ATS | ROI | size |\n|---|---|---|---|---|\n"
        "| PREMIUM | spread ≥25 | **62.3%** (302 bets, 8/8 seasons) | +18.8% | 2u |\n"
        "| STANDARD | spread 17-25 | 56.9% | +8.6% | 1u |\n\n"
        "Books inflate early lines on big-brand favorites; the model fades "
        "the most overvalued. **Spread-only** — these same dogs lose outright "
        "37% of the time, so never bet them as moneylines.\n\n"
        + plays[["commence", "tier", "units", "bet", "book_spread",
                 "fair_spread", "edge"]].to_markdown(index=False),
        encoding="utf-8")
    print(f"report -> {path}")
    print(plays.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
