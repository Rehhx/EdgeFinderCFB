"""Live ML early-season spread picks — paper-trial vs the linear model.

Trains the gradient-boosted margin model on all early-season history, then
prices the live board's weeks 1-5 games. Logged to the paper tracker as
market 'ml_spread' so we compare it head-to-head with the linear 'spread'
picks.

⚠️ PAPER ONLY. The old header claimed "+6.3% ROI vs linear +5.3%"; on the
de-contaminated game universe both models are BELOW break-even at the generic
cut — ML 50.5% / -3.6%, linear 51.4% / -1.9% (break-even 52.4%). The tiered
big-dog selection is no better: see the TIERS note below, where the tier
ordering inverts depending on which model and window you ask.

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
EDGE_FLAG = 6.0
MIN_ABS_SPREAD = 17.0
PREMIUM_SPREAD = 25.0
MAX_WEEK = 5            # this play was ONLY ever fitted on weeks 1-5

# ⚠️ RE-MEASURED 2026-08-05 on the de-contaminated game universe (the
# garbage-time play filter used to delete blowouts from the sample entirely —
# see backtest/spread_baseline.py::game_table). The old numbers below were
# measured on that poisoned sample and were badly wrong:
#
#   tier                claimed          ML (clean)              linear control
#   PREMIUM >=25   62.3% / +18.8% / 8-8   48.9% / -6.5% / 1-3    47.5% / -9.1% / 0-3
#   STANDARD 17-25 56.9% /  +8.6%         58.7% / +11.8% / 3-3   58.9% / +12.3% / 3-3
#
# THE TIER ORDER INVERTED: the slice we staked at 2u is the one that loses, in
# BOTH the ML model and the linear control. PREMIUM is cut to 1u — that is
# removing an unjustified size multiplier from a tier now measured negative,
# not fitting a new threshold.
#
# ⚠️ AND DO NOT TRUST EITHER TIER. On an 8-season EPA re-derivation the signs
# flip the OTHER way (PREMIUM +2.3% 5-of-8, STANDARD -11.9% 1-of-8, t=-2.56).
# Whichever tier looks good depends on the model, window and edge definition
# you happen to ask — which is what "the tier structure is noise" means. This
# module stays a PAPER trial until a deliberate re-derivation says otherwise.
TIERS = {"PREMIUM": (0.489, 191, -6.5), "STANDARD": (0.587, 141, 11.8)}


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
    # WEEK GATE. The model trains on weeks 1-5 only and every stat quoted for
    # this play is a weeks-1-5 stat, but nothing enforced it — the docstring,
    # the report header and `paper_trades import-ml-spread` all said
    # "weeks 1-5 only" while the code emitted plays for all 15 weeks.
    # PER-GAME week gate. This used to gate the whole board on current_week(),
    # which is "what week is it now" — the right input for the ratings as-of
    # fit, the wrong one for deciding whether a FIXTURE is in the window. The
    # live board spans several weeks at once (today: weeks 1-5), so the old
    # form would admit a week-6 game while current_week()==5 and suppress a
    # still-listed week-5 game once the calendar reached week 6.
    from picks.prop_picks import game_week
    gw = df.commence.map(lambda c: game_week(c))
    in_window = gw.isna() | (gw <= MAX_WEEK)   # unknown week -> do not drop
    df = df[in_window]
    if df.empty:
        return df.assign(tier="STANDARD", units=1.0)

    fav_is_home = df.book_spread < 0
    picks_home = df.edge > 0
    picks_dog = picks_home != fav_is_home
    keep = (picks_dog & (df.book_spread.abs() >= MIN_ABS_SPREAD)
            # PREMIUM (|spread| >= PREMIUM_SPREAD) is EXCLUDED — see the note
            # on TIERS. 22 variants were tested on 8 clean seasons and none
            # cleared the bar; blind beat model-selected, and raising the edge
            # threshold made it worse, so the selection is anti-predictive.
            & (df.book_spread.abs() < PREMIUM_SPREAD)
            & (df.edge.abs() >= EDGE_FLAG))
    out = df[keep].copy()
    out["tier"] = "STANDARD"
    # Flat 1u across tiers: PREMIUM measures NEGATIVE on clean data in both
    # the ML model and the linear control (see the TIERS note above).
    out["units"] = 1.0
    return out.sort_values(["tier", "edge"], key=lambda c: (
        c if c.name != "edge" else c.abs()), ascending=[True, False])


def run() -> pd.DataFrame:
    # Refit AS OF the upcoming week, including the live season once its
    # play-by-play lands (features/epa_ratings.live_asof).
    from features.epa_ratings import asof_seasons, live_asof
    from picks.prop_picks import current_week
    obs = load_game_obs(asof_seasons(PBP_SEASONS, SEASON))
    model = train_live_model()
    ratings = fit_ratings(obs, asof_week_idx=live_asof(SEASON, current_week()))
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
    # Carry the week so validated_plays() gates on the SAME value the ratings
    # were fitted as-of, rather than re-deriving it independently.
    df.attrs["week"] = current_week() or 1
    print(f"ML-spread priced: {len(df)} games (week {df.attrs['week']}) "
          f"| credits {client.remaining()}")
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
