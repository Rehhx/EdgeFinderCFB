"""Season paper-trading tracker: log picks, settle vs results, track ROI + CLV.

  python -m picks.paper_trades import-ml      # log current ML value picks
  python -m picks.paper_trades import-edges   # log flagged spread/total plays
  python -m picks.paper_trades add --market ml --pick "Tulane Green Wave" \
        --game "Tulane Green Wave @ Duke Blue Devils" --price 330
  python -m picks.paper_trades settle         # grade vs CFBD finals + CLV
  python -m picks.paper_trades report         # season P&L summary

Stakes are tier-based (PREMIUM/PRIME/BIG-DOG 1H = 2u, standard = 1u, ML = 0u
tracked-only). settle() scales pnl by stake once. Ledger:
warehouse/paper_trades.parquet.
"""
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR, PROJECT_ROOT

LEDGER = PARQUET_DIR / "paper_trades.parquet"
SEASON = 2026


COLUMNS = ["placed_at", "market", "game", "commence", "home_school",
           "away_school", "pick", "side", "line", "price", "stake", "unit_usd",
           "model_p", "edge", "stat", "player", "status", "pnl", "clv_pts",
           "legs"]


def load_ledger() -> pd.DataFrame:
    if LEDGER.exists():
        led = pd.read_parquet(LEDGER)
        for c in COLUMNS:  # schema may have grown since rows were logged
            if c not in led.columns:
                led[c] = np.nan
        return led
    return pd.DataFrame(columns=COLUMNS)


def save_ledger(df: pd.DataFrame) -> None:
    df.to_parquet(LEDGER, index=False)


def _school(name: str, match) -> str | None:
    tid = match(name)
    if tid is None:
        return None
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    m = teams[teams.id == tid]
    return m.school.iloc[0] if len(m) else None


# --- compounding bankroll -------------------------------------------------
# Stakes are a % of the CURRENT bankroll, not the starting one. This is the
# single biggest free lever we have: backtest/growth_paths.py puts flat 2% at a
# $1,814 median and compounded 2% at $1,990, with no new bets and no extra
# per-bet risk. Every row records the dollar value of 1u AT PLACEMENT TIME, so
# the history stays gradeable even as the unit size moves.
STARTING_BANKROLL = 1000.0
UNIT_PCT = 0.04            # 4% — the growth-path plan; 2% is the safe setting
# Q1 is a low-limit market. The cap is in DOLLARS and is the binding constraint
# on compounding: measured P($5k) runs 42.1% at a $25 cap, 46.6% at $50, 57.0%
# at $200 and only 57.9% at $500. Spread Q1 across books to reach ~$200; past
# that it stops mattering.
Q1_MAX_USD = 200.0


def current_bankroll(led: pd.DataFrame | None = None) -> float:
    """Starting bankroll plus realised dollar P&L. Open bets are not counted."""
    led = load_ledger() if led is None else led
    if led.empty or "unit_usd" not in led.columns:
        return STARTING_BANKROLL
    s = led[led.status.isin(["won", "lost", "push"])]
    if s.empty:
        return STARTING_BANKROLL
    return STARTING_BANKROLL + float(
        (s.pnl.fillna(0.0) * s.unit_usd.fillna(0.0)).sum())


def unit_usd(led: pd.DataFrame | None = None) -> float:
    """Dollar value of 1 unit right now."""
    return round(current_bankroll(led) * UNIT_PCT, 2)


def _stake_usd(market: str, stake_units: float, u: float) -> float:
    """Per-unit dollars for a row, honouring the Q1 dollar cap."""
    if market.startswith("q1") and stake_units > 0:
        return round(min(u, Q1_MAX_USD / stake_units), 2)
    return u


def _payout_dec(american: float) -> float:
    """Profit per 1 unit staked at an American price."""
    if not np.isfinite(american):
        return 100 / 110
    return 100 / abs(american) if american < 0 else american / 100


def _append(led: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    if len(new):
        u = unit_usd(led)
        new["unit_usd"] = [
            _stake_usd(str(r.get("market", "")), float(r.get("stake", 1.0) or 0), u)
            for r in rows]
    # `new` is column-less when a market has no qualifying play — an ordinary
    # Saturday, not an error. Guarding only on len(led) made the dedup KeyError
    # on the empty frame.
    if len(led) and len(new):
        dupe_keys = led[["market", "pick", "commence"]].apply(tuple, axis=1)
        new = new[~new[["market", "pick", "commence"]].apply(tuple, axis=1)
                  .isin(set(dupe_keys))]
    print(f"logged {len(new)} new paper bets")
    return pd.concat([led, new], ignore_index=True)


def import_ml() -> None:
    """⚠️ Moneyline is INFORMATIONAL ONLY — 8-season real-price backtest shows
    -5% ROI. Logged with stake 0 so it tracks without polluting P&L."""
    from picks.edge_report import team_matcher
    from picks.ml_value import build_picks, run
    df = run()
    picks = build_picks(df)
    clean = picks[~picks.overhaul_risk]
    match = team_matcher()
    now = datetime.now(timezone.utc).isoformat()
    rows = [{
        "placed_at": now, "market": "ml",
        "game": f"{r.away} @ {r.home}", "commence": r.commence,
        "home_school": _school(r.home, match),
        "away_school": _school(r.away, match),
        "pick": r.pick, "side": "home" if r.pick == r.home else "away",
        "line": np.nan, "price": float(r.best_price),
        "stake": 0.0,  # tracked but NOT bet (no validated ML edge)
        "model_p": r.conf, "edge": r.ev_cons, "status": "open",
        "pnl": np.nan, "clv_pts": np.nan,
    } for r in clean.itertuples()]
    save_ledger(_append(load_ledger(), rows))


def import_1h() -> None:
    """Log flagged first-half spread plays (the validated wks 1-5 edge)."""
    from picks.first_half_picks import EDGE_FLAG, run as fh_run
    df = fh_run()
    if df.empty:
        print("no 1H lines posted yet — run on game week.")
        return
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    id2school = dict(zip(teams.id, teams.school))
    # Use the SAME tiering as the report, so BIG-DOG 1H is staked at 2u and the
    # STANDARD week gate applies. Re-deriving the selection here is what let
    # the ledger drift from the published picks.
    from picks.first_half_picks import tiered
    plays = tiered(df)
    now = datetime.now(timezone.utc).isoformat()
    rows = [{
        "placed_at": now, "market": "1h_spread",
        "game": f"{r.away} @ {r.home}", "commence": r.commence,
        "home_school": id2school.get(r.home_id),
        "away_school": id2school.get(r.away_id),
        "pick": (r.home if r.edge > 0 else r.away) + " 1H",
        "side": "home" if r.edge > 0 else "away",
        "line": float(r.book_1h_spread), "price": -110.0,
        "stake": float(getattr(r, "units", 1.0)),  # BIG-DOG 1H = 2u
        "model_p": np.nan, "edge": float(abs(r.edge)),
        "status": "open", "pnl": np.nan, "clv_pts": np.nan,
    } for r in plays.itertuples()]
    save_ledger(_append(load_ledger(), rows))


def import_q1() -> None:
    """Log Q1 big-dog plays (the validated market-structure edge).

    No model is involved — the pick is purely "underdog on the Q1 line when the
    full spread is big", so there is no model_p or edge to record. Stake comes
    from the tier (PREMIUM |spread|>=25 = 2u).
    """
    from picks.q1_picks import run as q1_run
    df = q1_run()
    if df.empty:
        print("no Q1 lines posted yet — run on game week.")
        return
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    id2school = dict(zip(teams.id, teams.school))
    now = datetime.now(timezone.utc).isoformat()
    rows = [{
        "placed_at": now, "market": "q1_spread",
        "game": f"{r.away} @ {r.home}", "commence": r.commence,
        "home_school": id2school.get(r.home_id),
        "away_school": id2school.get(r.away_id),
        "pick": f"{r.dog} Q1 {r.q1_line:+g}",
        "side": r.dog_side,
        "line": float(r.q1_line), "price": float(r.price),
        "stake": float(r.units),
        "model_p": np.nan, "edge": np.nan,
        "status": "open", "pnl": np.nan, "clv_pts": np.nan,
    } for r in df.itertuples()]
    save_ledger(_append(load_ledger(), rows))


def import_teasers() -> None:
    """Log 2-team teasers on BIG-DOG 1H legs (backtest/teasers.py: 83.8% hit,
    +30.3% EV at +10 pts, validated on real same-week pairs).

    Legs are stored as JSON in `legs` so settle() can grade each one against
    its own TEASED number — the parlay settler cannot be reused here, because
    it matches leg text against settled singles and a teased line is not the
    line any single was placed at.
    """
    import json
    from picks.teaser_picks import build
    print("[!] RESEARCH-ONLY as of 2026-08-06. The '83.8% hit / +30.3% EV' "
          "headline quoted the JOINT rate as the LEG rate, on the contaminated "
          "sample. Clean: leg 83.6%, JOINT 69.9% -> EV +8.8% at -180, and "
          "-0.0 u/szn realised in the complete-book seasons (2024 +2.9, "
          "2025 -2.9). Teasers also stack correlated exposure on games already "
          "bet as 1H singles, so they are OUT of the book (bankroll_sim "
          "TEASER_MODE='off'). Logged at stake 0 for tracking only.")
    df = build()
    if df.empty:
        print("no teaser-eligible 1H legs yet — run on game week.")
        return
    # Legs MUST be stored under CFBD school names — that is what settle() keys
    # its first-half margins by. See the note in teaser_picks.build().
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    id2school = dict(zip(teams.id, teams.school))
    now = datetime.now(timezone.utc).isoformat()
    rows = [{
        "placed_at": now, "market": "teaser2",
        "game": f"{r.game1} + {r.game2}", "commence": r.commence,
        "home_school": np.nan, "away_school": np.nan,
        "pick": f"{r.leg1}; {r.leg2}", "side": np.nan,
        "line": float(r.tease_pts), "price": float(r.price),
        "stake": 0.0,  # research-only: tracked, not staked
        "model_p": np.nan, "edge": np.nan,
        "status": "open", "pnl": np.nan, "clv_pts": np.nan,
        "legs": json.dumps([
            {"home": id2school.get(r.home1_id), "away": id2school.get(r.away1_id),
             "side": r.side1, "line": r.line1},
            {"home": id2school.get(r.home2_id), "away": id2school.get(r.away2_id),
             "side": r.side2, "line": r.line2},
        ]),
    } for r in df.itertuples()]
    save_ledger(_append(load_ledger(), rows))


def import_ml_spread() -> None:
    """Log ML early-season spread plays (paper-trial vs linear 'spread')."""
    from picks.ml_spread_picks import run as mls_run, validated_plays
    df = mls_run()
    plays = validated_plays(df)
    if plays.empty:
        print("no validated ML-spread plays.")
        return
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    id2school = dict(zip(teams.id, teams.school))
    now = datetime.now(timezone.utc).isoformat()
    rows = [{
        "placed_at": now, "market": "ml_spread",
        "game": f"{r.away} @ {r.home}", "commence": r.commence,
        "home_school": id2school.get(r.home_id),
        "away_school": id2school.get(r.away_id),
        "pick": (r.home if r.edge > 0 else r.away),
        "side": "home" if r.edge > 0 else "away",
        "line": float(r.book_spread), "price": -110.0,
        # Flat 1u across tiers now — PREMIUM measures negative on the
        # de-contaminated sample (see picks/ml_spread_picks.py TIERS).
        "stake": float(getattr(r, "units", 1.0)),
        "model_p": np.nan, "edge": float(abs(r.edge)),
        "status": "open", "pnl": np.nan, "clv_pts": np.nan,
    } for r in plays.itertuples()]
    save_ledger(_append(load_ledger(), rows))


def import_parlay() -> None:
    """Log the recommended parlay as ONE bet (all legs must win).

    Legs are stored in `pick` (semicolon-joined) and settled by re-grading
    each leg's game; `line`/`side` hold the leg list encoded for settlement.
    """
    from picks.parlay_builder import recommended
    p = recommended()
    if not p:
        print("no +EV parlay available.")
        return
    legs = p["legs"]
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "placed_at": now, "market": "parlay",
        "game": f"{p['k']}-leg parlay",
        "commence": legs.iloc[0].get("commence", ""),
        "home_school": None, "away_school": None,
        "pick": "; ".join(legs.pick.tolist()),
        "side": "; ".join(legs.market.tolist()),
        "line": float(p["dec"]),           # decimal payout
        "price": float((p["dec"] - 1) * 100),  # american-equivalent
        "stake": 1.0, "model_p": round(p["prob"], 4),
        "edge": round(p["ev"], 4),
        "status": "open", "pnl": np.nan, "clv_pts": np.nan,
    }
    save_ledger(_append(load_ledger(), [row]))
    print(f"logged {p['k']}-leg parlay: {row['pick']} "
          f"({p['prob']:.1%} to hit, {p['ev']:+.0%} EV)")


def import_props() -> None:
    from picks.edge_report import team_matcher
    from picks.prop_picks import current_week, flag_picks, run as prop_run
    df = prop_run()
    if df.empty:
        return
    # MUST pass the week. flag_picks(df) defaults to week=None, which disables
    # the season-arc gate entirely — the ledger was logging weeks 1-4 props
    # (negative in every band x season cell) while prop_picks itself correctly
    # printed nothing. Silent: no traceback, just bets we researched away.
    picks = flag_picks(df, current_week())
    match = team_matcher()
    now = datetime.now(timezone.utc).isoformat()
    rows = [{
        "placed_at": now, "market": "prop", "game": r.game,
        "commence": r.commence,
        "home_school": _school(r.home, match),
        "away_school": _school(r.away, match),
        "pick": f"{r.player} {r.side} {r.line} {r.stat}",
        "side": r.side, "line": float(r.line), "price": float(r.price),
        "stake": float(getattr(r, "units", 1.0)),  # PRIME band = 2u
        "model_p": r.conf, "edge": r.ev,
        "stat": r.stat, "player": r.player,
        "status": "open", "pnl": np.nan, "clv_pts": np.nan,
    } for r in picks.itertuples()]
    save_ledger(_append(load_ledger(), rows))


def import_edges() -> None:
    from picks.edge_report import SPREAD_FLAG, TOTAL_FLAG, run, team_matcher
    df = run()
    match = team_matcher()
    now = datetime.now(timezone.utc).isoformat()
    # 8-season-validated spread play: dog side, big spread, strong disagreement
    fav_home = df.book_spread < 0
    picks_home = df.spread_edge > 0
    df = df.assign(validated_spread=(picks_home != fav_home)
                   & (df.book_spread.abs() >= 17)
                   & (df.spread_edge.abs() >= 6))
    rows = []
    for r in df.itertuples():
        base = {"placed_at": now, "game": f"{r.away} @ {r.home}",
                "commence": r.commence[:10],
                "home_school": _school(r.home, match),
                "away_school": _school(r.away, match),
                "stake": 1.0, "status": "open", "pnl": np.nan,
                "clv_pts": np.nan, "price": -110.0}
        if r.validated_spread:
            side = "home" if r.spread_edge > 0 else "away"
            rows.append(base | {
                "market": "spread", "side": side,
                "pick": r.home if side == "home" else r.away,
                "line": float(r.book_spread), "model_p": np.nan,
                "edge": float(abs(r.spread_edge))})
    save_ledger(_append(load_ledger(), rows))


def _payout(price: float) -> float:
    return 100 / abs(price) if price < 0 else price / 100


def settle() -> None:
    led = load_ledger()
    games = pd.read_parquet(CFBD_PARQUET_DIR / f"games_{SEASON}.parquet")
    done = games[games.homePoints.notna()]
    try:
        lines = pd.read_parquet(CFBD_PARQUET_DIR / f"lines_{SEASON}.parquet")
        close = lines.groupby(["homeTeam", "awayTeam"], as_index=False).agg(
            c_spread=("spread", "median"), c_total=("overUnder", "median"))
    except FileNotFoundError:
        close = None

    try:
        logs = pd.read_parquet(PARQUET_DIR / "player_game_logs.parquet")
        logs = logs[logs.season == SEASON]
    except FileNotFoundError:
        logs = pd.DataFrame()

    # first-half scores (official quarter line scores) for 1h_spread bets
    h1 = {}
    needs_h1 = led.market.astype(str).str.startswith(("1h_spread", "teaser"))
    if (led.status == "open").any() and needs_h1.any():
        from features.first_half import one_half_scores
        for r in one_half_scores([SEASON]).itertuples():
            h1[(r.homeTeam, r.awayTeam)] = r.h1_margin

    # first-QUARTER margins for q1_spread bets, from the same official
    # lineScores array (element 0). Graded on CFBD finals, never on PBP.
    q1 = {}
    if (led.status == "open").any() and (led.market == "q1_spread").any():
        for r in games[games.homeLineScores.notna()
                       & games.awayLineScores.notna()].itertuples():
            try:
                q1[(r.homeTeam, r.awayTeam)] = float(r.homeLineScores[0]) \
                    - float(r.awayLineScores[0])
            except (TypeError, IndexError, ValueError):
                continue

    n = 0
    # parlays settle LAST (they depend on their legs' outcomes)
    open_bets = led[led.status == "open"]
    open_idx = open_bets.index.to_numpy()
    for i, b in pd.concat([open_bets[open_bets.market != "parlay"],
                           open_bets[open_bets.market == "parlay"]]).iterrows():
        if b.market == "parlay":
            # every leg must have a settled single with the same pick text
            settled = led[(led.market != "parlay")
                          & led.status.isin(["won", "lost", "push"])]
            outcomes = []
            for leg in str(b.pick).split("; "):
                match = settled[settled.pick.astype(str).str.startswith(
                    leg.split(" +")[0].split(" ML ")[0])]
                if match.empty:
                    outcomes = None
                    break
                outcomes.append(match.iloc[0].status)
            if outcomes is None:
                continue  # legs not all graded yet
            if any(o == "lost" for o in outcomes):
                led.loc[i, "status"] = "lost"
                led.loc[i, "pnl"] = -1.0
            elif all(o in ("won", "push") for o in outcomes):
                led.loc[i, "status"] = "won"
                led.loc[i, "pnl"] = float(b.line) - 1.0  # decimal payout - stake
            n += 1
            continue
        if str(b.market).startswith("teaser"):
            # Each leg is graded against its OWN teased number. Every leg must
            # cover; a push on any leg voids the ticket the way most books
            # settle a 2-teamer (push+win = no action, not a win).
            import json
            try:
                legs = json.loads(b.legs) if pd.notna(b.legs) else None
            except (TypeError, ValueError):
                legs = None
            if not legs:
                continue
            covers = []
            for lg in legs:
                hm = h1.get((lg["home"], lg["away"]))
                if hm is None or pd.isna(hm):
                    covers = None
                    break
                # line is the DOG's teased points; side says which team it is
                covers.append((hm + lg["line"]) if lg["side"] == "home"
                              else (-hm + lg["line"]))
            if covers is None:
                continue                      # a leg has not been played yet
            if any(c < 0 for c in covers):
                led.loc[i, "status"], led.loc[i, "pnl"] = "lost", -1.0
            elif any(c == 0 for c in covers):
                led.loc[i, "status"], led.loc[i, "pnl"] = "push", 0.0
            else:
                led.loc[i, "status"] = "won"
                led.loc[i, "pnl"] = _payout_dec(float(b.price))
            n += 1
            continue
        if b.market == "q1_spread":
            qm = q1.get((b.home_school, b.away_school))
            if qm is None or pd.isna(qm):
                continue
            # b.line is the DOG's points; b.side says which team the dog is
            cover = (qm + b.line) if b.side == "home" else (-qm + b.line)
            led.loc[i, "status"] = ("push" if cover == 0
                                    else "won" if cover > 0 else "lost")
            led.loc[i, "pnl"] = (0.0 if cover == 0
                                 else _payout(b.price) if cover > 0 else -1.0)
            n += 1
            continue
        if b.market == "1h_spread":
            hm = h1.get((b.home_school, b.away_school))
            if hm is None or pd.isna(hm):
                continue
            cover = hm + b.line if b.side == "home" else -(hm + b.line)
            led.loc[i, "status"] = ("push" if cover == 0
                                    else "won" if cover > 0 else "lost")
            led.loc[i, "pnl"] = (0.0 if cover == 0
                                 else _payout(b.price) if cover > 0 else -1.0)
            n += 1
            continue
        g = done[(done.homeTeam == b.home_school)
                 & (done.awayTeam == b.away_school)]
        if g.empty:
            continue
        g = g.iloc[0]
        margin = g.homePoints - g.awayPoints
        total = g.homePoints + g.awayPoints
        if b.market == "prop":
            if logs.empty:
                continue
            row = logs[(logs.week == g.week) & (logs.player == b.player)]
            if row.empty:
                led.loc[i, "status"] = "void"  # DNP / no stat line
                led.loc[i, "pnl"] = 0.0
                continue
            actual = float(row.iloc[0][b.stat])
            diff = actual - b.line
            won = diff > 0 if b.side == "over" else diff < 0
            led.loc[i, "status"] = ("push" if diff == 0
                                    else "won" if won else "lost")
            led.loc[i, "pnl"] = (0.0 if diff == 0
                                 else _payout(b.price) if won else -1.0)
            n += 1
            continue
        if b.market == "ml":
            won = margin > 0 if b.side == "home" else margin < 0
            led.loc[i, "pnl"] = _payout(b.price) if won else -1.0
            led.loc[i, "status"] = "won" if won else "lost"
        elif b.market in ("spread", "ml_spread"):
            cover = margin + b.line if b.side == "home" else -(margin + b.line)
            led.loc[i, "status"] = ("push" if cover == 0
                                    else "won" if cover > 0 else "lost")
            led.loc[i, "pnl"] = (0.0 if cover == 0
                                 else _payout(b.price) if cover > 0 else -1.0)
        elif b.market == "total":
            diff = total - b.line
            won = diff > 0 if b.side == "over" else diff < 0
            led.loc[i, "status"] = ("push" if diff == 0
                                    else "won" if won else "lost")
            led.loc[i, "pnl"] = (0.0 if diff == 0
                                 else _payout(b.price) if won else -1.0)
        if close is not None and b.market in ("spread", "total"):
            c = close[(close.homeTeam == b.home_school)
                      & (close.awayTeam == b.away_school)]
            if len(c):
                if b.market == "spread":
                    mv = c.c_spread.iloc[0] - b.line
                    led.loc[i, "clv_pts"] = mv if b.side == "away" else -mv
                else:
                    mv = c.c_total.iloc[0] - b.line
                    led.loc[i, "clv_pts"] = mv if b.side == "under" else -mv
        n += 1
    # scale P&L by stake (PREMIUM-tier spread plays are 2u). Only for rows
    # settled in THIS pass, so re-running settle never double-scales.
    just = led.index.isin(open_idx) & led.status.isin(["won", "lost"])
    led.loc[just, "pnl"] = led.loc[just, "pnl"] * led.loc[just, "stake"].fillna(1.0)
    save_ledger(led)
    print(f"settled {n} bets")


def bankroll() -> None:
    """Current bankroll, unit size, and progress toward the season target."""
    led = load_ledger()
    bank = current_bankroll(led)
    u = unit_usd(led)
    growth = bank / STARTING_BANKROLL - 1
    print(f"\n=== BANKROLL ===")
    print(f"  start        ${STARTING_BANKROLL:,.0f}")
    print(f"  current      ${bank:,.2f}   ({growth:+.1%})")
    print(f"  1u is now    ${u:,.2f}   ({UNIT_PCT:.0%} of current — COMPOUNDING)")
    print(f"  2u plays     ${2*u:,.2f}")
    print(f"  Q1 capped at ${Q1_MAX_USD:,.0f} total across books "
          f"(binding above ${Q1_MAX_USD/2/UNIT_PCT:,.0f} bankroll on 2u)")
    if led.empty or not led.status.isin(["won", "lost", "push"]).any():
        print("  no settled bets yet — unit size is still at the opening value")
    # rows logged before compounding existed carry no unit_usd and would
    # silently settle for $0. Say so rather than quietly under-counting.
    if not led.empty:
        stale = int(led.unit_usd.isna().sum())
        if stale:
            # ASCII only: this runs under Task Scheduler and redirection, where
            # the console is cp1252 and an emoji raises UnicodeEncodeError.
            print(f"  [!] {stale} row(s) predate compounding and have no "
                  "unit_usd — they contribute $0. Backfill with "
                  "`backfill-units` if they are real bets.")
    open_n = int((led.status == "open").sum()) if not led.empty else 0
    if open_n:
        print(f"  [!] {open_n} bets still open - bankroll excludes them until "
              "they settle")


def backfill_units() -> None:
    """Give pre-compounding rows a unit_usd at the OPENING unit size.

    Only touches rows where it is missing, and uses the opening value rather
    than today's — those bets were sized before compounding existed, so
    pretending they were staked at the current unit would invent P&L.
    """
    led = load_ledger()
    miss = led.unit_usd.isna()
    if not miss.any():
        print("nothing to backfill")
        return
    opening = round(STARTING_BANKROLL * UNIT_PCT, 2)
    led.loc[miss, "unit_usd"] = [
        _stake_usd(str(m), float(st) if pd.notna(st) else 1.0, opening)
        for m, st in zip(led.loc[miss, "market"], led.loc[miss, "stake"])]
    save_ledger(led)
    print(f"backfilled {int(miss.sum())} rows at the opening unit ${opening:,.2f}")


def report() -> None:
    led = load_ledger()
    if led.empty:
        print("ledger empty — run import-ml / import-edges first")
        return
    print(f"paper ledger: {len(led)} bets "
          f"({(led.status == 'open').sum()} open)")
    open_b = led[led.status == "open"]
    if len(open_b):
        if open_b.model_p.notna().any():
            bc = open_b.loc[open_b.model_p.idxmax()]
            print(f"best confidence open: {bc.market} {bc.pick} "
                  f"({bc.game}) — {bc.model_p:.0%}")
        if open_b.edge.notna().any():
            be = open_b.loc[open_b.edge.idxmax()]
            unit = "pts" if be.market in ("spread", "total") else "EV"
            print(f"best edge open: {be.market} {be.pick} "
                  f"({be.game}) — {be.edge:.2f} {unit}")
    settled = led[led.status.isin(["won", "lost", "push"])]
    if len(settled):
        settled = settled.copy()
        settled["pnl_usd"] = (settled.pnl.fillna(0.0)
                              * settled.unit_usd.fillna(0.0))
        g = settled.groupby("market").agg(
            bets=("pnl", "size"),
            won=("status", lambda s: (s == "won").sum()),
            pnl_u=("pnl", "sum"), roi=("pnl", "mean"),
            pnl_usd=("pnl_usd", "sum"),
            clv=("clv_pts", "mean")).round(3)
        print(g.to_string())
        bankroll()
        print(f"\nTOTAL: {settled.pnl.sum():+.2f}u over {len(settled)} bets "
              f"({settled.pnl.mean() * 100:+.1f}% ROI)")
    day = datetime.now(timezone.utc).date().isoformat()
    path = PROJECT_ROOT / "reports" / f"paper_trades_{day}.md"
    path.write_text(f"# Paper Trades — {day}\n\n"
                    + led.to_markdown(index=False), encoding="utf-8")
    print(f"report -> {path}")


def add(args) -> None:
    led = load_ledger()
    from picks.edge_report import team_matcher
    match = team_matcher()
    away, home = [s.strip() for s in args.game.split("@")]
    rows = [{
        "placed_at": datetime.now(timezone.utc).isoformat(),
        "market": args.market, "game": args.game, "commence": args.date,
        "home_school": _school(home, match), "away_school": _school(away, match),
        "pick": args.pick, "side": args.side or "",
        "line": args.line, "price": args.price, "stake": 1.0,
        "model_p": np.nan, "status": "open", "pnl": np.nan, "clv_pts": np.nan}]
    save_ledger(_append(led, rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["import-ml", "import-edges",
                                    "import-props", "import-1h", "import-q1",
                                    "import-ml-spread", "import-parlay", "import-teasers",
                                    "settle", "report", "bankroll",
                                    "backfill-units", "add"])
    ap.add_argument("--market", default="ml")
    ap.add_argument("--game", help='"Away Full Name @ Home Full Name"')
    ap.add_argument("--pick", default="")
    ap.add_argument("--side", default="")
    ap.add_argument("--line", type=float, default=np.nan)
    ap.add_argument("--price", type=float, default=-110)
    ap.add_argument("--date", default="")
    a = ap.parse_args()
    {"import-ml": import_ml, "import-edges": import_edges,
     "import-props": import_props, "import-1h": import_1h,
     "import-q1": import_q1,
     "import-ml-spread": import_ml_spread, "import-parlay": import_parlay,
     "import-teasers": import_teasers,
     "settle": settle, "report": report, "bankroll": bankroll,
     "backfill-units": backfill_units}.get(
        a.cmd, lambda: add(a))() if a.cmd != "add" else add(a)
