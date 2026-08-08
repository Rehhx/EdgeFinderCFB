"""In-season player-prop picks: model projections vs live prop lines.

Projections come from `models.props.project_upcoming()` — the SAME pipeline
every backtest measures. It carries trailing (shift-1 EWMA) player share,
team volume and efficiency, the season-opening priors that seed them, and the
opponent allowed-efficiency factor. In week 1 there is no current-season form
so it falls back to the priors alone; the advantage accumulates as in-season
role changes do, which is the mechanism behind the season-arc gate below.

Pricing uses the backtest-validated stack: recalibration, NB receptions,
gamma tails on yardage, and the model/market logit blend. Bets flagged at
blended EV >= MIN_EV with line shopping.

Run on game week (props post a few days before kickoff):
  python -m picks.prop_picks
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR, PROJECT_ROOT
from ingestion.odds_client import NCAAF, PROP_MARKETS, OddsClient
from picks.edge_report import payout, team_matcher

SEASON = 2026
# RE-DERIVED 2026-08-04 on sign-bug-fixed data (backtest/props_ev_sweep.py).
# The old floor of 0.05 was set when the 3-5% band read "+2.2%, 2025 -8.8%,
# unreliable" — that was measured on corrupted data. Clean OOS sub-bands
# (weeks 5+, 2+ books, 2024-25):
#   EV 0.02-0.03  121 bets   -1.6%  t=-0.19   2024 +7.1   2025 -10.2  -> no
#   EV 0.03-0.04   99 bets  +11.8%  t=+1.26   2024 +37.3  2025  -4.1  -> no
#   EV 0.04-0.05   98 bets  +21.4%  t=+2.32   2024 +29.1  2025 +14.8  -> YES
#   EV 0.05-0.06   84 bets  +18.1%  t=+1.78   2024 +24.9  2025 +12.0
#   EV 0.06-0.08  108 bets   +8.5%  t=+0.94   2024  +0.9  2025 +14.3
# The 0.04-0.05 band grades BETTER than the band above it and is positive in
# both seasons; 0.03-0.04 fails 2025, so the floor stops at 0.04 — the cut is
# set by the both-seasons rule, not by maximising units. Paired week-block
# bootstrap of the change: +21.1 u/szn, 90% CI [+6.7, +35.4], P(gain)=99%.
MIN_EV = 0.04

# EV bands remain NON-MONOTONE — more EV is not more edge. Clean OOS bands:
#   0.04-0.05   PROBE, 1u — newest evidence, provisional (see below)
#   0.05-0.08   PRIME, 2u — the engine, both seasons +
#   0.08-0.12  113 bets  -5.9%  t=-0.66  (2024 -15.9, 2025 +2.6) -> NEGATIVE
#   0.12+       73 bets +17.4%  t=+1.59  (2024 +3.1, 2025 +35.9) -> imbalanced
# Very high "EV" usually signals a stale line or model error rather than real
# edge, so the top band is staked DOWN, not up. (Cutting 0.08+ entirely was
# considered; it is kept at 1u because the 0.12+ slice is positive and the
# 0.08-0.12 loss is within noise, but this is the weakest part of the book.)
#
# PROBE is staked at 1u rather than 2u ON PURPOSE. Its measured ROI (+21.4%)
# is the best of any band, but it rests on two seasons and was only found
# after today's sign-bug fix, so it is the least-validated slice in the book.
# The bankroll maths agrees with the caution: at 1u the change buys +$46 of
# median for +0.9pp P(losing season); pushing it to 2u buys a further +$45 for
# +1.3pp — a strictly worse trade, and on the newest evidence.
# PER-STAT STAKE TILT (2026-08-06). rush_yds is the ONLY prop stat whose
# bootstrap CI excludes zero, and it ranks #1 in ALL THREE seasons
# (+23.8 / +21.7 / +17.5); pass_yds and receptions swap 2nd/3rd between
# windows, so only the top rank is treated as signal. The mechanism agrees:
# rush share (corr 0.596) and rush attempts (0.541) are the best-predicted
# components in the whole model (HANDOFF §18).
#
# Validated OUT OF SAMPLE: a tilt ranked on 2023-24 and applied to 2025 alone
# lifts ROI +15.4% -> +16.6% on slightly LESS stake. Through the book it is
# +7.8 u/szn with median $1,872 -> $1,988, 5th %ile $957 -> $1,024 and
# P(losing season) 5.7% -> 4.6% for only 4.5% more staked — better at BOTH
# ends, which pure leverage never is.
#
# Multipliers are deliberately modest (not the sweep optimum of 1.5x, which
# gains more units only by staking 17% more). PROVISIONAL: re-check after 2026.
STAT_STAKE_MULT = {"rush_yds": 1.25, "pass_yds": 0.85, "receptions": 0.85}

PROBE_LO, PROBE_HI = 0.04, 0.05
PRIME_LO, PRIME_HI = 0.05, 0.08
PROBE_UNITS = 1.0
PRIME_UNITS, STANDARD_UNITS = 2.0, 1.5   # see the stake note above
MAX_EVENTS = 80

# The props edge RISES through the season (OOS 2024+2025, production metric):
#   wk 1-4   126/szn  51.0%  -2.3%   (2024 -2.7, 2025 -2.1)  <- no edge
#   wk 5-8   102/szn  55.1%  +4.1%   (2024 +1.9, 2025 +5.8)
#   wk 9-15  161/szn  61.2% +16.2%   (2024 +14.8, 2025 +17.6)
# Monotone, same direction in both seasons, and it survives the MEDIAN-price
# control (wk1-4 -6.1% vs wk9-15 +12.8%), so it is not a line-shopping
# artifact. Permutation test on the wk>=5 vs wk<=4 gap: p=0.028.
# Mechanism (best read): our projections are EWMA-weighted on current-season
# form while the book's prop line is anchored to a season-long average, so our
# advantage GROWS as in-season role changes accumulate. In weeks 1-4 neither
# side has current data and the book's prior is at least as good as ours.
#
# RE-DERIVED 2026-08-04 on data with the home_id sign bug FIXED. This matters
# because the bug was 81% concentrated in weeks 1-5 — precisely the weeks this
# filter governs — so the old gate was fitted on the corrupted region.
# Corrected 2024-25 OOS ROI by band:
#            wk1-4    wk5-8    wk9-15
#   PRIME    -5.2%   +14.2%   +11.7%
#   STANDARD -4.1%    -6.2%   +10.9%
# wk1-4 is now negative in ALL FOUR band x season cells (PRIME -7.8/-3.0,
# STANDARD -6.2/-3.2), so PRIME is no longer kept year-round — pre-fix it read
# +2.6% there and survived only on the bug's help. STANDARD in wk5-8 is
# -39.8%/+22.1% by season, i.e. no reliable signal, so it waits for week 9.
#
# Gate comparison (units/szn, per-season split in brackets):
#   STANDARD wk5+ (old)          226/szn  +23.5u  [+8.3 / +38.7]
#   drop all wk1-4               189/szn  +27.4u  [+13.8 / +41.1]
#   drop wk1-4, STANDARD wk9+    147/szn  +30.0u  [+28.9 / +31.2]  <- SHIPPED
# Chosen for SEASON BALANCE, not for being the maximum: the old gate drew 82%
# of its units from one season, the new one is near-identical in both. Note
# wk1-4 pooled is only t=-0.66 on its own — the case rests on 4-of-4 cell
# consistency, not significance. Re-check after 2026.
MIN_WEEK_PRIME = 5
MIN_WEEK_STANDARD = 9
from features.epa_ratings import POSTSEASON_WEEK

# fitted by backtest/props_vs_book.py + models/props.py; auto-loaded from
# warehouse/model_coefs.json when present (scripts/august_refit.py refreshes)
from ingestion.config import load_coefs

_props = load_coefs("props", {})
RECAL = _props.get("recal", {
    "rush_yds": (13.62, 0.87, 40.89), "rec_yds": (11.81, 0.81, 33.85),
    "receptions": (1.02, 0.78, 2.11), "pass_yds": (101.17, 0.60, 84.72)})
NB_R = _props.get("nb_r", 25)
BLEND_W = _props.get("blend_w", 0.20)
MARKET_TO_STAT = {
    "player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds",
    "player_receptions": "receptions", "player_reception_yds": "rec_yds"}
# rec_yds loses under both normal and gamma tails across 2024+2025 — the
# market prices receiving yards better than we do. Excluding it lifts the
# book-graded edge from +3.6% to +7.0% ROI. Priced below but never bet.
EXCLUDE_STATS = ("rec_yds",)
YARDAGE = ("rush_yds", "rec_yds", "pass_yds")  # right-skewed -> gamma tail


# REMOVED 2026-08-05: opening_projections() / project_for_spread().
# They were a SECOND, preseason-only implementation of the props model —
# no trailing EWMA, no opponent factor — and run() used them, so the live
# book was priced by a weaker model than every backtest measured while
# model_coefs.json shipped slopes fitted on the OTHER one. Live now calls
# models.props.project_upcoming(), the same pipeline the backtests use;
# it reproduces prop_projections.parquet exactly on a past week (corr
# 1.0000, max diff 0.0000 over 1,069 players). Do not reintroduce a
# separate live projector — that drift is the bug.

def p_over_model(stat: str, proj_raw: float, line: float,
                 mu_total: float | None = None) -> float:
    """P(stat > line). `mu_total` is the 0-1 offense-vs-defense matchup score
    (features/matchups.py); it enters exactly as in the backtest recalibration
    so live pricing matches the validated model."""
    coefs = RECAL[stat]
    a, b, sigma = coefs[:3]
    c_mu = coefs[3] if len(coefs) > 3 else 0.0
    mu = a + b * proj_raw
    if mu_total is not None and not pd.isna(mu_total):
        mu += c_mu * (mu_total - 0.5) * proj_raw
    if stat == "receptions":
        mu = max(mu, 0.1)
        return float(1 - nbinom.cdf(np.floor(line), NB_R, NB_R / (NB_R + mu)))
    if stat in YARDAGE:  # right-skewed: gamma tail, not normal
        from scipy.stats import gamma
        mu = max(mu, 1.0)
        return float(gamma.sf(max(line, 0.01), a=(mu / sigma) ** 2,
                              scale=sigma ** 2 / mu))
    return float(1 - norm.cdf((line - mu) / sigma))


def current_week(season: int = SEASON) -> int | None:
    """CFB week for today, from the schedule. None if the season hasn't begun."""
    path = CFBD_PARQUET_DIR / f"games_{season}.parquet"
    if not path.exists():
        return None
    g = pd.read_parquet(path, columns=["week", "startDate", "seasonType"])
    g["start"] = pd.to_datetime(g.startDate, format="ISO8601", utc=True)
    # CFBD labels EVERY postseason game `week=1`. Taken literally, that made
    # this function return 1 from mid-December on — which suppressed all props
    # (gate needs >=5), reverted 1H to the loose early-season threshold, and
    # made the ratings refit as-of week 1, discarding the entire season at the
    # moment it is most complete. Order the postseason AFTER the regular
    # season, matching features/epa_ratings.POSTSEASON_WEEK.
    from features.epa_ratings import POSTSEASON_WEEK
    g["week"] = np.where(g.seasonType.astype(str) == "postseason",
                         POSTSEASON_WEEK, g.week)
    now = pd.Timestamp.now(tz="UTC")
    # STRICTLY ahead. A 2-day lookback kept Saturday's finished games "ahead",
    # so 12 of 15 Sunday/Monday run-slots — the documented time to run this —
    # returned the week that had just ENDED. Cost: all props suppressed on the
    # week-5 slate (the first live week) and STANDARD suppressed on week 9 (the
    # engine). Verified that strict does NOT advance mid-slate: it holds the
    # current week through late games still in progress and only rolls over
    # after the last whistle.
    ahead = g[g.start >= now]
    if ahead.empty:                      # season over -> treat as late season
        return int(g.week.max())
    return int(ahead.sort_values("start").week.iloc[0])


def flag_picks(df: pd.DataFrame, week: int | None = None) -> pd.DataFrame:
    """Select and tier bets, then apply the season-arc gate: nothing at all
    before MIN_WEEK_PRIME, and STANDARD-band props only from MIN_WEEK_STANDARD
    (see the note above). `week=None` means the week could not be determined,
    so no week gate is applied; run() warns when that happens."""
    if df.empty:
        return df
    out = df[(df.ev >= MIN_EV) & (df.n_books >= 2)
             & (~df.stat.isin(EXCLUDE_STATS))].copy()
    probe = out.ev.between(PROBE_LO, PROBE_HI, inclusive="left")
    prime = out.ev.between(PRIME_LO, PRIME_HI, inclusive="left")
    out["tier"] = np.where(probe, "PROBE",
                           np.where(prime, "PRIME", "STANDARD"))
    out["units"] = (np.where(probe, PROBE_UNITS,
                             np.where(prime, PRIME_UNITS, STANDARD_UNITS))
                    * out.stat.map(STAT_STAKE_MULT).fillna(1.0).values)
    if week is not None:
        if week >= POSTSEASON_WEEK:
            # Bowls/CFP are OUT OF SAMPLE for every threshold in this file —
            # the season arc was fitted on weeks 1-15 of the regular season.
            # And the specific hazard is one the model cannot see: projections
            # come from trailing usage, while bowl rosters are gutted by
            # opt-outs, portal departures and long layoffs. The market prices
            # that; we have no feature for it. Do not bet it.
            return out.iloc[0:0]
        if week < MIN_WEEK_PRIME:
            return out.iloc[0:0]          # no props edge exists this early
        if week < MIN_WEEK_STANDARD:
            out = out[(probe | prime).values]   # STANDARD waits for week 9
    order = {"PRIME": 0, "PROBE": 1, "STANDARD": 2}
    return out.sort_values(["tier", "ev"], key=lambda s: s.map(order)
                           if s.name == "tier" else -s)


def game_week(commence, season: int = SEASON):
    """CFB week for a given kickoff date — the GAME's week, not today's.

    Week gates must be evaluated per GAME. `current_week()` answers "what week
    is it now", which is the right input for the ratings as-of fit but the
    WRONG one for deciding whether a specific fixture is inside a play's
    validated window: the live board routinely spans several weeks at once, so
    gating the whole board on today's week both admits games past the window
    and suppresses in-window games once the calendar rolls past them.
    """
    path = CFBD_PARQUET_DIR / f"games_{season}.parquet"
    if not path.exists():
        return None
    g = pd.read_parquet(path, columns=["week", "startDate", "seasonType"])
    g["d"] = pd.to_datetime(g.startDate, utc=True, format="ISO8601").dt.date
    g["week"] = np.where(g.seasonType.astype(str) == "postseason",
                         POSTSEASON_WEEK, g.week)
    try:
        d = pd.Timestamp(str(commence)[:10]).date()
    except Exception:
        return None
    span = g.groupby("week").d.agg(["min", "max"])
    hit = span[(span["min"] <= d) & (span["max"] >= d)]
    return int(hit.index[0]) if len(hit) else None


def _schedule() -> dict:
    """(home_id, away_id) -> (game_id, week) for the live season."""
    try:
        g = pd.read_parquet(CFBD_PARQUET_DIR / f"games_{SEASON}.parquet")
    except FileNotFoundError:
        return {}
    g = g[g.homeId.notna() & g.awayId.notna()].copy()
    # Normalise the postseason week here too. `build_table`'s as-of guard drops
    # every logged row at or after the earliest future week, so a bowl arriving
    # as CFBD's literal `week=1` would wipe the whole season's trailing
    # features instead of using them.
    g["week"] = np.where(g.seasonType.astype(str) == "postseason",
                         POSTSEASON_WEEK, g.week)
    return {(int(r.homeId), int(r.awayId)): (float(r.id), float(r.week))
            for r in g.itertuples()}


def run() -> pd.DataFrame:
    client = OddsClient()
    events = client.get(f"/sports/{NCAAF}/odds",
                        {"markets": "spreads", "regions": "us"}, refresh=True)
    match = team_matcher()
    logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4)
                             / (1 - np.clip(p, 1e-4, 1 - 1e-4)))

    # offense-vs-defense matchup scores for the current week (empty preseason)
    mu_lookup: dict = {}
    try:
        from features.matchups import matchup_features
        cur = matchup_features("rec_yds")
        cur = cur[cur.season == SEASON]
        if len(cur):
            cur = cur[cur.week == cur.week.max()]
            mu_lookup = {(int(r.team_id), int(r.opp_id)): float(r.mu_total)
                         for r in cur.itertuples()}
    except Exception:
        pass

    # PASS 1 — resolve every board game to ids and a market spread, then run
    # the projection ONCE through models.props. This replaces a per-team call
    # to a preseason-only projector that had no trailing form and no opponent
    # factor: the live book was priced by a different, weaker model than every
    # backtest measured. See models/props.py::project_upcoming.
    sched, meta, grows = _schedule(), [], []
    week = current_week() or 1
    for e in events[:MAX_EVENTS]:
        home_id, away_id = match(e["home_team"]), match(e["away_team"])
        if not home_id or not away_id:
            continue
        spreads = [o["point"] for bk in e.get("bookmakers", [])
                   for mkt in bk.get("markets", []) if mkt["key"] == "spreads"
                   for o in mkt["outcomes"] if o["name"] == e["home_team"]]
        home_spread = float(np.median(spreads)) if spreads else 0.0
        gid, wk = sched.get((int(home_id), int(away_id)),
                            (float(hash((home_id, away_id)) % 10**8), float(week)))
        meta.append((e, home_id, away_id, home_spread, gid, wk))
        # team_spread is positive when THIS team is the underdog
        grows += [
            {"season": SEASON, "week": wk, "game_id": gid, "team_id": int(home_id),
             "opp_id": float(away_id), "team_spread": home_spread},
            {"season": SEASON, "week": wk, "game_id": gid, "team_id": int(away_id),
             "opp_id": float(home_id), "team_spread": -home_spread}]

    proj_all = pd.DataFrame()
    if grows:
        from models.props import project_upcoming
        try:
            proj_all = project_upcoming(pd.DataFrame(grows))
        except Exception as exc:                 # never kill the whole report
            print(f"  projection failed: {type(exc).__name__}: {exc}")
    if proj_all.empty:
        print("no projections available — cannot price props")
        return pd.DataFrame()

    rows, with_props = [], 0
    for e, home_id, away_id, home_spread, gid, wk in meta:
        odds = client.get(f"/sports/{NCAAF}/events/{e['id']}/odds",
                          {"markets": ",".join(PROP_MARKETS), "regions": "us",
                           "oddsFormat": "american"}, refresh=True)
        if not odds.get("bookmakers"):
            continue
        with_props += 1

        proj = proj_all[proj_all.game_id == gid]
        if proj.empty:
            continue
        home_players = set(proj[proj.team_id == int(home_id)].player)
        proj_map = proj.drop_duplicates("player").set_index("player")

        quotes = []
        for bk in odds["bookmakers"]:
            for mkt in bk.get("markets", []):
                stat = MARKET_TO_STAT.get(mkt["key"])
                if not stat:
                    continue
                for o in mkt.get("outcomes", []):
                    quotes.append({"book": bk["key"], "stat": stat,
                                   "player": o.get("description"),
                                   "side": o["name"], "point": o.get("point"),
                                   "price": o.get("price")})
        q = pd.DataFrame(quotes).dropna(subset=["player", "point", "price"])
        if q.empty:
            continue
        q = q[q.price.between(-200, 200)]
        piv = q.pivot_table(index=["stat", "player", "book", "point"],
                            columns="side", values="price",
                            aggfunc="first").reset_index().dropna()
        piv["pay_over"] = piv.Over.map(payout)
        piv["pay_under"] = piv.Under.map(payout)
        modal = piv.groupby(["stat", "player"]).point.agg(
            lambda s: s.mode().iloc[0]).rename("line")
        piv = piv.merge(modal, on=["stat", "player"])
        piv = piv[piv.point == piv.line]
        cons = piv.groupby(["stat", "player"], as_index=False).agg(
            line=("point", "first"),
            pay_o_med=("pay_over", "median"), pay_u_med=("pay_under", "median"),
            pay_o_best=("pay_over", "max"), pay_u_best=("pay_under", "max"),
            n_books=("book", "nunique"))

        for r in cons.itertuples():
            if r.player not in proj_map.index:
                continue
            proj_raw = proj_map.loc[r.player, f"proj_{r.stat}"]
            if isinstance(proj_raw, pd.Series) or pd.isna(proj_raw):
                continue
            mu_t = mu_lookup.get(
                (home_id if r.player in home_players else away_id,
                 away_id if r.player in home_players else home_id))
            pm = p_over_model(r.stat, proj_raw, r.line, mu_t)
            imp_o = 1 / (1 + r.pay_o_med)
            imp_u = 1 / (1 + r.pay_u_med)
            p_mkt = imp_o / (imp_o + imp_u)
            p_bl = 1 / (1 + np.exp(-(BLEND_W * logit(pm)
                                     + (1 - BLEND_W) * logit(p_mkt))))
            ev_o = p_bl * r.pay_o_best - (1 - p_bl)
            ev_u = (1 - p_bl) * r.pay_u_best - p_bl
            side = "over" if ev_o >= ev_u else "under"
            pay = r.pay_o_best if side == "over" else r.pay_u_best
            price = round(100 * pay) if pay >= 1 else -round(100 / pay)
            rows.append({
                "game": f"{e['away_team']} @ {e['home_team']}",
                "away": e["away_team"], "home": e["home_team"],
                "commence": e["commence_time"][:10], "stat": r.stat,
                "player": r.player, "line": r.line,
                "proj": round(RECAL[r.stat][0] + RECAL[r.stat][1] * proj_raw, 1),
                "side": side, "price": int(price),
                "p_model": round(pm, 3), "p_blend": round(p_bl, 3),
                "conf": round(p_bl if side == "over" else 1 - p_bl, 3),
                "ev": round(max(ev_o, ev_u), 3),
                "n_books": r.n_books,
            })

    print(f"events with props posted: {with_props} "
          f"| credits left: {client.remaining()}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("No player props posted yet — run again closer to game day.")
        return df
    week = current_week()
    if week is None:
        print(f"WARNING: no games_{SEASON}.parquet — cannot tell the week, so "
              f"the season-arc gate (nothing before wk{MIN_WEEK_PRIME}, "
              f"STANDARD from wk{MIN_WEEK_STANDARD}) is NOT applied.")
    picks = flag_picks(df, week)
    day = datetime.now(timezone.utc).date().isoformat()
    path = PROJECT_ROOT / "reports" / f"prop_picks_{day}.md"
    callout = ""
    if len(picks):
        be = picks.iloc[0]
        bc = picks.loc[picks.conf.idxmax()]
        callout = (
            f"**Best edge:** {be.player} {be.side} {be.line} {be.stat} "
            f"({be.ev:+.1%} EV, conf {be.conf:.0%})  \n"
            f"**Best confidence:** {bc.player} {bc.side} {bc.line} {bc.stat} "
            f"(conf {bc.conf:.0%}, {bc.ev:+.1%} EV)\n\n")
    # coverage-injury watch: opponents missing their top coverage DB →
    # opposing WR/QB overs are a market-slow spot (free PFF approximation).
    cov = ""
    try:
        from features.defense_players import coverage_injury_flags
        flags = coverage_injury_flags()
        boards = set(picks.home) | set(picks.away)
        flags = flags[flags.team.isin(boards)]
        if len(flags):
            cov = ("\n## Coverage-injury watch (unbacktested — paper only)\n"
                   "Top coverage DB out/questionable → lean opposing WR/QB "
                   "OVERS:\n\n" + flags.to_markdown(index=False) + "\n")
    except Exception:
        pass
    n_prime = int((picks.tier == "PRIME").sum()) if len(picks) else 0
    early = week is not None and week < MIN_WEEK_STANDARD
    wk_note = (
        f"\n> **Week {week} — PRIME only.** The STANDARD band is suppressed "
        f"before week {MIN_WEEK_STANDARD}: in weeks 5–8 it split −39.8% / "
        "+22.1% by season, i.e. no reliable signal. Our edge comes from "
        "current-season form the book under-weights, and it takes most of a "
        "season to accumulate.\n" if early else "")
    path.write_text(
        f"# Prop Picks — {day}"
        + (f" (week {week})" if week else "") + "\n\n"
        f"{n_prime} **PRIME** ({PRIME_UNITS:g}u) + "
        f"{len(picks)-n_prime} STANDARD ({STANDARD_UNITS:g}u)\n"
        + wk_note +
        "\n| tier | EV band | weeks | OOS win% | ROI | size |\n"
        "|---|---|---|---|---|---|\n"
        f"| PRIME | {PRIME_LO:.0%}–{PRIME_HI:.0%} | ≥{MIN_WEEK_PRIME} | "
        f"**59.4%** (both seasons +) | +12.7% | {PRIME_UNITS:g}u |\n"
        f"| PROBE | {PROBE_LO:.0%}–{PROBE_HI:.0%} | ≥{MIN_WEEK_PRIME} | 60.2% "
        f"(t=+2.32, both seasons +) | +21.4% | {PROBE_UNITS:g}u |\n"
        f"| STANDARD | ≥{PRIME_HI:.0%} | ≥{MIN_WEEK_STANDARD} only | 58.3% "
        f"| +10.9% | {STANDARD_UNITS:g}u |\n\n"
        "*PROBE grades best of all but rests on two seasons and was only found "
        "after the 2026-08-04 sign-bug fix, so it is staked at 1u until a "
        "third season confirms it.*\n\n"
        "*EV bands are non-monotone: very high EV usually means a stale line "
        f"or model error, so the top band is staked DOWN. Sub-{MIN_EV:.0%} EV "
        "is skipped (the 3–4% band was +37.3% in 2024 but −4.1% in 2025). "
        "The edge also rises through the season — props are skipped entirely "
        f"before week {MIN_WEEK_PRIME}, where all four band×season cells were "
        "negative.*\n\n"
        + callout + picks.to_markdown(index=False) + cov, encoding="utf-8")
    print(f"report -> {path}")
    print(picks.head(15).to_string(index=False))
    return df


if __name__ == "__main__":
    run()
