"""In-season player-prop picks: opening projections vs live prop lines.

Projections for the upcoming week come from season-opening player priors
(returning players + 2026 portal transfers carrying their old-school
profiles) times expected team volume (prev-season pace, spread-adjusted).
Pricing uses the backtest-validated stack: 2024-fitted recalibration,
NB receptions, and the 0.2-model/0.8-market logit blend that was +4.2% ROI
out-of-sample in 2025. Bets flagged at blended EV >= 3% with line shopping.

Run on game week (props post a few days before kickoff):
  python -m picks.prop_picks
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR, PROJECT_ROOT
from ingestion.odds_client import NCAAF, PROP_MARKETS, OddsClient
from models.props import player_priors
from picks.edge_report import payout, team_matcher

SEASON = 2026
MIN_EV = 0.05

# EV bands are NON-MONOTONE — more EV is not more edge. Isolated OOS bands:
#   3-5%  138/szn  54.2%  +2.2%  (2024 +18.8, 2025 -8.8)  -> SKIP, unreliable
#   5-8%  160/szn  59.5% +12.7%  (2024 +17.8, 2025 +8.3)  -> PRIME, 2u
#   8%+   230/szn  54.0%  +3.1%  (2024 -1.7,  2025 +6.6)  -> 1u
# Very high "EV" usually signals a stale line or model error rather than real
# edge, so the top band is staked DOWN, not up.
PRIME_LO, PRIME_HI = 0.05, 0.08
PRIME_UNITS, STANDARD_UNITS = 2.0, 1.0
MAX_EVENTS = 80

# fitted by backtest/props_vs_book.py + models/props.py; auto-loaded from
# warehouse/model_coefs.json when present (scripts/august_refit.py refreshes)
from ingestion.config import load_coefs

_props = load_coefs("props", {})
RECAL = _props.get("recal", {
    "rush_yds": (13.62, 0.87, 40.89), "rec_yds": (11.81, 0.81, 33.85),
    "receptions": (1.02, 0.78, 2.11), "pass_yds": (101.17, 0.60, 84.72)})
NB_R = _props.get("nb_r", 25)
BLEND_W = _props.get("blend_w", 0.20)
VOL_COEFS = load_coefs("prop_volume", {
    "rush": (0.61, -0.05, 12.89), "pass": (0.77, -0.0, 6.89)})
MARKET_TO_STAT = {
    "player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds",
    "player_receptions": "receptions", "player_reception_yds": "rec_yds"}
# rec_yds loses under both normal and gamma tails across 2024+2025 — the
# market prices receiving yards better than we do. Excluding it lifts the
# book-graded edge from +3.6% to +7.0% ROI. Priced below but never bet.
EXCLUDE_STATS = ("rec_yds",)
YARDAGE = ("rush_yds", "rec_yds", "pass_yds")  # right-skewed -> gamma tail


def opening_projections() -> pd.DataFrame:
    """Per (team_id, player): projected stats for a season-opening game."""
    logs = pd.read_parquet(PARQUET_DIR / "player_game_logs.parquet")
    pri = player_priors(logs)
    pri = pri[pri.season == SEASON]

    team_prev = logs[logs.season == SEASON - 1].groupby(
        ["team_id", "game_id"], as_index=False).agg(
        rush=("rush_att", "sum"), pas=("pass_att", "sum")) \
        .groupby("team_id")[["rush", "pas"]].mean()
    lg = logs[logs.season == SEASON - 1]
    league = {
        "ypc": (lg.rush_yds.sum() / lg.rush_att.sum()),
        "ypt": (lg.rec_yds.sum() / lg.targets.sum()),
        "catch": (lg.receptions.sum() / lg.targets.sum()),
        "ypa": (lg.pass_yds.sum() / lg.pass_att.sum()),
    }

    pri = pri.join(team_prev, on="team_id")

    def blend(prior, n, key, k):
        return (prior.fillna(league[key]) * n + league[key] * k) / (n + k)

    n_r = pri.prior_w * pri.prior_rush_att_pg.fillna(0)
    n_t = pri.prior_w * pri.prior_targets_pg.fillna(0)
    n_p = pri.prior_w * pri.prior_pass_att_pg.fillna(0)
    pri["exp_rush_att_base"] = pri.rush
    pri["exp_pass_att_base"] = pri.pas
    pri["ypc_b"] = blend(pri.prior_ypc, n_r, "ypc", 40)
    pri["ypt_b"] = blend(pri.prior_ypt, n_t, "ypt", 25)
    pri["catch_b"] = blend(pri.prior_catch, n_t, "catch", 15)
    pri["ypa_b"] = blend(pri.prior_ypa, n_p, "ypa", 80)
    return pri


def project_for_spread(pri: pd.DataFrame, team_id: int,
                       team_spread: float) -> pd.DataFrame:
    p = pri[pri.team_id == team_id].copy()
    if p.empty:
        return p
    b0, b1, b2 = VOL_COEFS["rush"]
    rush_att = b0 * p.exp_rush_att_base + b1 * team_spread + b2
    b0, b1, b2 = VOL_COEFS["pass"]
    pass_att = b0 * p.exp_pass_att_base + b1 * team_spread + b2
    p["proj_rush_yds"] = p.prior_rush_share * rush_att * p.ypc_b
    p["proj_targets"] = p.prior_tgt_share * pass_att
    p["proj_receptions"] = p.proj_targets * p.catch_b
    p["proj_rec_yds"] = p.proj_targets * p.ypt_b
    p["proj_pass_yds"] = p.prior_qb_share * pass_att * p.ypa_b
    return p


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


def flag_picks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[(df.ev >= MIN_EV) & (df.n_books >= 2)
             & (~df.stat.isin(EXCLUDE_STATS))].copy()
    prime = out.ev.between(PRIME_LO, PRIME_HI, inclusive="left")
    out["tier"] = np.where(prime, "PRIME", "STANDARD")
    out["units"] = np.where(prime, PRIME_UNITS, STANDARD_UNITS)
    return out.sort_values(["tier", "ev"], ascending=[True, False])


def run() -> pd.DataFrame:
    client = OddsClient()
    events = client.get(f"/sports/{NCAAF}/odds",
                        {"markets": "spreads", "regions": "us"}, refresh=True)
    match = team_matcher()
    pri = opening_projections()
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

    rows, with_props = [], 0
    for e in events[:MAX_EVENTS]:
        home_id, away_id = match(e["home_team"]), match(e["away_team"])
        if not home_id or not away_id:
            continue
        spreads = [o["point"] for bk in e.get("bookmakers", [])
                   for mkt in bk.get("markets", []) if mkt["key"] == "spreads"
                   for o in mkt["outcomes"] if o["name"] == e["home_team"]]
        home_spread = float(np.median(spreads)) if spreads else 0.0

        odds = client.get(f"/sports/{NCAAF}/events/{e['id']}/odds",
                          {"markets": ",".join(PROP_MARKETS), "regions": "us",
                           "oddsFormat": "american"}, refresh=True)
        if not odds.get("bookmakers"):
            continue
        with_props += 1

        home_side = project_for_spread(pri, home_id, home_spread)
        away_side = project_for_spread(pri, away_id, -home_spread)
        home_players = set(home_side.player)
        proj = pd.concat([home_side, away_side])
        proj_map = proj.set_index("player")

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
    picks = flag_picks(df)
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
    path.write_text(
        f"# Prop Picks — {day}\n\n{n_prime} **PRIME** ({PRIME_UNITS:g}u) + "
        f"{len(picks)-n_prime} STANDARD ({STANDARD_UNITS:g}u)\n\n"
        "| tier | EV band | OOS win% | ROI | size |\n|---|---|---|---|---|\n"
        f"| PRIME | {PRIME_LO:.0%}–{PRIME_HI:.0%} | **59.5%** (160/szn, both "
        f"seasons +) | +12.7% | {PRIME_UNITS:g}u |\n"
        f"| STANDARD | ≥{PRIME_HI:.0%} | 54.0% (230/szn) | +3.1% | "
        f"{STANDARD_UNITS:g}u |\n\n"
        "*EV bands are non-monotone: very high EV usually means a stale line "
        "or model error, so the top band is staked DOWN. Sub-5% EV is skipped "
        "(3–5% band was +18.8% in 2024 but −8.8% in 2025).*\n\n"
        + callout + picks.to_markdown(index=False) + cov, encoding="utf-8")
    print(f"report -> {path}")
    print(picks.head(15).to_string(index=False))
    return df


if __name__ == "__main__":
    run()
