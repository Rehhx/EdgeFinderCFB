"""Weekly edge report: model fair lines vs live market for upcoming games.

Pulls current NCAAF spreads/totals/moneylines (The Odds API), predicts every
game with the Phase-1 spread stack (EPA ratings + roster prior, full prior
weight in week 1) and the game-sim totals model, and writes:
  - reports/edge_report_<date>.md   ranked edges + recommended plays
  - warehouse/parquet/clv_log.parquet  our fair line + market now, so every
    report run builds the CLV history to grade later against closing lines

  python -m picks.edge_report
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from features.epa_ratings import SEASON_STRIDE, fit_ratings, load_game_obs
from features.roster_priors import preseason_priors
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR, PROJECT_ROOT
from ingestion.odds_client import NCAAF, OddsClient
from models.game_sim import GameSim, coach_priors

SEASON = 2026
PBP_SEASONS = [2021, 2022, 2023, 2024, 2025]

# fitted by the walk-forward backtests; warehouse/model_coefs.json is
# written whenever those backtests run (scripts/august_refit.py automates it)
from ingestion.config import load_coefs

SPREAD_COEFS = load_coefs(
    "spread", {"epa": 65.73, "prior": 75.07, "hfa": 4.2, "const": -1.6})
TOTAL_COEFS = np.array(load_coefs(
    "total", [60.34, -0.47, 8.52, 1.89, 3.25, 33.98]))
MARGIN_SIGMA = 16.0  # margin residual sd for win-prob conversion

SPREAD_FLAG = 4.0  # backtested early-season thresholds
TOTAL_FLAG = 5.0
ML_EV_FLAG = 0.05

ALIASES = {"appalachian state mountaineers": "App State",
           "central florida knights": "UCF",
           "connecticut huskies": "Connecticut",
           "mississippi rebels": "Ole Miss",
           "hawaii rainbow warriors": "Hawai'i",
           "san jose state spartans": "San José State",
           "louisiana ragin cajuns": "Louisiana",
           "umass minutemen": "Massachusetts",
           "southern mississippi golden eagles": "Southern Miss",
           "louisiana monroe warhawks": "UL Monroe"}


def team_matcher():
    t = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    by_full = {f"{r.school} {r.mascot}".lower(): r.id for r in t.itertuples()}
    by_school = {r.school.lower(): r.id for r in t.itertuples()}

    def match(name: str):
        # exact matches only — a prefix fallback once mapped FCS
        # "Arkansas Pine Bluff" onto Arkansas and invented a 43-pt edge
        n = name.lower().replace("'", "'")
        if n in ALIASES:
            return by_school.get(ALIASES[n].lower())
        return by_full.get(n)
    return match


def payout(price: float) -> float:
    return 100 / abs(price) if price < 0 else price / 100


def market_consensus(events: list) -> pd.DataFrame:
    rows = []
    for e in events:
        spreads, totals, ml_home, ml_away = [], [], [], []
        for bk in e.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                for o in mkt.get("outcomes", []):
                    if mkt["key"] == "spreads" and o["name"] == e["home_team"]:
                        spreads.append(o["point"])
                    elif mkt["key"] == "totals" and o["name"] == "Over":
                        totals.append(o["point"])
                    elif mkt["key"] == "h2h":
                        (ml_home if o["name"] == e["home_team"]
                         else ml_away).append(o["price"])
        rows.append({
            "event_id": e["id"], "commence": e["commence_time"],
            "home": e["home_team"], "away": e["away_team"],
            "book_spread": np.median(spreads) if spreads else np.nan,
            "book_total": np.median(totals) if totals else np.nan,
            "ml_home_best": max(ml_home, key=payout) if ml_home else np.nan,
            "ml_away_best": max(ml_away, key=payout) if ml_away else np.nan,
            "n_books": len(e.get("bookmakers", [])),
        })
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    obs = load_game_obs(PBP_SEASONS)
    ratings = fit_ratings(obs, asof_week_idx=SEASON * SEASON_STRIDE + 1)
    priors = preseason_priors(SEASON, obs)
    pri_floor = priors.quantile(0.10)
    cpriors = coach_priors()
    sim = GameSim(ratings, cpriors, TOTAL_COEFS,
                  cpriors.pace_prior.median(), cpriors.proe_prior.median())

    client = OddsClient()
    events = client.get(f"/sports/{NCAAF}/odds",
                        {"markets": "h2h,spreads,totals", "regions": "us",
                         "oddsFormat": "american"}, refresh=True)
    print(f"{len(events)} events on the board | credits left: {client.remaining()}")
    mkt = market_consensus(events)

    match = team_matcher()
    mkt["home_id"] = mkt.home.map(match)
    mkt["away_id"] = mkt.away.map(match)
    unmatched = mkt[mkt.home_id.isna() | mkt.away_id.isna()]
    if len(unmatched):
        bad = pd.concat([unmatched[unmatched.home_id.isna()].home,
                         unmatched[unmatched.away_id.isna()].away])
        print(f"{len(unmatched)} events skipped (non-FBS or unmatched): "
              f"{bad.unique()[:6].tolist()}")
    mkt = mkt.dropna(subset=["home_id", "away_id"]).astype(
        {"home_id": int, "away_id": int})

    c = SPREAD_COEFS
    out = []
    for g in mkt.itertuples():
        epa_diff = ratings.net(g.home_id) - ratings.net(g.away_id)
        prior_diff = (priors.get(g.home_id, pri_floor)
                      - priors.get(g.away_id, pri_floor))
        pred_margin = (c["epa"] * epa_diff + c["prior"] * prior_diff
                       + c["hfa"] + c["const"])
        ph, pa = sim.predict(g.home_id, g.away_id, SEASON)
        p_home = norm.cdf(pred_margin / MARGIN_SIGMA)
        row = {
            "commence": g.commence, "home": g.home, "away": g.away,
            "fair_spread": -round(pred_margin, 1),
            "book_spread": g.book_spread,
            "spread_edge": pred_margin - (-g.book_spread)
            if pd.notna(g.book_spread) else np.nan,
            "fair_total": round(ph + pa, 1), "book_total": g.book_total,
            "total_edge": ph + pa - g.book_total
            if pd.notna(g.book_total) else np.nan,
            "p_home": round(p_home, 3),
            "ml_home_best": g.ml_home_best, "ml_away_best": g.ml_away_best,
            "ev_ml_home": p_home * payout(g.ml_home_best) - (1 - p_home)
            if pd.notna(g.ml_home_best) else np.nan,
            "ev_ml_away": (1 - p_home) * payout(g.ml_away_best) - p_home
            if pd.notna(g.ml_away_best) else np.nan,
            "n_books": g.n_books,
            "both_rated": (g.home_id in ratings.ratings.index
                           and g.away_id in ratings.ratings.index),
        }
        out.append(row)
    df = pd.DataFrame(out)

    log = df.assign(as_of=datetime.now(timezone.utc).isoformat(),
                    season=SEASON)
    dest = PARQUET_DIR / "clv_log.parquet"
    if dest.exists():
        log = pd.concat([pd.read_parquet(dest), log], ignore_index=True)
    log.to_parquet(dest, index=False)
    return df


def write_report(df: pd.DataFrame) -> str:
    day = datetime.now(timezone.utc).date().isoformat()
    rep_dir = PROJECT_ROOT / "reports"
    rep_dir.mkdir(exist_ok=True)
    path = rep_dir / f"edge_report_{day}.md"

    def fmt(d, cols):
        return d[cols].to_markdown(index=False, floatfmt=".1f")

    from scipy.special import expit, logit as slogit
    df = df.copy()
    for col, edge_col in [("conf_spread", "spread_edge"),
                          ("conf_total", "total_edge")]:
        p_raw = norm.cdf(df[edge_col].abs() / MARGIN_SIGMA).clip(1e-4, 1 - 1e-4)
        df[col] = expit(0.2 * slogit(p_raw)).round(3)
    # SPREADS: the validated early-season edge is BIG-spread underdogs the
    # model likes (8-season: |spread|>=17, dog side, edge>=6 -> 59% ATS). The
    # old |spread|<=21 guard suppressed exactly this — so spreads use the full
    # board; the dog-side big-spread plays are the money ones.
    fav_home = df.book_spread < 0
    picks_home = df.spread_edge > 0
    df["dog_side"] = picks_home != fav_home
    df["validated_spread"] = (df.dog_side & (df.book_spread.abs() >= 17)
                              & (df.spread_edge.abs() >= 6))
    spread_plays = df[df.spread_edge.abs() >= SPREAD_FLAG].sort_values(
        "validated_spread", ascending=False).sort_values(
        "spread_edge", key=abs, ascending=False, kind="stable")
    # TOTALS/ML keep the blowout guard (totals have no edge; big ML dogs lose)
    playable = df[df.book_spread.abs() <= 21].copy()
    total_plays = playable[playable.total_edge.abs() >= TOTAL_FLAG] \
        .sort_values("total_edge", key=abs, ascending=False)
    ml = playable.melt(id_vars=["commence", "home", "away"],
                 value_vars=["ev_ml_home", "ev_ml_away"],
                 var_name="side", value_name="ev").dropna()
    ml_plays = ml[ml.ev >= ML_EV_FLAG].sort_values("ev", ascending=False)

    lines = [
        f"# Edge Report — {day} (2026 season opening board)",
        f"\nGames priced: {len(df)} | thresholds: spread ±{SPREAD_FLAG}, "
        f"total ±{TOTAL_FLAG}, ML EV ≥{ML_EV_FLAG:.0%}",
        "\n> Week-1 numbers rest on decayed 2025 ratings + roster priors "
        "(2026 returning production/rosters not yet published — priors are "
        "partial). Treat as opening-line value hunting; sizes small, "
        "quarter-Kelly cap.\n",
        "## Top plays",
        (f"**Best spread edge:** "
         f"{spread_plays.iloc[0].away} @ {spread_plays.iloc[0].home} — "
         f"{abs(spread_plays.iloc[0].spread_edge):.1f} pts "
         f"(calibrated conf {spread_plays.iloc[0].conf_spread:.0%})"
         if len(spread_plays) else "no spread plays"),
        (f"**Best total edge:** "
         f"{total_plays.iloc[0].away} @ {total_plays.iloc[0].home} "
         f"{'OVER' if total_plays.iloc[0].total_edge > 0 else 'UNDER'} "
         f"{total_plays.iloc[0].book_total} — "
         f"{abs(total_plays.iloc[0].total_edge):.1f} pts "
         f"(calibrated conf {total_plays.iloc[0].conf_total:.0%})\n"
         if len(total_plays) else "no total plays\n"),
        f"## Spread plays ({len(spread_plays)})",
        "positive edge = model likes HOME side vs number; conf is "
        "market-shrunk (raw model probs are overconfident vs a line)\n",
        fmt(spread_plays, ["commence", "away", "home", "fair_spread",
                           "book_spread", "spread_edge", "conf_spread"]),
        f"\n## Total plays ({len(total_plays)})",
        "positive edge = model over\n",
        fmt(total_plays, ["commence", "away", "home", "fair_total",
                          "book_total", "total_edge", "conf_total"]),
        f"\n## Moneyline value ({len(ml_plays)})\n",
        ml_plays.to_markdown(index=False, floatfmt=".2f"),
        "\n## Edge improvement roadmap",
        "1. **Bet openers, grade at close** — our validated edge is the "
        "market's early slowness; every run logs to clv_log.parquet, CLV is "
        "the scoreboard.",
        "2. **Line-shop every play** (worth ~1%+); this report already uses "
        "best ML price, spread/total shopping needs per-book output.",
        "3. **Props**: main lines are efficient (backtested −4%); attack via "
        "injury/depth-timing (layer is live), discrete receptions model, "
        "coach game-script → volume, thin derivative markets.",
        "4. **Weeks 1–5 are the window** — 55%+ ATS backtested; discipline "
        "drops after week 6 (no edge mid-season, don't force bets).",
        "5. Re-fit all coefficients after 2025 bowls final + re-run wiki/"
        "portal scrapes late August when rosters publish.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {path}")
    return str(path)


if __name__ == "__main__":
    df = run()
    write_report(df)
    playable = df[df.book_spread.abs() <= 21]
    sp = playable[playable.spread_edge.abs() >= SPREAD_FLAG]
    print(f"\nplayable games: {len(playable)} | flagged: {len(sp)} spreads, "
          f"{int((playable.total_edge.abs() >= TOTAL_FLAG).sum())} totals")
    print(sp.sort_values('spread_edge', key=abs, ascending=False)
          [["away", "home", "fair_spread", "book_spread", "spread_edge"]]
          .head(10).to_string(index=False))
