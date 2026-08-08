"""August refit: pull newly-published 2026 data and re-fit all model coefs.

Designed to run unattended (Windows Task Scheduler, weekly from mid-August):
  1. Fresh pulls of 2026 returning production, portal, games/lines,
     bulk rosters, and the Wikipedia staff scrape (fills late G5 pages).
  2. Rebuild coach DB + player logs derivatives.
  3. Re-run the fitting backtests — each writes its coefficients to
     warehouse/model_coefs.json, which the picks/ modules auto-load.
  4. Write reports/refit_<date>.md summarizing what changed.
Idempotent and safe to re-run; steps that fail (e.g. data not yet
published) are logged and retried on the next scheduled run.

  python -m scripts.august_refit
"""
import json
import subprocess
import sys
from datetime import date

import pandas as pd
import requests

from ingestion.cfbd_client import CFBDClient
from ingestion.config import CFBD_PARQUET_DIR, COEFS_PATH, PARQUET_DIR, PROJECT_ROOT

PY = sys.executable
LIVE_SEASON = 2026
LOG: list[str] = []


def step(name: str, fn) -> bool:
    try:
        msg = fn()
        LOG.append(f"- OK **{name}**: {msg}")
        return True
    except Exception as e:
        LOG.append(f"- FAIL **{name}**: {type(e).__name__}: {e}")
        return False


# REQUIRED: nothing can be priced without the schedule and the lines.
# OPTIONAL: CFBD publishes these late; the callers already degrade (see
# features/roster_priors.py::roster_features, which falls back to league-median
# returning production and to NaN OL continuity).
CFBD_2026 = [
    # name,             endpoint,             required
    ("games_2026",      "/games",             True),
    ("lines_2026",      "/lines",             True),
    ("portal_2026",     "/player/portal",     False),
    ("returning_2026",  "/player/returning",  False),
]


def pull_cfbd_2026() -> str:
    """Pull each 2026 dataset INDEPENDENTLY.

    This used to be one loop that raised on the first empty response, and
    `returning_2026` was first in it. So while returning production was
    unpublished — still true on 2026-08-05 — the schedule and the BETTING LINES
    were never pulled at all, on every weekly run. One unpublished preseason
    stat silently blocked the two datasets the whole book depends on.

    An empty OPTIONAL pull is now reported and skipped WITHOUT overwriting the
    last good parquet; an empty REQUIRED pull is the only thing that fails the
    step. Never write an empty file over real data — a zero-row roster makes
    every prior-season contributor look departed.
    """
    c = CFBDClient()
    ok, missing, failed = [], [], []
    for name, ep, required in CFBD_2026:
        dest = CFBD_PARQUET_DIR / f"{name}.parquet"
        try:
            data = c.get(ep, {"year": 2026}, refresh=True)
        except Exception as e:
            (failed if required else missing).append(f"{name}({type(e).__name__})")
            continue
        if not data:
            missing.append(f"{name}=unpublished"
                           + (" **REQUIRED**" if required else ""))
            if required:
                failed.append(name)
            continue
        # /lines is NESTED (one row per game, a `lines` list per book). It must
        # be written with the SAME flattener run_ingest uses for 2015-2025, or
        # picks/paper_trades.py::settle() KeyErrors on spread/overUnder and
        # Sunday grading dies silently.
        from ingestion.run_ingest import flatten_lines
        df = flatten_lines(data) if ep == "/lines" else pd.json_normalize(data)
        df.to_parquet(dest, index=False)
        ok.append(f"{name}={len(df)}")

    msg = ", ".join(ok) or "nothing pulled"
    if missing:
        msg += " | not yet published: " + ", ".join(missing)
    msg += f" (calls used {c.calls_used()}/1000)"
    if failed:
        raise RuntimeError(f"REQUIRED 2026 data missing: {', '.join(failed)}. {msg}")
    return msg


def pull_pbp_live() -> str:
    """Re-download the LIVE season's play-by-play. Must run every week.

    `download_season` short-circuits to "cached" whenever the file exists, which
    is right for a finished season and wrong for the current one — that file
    GROWS every Saturday. Without a forced re-pull the live season's PBP would
    freeze at whatever week it was first fetched, and the weekly ratings refit
    (features/epa_ratings.live_asof) would have nothing new to fit on.

    404s until the season opens; `step()` logs that and retries next run.
    """
    from ingestion.bulk_pbp import download_season
    info = download_season(LIVE_SEASON, force=True)
    return f"{info.get('rows', 0):,} plays ({info.get('status')})"


def pull_ratings() -> str:
    """Refresh SP+/FPI power ratings (used as ml_spread ensemble features)."""
    import pandas as pd
    c = CFBDClient()
    sp, fpi = [], []
    for y in range(2015, 2027):
        for r in c.get("/ratings/sp", {"year": y}):
            if r.get("team"):
                sp.append({"season": y, "team": r["team"], "sp": r["rating"],
                           "sp_off": r.get("offense"), "sp_def": r.get("defense")})
        try:
            for r in c.get("/ratings/fpi", {"year": y}):
                if r.get("team"):
                    fpi.append({"season": y, "team": r["team"], "fpi": r["fpi"]})
        except Exception:
            pass
    pd.DataFrame(sp).to_parquet(CFBD_PARQUET_DIR / "sp_ratings.parquet", index=False)
    if fpi:
        pd.DataFrame(fpi).to_parquet(CFBD_PARQUET_DIR / "fpi_ratings.parquet", index=False)
    return f"SP+ {len(sp)} rows, FPI {len(fpi)} rows"


def pull_rosters_2026() -> str:
    url = ("https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/"
           "main/rosters/parquet/rosters_2026.parquet")
    dest = PARQUET_DIR / "rosters" / "rosters_2026.parquet"
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"rosters_2026 not published yet ({r.status_code})")
    dest.write_bytes(r.content)
    return f"{len(r.content) / 1e6:.1f} MB"


def pull_cfbd_roster_2026() -> str:
    """CFBD /roster for 2026 — required for VACATED SHARE in player props.

    Separate from pull_rosters_2026() above, which uses cfbfastR and a
    different schema. The vacated-share feature keys off CFBD names/teams so
    it must come from the CFBD endpoint.

    This returned ZERO rows on 2026-08-04 because the roster was not published
    yet, so it MUST be re-run closer to week 1. Raises if still empty, rather
    than silently writing an empty parquet that would make every prior-season
    contributor look departed.
    """
    import pandas as pd
    from ingestion.cfbd_client import CFBDClient
    from ingestion.config import CFBD_PARQUET_DIR
    df = pd.json_normalize(CFBDClient().get("/roster", {"year": 2026}))
    if len(df) < 1000:
        raise RuntimeError(
            f"roster_2026 has only {len(df)} rows — not published yet. "
            "Vacated share CANNOT be computed until this fills; re-run later.")
    dest = CFBD_PARQUET_DIR / "roster_2026.parquet"
    df.to_parquet(dest, index=False)
    return f"{len(df):,} players"


def run_module(mod: str) -> str:
    r = subprocess.run([PY, "-m", mod], capture_output=True, text=True,
                       cwd=PROJECT_ROOT, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr
                           else f"exit {r.returncode}")
    return "ran"


def big_dog_picks() -> str:
    """Short & sweet: the validated big-dog spread edge, current board."""
    try:
        import pandas as pd
        from picks.ml_spread_picks import run as mls_run, validated_plays
        picks = validated_plays(mls_run())
        if picks.empty:
            return "\n## Big-dog spread picks — none on the board yet\n"
        teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
        s = dict(zip(teams.id, teams.school))
        lines = ["\n## Big-dog spread picks (wks 1-5 edge: ~59% ATS, +12.8% ROI)"]
        for r in picks.itertuples():
            dog_id, fav_id = ((r.home_id, r.away_id) if r.edge > 0
                              else (r.away_id, r.home_id))
            dog, fav = s.get(dog_id, "?"), s.get(fav_id, "?")
            lines.append(f"- pick {dog} spread vs {fav} "
                         f"(+{abs(r.book_spread):g}) — edge {abs(r.edge):g} pts")
        return "\n".join(lines) + "\n"
    except Exception as e:
        return f"\n## Big-dog spread picks — unavailable ({e})\n"


def main() -> None:
    old = (json.loads(COEFS_PATH.read_text(encoding="utf-8"))
           if COEFS_PATH.exists() else {})

    data_ok = step("CFBD 2026 pulls", pull_cfbd_2026)
    step("bulk rosters 2026", pull_rosters_2026)
    step("CFBD roster 2026 (vacated share)", pull_cfbd_roster_2026)
    step(f"live play-by-play {LIVE_SEASON} (feeds the weekly ratings refit)",
         pull_pbp_live)
    step("SP+/FPI power ratings", pull_ratings)
    step("wiki staff 2021-2026",
         lambda: run_module("ingestion.scrapers.wiki_staff") and "ran")
    step("coach DB rebuild", lambda: run_module("features.coach_db"))
    step("defense profiles", lambda: run_module("features.defense_profiles"))
    step("defense players", lambda: run_module("features.defense_players"))
    step("matchup stats (off vs def pass/rush)",
         lambda: run_module("features.matchups"))
    step("depth chart snapshot",
         lambda: run_module("ingestion.scrapers.ourlads_depth"))

    # re-fit everything (each module writes model_coefs.json)
    step("spread coefs (backtest.spread_phase1)",
         lambda: run_module("backtest.spread_phase1"))
    step("totals coefs (models.game_sim)", lambda: run_module("models.game_sim"))
    step("1H coefs (backtest.first_half)",
         lambda: run_module("backtest.first_half"))
    step("prop projections (models.props)", lambda: run_module("models.props"))
    step("prop recal/blend (backtest.props_vs_book)",
         lambda: run_module("backtest.props_vs_book"))

    new = (json.loads(COEFS_PATH.read_text(encoding="utf-8"))
           if COEFS_PATH.exists() else {})
    day = date.today().isoformat()
    rep = PROJECT_ROOT / "reports" / f"refit_{day}.md"
    rep.parent.mkdir(exist_ok=True)
    rep.write_text("\n".join(
        [f"# August refit — {day}",
         big_dog_picks(),
         f"\n2026 data available: {'YES' if data_ok else 'NOT YET (will retry next run)'}\n",
         *LOG,
         "\n## Coefficients (old -> new)",
         "```json", json.dumps({"old": old, "new": new}, indent=2), "```",
         "\nPicks modules auto-load the new values. Next: review "
         "reports/edge_report + ml_value regenerated numbers before betting."]),
        encoding="utf-8")
    print(f"refit report -> {rep}")
    for line in LOG:
        print(line)


if __name__ == "__main__":
    main()
