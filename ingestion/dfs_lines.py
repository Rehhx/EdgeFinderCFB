"""Log DFS pick'em lines (Underdog live; PrizePicks by manual import).

WHY THIS EXISTS
Traditional US books rarely post NCAAF rush attempts / pass attempts /
completions, and The Odds API's historical archive has none of them — so the
volume markets CANNOT be backtested the way every other play in this repo was
(HANDOFF §16). DFS pick'em books do post them, typically near the median at
roughly even money ("23.5 completions"). Those lines are set for engagement
rather than sharp balance, which is where a model edge can live.

Since there is no history to backtest, we grade them FORWARD: log every line we
see with an `as_of` stamp, then settle against `player_game_logs` (which now
carries attempts, completions and touchdowns) once the games are played. After
a real sample, `report()` says whether the projections beat the lines.

  python -m ingestion.dfs_lines fetch          # pull + append Underdog
  python -m ingestion.dfs_lines import-pp FILE # add a PrizePicks CSV export
  python -m ingestion.dfs_lines grade          # settle + score vs our model

SOURCES
- Underdog: a public, unauthenticated JSON endpoint. Polled politely, cached
  with an as_of stamp, no credentials and no bet placement.
- PrizePicks: their API sits behind a DataDome CAPTCHA challenge (verified
  403 + geo.captcha-delivery.com). That is a deliberate access control, so
  this module does NOT try to defeat it. Export what you see and feed it to
  `import-pp`; it normalises into the identical schema and grades the same way.
"""
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

UNDERDOG_URL = "https://api.underdogfantasy.com/beta/v6/over_under_lines"
HEADERS = {"User-Agent": "cfb-model/1.0 (personal betting research)",
           "Accept": "application/json"}
DEST = PARQUET_DIR / "dfs_lines.parquet"
SEASON = 2026

# Underdog stat keys -> our player_game_logs columns. Their per-game vocabulary
# is `<verb>ing_yds` style; attempts/completions were not in the preseason
# slate when this was written, so several plausible spellings are mapped and
# ANY unmapped key is still logged (with stat=None) rather than dropped —
# that is how we discover the real key when CFB game props post.
STAT_MAP = {
    "rushing_yds": "rush_yds", "rush_yds": "rush_yds",
    "receiving_yds": "rec_yds", "rec_yds": "rec_yds",
    "passing_yds": "pass_yds", "pass_yds": "pass_yds",
    "receptions": "receptions",
    "passing_completions": "pass_comp", "pass_completions": "pass_comp",
    "completions": "pass_comp",
    "passing_attempts": "pass_att", "pass_attempts": "pass_att",
    "rushing_attempts": "rush_att", "rush_attempts": "rush_att",
    "carries": "rush_att",
    "passing_tds": "pass_td", "rushing_tds": "rush_td",
    "receiving_tds": "rec_td",
}
# the markets our model can actually price (see models/props.py::project)
PRICEABLE = {"rush_yds", "rec_yds", "pass_yds", "receptions",
             "rush_att", "pass_att", "pass_comp"}

COLUMNS = ["as_of", "source", "line_id", "player", "team", "opponent",
           "scheduled_at", "scope", "raw_stat", "stat", "line",
           "over_mult", "under_mult", "over_price", "under_price"]


def _scope(raw) -> str:
    """season-long future, sub-period prop, or a full-game prop."""
    if not isinstance(raw, str) or not raw or raw.lower() in ("nan", "none"):
        return "unknown"
    if raw.startswith("season_"):
        return "season"
    if raw.startswith("period_"):
        return "period"
    return "game"


def fetch_underdog() -> dict:
    r = requests.get(UNDERDOG_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def parse_underdog(raw: dict, sport: str = "CFB") -> pd.DataFrame:
    players = {p["id"]: p for p in raw.get("players", [])
               if p.get("sport_id") == sport}
    apps = {a["id"]: a for a in raw.get("appearances", [])
            if a.get("player_id") in players}
    games = {g["id"]: g for g in raw.get("games", [])}
    now = datetime.now(timezone.utc).isoformat()

    rows = []
    for ln in raw.get("over_under_lines", []):
        ou = ln.get("over_under") or {}
        ast = ou.get("appearance_stat") or {}
        app = apps.get(ast.get("appearance_id"))
        if app is None:
            continue
        p = players[app["player_id"]]
        g = games.get(app.get("match_id")) or {}
        over = under = None
        for o in ln.get("options", []):
            (over := o) if o.get("choice") == "higher" else (under := o)
        raw_stat = ast.get("stat")
        rows.append({
            "as_of": now, "source": "underdog", "line_id": ln.get("id"),
            "player": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
            "team": p.get("team_id"), "opponent": g.get("abbreviated_title"),
            "scheduled_at": g.get("scheduled_at"),
            "scope": _scope(raw_stat), "raw_stat": raw_stat,
            "stat": STAT_MAP.get(raw_stat),
            "line": pd.to_numeric(ln.get("stat_value"), errors="coerce"),
            "over_mult": pd.to_numeric((over or {}).get("payout_multiplier"),
                                       errors="coerce"),
            "under_mult": pd.to_numeric((under or {}).get("payout_multiplier"),
                                        errors="coerce"),
            "over_price": pd.to_numeric((over or {}).get("american_price"),
                                        errors="coerce"),
            "under_price": pd.to_numeric((under or {}).get("american_price"),
                                         errors="coerce"),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def import_prizepicks_csv(path: str) -> pd.DataFrame:
    """Normalise a manually exported PrizePicks table.

    Accepts any CSV with columns for the player, the stat and the line; header
    names are matched loosely so an export or a hand-built sheet both work.
    """
    df = pd.read_csv(path)
    # match headers loosely: "Stat Type", "stat_type" and "stat type" all work
    low = {}
    for c in df.columns:
        k = c.lower().strip()
        low.setdefault(k, c)
        low.setdefault(k.replace(" ", "_"), c)
        low.setdefault(k.replace("_", " "), c)

    def col(*names):
        for n in names:
            if n in low:
                return df[low[n]]
        return pd.Series([None] * len(df))

    raw_stat = (col("stat", "stat_type", "market", "type")
                .astype("object").fillna("").astype(str))
    out = pd.DataFrame({
        "as_of": datetime.now(timezone.utc).isoformat(),
        "source": "prizepicks",
        "line_id": col("id", "line_id"),
        "player": col("player", "name", "player_name"),
        "team": col("team"), "opponent": col("opponent", "opp", "game"),
        "scheduled_at": col("start_time", "scheduled_at", "date"),
        "raw_stat": raw_stat,
        "stat": raw_stat.str.lower().str.replace(" ", "_").map(STAT_MAP),
        "line": pd.to_numeric(col("line", "line_score", "value"),
                              errors="coerce"),
        "over_mult": pd.to_numeric(col("over_mult"), errors="coerce"),
        "under_mult": pd.to_numeric(col("under_mult"), errors="coerce"),
        "over_price": pd.to_numeric(col("over_price"), errors="coerce"),
        "under_price": pd.to_numeric(col("under_price"), errors="coerce"),
    })
    out["scope"] = out.raw_stat.map(_scope)
    return out[COLUMNS]


def fetch() -> None:
    raw = fetch_underdog()
    df = parse_underdog(raw)
    before = len(pd.read_parquet(DEST)) if DEST.exists() else 0
    if not df.empty:
        keep = df[df.line.notna()]
        _write(keep)
    after = len(pd.read_parquet(DEST)) if DEST.exists() else 0
    g = df[df.scope == "game"]
    print(f"underdog: {len(df)} CFB lines ({len(g)} game props), "
          f"{after - before} new/moved rows logged -> {DEST}")
    unmapped = sorted(set(df[df.stat.isna()].raw_stat.dropna()))
    if unmapped:
        print(f"  UNMAPPED stat keys (add to STAT_MAP if priceable): {unmapped}")
    if not g.empty:
        print(g.groupby("raw_stat").size().to_string())


def _write(df: pd.DataFrame) -> None:
    """Append only lines that are NEW or have MOVED.

    Same rule as the CLV log (README discipline #22): de-duplicate on the
    signal, not the clock. A genuine line move is always recorded; re-running
    against an unchanged board costs nothing.
    """
    key = ["source", "player", "raw_stat"]
    if DEST.exists():
        old = pd.read_parquet(DEST)
        last = (old.sort_values("as_of").groupby(key, as_index=False).last()
                   [key + ["line"]].rename(columns={"line": "prev"}))
        m = df.merge(last, on=key, how="left")
        df = df[~((m.line == m.prev) & m.prev.notna()).values]
        if df.empty:
            return
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(DEST, index=False)


def grade() -> pd.DataFrame:
    """Settle logged game props against real player logs and score the model.

    Uses the LAST line seen per (source, player, stat) before kickoff.
    """
    if not DEST.exists():
        print("no dfs_lines.parquet yet — run `fetch` on game week.")
        return pd.DataFrame()
    d = pd.read_parquet(DEST)
    d = d[(d.scope == "game") & d.stat.isin(PRICEABLE) & d.line.notna()]
    if d.empty:
        print("no gradeable game props logged yet "
              "(season-long markets are skipped).")
        return pd.DataFrame()
    d = d.sort_values("as_of").groupby(
        ["source", "player", "stat"], as_index=False).last()

    logs = pd.read_parquet(PARQUET_DIR / "player_game_logs.parquet")
    logs = logs[logs.season == SEASON]
    if logs.empty:
        print(f"no {SEASON} player logs yet — grade after games are played.")
        return pd.DataFrame()
    long = logs.melt(id_vars=["season", "week", "player"],
                     value_vars=sorted(PRICEABLE & set(logs.columns)),
                     var_name="stat", value_name="actual")
    m = d.merge(long, on=["player", "stat"], how="inner")
    m = m[m.actual.notna()]
    if m.empty:
        print("logged lines do not match any played games yet.")
        return m
    m["result"] = pd.cut(m.actual - m.line, [-1e9, -1e-9, 1e-9, 1e9],
                         labels=["under", "push", "over"])

    proj = pd.read_parquet(PARQUET_DIR / "prop_projections.parquet")
    pcol = {s: f"proj_{s}" for s in PRICEABLE}
    plong = proj[proj.season == SEASON].melt(
        id_vars=["season", "week", "player"],
        value_vars=[c for c in pcol.values() if c in proj.columns],
        var_name="pstat", value_name="proj")
    plong["stat"] = plong.pstat.str.replace("proj_", "", regex=False)
    m = m.merge(plong[["season", "week", "player", "stat", "proj"]],
                on=["season", "week", "player", "stat"], how="left")
    m["model_side"] = pd.NA
    ok = m.proj.notna()
    m.loc[ok, "model_side"] = (m.loc[ok, "proj"] > m.loc[ok, "line"]).map(
        {True: "over", False: "under"})
    graded = m[m.model_side.notna() & (m.result != "push")]
    m["won"] = (m.model_side == m.result)

    print(f"\nDFS forward grading — {len(m)} settled lines "
          f"({len(graded)} with a model side, pushes excluded)\n")
    if len(graded):
        print(f"{'source':<12} {'stat':<12} {'n':>5} {'model hit%':>11}")
        for (src, st), g in graded.groupby(["source", "stat"]):
            print(f"{src:<12} {st:<12} {len(g):>5} "
                  f"{(g.model_side == g.result).mean():>10.1%}")
        hit = (graded.model_side == graded.result).mean()
        print(f"\nOVERALL model hit rate: {hit:.1%} on {len(graded)} lines")
        print("A ~50/50 DFS line means break-even is roughly 50% before the")
        print("multi-pick structure; you need a real margin over that.")
    return m


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        fetch()
    elif cmd == "import-pp":
        df = import_prizepicks_csv(sys.argv[2])
        _write(df[df.line.notna()])
        print(f"prizepicks: imported {len(df)} rows -> {DEST}")
        un = sorted(set(df[df.stat.isna()].raw_stat.dropna()))
        if un:
            print(f"  UNMAPPED stat names: {un}")
    elif cmd == "grade":
        grade()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
