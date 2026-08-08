"""Scrape Ourlads NCAA depth charts (all FBS teams) into depth_charts.parquet.

Team pages are static HTML; the team list comes from the dropdown on any
team page. Snapshots are dated (as_of) so weekly runs build a history.
Raw HTML cached under warehouse/raw/ourlads/{date}/.

  python -m ingestion.scrapers.ourlads_depth
"""
import re
import time
from datetime import date

import pandas as pd
import requests

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR, RAW_DIR

BASE = "https://www.ourlads.com/ncaa-football-depth-charts/depth-chart"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0"}
SEED_PAGE = f"{BASE}/boston-college/90153"

ROW_RE = re.compile(
    r"<td class='row-dc-(?:wht|grey)'>([A-Z0-9-]{1,6})</td>(.*?)</tr>", re.S)
NAME_RE = re.compile(r"class='lc_[a-z]*'>([^<]+)</a>")
OPT_RE = re.compile(
    r"<option value=[\"']s=([a-z0-9-]+)&amp;id=(\d+)[\"'][^>]*>([^<]+)</option>")


def fetch(url: str, session: requests.Session, tries: int = 3) -> str:
    """GET with retries. A silently dropped team is a permanently missing
    depth chart for that week — 20 of 138 teams were being lost this way."""
    last = None
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            time.sleep(1.5)
            return r.text
        except requests.RequestException as e:
            last = e
            time.sleep(2.0 * (attempt + 1))     # back off, then try again
    raise last


def team_list(session: requests.Session) -> list[tuple[str, str, str]]:
    html = fetch(SEED_PAGE, session)
    teams = OPT_RE.findall(html)
    return [(slug, tid, name) for slug, tid, name in teams if slug]


def parse_chart(html: str) -> list[dict]:
    rows = []
    for pos, body in ROW_RE.findall(html):
        rank = 0
        for raw in NAME_RE.findall(body):
            raw = raw.strip()
            if not raw:
                continue
            rank += 1
            # "McKenzie, Mason RS JR/TR" -> name + class tokens
            m = re.match(r"([^,]+),\s*(\S+)(.*)", raw)
            if m:
                name = f"{m.group(2)} {m.group(1)}"
                cls = m.group(3).strip()
            else:
                name, cls = raw, ""
            rows.append({"pos": pos, "rank": rank, "player": name,
                         "class": cls})
    return rows


# Ourlads display name -> CFBD school
NAME_ALIASES = {
    "appalachian state mountaineers": "App State",
    "central florida knights": "UCF",
    "connecticut huskies": "UConn",
    "mississippi rebels": "Ole Miss",
    # CFBD spells these with diacritics; Ourlads does not
    "hawaii rainbow warriors": "Hawai'i",
    "san jose state spartans": "San José State",
    "louisiana-monroe warhawks": "UL Monroe",
    "miami (ohio) redhawks": "Miami (OH)",
}


def map_team_ids(df: pd.DataFrame) -> pd.DataFrame:
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    teams["full"] = (teams.school + " " + teams.mascot).str.lower()
    by_full = dict(zip(teams.full, teams.id))
    by_school = dict(zip(teams.school.str.lower(), teams.id))

    def match(name: str):
        n = name.lower().replace("&amp;", "&")
        if n in NAME_ALIASES:
            return by_school.get(NAME_ALIASES[n].lower())
        if n in by_full:
            return by_full[n]
        for school, tid in by_school.items():
            if n.startswith(school):
                return tid
        return None

    df["team_id"] = df.team_name.map(match)
    return df


def scrape() -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(HEADERS)
    as_of = date.today().isoformat()
    raw_dir = RAW_DIR / "ourlads" / as_of
    raw_dir.mkdir(parents=True, exist_ok=True)

    teams = team_list(session)
    print(f"{len(teams)} teams in Ourlads dropdown")
    all_rows: list[dict] = []
    missed: list[str] = []
    for i, (slug, tid, name) in enumerate(teams):
        cache = raw_dir / f"{slug}.html"
        if cache.exists():
            html = cache.read_text(encoding="utf-8")
        else:
            try:
                html = fetch(f"{BASE}/{slug}/{tid}", session)
            except requests.RequestException as e:
                print(f"  miss {slug}: {e}")
                missed.append(name)
                continue
            cache.write_text(html, encoding="utf-8")
        rows = parse_chart(html)
        if not rows:
            missed.append(name)
        for row in rows:
            all_rows.append(row | {"team_name": name, "as_of": as_of})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(teams)} scraped")

    df = map_team_ids(pd.DataFrame(all_rows))
    unmatched = sorted(df[df.team_id.isna()].team_name.unique())
    if unmatched:
        print(f"  [!] {len(unmatched)} team names unmatched to CFBD ids: "
              f"{unmatched}")
    if missed:
        print(f"  [!] {len(missed)} teams returned NO depth chart "
              f"(permanently lost for this week): {missed}")
    print(f"  captured {df.team_id.nunique()} of {len(teams)} teams")

    dest = PARQUET_DIR / "depth_charts.parquet"
    if dest.exists():  # keep history, replace same-day snapshot
        old = pd.read_parquet(dest)
        df = pd.concat([old[old.as_of != as_of], df], ignore_index=True)
    df.to_parquet(dest, index=False)
    print(f"depth_charts: {len(df)} rows ({df.as_of.nunique()} snapshots) -> {dest}")
    return df


MAX_SNAPSHOT_AGE_DAYS = 10   # weekly cadence + slack; louder than a silent stall


def history() -> pd.DataFrame:
    """Report the accumulated snapshot history and shout if it has stalled.

    Depth charts are the ONLY as-of source for who is expected to play, and
    they cannot be reconstructed after the fact — a week not captured is gone
    forever. `models/props.py` shows player SHARE is 55% of the rush-attempt
    error and the one component with real signal left to sharpen, but the
    2023-25 backtest could not use depth charts at all because only two
    snapshots existed, both from 2026-07. This function exists so the history
    building toward 2027 fails LOUDLY rather than quietly.
    """
    dest = PARQUET_DIR / "depth_charts.parquet"
    if not dest.exists():
        print("[!] no depth_charts.parquet yet — run the scraper.")
        return pd.DataFrame()
    d = pd.read_parquet(dest)
    snaps = (d.groupby("as_of")
              .agg(rows=("player", "size"), teams=("team_id", "nunique"))
              .sort_index())
    print(f"depth-chart history: {len(snaps)} snapshots, {len(d):,} rows")
    print(snaps.tail(12).to_string())
    newest = pd.to_datetime(snaps.index.max())
    age = (pd.Timestamp.now().normalize() - newest).days
    if age > MAX_SNAPSHOT_AGE_DAYS:
        print(f"\n[!] NEWEST SNAPSHOT IS {age} DAYS OLD — the weekly capture "
              f"has stalled. Every missed week is permanently lost; depth "
              f"charts cannot be backfilled.")
    else:
        print(f"\nnewest snapshot {newest.date()} ({age}d old) — healthy")
    thin = snaps[snaps.teams < 100]
    if len(thin):
        print(f"[!] {len(thin)} snapshot(s) cover <100 teams (partial scrape): "
              f"{list(thin.index)}")
    return snaps


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "history":
        history()
    else:
        scrape()
        history()
