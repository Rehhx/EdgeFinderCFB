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


def fetch(url: str, session: requests.Session) -> str:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    time.sleep(1.5)
    return r.text


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
    "connecticut huskies": "Connecticut",
    "mississippi rebels": "Ole Miss",
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
    all_rows = []
    for i, (slug, tid, name) in enumerate(teams):
        cache = raw_dir / f"{slug}.html"
        if cache.exists():
            html = cache.read_text(encoding="utf-8")
        else:
            try:
                html = fetch(f"{BASE}/{slug}/{tid}", session)
            except requests.RequestException as e:
                print(f"  miss {slug}: {e}")
                continue
            cache.write_text(html, encoding="utf-8")
        for row in parse_chart(html):
            all_rows.append(row | {"team_name": name, "as_of": as_of})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(teams)} scraped")

    df = map_team_ids(pd.DataFrame(all_rows))
    unmatched = df[df.team_id.isna()].team_name.nunique()
    if unmatched:
        print(f"  {unmatched} team names unmatched to CFBD ids")

    dest = PARQUET_DIR / "depth_charts.parquet"
    if dest.exists():  # keep history, replace same-day snapshot
        old = pd.read_parquet(dest)
        df = pd.concat([old[old.as_of != as_of], df], ignore_index=True)
    df.to_parquet(dest, index=False)
    print(f"depth_charts: {len(df)} rows ({df.as_of.nunique()} snapshots) -> {dest}")
    return df


if __name__ == "__main__":
    scrape()
