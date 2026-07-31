"""Scrape HC/OC/DC per team-season from Wikipedia season-page infoboxes.

Pages like "2024 Alabama Crimson Tide football team" carry
| head_coach = ..., | off_coach = ..., | def_coach = ... in the infobox.
Uses the batched query API (50 titles per request) — the whole 2021-2025
FBS scrape is ~15 requests. Wikitext per page is cached to warehouse/raw/wiki.

  python -m ingestion.scrapers.wiki_staff 2021 2025
"""
import re
import sys
import time
import unicodedata

import pandas as pd
import requests

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR, RAW_DIR

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "CFBResearch/0.1 (personal research project)"}
RAW_WIKI = RAW_DIR / "wiki"
RAW_WIKI.mkdir(parents=True, exist_ok=True)
BATCH = 50

# CFBD school -> "School Mascot" as Wikipedia titles it
ALIASES = {
    "App State": "Appalachian State Mountaineers",
    "Miami (OH)": "Miami RedHawks",
    "UL Monroe": "Louisiana–Monroe Warhawks",
    "Massachusetts": "UMass Minutemen",
    "Florida International": "FIU Panthers",
    "Delaware": "Delaware Fightin' Blue Hens",
    "Southern Miss": "Southern Miss Golden Eagles",
    "Hawai'i": "Hawaii Rainbow Warriors",
}

FIELD_RE = {
    "hc": re.compile(r"\|\s*head_coach\s*=\s*(.+)"),
    "oc": re.compile(r"\|\s*(?:off_coach|cooff_coach1)\s*=\s*(.+)"),
    "dc": re.compile(r"\|\s*(?:def_coach|codef_coach1)\s*=\s*(.+)"),
}


def clean_name(raw: str) -> str | None:
    """'[[Kalen DeBoer]]<ref.../>' -> 'Kalen DeBoer'."""
    raw = re.sub(r"<ref[^<]*?(/>|</ref>)", "", raw)
    m = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", raw)
    name = (m.group(2) or m.group(1)) if m else raw
    name = re.sub(r"[{}\[\]<>|]", "", name).strip()
    name = re.sub(r"\s*\(.*?\)\s*", "", name)
    if not name or "=" in name or len(name) < 4:
        return None  # captured a stray template parameter, not a name
    return name


def _cache_path(title: str):
    return RAW_WIKI / (re.sub(r"[^\w]+", "_", title) + ".txt")


def fetch_batch(titles: list[str], session: requests.Session) -> dict[str, str]:
    """Requested title -> wikitext, following normalization + redirects."""
    for attempt in range(6):
        r = session.post(WIKI_API, data={
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "titles": "|".join(titles), "redirects": "1",
            "format": "json", "formatversion": "2",
        }, timeout=60)
        if r.status_code == 429:
            wait = max(int(r.headers.get("Retry-After", 0)), 15 * (attempt + 1))
            print(f"  429 throttled, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        break
    q = r.json()["query"]

    mapping = {t: t for t in titles}
    for step in q.get("normalized", []) + q.get("redirects", []):
        for k, v in mapping.items():
            if v == step["from"]:
                mapping[k] = step["to"]

    content = {}
    for p in q.get("pages", []):
        if p.get("missing") or "revisions" not in p:
            continue
        content[p["title"]] = p["revisions"][0]["slots"]["main"]["content"]

    out = {}
    for t in titles:
        text = content.get(mapping[t])
        if text:
            out[t] = text
            _cache_path(t).write_text(text, encoding="utf-8")
    time.sleep(3.0)
    return out


def scrape(years: list[int]) -> pd.DataFrame:
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    session = requests.Session()
    session.headers.update(HEADERS)

    def title_for(year, school, mascot, ascii_fallback=False):
        name = ALIASES.get(school, f"{school} {mascot}")
        t = f"{year} {name} football team"
        if ascii_fallback:
            t = unicodedata.normalize("NFKD", t).encode(
                "ascii", "ignore").decode().replace("'", "")
        return t

    keys = [(y, t.school, t.mascot, t.id)
            for y in years for t in teams.itertuples()]
    texts: dict[tuple, str] = {}

    for ascii_fallback in (False, True):
        pending = [k for k in keys if k not in texts]
        titles = {k: title_for(k[0], k[1], k[2], ascii_fallback) for k in pending}
        # serve from cache first
        for k, t in list(titles.items()):
            c = _cache_path(t)
            if c.exists():
                texts[k] = c.read_text(encoding="utf-8")
                del titles[k]
        todo = list(titles.items())
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            got = fetch_batch([t for _, t in chunk], session)
            for k, t in chunk:
                if t in got:
                    texts[k] = got[t]
        if not ascii_fallback:
            print(f"after primary pass: {len(texts)}/{len(keys)} pages")

    missed = [k[:2] for k in keys if k not in texts]
    if missed:
        print(f"missed pages ({len(missed)}): {missed[:10]}"
              f"{'...' if len(missed) > 10 else ''}")

    rows = []
    for (year, school, mascot, team_id), text in texts.items():
        row = {"season": year, "school": school, "team_id": team_id}
        for key, rx in FIELD_RE.items():
            m = rx.search(text)
            row[key] = clean_name(m.group(1)) if m else None
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["school", "season"])
    dest = PARQUET_DIR / "wiki_staff.parquet"
    df.to_parquet(dest, index=False)
    print(f"saved {len(df)} rows -> {dest}")
    for c in ("hc", "oc", "dc"):
        print(f"  {c} coverage: {df[c].notna().mean():.0%}")
    return df


if __name__ == "__main__":
    y0, y1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2021, 2025)
    scrape(list(range(y0, y1 + 1)))
