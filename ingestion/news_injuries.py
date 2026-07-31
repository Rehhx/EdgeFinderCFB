"""Injury/availability feeds: ESPN structured injuries + Claude news extraction.

1. ESPN injuries endpoint (free, structured): player, team, status, comment.
   Stale entries filtered by date. Appended snapshots -> espn_injuries.parquet.
2. ESPN CFB news feed -> Claude (CLAUDE_MODEL_FAST) extracts structured
   availability facts (injury/suspension/transfer/role change) that the
   injuries endpoint misses -> news_extractions.parquet.

  python -m ingestion.news_injuries            # both feeds
  python -m ingestion.news_injuries --no-llm   # skip Claude step
"""
import json
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from ingestion.config import (ANTHROPIC_API_KEY, CLAUDE_MODEL_FAST,
                              PARQUET_DIR, RAW_DIR)

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
MAX_AGE_DAYS = 90
RAW_NEWS = RAW_DIR / "news"
RAW_NEWS.mkdir(parents=True, exist_ok=True)

EXTRACT_PROMPT = """You extract player-availability facts from college football news.
From the articles below, extract every concrete fact about a PLAYER's availability
or role: injury, suspension, transfer, opt-out, position battle won/lost, starter
named, retirement. Ignore rumors, coach quotes without substance, and team-level news.

Return ONLY a JSON array (possibly empty), one object per fact:
{"player": "...", "team": "...", "kind": "injury|suspension|transfer|role_change|other",
 "status": "out|questionable|active|unknown", "detail": "<=15 words", "headline_idx": N}

Articles:
"""


def fetch_espn_injuries() -> pd.DataFrame:
    data = requests.get(f"{ESPN}/injuries", timeout=30).json()
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    rows = []
    for team in data.get("injuries", []):
        for inj in team.get("injuries", []):
            d = inj.get("date")
            if d and datetime.fromisoformat(d.replace("Z", "+00:00")) < cutoff:
                continue
            ath = inj.get("athlete", {})
            rows.append({
                "as_of": datetime.now(timezone.utc).date().isoformat(),
                "team": team.get("displayName"),
                "player": ath.get("displayName"),
                "position": ath.get("position", {}).get("abbreviation"),
                "status": inj.get("status"),
                "date": d,
                "comment": inj.get("shortComment"),
            })
    df = pd.DataFrame(rows)
    dest = PARQUET_DIR / "espn_injuries.parquet"
    if dest.exists() and len(df):
        old = pd.read_parquet(dest)
        df = pd.concat([old[old.as_of != df.as_of.iloc[0]], df],
                       ignore_index=True)
    if len(df):
        df.to_parquet(dest, index=False)
    print(f"espn_injuries: {len(df)} rows -> {dest.name}")
    return df


def fetch_news(limit: int = 50) -> list[dict]:
    data = requests.get(f"{ESPN}/news", params={"limit": limit}, timeout=30).json()
    arts = [{
        "headline": a.get("headline", ""),
        "description": a.get("description", ""),
        "published": a.get("published", ""),
    } for a in data.get("articles", [])]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (RAW_NEWS / f"espn_news_{stamp}.json").write_text(
        json.dumps(arts), encoding="utf-8")
    return arts


def extract_with_claude(articles: list[dict]) -> pd.DataFrame:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    body = "\n".join(
        f"[{i}] {a['headline']} — {a['description']}"
        for i, a in enumerate(articles))
    msg = client.messages.create(
        model=CLAUDE_MODEL_FAST, max_tokens=2000,
        messages=[{"role": "user", "content": EXTRACT_PROMPT + body}])
    text = next(b.text for b in msg.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    facts = json.loads(text)
    df = pd.DataFrame(facts)
    if len(df):
        df["as_of"] = datetime.now(timezone.utc).date().isoformat()
        df["published"] = df.headline_idx.map(
            lambda i: articles[i]["published"] if 0 <= i < len(articles) else None)
    dest = PARQUET_DIR / "news_extractions.parquet"
    if dest.exists() and len(df):
        old = pd.read_parquet(dest)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["player", "kind", "detail"], keep="last")
    if len(df):
        df.to_parquet(dest, index=False)
    print(f"news_extractions: {len(df)} facts -> {dest.name}")
    return df


if __name__ == "__main__":
    fetch_espn_injuries()
    articles = fetch_news()
    print(f"news feed: {len(articles)} articles")
    if "--no-llm" in sys.argv or not ANTHROPIC_API_KEY:
        print("skipping Claude extraction")
    else:
        out = extract_with_claude(articles)
        if len(out):
            print(out[["player", "team", "kind", "status", "detail"]]
                  .head(10).to_string(index=False))
