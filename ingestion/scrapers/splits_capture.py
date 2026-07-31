"""DIY betting-splits capture — build our own proprietary splits history.

Scrapes the FREE, public Action Network public-betting page (no paywall, no
login — same category as our Ourlads/Wikipedia scrapers) and extracts, per
game / market / side: the line, price, **tickets% (share of bets)** and
**money% (share of handle)**.

Why it matters: the strongest documented angle in betting research is fading
the public's overbet favorite (>75% handle side goes ~47% ATS). Our big-dog
spread edge is only a *proxy* for that. Vendors sell this data but not with
history, and we refuse to backtest-free bets — so we capture our own,
timestamped, weekly, all season. By 2027 we own a backtestable dataset.

Run 2-3x/week in-season (lines move; splits build through the week):
  python -m ingestion.scrapers.splits_capture
"""
import json
import re
from datetime import datetime, timezone

import pandas as pd
import requests

from ingestion.config import PARQUET_DIR, RAW_DIR

URL = "https://www.actionnetwork.com/ncaaf/public-betting"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}
DEST = PARQUET_DIR / "betting_splits.parquet"
RAW = RAW_DIR / "splits"
MARKETS = ("moneyline", "spread", "total")


def fetch() -> dict:
    r = requests.get(URL, headers=HEADERS, timeout=45)
    r.raise_for_status()
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                  r.text, re.S)
    if not m:
        raise RuntimeError("__NEXT_DATA__ not found (page layout changed)")
    return json.loads(m.group(1))


def parse(data: dict, as_of: str) -> pd.DataFrame:
    sb = data["props"]["pageProps"]["scoreboardResponse"]
    rows = []
    for g in sb.get("games", []):
        teams = {t["id"]: t.get("full_name") or t.get("display_name")
                 for t in (g.get("teams") or [])}
        home = teams.get(g.get("home_team_id"))
        away = teams.get(g.get("away_team_id"))
        for book_id, mk in (g.get("markets") or {}).items():
            event = (mk or {}).get("event") or {}
            for market in MARKETS:
                for o in event.get(market) or []:
                    bi = o.get("bet_info") or {}
                    tix = (bi.get("tickets") or {}).get("percent")
                    mon = (bi.get("money") or {}).get("percent")
                    if tix in (None, 0) and mon in (None, 0):
                        continue  # not populated yet (preseason / no action)
                    rows.append({
                        "as_of": as_of, "game_id": g.get("id"),
                        "season": g.get("season"), "week": g.get("week"),
                        "start_time": g.get("start_time"),
                        "home": home, "away": away,
                        "book_id": str(book_id), "market": market,
                        "side": o.get("side"), "team_id": o.get("team_id"),
                        "line": o.get("value"), "price": o.get("odds"),
                        "tickets_pct": tix, "money_pct": mon,
                    })
    return pd.DataFrame(rows)


def capture() -> pd.DataFrame:
    as_of = datetime.now(timezone.utc).isoformat(timespec="minutes")
    data = fetch()
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = as_of.replace(":", "").replace("-", "")
    (RAW / f"an_{stamp}.json").write_text(json.dumps(data), encoding="utf-8")

    df = parse(data, as_of)
    if df.empty:
        print(f"{as_of}: page fetched OK but no splits populated yet "
              "(expected in preseason — splits appear once books take action).")
        return df

    if DEST.exists():  # append; keep one row per (as_of, game, book, market, side)
        old = pd.read_parquet(DEST)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["as_of", "game_id", "book_id", "market", "side"],
            keep="last")
    df.to_parquet(DEST, index=False)
    snaps = df.as_of.nunique()
    print(f"{as_of}: captured {len(parse(data, as_of)):,} rows "
          f"({df.game_id.nunique()} games) | history: {len(df):,} rows "
          f"across {snaps} snapshots -> {DEST.name}")
    return df


def public_fade_candidates(min_money_pct: float = 75.0) -> pd.DataFrame:
    """Latest snapshot: sides with >=X% of HANDLE — fade candidates."""
    if not DEST.exists():
        return pd.DataFrame()
    d = pd.read_parquet(DEST)
    d = d[(d.as_of == d.as_of.max()) & (d.market == "spread")]
    heavy = d[d.money_pct >= min_money_pct]
    return heavy[["away", "home", "side", "line", "money_pct", "tickets_pct"]]


if __name__ == "__main__":
    capture()
