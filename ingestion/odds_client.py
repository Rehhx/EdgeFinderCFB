"""The Odds API client (the-odds-api.com) with credit tracking + raw caching.

Every response is cached to warehouse/raw/odds keyed by endpoint+params, and
credit usage (x-requests-used / remaining headers) is logged to
odds_usage.jsonl. Historical endpoints burn credits fast — always check
remaining() before large pulls.
"""
import hashlib
import json
import time
from datetime import datetime, timezone

import requests

from ingestion.config import ODDS_API_KEY, RAW_DIR

BASE = "https://api.the-odds-api.com/v4"
RAW_ODDS = RAW_DIR / "odds"
RAW_ODDS.mkdir(parents=True, exist_ok=True)
USAGE_LOG = RAW_ODDS / "odds_usage.jsonl"

NCAAF = "americanfootball_ncaaf"
PROP_MARKETS = ["player_pass_yds", "player_rush_yds",
                "player_receptions", "player_reception_yds"]


class OddsClient:
    def __init__(self, api_key: str = ODDS_API_KEY):
        if not api_key:
            raise RuntimeError("ODDS_API_KEY is not set in .env")
        self.key = api_key
        self.session = requests.Session()
        self.last_remaining = None

    def _cache_path(self, endpoint: str, params: dict):
        h = hashlib.sha1(json.dumps([endpoint, params],
                                    sort_keys=True).encode()).hexdigest()[:16]
        safe = endpoint.strip("/").replace("/", "_")[:80]
        return RAW_ODDS / f"{safe}_{h}.json"

    def get(self, endpoint: str, params: dict | None = None,
            refresh: bool = False):
        params = dict(params or {})
        cache = self._cache_path(endpoint, params)
        if cache.exists() and not refresh:
            return json.loads(cache.read_text(encoding="utf-8"))["data"]

        r = self.session.get(f"{BASE}{endpoint}",
                             params=params | {"apiKey": self.key}, timeout=60)
        used = r.headers.get("x-requests-used")
        remaining = r.headers.get("x-requests-remaining")
        self.last_remaining = remaining
        with open(USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "endpoint": endpoint, "status": r.status_code,
                "used": used, "remaining": remaining,
            }) + "\n")
        r.raise_for_status()
        data = r.json()
        cache.write_text(json.dumps(
            {"as_of": datetime.now(timezone.utc).isoformat(),
             "params": {k: v for k, v in params.items()},
             "data": data}), encoding="utf-8")
        time.sleep(0.4)
        return data

    def remaining(self) -> str | None:
        return self.last_remaining
