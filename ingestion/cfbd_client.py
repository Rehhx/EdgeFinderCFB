"""Minimal CollegeFootballData API client.

Free tier is 1,000 calls/month, so every response is cached to disk
(warehouse/raw/cfbd) with an as_of timestamp and re-served from cache,
and every real API hit is appended to usage_log.jsonl for quota tracking.
"""
import hashlib
import json
import time
from datetime import datetime, timezone

import requests

from .config import CFBD_API_KEY, CFBD_BASE_URL, CFBD_RAW_DIR

USAGE_LOG = CFBD_RAW_DIR / "usage_log.jsonl"


class CFBDClient:
    def __init__(self, api_key: str = CFBD_API_KEY):
        if not api_key:
            raise RuntimeError("CFBD_API_KEY is not set in .env")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _cache_path(self, endpoint: str, params: dict):
        key = hashlib.sha1(
            json.dumps([endpoint, params], sort_keys=True).encode()
        ).hexdigest()[:16]
        safe = endpoint.strip("/").replace("/", "_")
        return CFBD_RAW_DIR / f"{safe}_{key}.json"

    def get(self, endpoint: str, params: dict | None = None, refresh: bool = False):
        """GET an endpoint, serving from disk cache unless refresh=True."""
        params = params or {}
        cache = self._cache_path(endpoint, params)
        if cache.exists() and not refresh:
            return json.loads(cache.read_text(encoding="utf-8"))["data"]

        resp = self.session.get(
            f"{CFBD_BASE_URL}{endpoint}", params=params, timeout=60
        )
        with open(USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "endpoint": endpoint,
                "params": params,
                "status": resp.status_code,
            }) + "\n")
        resp.raise_for_status()
        data = resp.json()

        cache.write_text(json.dumps({
            "as_of": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "params": params,
            "data": data,
        }), encoding="utf-8")
        time.sleep(1)  # be polite to the free tier
        return data

    def calls_used(self) -> int:
        if not USAGE_LOG.exists():
            return 0
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        return sum(
            1 for line in USAGE_LOG.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["ts"].startswith(month)
        )
