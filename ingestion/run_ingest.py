"""Phase 0 ingestion orchestrator.

  python -m ingestion.run_ingest --pbp 2023 2024 2025
  python -m ingestion.run_ingest --cfbd
  python -m ingestion.run_ingest --all
"""
import argparse

import pandas as pd

from . import bulk_pbp
from .cfbd_client import CFBDClient
from .config import CFBD_PARQUET_DIR

# (name, endpoint, params) — kept small: free tier is 1,000 calls/month.
CFBD_PULLS = [
    ("teams_fbs",            "/teams/fbs",          {}),
    ("venues",               "/venues",             {}),
    ("coaches",              "/coaches",            {"minYear": 2013}),
    ("games_2022",           "/games",              {"year": 2022}),
    ("games_2023",           "/games",              {"year": 2023}),
    ("games_2024",           "/games",              {"year": 2024}),
    ("games_2025",           "/games",              {"year": 2025}),
    ("lines_2022",           "/lines",              {"year": 2022}),
    ("lines_2023",           "/lines",              {"year": 2023}),
    ("lines_2024",           "/lines",              {"year": 2024}),
    ("lines_2025",           "/lines",              {"year": 2025}),
    ("recruiting_teams",     "/recruiting/teams",   {}),
    ("portal_2023",          "/player/portal",      {"year": 2023}),
    ("portal_2024",          "/player/portal",      {"year": 2024}),
    ("portal_2025",          "/player/portal",      {"year": 2025}),
    ("portal_2026",          "/player/portal",      {"year": 2026}),
    ("returning_2022",       "/player/returning",   {"year": 2022}),
    ("returning_2023",       "/player/returning",   {"year": 2023}),
    ("returning_2024",       "/player/returning",   {"year": 2024}),
    ("returning_2025",       "/player/returning",   {"year": 2025}),
]


def flatten_lines(rows: list) -> pd.DataFrame:
    """One row per (game, book) from the nested /lines response."""
    out = []
    for g in rows:
        base = {k: g.get(k) for k in (
            "id", "season", "week", "seasonType", "startDate",
            "homeTeam", "awayTeam", "homeScore", "awayScore")}
        for line in g.get("lines", []):
            out.append(base | {
                "provider": line.get("provider"),
                "spread": line.get("spread"),
                "spreadOpen": line.get("spreadOpen"),
                "overUnder": line.get("overUnder"),
                "overUnderOpen": line.get("overUnderOpen"),
                "homeMoneyline": line.get("homeMoneyline"),
                "awayMoneyline": line.get("awayMoneyline"),
            })
    return pd.DataFrame(out)


def ingest_cfbd() -> None:
    client = CFBDClient()
    for name, endpoint, params in CFBD_PULLS:
        data = client.get(endpoint, params)
        df = flatten_lines(data) if endpoint == "/lines" else pd.json_normalize(data)
        dest = CFBD_PARQUET_DIR / f"{name}.parquet"
        df.to_parquet(dest, index=False)
        print(f"{name:20s} {len(df):7,d} rows -> {dest.name}")
    print(f"\nCFBD calls used this month: {client.calls_used()} / 1000")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbp", nargs="*", type=int, help="seasons to bulk-download")
    ap.add_argument("--cfbd", action="store_true", help="pull CFBD core tables")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all or args.pbp is not None:
        bulk_pbp.main(args.pbp or [2023, 2024, 2025])
    if args.all or args.cfbd:
        ingest_cfbd()


if __name__ == "__main__":
    main()
