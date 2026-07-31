"""Bulk play-by-play download from sportsdataverse GitHub (free, no API key).

Seasons 2020+ come from sportsdataverse-data release assets (kept current);
older seasons fall back to the cfbfastR-data repo files.
"""
import sys

import pyarrow.parquet as pq
import requests

from .config import CFBFASTR_PBP_URL, PBP_DIR, SDV_PBP_URL

CHUNK = 1 << 20  # 1 MiB


def pbp_url(year: int) -> str:
    return (SDV_PBP_URL if year >= 2020 else CFBFASTR_PBP_URL).format(year=year)


def download_season(year: int, force: bool = False) -> dict:
    dest = PBP_DIR / f"play_by_play_{year}.parquet"
    if dest.exists() and not force:
        meta = pq.read_metadata(dest)
        return {"year": year, "status": "cached", "rows": meta.num_rows}

    url = pbp_url(year)
    tmp = dest.with_suffix(".parquet.part")
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
    tmp.replace(dest)

    meta = pq.read_metadata(dest)  # validates the file parses as parquet
    return {
        "year": year,
        "status": "downloaded",
        "rows": meta.num_rows,
        "cols": meta.num_columns,
        "mb": round(dest.stat().st_size / 1e6, 1),
    }


def main(years: list[int]) -> None:
    for year in years:
        info = download_season(year)
        print(info)


if __name__ == "__main__":
    main([int(y) for y in sys.argv[1:]] or [2023, 2024, 2025])
