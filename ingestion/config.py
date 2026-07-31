"""Central config: loads .env and defines warehouse paths."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL_FAST = os.getenv("CLAUDE_MODEL_FAST", "claude-sonnet-5")

PARQUET_DIR = PROJECT_ROOT / os.getenv("PARQUET_DIR", "./warehouse/parquet")
RAW_DIR = PROJECT_ROOT / os.getenv("RAW_DIR", "./warehouse/raw")

PBP_DIR = PARQUET_DIR / "pbp"
CFBD_PARQUET_DIR = PARQUET_DIR / "cfbd"
CFBD_RAW_DIR = RAW_DIR / "cfbd"

for _d in (PBP_DIR, CFBD_PARQUET_DIR, CFBD_RAW_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CFBD_BASE_URL = "https://api.collegefootballdata.com"

# fitted model coefficients, written by the backtests and read by picks/
COEFS_PATH = PROJECT_ROOT / "warehouse" / "model_coefs.json"


def save_coefs(section: str, values) -> None:
    import json
    from datetime import date
    data = {}
    if COEFS_PATH.exists():
        data = json.loads(COEFS_PATH.read_text(encoding="utf-8"))
    data[section] = values
    data.setdefault("updated", {})[section] = date.today().isoformat()
    COEFS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_coefs(section: str, default=None):
    import json
    if COEFS_PATH.exists():
        data = json.loads(COEFS_PATH.read_text(encoding="utf-8"))
        if section in data:
            return data[section]
    return default

# sportsdataverse-data release assets: seasons 2020+ (kept current).
# cfbfastR-data repo holds the older 2002-2021 files.
SDV_PBP_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_cfb_pbp/play_by_play_{year}.parquet"
)
CFBFASTR_PBP_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/"
    "pbp/parquet/play_by_play_{year}.parquet"
)
