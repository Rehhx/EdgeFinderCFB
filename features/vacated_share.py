"""Vacated share: how much of last season's usage is NOT coming back.

`models/props.py` uses the portal only for ARRIVALS — a transfer inherits his
origin-school usage. DEPARTURES were never modelled: the leaving player's prior
row is deleted and his share simply evaporates, leaving every returning
team-mate with a prior computed while that player was still there.

Measured effect on returning players' week 1-4 usage is a 38% monotone swing
(actual/prior runs 0.844 -> 1.169 across vacancy quartiles). The tell is that
ACTUAL share is nearly flat while the PRIOR falls steeply: the outcome is fine,
the prior is wrong. Backups on gutted rosters inherit the vacated work.

Pregame-safe because it keys off the CFBD ROSTER, not on who ends up playing:
    portal departures alone capture 17% of vacated targets (corr 0.188)
    the roster captures                97%                 (corr 0.969)

  python -m features.vacated_share
"""
import numpy as np
import pandas as pd

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

# vacated(S) needs season S-1 LOGS and season S ROSTER, so the usable span is
# bounded by both. Logs start 2022 and rosters start 2022 -> 2023 is the first
# computable season.
FIRST_SEASON = 2023
SHARES = {"tgt": "targets", "rush": "rush_att", "pass": "pass_att"}


def _roster(season: int, school_to_id: dict) -> pd.DataFrame | None:
    path = CFBD_PARQUET_DIR / f"roster_{season}.parquet"
    if not path.exists():
        return None
    r = pd.read_parquet(path)
    if len(r) < 1000:
        # An empty/short roster would mark EVERY prior contributor as departed
        # and push vacated share to ~1.0 for every team. Refuse rather than
        # silently poison the feature.
        raise RuntimeError(
            f"roster_{season} has only {len(r)} rows — not published yet. "
            "Vacated share cannot be computed for this season.")
    r["player"] = (r.firstName.astype(str).str.strip() + " "
                   + r.lastName.astype(str).str.strip())
    r["team_id"] = r.team.map(school_to_id)
    r = r.dropna(subset=["team_id"])
    return pd.DataFrame({"season": season,
                         "team_id": r.team_id.astype("int64"),
                         "player": r.player}).drop_duplicates()


def build() -> pd.DataFrame:
    """(season, team_id) -> vacated share of targets / carries / pass attempts."""
    logs = pd.read_parquet(PARQUET_DIR / "player_game_logs.parquet")
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    s2i = dict(zip(teams.school, teams.id))

    # A player's share must use the team's FULL-SEASON attempts. Using only the
    # games he appeared in inflates part-season players and made team shares
    # sum to 1.9 instead of 1.0 the first time this was measured.
    tot = logs.groupby(["season", "team_id"], as_index=False).agg(
        **{f"tot_{k}": (col, "sum") for k, col in SHARES.items()})
    pl = logs.groupby(["season", "team_id", "player"], as_index=False).agg(
        **{k: (col, "sum") for k, col in SHARES.items()})
    pl = pl.merge(tot, on=["season", "team_id"])
    for k in SHARES:
        pl[f"{k}_share"] = pl[k] / pl[f"tot_{k}"].clip(lower=1)
    pl = pl.astype({"season": "int64", "team_id": "int64"})

    seasons = [s for s in sorted(pl.season.unique()) if s >= FIRST_SEASON]
    rosters = []
    for s in seasons:
        r = _roster(s, s2i)
        if r is not None:
            rosters.append(r)
    if not rosters:
        raise FileNotFoundError("no roster_*.parquet — run ingestion/run_ingest")
    ros = pd.concat(rosters).assign(on_roster=1)

    prev = pl.copy()
    prev["season"] = prev.season + 1          # last season's usage, this season
    prev = prev[prev.season.isin(seasons)]
    prev = prev.merge(ros, on=["season", "team_id", "player"], how="left")
    gone = prev[prev.on_roster.isna()]
    vac = gone.groupby(["season", "team_id"], as_index=False).agg(
        **{f"vac_{k}": (f"{k}_share", "sum") for k in SHARES})
    # a team with no departures has no row; that is genuinely zero vacancy
    allt = pl[pl.season.isin(seasons)][["season", "team_id"]].drop_duplicates()
    vac = allt.merge(vac, on=["season", "team_id"], how="left").fillna(0.0)
    for k in SHARES:
        vac[f"vac_{k}"] = vac[f"vac_{k}"].clip(0.0, 1.0)
    return vac


def main() -> None:
    v = build()
    print(f"team-seasons: {len(v)}  ({v.season.min()}-{v.season.max()})")
    print(v.groupby("season")[[f"vac_{k}" for k in SHARES]].mean().round(3).to_string())
    dest = PARQUET_DIR / "vacated_share.parquet"
    v.to_parquet(dest, index=False)
    print(f"\nsaved: {dest}")


if __name__ == "__main__":
    main()
