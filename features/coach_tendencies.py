"""Per-coach play-calling tendency profiles from play-by-play (2021+).

One row per (playcaller, season). Playcaller = OC from the coach DB where
known, else HC — keyed to the person, not the team, so profiles follow
coaches across jobs.

Metrics:
  proe            pass rate over expectation (league expectation by season,
                  down, distance bucket, score bucket, period)
  pass_rate_neutral, epa_pass, epa_rush, explosive_pass_rate
  pace_neutral    seconds per offensive play on neutral-state drives
  turtle_delta    pass rate leading 9+ in 2H minus neutral pass rate
  chase_delta     pass rate trailing 9+ in 2H minus neutral pass rate
  fourth_go_rate  go rate on 4th-and-<=2 in the 40-70 yds-to-endzone band
  second_half_adj mean per-game (2H off EPA/play - 1H off EPA/play)
  script_pass_delta first-15-plays pass rate minus neutral pass rate

  python -m features.coach_tendencies
"""
import numpy as np
import pandas as pd

from ingestion.config import PARQUET_DIR, PBP_DIR

SEASONS = [2021, 2022, 2023, 2024, 2025]

COLS = [
    "game_id", "week", "period", "down", "distance", "pass", "rush",
    "completion", "EPA", "wp_before", "pos_team", "pos_team_score",
    "def_pos_team_score", "yds_receiving", "yds_punted", "fg_attempt",
    "start.TimeSecsRem", "end.TimeSecsRem", "drive.id", "start.yardsToEndzone",
    "start.down", "start.distance",
]


def load_season(season: int) -> pd.DataFrame:
    df = pd.read_parquet(PBP_DIR / f"play_by_play_{season}.parquet", columns=COLS)
    df = df.rename(columns={
        "start.TimeSecsRem": "secs_start", "end.TimeSecsRem": "secs_end",
        "drive.id": "drive_id", "start.yardsToEndzone": "yds_to_ez"})
    # older files: cleaned down/distance mostly null -> fall back to raw
    df["down"] = df["down"].fillna(df["start.down"])
    df["distance"] = df["distance"].fillna(df["start.distance"])
    # older season files store bools as object and ids as float
    for c in ("pass", "rush", "completion", "fg_attempt"):
        df[c] = df[c].fillna(False).astype(bool)
    df = df[df.pos_team.notna()]
    df["pos_team"] = df.pos_team.astype(int)
    df["season"] = season
    df["diff"] = df.pos_team_score - df.def_pos_team_score
    df["scrimmage"] = df["pass"] | df["rush"]
    df["neutral"] = (
        (df.period <= 3) & df["diff"].between(-8, 8)
        & df.wp_before.between(0.05, 0.95) & df.down.between(1, 3))
    df["dist_b"] = pd.cut(df.distance, [-1, 3, 6, 99], labels=["s", "m", "l"])
    df["diff_b"] = pd.cut(df["diff"], [-99, -9, -1, 8, 99],
                          labels=["down9", "down", "up", "up9"])
    return df


def playcaller_map() -> pd.DataFrame:
    cdb = pd.read_parquet(PARQUET_DIR / "coach_db.parquet")
    cdb = cdb[cdb.team_id.notna()].copy()
    cdb["playcaller"] = cdb.oc.fillna(cdb.hc)
    return cdb[["team_id", "season", "playcaller", "hc"]].astype(
        {"team_id": int})


def drive_pace(df: pd.DataFrame) -> pd.Series:
    """Seconds per scrimmage play on neutral drives, per (pos_team)."""
    d = df[df.scrimmage & df.neutral & df.drive_id.notna()]
    g = d.groupby(["pos_team", "game_id", "drive_id"]).agg(
        secs=("secs_start", "max"), secs_end=("secs_end", "min"),
        plays=("scrimmage", "size"))
    g["elapsed"] = g.secs - g.secs_end
    g = g[(g.elapsed > 0) & (g.elapsed < 600) & (g.plays >= 3)]
    pace = g.groupby("pos_team").apply(
        lambda x: x.elapsed.sum() / x.plays.sum())
    return pace.clip(10, 45)


def season_profiles(season: int) -> pd.DataFrame:
    df = load_season(season)
    sc = df[df.scrimmage].copy()

    # league expected pass rate for PROE
    exp = sc[sc.down.notna()].groupby(
        ["down", "dist_b", "diff_b", "period"], observed=True
    )["pass"].mean().rename("exp_pass")
    sc = sc.join(exp, on=["down", "dist_b", "diff_b", "period"])
    sc["proe_play"] = sc["pass"].astype(float) - sc.exp_pass

    neu = sc[sc.neutral]
    second_half = sc[sc.period.isin([3, 4])]
    first_half = sc[sc.period.isin([1, 2])]

    prof = pd.DataFrame({
        "n_plays": sc.groupby("pos_team").size(),
        "proe": sc.groupby("pos_team").proe_play.mean(),
        "pass_rate_neutral": neu.groupby("pos_team")["pass"].mean(),
        "epa_pass": neu[neu["pass"]].groupby("pos_team").EPA.mean(),
        "epa_rush": neu[neu.rush].groupby("pos_team").EPA.mean(),
        "explosive_pass_rate": sc[sc["pass"]].groupby("pos_team").apply(
            lambda x: ((x.completion == 1) & (x.yds_receiving >= 20)).mean()),
        "pace_neutral": drive_pace(df),
        "turtle_delta": second_half[second_half.diff_b == "up9"]
            .groupby("pos_team")["pass"].mean(),
        "chase_delta": second_half[second_half.diff_b == "down9"]
            .groupby("pos_team")["pass"].mean(),
        "second_half_adj": (
            second_half.groupby(["pos_team", "game_id"]).EPA.mean()
            - first_half.groupby(["pos_team", "game_id"]).EPA.mean()
        ).groupby("pos_team").mean(),
    })
    prof["turtle_delta"] -= prof.pass_rate_neutral
    prof["chase_delta"] -= prof.pass_rate_neutral

    # 4th-down aggressiveness in the no-man's-land band
    f4 = df[(df.down == 4) & (df.distance <= 2)
            & df.yds_to_ez.between(40, 70)].copy()
    f4["go"] = f4.scrimmage
    f4["attempt"] = f4.scrimmage | f4.yds_punted.notna() | f4.fg_attempt
    f4 = f4[f4.attempt]
    prof["fourth_go_rate"] = f4.groupby("pos_team").go.mean()

    # scripted opening: first 15 scrimmage plays per game
    sc["play_no"] = sc.groupby(["pos_team", "game_id"]).cumcount()
    script = sc[sc.play_no < 15]
    prof["script_pass_delta"] = (
        script.groupby("pos_team")["pass"].mean() - prof.pass_rate_neutral)

    prof["season"] = season
    return prof.reset_index().rename(columns={"pos_team": "team_id"})


def build() -> pd.DataFrame:
    prof = pd.concat([season_profiles(s) for s in SEASONS], ignore_index=True)
    pc = playcaller_map()
    out = prof.merge(pc, on=["team_id", "season"], how="inner")
    out = out[out.n_plays >= 300]
    dest = PARQUET_DIR / "coach_tendencies.parquet"
    out.to_parquet(dest, index=False)
    print(f"coach_tendencies: {len(out)} playcaller-seasons -> {dest}")
    return out


if __name__ == "__main__":
    build()
