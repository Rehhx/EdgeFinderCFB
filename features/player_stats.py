"""Player game logs built from play-by-play (2021+).

One row per (season, week, game_id, team_id, player): rush att/yds,
targets/receptions/receiving yds, pass att/comp/yds. Full-game stats
(no garbage-time filter — props settle on official stat lines).

  python -m features.player_stats
"""
import pandas as pd

from ingestion.config import PARQUET_DIR, PBP_DIR

# every season with a downloaded play-by-play file (auto-includes 2026
# once the in-season weekly `bulk_pbp 2026` refresh starts)
SEASONS = sorted(
    int(p.stem.split("_")[-1]) for p in PBP_DIR.glob("play_by_play_*.parquet"))

COLS = [
    "game_id", "season", "week", "pos_team", "pass", "rush", "sack",
    "completion", "yds_rushed", "yds_receiving", "statYardage",
    "rusher_player_name", "receiver_player_name", "passer_player_name",
]


def load_season(season: int) -> pd.DataFrame:
    df = pd.read_parquet(PBP_DIR / f"play_by_play_{season}.parquet", columns=COLS)
    for c in ("pass", "rush", "sack", "completion"):
        df[c] = df[c].fillna(False).astype(bool)
    df = df[df.pos_team.notna()]
    df["pos_team"] = df.pos_team.astype(int)
    df["season"] = season
    # 2025 file: yds_rushed/yds_receiving ~43% null; statYardage is complete
    # and agrees 99% where both exist -> coalesce within the play-type mask
    df.loc[df.rush, "yds_rushed"] = df.loc[df.rush, "yds_rushed"].fillna(
        df.loc[df.rush, "statYardage"])
    caught = df["pass"] & ~df.sack & df.completion
    df.loc[caught, "yds_receiving"] = df.loc[caught, "yds_receiving"].fillna(
        df.loc[caught, "statYardage"])
    return df


def season_logs(season: int) -> pd.DataFrame:
    df = load_season(season)
    key = ["season", "week", "game_id", "pos_team"]

    rush = df[df.rush & df.rusher_player_name.notna()]
    rush_g = rush.groupby(key + ["rusher_player_name"]).agg(
        rush_att=("rush", "size"), rush_yds=("yds_rushed", "sum")
    ).reset_index().rename(columns={"rusher_player_name": "player"})

    tgt = df[df["pass"] & ~df.sack & df.receiver_player_name.notna()]
    tgt_g = tgt.groupby(key + ["receiver_player_name"]).agg(
        targets=("pass", "size"), receptions=("completion", "sum"),
        rec_yds=("yds_receiving", lambda y: y.fillna(0).sum()),
    ).reset_index().rename(columns={"receiver_player_name": "player"})

    pas = df[df["pass"] & ~df.sack & df.passer_player_name.notna()]
    pas_g = pas.groupby(key + ["passer_player_name"]).agg(
        pass_att=("pass", "size"), pass_comp=("completion", "sum"),
        pass_yds=("yds_receiving", lambda y: y.fillna(0).sum()),
    ).reset_index().rename(columns={"passer_player_name": "player"})

    out = rush_g.merge(tgt_g, on=key + ["player"], how="outer") \
                .merge(pas_g, on=key + ["player"], how="outer")
    stat_cols = ["rush_att", "rush_yds", "targets", "receptions", "rec_yds",
                 "pass_att", "pass_comp", "pass_yds"]
    out[stat_cols] = out[stat_cols].fillna(0)
    return out.rename(columns={"pos_team": "team_id"})


def build() -> pd.DataFrame:
    logs = pd.concat([season_logs(s) for s in SEASONS], ignore_index=True)
    dest = PARQUET_DIR / "player_game_logs.parquet"
    logs.to_parquet(dest, index=False)
    print(f"player_game_logs: {len(logs):,} player-games -> {dest}")
    return logs


if __name__ == "__main__":
    build()
