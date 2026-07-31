"""Availability layer: depth charts + injuries + news -> projection adjustments.

availability_table() merges the latest Ourlads depth chart snapshot with
ESPN injuries and Claude news extractions into one per-player status table.

apply_availability(proj) adjusts prop projections:
  - players with status 'out' are zeroed
  - the next man up at the same position inherits 60% of the starter's
    projected volume-driven stats (documented heuristic until we can fit
    it on historical injury replacements)
"""
import pandas as pd

from ingestion.config import PARQUET_DIR

BACKUP_INHERIT = 0.60
OUT_STATUSES = {"out", "doubtful", "suspended", "injured reserve"}
PROJ_COLS = ["proj_rush_yds", "proj_targets", "proj_receptions",
             "proj_rec_yds", "proj_pass_yds"]


def latest_depth() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "depth_charts.parquet")
    return d[d.as_of == d.as_of.max()]


def availability_table() -> pd.DataFrame:
    depth = latest_depth()[["team_id", "team_name", "pos", "rank",
                            "player", "as_of"]].copy()
    depth["status"] = "active"

    try:
        inj = pd.read_parquet(PARQUET_DIR / "espn_injuries.parquet")
        inj = inj[inj.as_of == inj.as_of.max()]
        out_players = set(
            inj[inj.status.str.lower().isin(OUT_STATUSES)].player.dropna())
        q_players = set(
            inj[inj.status.str.lower() == "questionable"].player.dropna())
    except FileNotFoundError:
        out_players, q_players = set(), set()

    try:
        news = pd.read_parquet(PARQUET_DIR / "news_extractions.parquet")
        out_players |= set(news[news.status == "out"].player.dropna())
    except FileNotFoundError:
        pass

    depth.loc[depth.player.isin(q_players), "status"] = "questionable"
    depth.loc[depth.player.isin(out_players), "status"] = "out"
    return depth


def apply_availability(proj: pd.DataFrame,
                       avail: pd.DataFrame | None = None) -> pd.DataFrame:
    """Zero out 'out' players and promote their direct backups."""
    avail = availability_table() if avail is None else avail
    proj = proj.copy()
    proj["avail_status"] = proj.player.map(
        avail.set_index("player").status.to_dict()).fillna("unknown")

    outs = avail[avail.status == "out"]
    for o in outs.itertuples():
        mask_out = (proj.player == o.player)
        if not mask_out.any():
            continue
        starter_proj = proj.loc[mask_out, PROJ_COLS].sum()
        backup = avail[(avail.team_id == o.team_id) & (avail.pos == o.pos)
                       & (avail["rank"] == o.rank + 1)]
        proj.loc[mask_out, PROJ_COLS] = 0.0
        if len(backup):
            mask_b = proj.player == backup.iloc[0].player
            for c in PROJ_COLS:
                proj.loc[mask_b, c] = (
                    proj.loc[mask_b, c] + BACKUP_INHERIT * starter_proj[c])
    return proj


if __name__ == "__main__":
    a = availability_table()
    print(f"availability table: {len(a)} depth-chart slots, "
          f"{(a.status == 'out').sum()} out, "
          f"{(a.status == 'questionable').sum()} questionable")
    print(a.head(8).to_string(index=False))
