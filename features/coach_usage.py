"""Coach deployment profiles: how play-callers distribute touches.

Per playcaller-season, from player logs:
  rb1_share  : lead back's share of team rush attempts
  wr1_share  : top target's share of team pass attempts
  rush_hhi   : concentration of the RB room (Herfindahl index)
  n_rb_used  : backs with >=10% of a game's carries on average

Also answers: does a STAR running back change the playbook? Compares each
coach's neutral pass rate in seasons where his lead back was a top-quartile
talent vs his own other seasons (within-coach, so scheme identity cancels).

  python -m features.coach_usage
"""
import pandas as pd

from ingestion.config import PARQUET_DIR

SEASONS = [2021, 2022, 2023, 2024, 2025]


def team_season_usage() -> pd.DataFrame:
    logs = pd.read_parquet(PARQUET_DIR / "player_game_logs.parquet")
    key = ["season", "week", "game_id", "team_id"]
    team = logs.groupby(key, as_index=False).agg(
        t_rush=("rush_att", "sum"), t_pass=("pass_att", "sum"))
    d = logs.merge(team, on=key)
    season_tot = d.groupby(["season", "team_id"]).agg(
        t_rush=("rush_att", "sum"), t_pass=("pass_att", "sum"))

    pl = d.groupby(["season", "team_id", "player"]).agg(
        rush=("rush_att", "sum"), tgt=("targets", "sum"),
        rush_yds=("rush_yds", "sum")).join(season_tot,
                                           on=["season", "team_id"])
    pl["rush_share"] = pl.rush / pl.t_rush.clip(lower=1)
    pl["tgt_share"] = pl.tgt / pl.t_pass.clip(lower=1)

    rows = []
    for (season, tid), g in pl.groupby(level=["season", "team_id"]):
        backs = g[g.rush_share >= 0.05].sort_values("rush_share",
                                                    ascending=False)
        rb1 = backs.iloc[0] if len(backs) else None
        rows.append({
            "season": season, "team_id": tid,
            "rb1_share": rb1.rush_share if rb1 is not None else None,
            "rb1_yds": rb1.rush_yds if rb1 is not None else 0,
            "wr1_share": g.tgt_share.max(),
            "rush_hhi": (backs.rush_share ** 2).sum() if len(backs) else None,
            "n_rb_used": (g.rush_share >= 0.10).sum(),
        })
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    usage = team_season_usage()
    cdb = pd.read_parquet(PARQUET_DIR / "coach_db.parquet")
    cdb = cdb[cdb.team_id.notna()].astype({"team_id": int})
    cdb["playcaller"] = cdb.oc.fillna(cdb.hc)
    prof = usage.merge(cdb[["season", "team_id", "playcaller"]],
                       on=["season", "team_id"])
    tend = pd.read_parquet(PARQUET_DIR / "coach_tendencies.parquet")
    prof = prof.merge(
        tend[["season", "team_id", "pass_rate_neutral", "proe"]],
        on=["season", "team_id"], how="left")
    dest = PARQUET_DIR / "coach_usage.parquet"
    prof.to_parquet(dest, index=False)

    # stability: is RB1 share a coach trait?
    p = prof.sort_values(["playcaller", "season"])
    p["next"] = p.season + 1
    pairs = p.merge(p, left_on=["playcaller", "next"],
                    right_on=["playcaller", "season"], suffixes=("", "_y"))
    print(f"coach_usage: {len(prof)} playcaller-seasons -> {dest.name}")
    for m in ("rb1_share", "wr1_share", "rush_hhi", "n_rb_used"):
        print(f"  y/y stability {m:10s}: r = {pairs[m].corr(pairs[f'{m}_y']):.2f} "
              f"({len(pairs)} pairs)")

    # star-RB playbook shift, within-coach
    prof["star_rb"] = prof.rb1_yds >= prof.rb1_yds.quantile(0.75)
    multi = prof.groupby("playcaller").filter(
        lambda g: g.star_rb.nunique() == 2 and len(g) >= 2)
    if len(multi):
        delta = multi.groupby(["playcaller", "star_rb"]) \
            .pass_rate_neutral.mean().unstack()
        delta = (delta[True] - delta[False]).dropna()
        print(f"\nSTAR-RB EFFECT (within-coach, {len(delta)} coaches): "
              f"neutral pass rate with a star back is "
              f"{delta.mean():+.1%} (median {delta.median():+.1%}) vs the "
              f"same coach's other seasons")
        share_d = multi.groupby(["playcaller", "star_rb"]) \
            .rb1_share.mean().unstack()
        share_d = (share_d[True] - share_d[False]).dropna()
        print(f"  and the lead back's carry share rises {share_d.mean():+.1%}")
    return prof


if __name__ == "__main__":
    build()
