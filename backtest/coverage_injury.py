"""Backtest the coverage-injury edge: when a team's top coverage DB sits out,
do opposing pass catchers beat expectation — and beat the book?

Injury proxy: CFBD per-game defensive box scores (1 call/week). A team's
season-identified top-PD defensive back who is ABSENT from a game's
defensive box (while present in >=4 other games) = did-not-play that week.

Two tests:
  1. On-field effect: opposing offense pass EPA vs its own season baseline,
     split by whether the defense was missing its top DB.
  2. Betting: opposing pass-catcher props (rec_yds/receptions/pass_yds) —
     over-hit rate and ROI at posted prices, DB-out vs DB-in.

  python -m backtest.coverage_injury
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import consensus_lines, payout
from features.defense_players import pull_defensive_stats
from ingestion.cfbd_client import CFBDClient
from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR, PBP_DIR

SEASONS = [2024, 2025]
WEEKS = range(1, 16)
DB_POS = {"CB", "S", "DB", "SAF", "FS", "SS"}
MIN_GAMES = 4  # a "regular" defender must appear this often to count


def team_id_map() -> dict[str, int]:
    t = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    return dict(zip(t.school, t.id))


def pull_presence() -> pd.DataFrame:
    """Long table of (season, week, game_id, team_id, playerId) for every
    player with a defensive box-score line."""
    c = CFBDClient()
    name_to_id = team_id_map()
    rows = []
    for season in SEASONS:
        for week in WEEKS:
            data = c.get("/games/players", {"year": season, "week": week})
            for g in data:
                for t in g.get("teams", []):
                    tid = name_to_id.get(t["team"])
                    if tid is None:
                        continue
                    seen = set()
                    for cat in t.get("categories", []):
                        if cat["name"] != "defensive":
                            continue
                        for typ in cat["types"]:
                            for a in typ["athletes"]:
                                seen.add((a["id"], a["name"]))
                    for pid, name in seen:
                        rows.append({"season": season, "week": week,
                                     "game_id": int(g["id"]), "team_id": tid,
                                     "playerId": pid, "player": name})
    print(f"defensive presence rows: {len(rows):,} (calls used {c.calls_used()})")
    return pd.DataFrame(rows)


def key_db_absence(pres: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team_id): was the team's top coverage DB absent?"""
    stats = pull_defensive_stats()
    stats["cover"] = stats.get("PD", 0) + 2 * stats.get("INT", 0)
    dbs = stats[stats.position.isin(DB_POS)].copy()
    name_to_id = team_id_map()
    dbs["team_id"] = dbs.team.map(name_to_id)
    dbs = dbs.dropna(subset=["team_id"]).astype({"team_id": int})
    # top coverage DB per (season, team) by playerId
    top = dbs.sort_values("cover", ascending=False).drop_duplicates(
        ["season", "team_id"])[["season", "team_id", "playerId", "player", "cover"]]
    top = top.rename(columns={"playerId": "key_pid", "player": "key_db"})

    # weeks the team actually played (appeared on defense)
    team_weeks = pres[["season", "week", "team_id"]].drop_duplicates()
    # weeks the key DB appeared
    key_present = pres.merge(top[["season", "team_id", "key_pid"]],
                             on=["season", "team_id"])
    key_present = key_present[key_present.playerId == key_present.key_pid][
        ["season", "team_id", "week"]].drop_duplicates()
    key_present["played"] = True

    tw = team_weeks.merge(top, on=["season", "team_id"], how="inner").merge(
        key_present, on=["season", "team_id", "week"], how="left")
    tw["played"] = tw.played.fillna(False).astype(bool)
    # only count teams whose key DB is a real regular (played >= MIN_GAMES)
    games_played = tw.groupby(["season", "team_id"]).played.sum()
    regular = games_played[games_played >= MIN_GAMES].index
    tw = tw.set_index(["season", "team_id"]).loc[
        tw.set_index(["season", "team_id"]).index.isin(regular)].reset_index()
    tw["key_db_out"] = (~tw.played).astype(bool)
    return tw[["season", "week", "team_id", "key_db_out", "key_db", "cover"]]


def offense_pass_epa() -> pd.DataFrame:
    """Per (season, game_id, off team) pass EPA/play, from PBP."""
    out = []
    for s in SEASONS:
        df = pd.read_parquet(PBP_DIR / f"play_by_play_{s}.parquet",
                             columns=["game_id", "week", "pos_team", "pass",
                                      "EPA", "wp_before"])
        df["pass"] = df["pass"].fillna(False).astype(bool)
        df = df[df["pass"] & df.EPA.notna() & df.pos_team.notna()
                & df.wp_before.between(0.05, 0.95)]
        g = df.groupby(["game_id", "pos_team"], as_index=False).agg(
            pass_epa=("EPA", "mean"), week=("week", "first"))
        g["season"] = s
        g["pos_team"] = g.pos_team.astype(int)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def effect_test(absence: pd.DataFrame) -> None:
    ope = offense_pass_epa()
    ope["off_season_mean"] = ope.groupby(["season", "pos_team"]).pass_epa.transform("mean")
    ope["delta"] = ope.pass_epa - ope.off_season_mean

    # need offense's opponent per game to join the DEFENSE's key_db_out
    games = pd.concat([
        pd.read_parquet(CFBD_PARQUET_DIR / f"games_{s}.parquet")[
            ["id", "homeId", "awayId"]].assign(season=s) for s in SEASONS])
    ope = ope.merge(games, left_on=["game_id", "season"],
                    right_on=["id", "season"], how="left")
    ope["def_id"] = np.where(ope.pos_team == ope.homeId, ope.awayId, ope.homeId)
    m = ope.merge(absence, left_on=["season", "week", "def_id"],
                  right_on=["season", "week", "team_id"], how="inner")
    print("\n=== ON-FIELD EFFECT: opposing pass EPA vs offense's own season "
          "baseline ===")
    g = m.groupby("key_db_out").agg(n=("delta", "size"),
                                    mean_delta=("delta", "mean"))
    print(g.round(4).to_string())
    if {True, False} <= set(g.index):
        lift = g.loc[True, "mean_delta"] - g.loc[False, "mean_delta"]
        print(f"pass-EPA lift when opp top DB OUT: {lift:+.4f} EPA/play "
              f"({int(g.loc[True,'n'])} DB-out games)")


def betting_test(absence: pd.DataFrame) -> None:
    proj = pd.read_parquet(PARQUET_DIR / "prop_projections.parquet")
    pass_stats = ["rec_yds", "receptions", "pass_yds"]
    frames = []
    for stat in pass_stats:
        p = proj[["season", "week", "player", "opp_id", stat]].rename(
            columns={stat: "actual"})
        p["stat"] = stat
        frames.append(p)
    pl = pd.concat(frames).dropna(subset=["opp_id"])
    pl["opp_id"] = pl.opp_id.astype(int)

    lines = consensus_lines()
    m = lines.merge(pl, on=["season", "week", "stat", "player"], how="inner")
    m = m.merge(absence.rename(columns={"team_id": "opp_id"}),
                on=["season", "week", "opp_id"], how="left")
    m["key_db_out"] = m.key_db_out.fillna(False).astype(bool)

    m["over_won"] = np.where(m.actual == m.line, np.nan, m.actual > m.line)
    print("\n=== BETTING: opposing pass-catcher OVERS, DB-out vs DB-in ===")
    print(f"matched pass props: {len(m):,} "
          f"({int(m.key_db_out.sum())} vs a DB-depleted secondary)")
    for label, sub in [("opp top DB IN", m[~m.key_db_out]),
                       ("opp top DB OUT", m[m.key_db_out])]:
        s = sub[sub.over_won.notna()]
        if len(s) < 10:
            print(f"  {label}: n={len(s)} (too few)")
            continue
        over_rate = s.over_won.mean()
        # flat bet the OVER at best price
        roi = (s.over_won * s.pay_over_best - (1 - s.over_won)).mean()
        print(f"  {label:16s}: {len(s):5d} props, over-hit {over_rate:.1%}, "
              f"blind-over ROI {roi * 100:+.1f}%")


if __name__ == "__main__":
    pres = pull_presence()
    pres.to_parquet(PARQUET_DIR / "def_presence.parquet", index=False)
    absence = key_db_absence(pres)
    absence.to_parquet(PARQUET_DIR / "key_db_absence.parquet", index=False)
    n_out = int(absence.key_db_out.sum())
    print(f"\nkey-DB-out games: {n_out} of {len(absence)} team-games "
          f"({n_out / len(absence):.1%})")
    effect_test(absence)
    betting_test(absence)
