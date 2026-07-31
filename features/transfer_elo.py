"""Transfer translation ("transfer Elo"): how production travels across tiers.

For every portal player 2022-2025 with real usage at the origin school,
measures what happened the next season at the destination:
  share_ratio : usage share at new school / share at old school
  eff_ratio   : efficiency (ypc / ypt / ypa) new / old
bucketed by tier move (G5->P4, P4->G5, P4->P4, G5->G5) and role.
Fitted multipliers are saved to model_coefs.json and used by
models/props.player_priors instead of the old flat 0.8 discount.

  python -m features.transfer_elo
"""
import numpy as np
import pandas as pd

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR, save_coefs

P4 = {"ACC", "Big Ten", "Big 12", "SEC"}
MIN_SEASONS = {"rb": 4, "wr": 3, "qb": 8}  # min per-game usage at origin


def tier_map() -> dict[int, str]:
    t = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    return {r.id: ("P4" if r.conference in P4
                   or r.school == "Notre Dame" else "G5")
            for r in t.itertuples()}


def season_usage(logs: pd.DataFrame) -> pd.DataFrame:
    key = ["season", "week", "game_id", "team_id"]
    team = logs.groupby(key, as_index=False).agg(
        t_rush=("rush_att", "sum"), t_pass=("pass_att", "sum"))
    d = logs.merge(team, on=key)
    agg = d.groupby(["season", "team_id", "player"]).agg(
        games=("game_id", "nunique"),
        rush_att=("rush_att", "sum"), rush_yds=("rush_yds", "sum"),
        targets=("targets", "sum"), rec_yds=("rec_yds", "sum"),
        pass_att=("pass_att", "sum"), pass_yds=("pass_yds", "sum"),
        t_rush=("t_rush", "sum"), t_pass=("t_pass", "sum")).reset_index()
    agg["rush_share"] = agg.rush_att / agg.t_rush.clip(lower=1)
    agg["tgt_share"] = agg.targets / agg.t_pass.clip(lower=1)
    agg["qb_share"] = agg.pass_att / agg.t_pass.clip(lower=1)
    agg["ypc"] = agg.rush_yds / agg.rush_att.replace(0, np.nan)
    agg["ypt"] = agg.rec_yds / agg.targets.replace(0, np.nan)
    agg["ypa"] = agg.pass_yds / agg.pass_att.replace(0, np.nan)
    return agg


def build() -> pd.DataFrame:
    logs = pd.read_parquet(PARQUET_DIR / "player_game_logs.parquet")
    usage = season_usage(logs)
    tiers = tier_map()
    teams = pd.read_parquet(CFBD_PARQUET_DIR / "teams_fbs.parquet")
    school_to_id = dict(zip(teams.school, teams.id))

    rows = []
    for season in range(2022, 2026):
        p = pd.read_parquet(CFBD_PARQUET_DIR / f"portal_{season}.parquet")
        p = p.dropna(subset=["origin", "destination"])
        p["player"] = p.firstName.str.strip() + " " + p.lastName.str.strip()
        p["from_id"] = p.origin.map(school_to_id)
        p["to_id"] = p.destination.map(school_to_id)
        p = p.dropna(subset=["from_id", "to_id"]).astype(
            {"from_id": int, "to_id": int})

        old = usage[usage.season == season - 1].rename(
            columns={"team_id": "from_id"})
        new = usage[usage.season == season].rename(
            columns={"team_id": "to_id"})
        m = p[["player", "from_id", "to_id"]] \
            .merge(old, on=["player", "from_id"]) \
            .merge(new, on=["player", "to_id"], suffixes=("_old", "_new"))
        m = m[m.games_new >= 4]

        for r in m.itertuples():
            move = f"{tiers.get(r.from_id, 'G5')}->{tiers.get(r.to_id, 'G5')}"
            if r.rush_att_old / r.games_old >= MIN_SEASONS["rb"]:
                rows.append({"season": season, "move": move, "role": "rb",
                             "player": r.player,
                             "share_ratio": r.rush_share_new / max(r.rush_share_old, 1e-3),
                             "eff_ratio": r.ypc_new / r.ypc_old
                             if r.ypc_old and r.ypc_old > 1 else np.nan})
            if r.targets_old / r.games_old >= MIN_SEASONS["wr"]:
                rows.append({"season": season, "move": move, "role": "wr",
                             "player": r.player,
                             "share_ratio": r.tgt_share_new / max(r.tgt_share_old, 1e-3),
                             "eff_ratio": r.ypt_new / r.ypt_old
                             if r.ypt_old and r.ypt_old > 1 else np.nan})
            if r.pass_att_old / r.games_old >= MIN_SEASONS["qb"]:
                rows.append({"season": season, "move": move, "role": "qb",
                             "player": r.player,
                             "share_ratio": r.qb_share_new / max(r.qb_share_old, 1e-3),
                             "eff_ratio": r.ypa_new / r.ypa_old
                             if r.ypa_old and r.ypa_old > 1 else np.nan})

    df = pd.DataFrame(rows)
    df["share_ratio"] = df.share_ratio.clip(0, 3)
    df["eff_ratio"] = df.eff_ratio.clip(0.2, 2.5)
    dest = PARQUET_DIR / "transfer_translation.parquet"
    df.to_parquet(dest, index=False)

    print(f"matched transfers with usage both seasons: {len(df)} role-rows\n")
    tab = df.groupby(["move", "role"]).agg(
        n=("player", "size"),
        share=("share_ratio", "median"),
        eff=("eff_ratio", "median")).round(2)
    print(tab.to_string())

    # save per-move share multipliers (median across roles, shrunk to 0.8)
    mult = df.groupby("move").agg(n=("player", "size"),
                                  share=("share_ratio", "median"),
                                  eff=("eff_ratio", "median"))
    k = 25  # shrink small buckets toward the old flat prior
    out = {}
    for mv, r in mult.iterrows():
        out[mv] = {
            "share": round((r.share * r.n + 0.8 * k) / (r.n + k), 3),
            "eff": round((r.eff * r.n + 1.0 * k) / (r.n + k), 3),
            "n": int(r.n),
        }
    save_coefs("transfer_translation", out)
    print("\nsaved multipliers:", out)

    fit_star_bump(usage, tiers, out)
    return df


def fit_star_bump(usage: pd.DataFrame, tiers: dict, mults: dict) -> None:
    """How much extra next-season share does a STAR back get, beyond what
    his raw prior share predicts? Fit on all season-to-season lead-back
    pairs (returning or transferred), with tier multipliers applied first
    so the bump measures the residual star effect."""
    u = usage[usage.rush_att / usage.games >= 4].copy()
    u["rush_ypg"] = u.rush_yds / u.games
    thresh = float(u.rush_ypg.quantile(0.75))

    nxt = usage.copy()
    nxt["season"] -= 1
    pairs = u.merge(
        nxt[["season", "player", "team_id", "rush_share", "games"]],
        on=["season", "player"], suffixes=("", "_new"))
    pairs = pairs[pairs.games_new >= 4]
    move = (pairs.team_id.map(tiers).fillna("G5") + "->"
            + pairs.team_id_new.map(tiers).fillna("G5"))
    same_team = pairs.team_id == pairs.team_id_new
    mult = np.where(same_team, 1.0,
                    move.map(lambda mv: mults.get(mv, {}).get("share", 0.8)))
    x_base = pairs.rush_share * mult
    star = (pairs.rush_ypg >= thresh).astype(float)
    X = np.column_stack([x_base, star, np.ones(len(pairs))])
    coef, *_ = np.linalg.lstsq(X, pairs.rush_share_new, rcond=None)
    bump = float(coef[1])
    print(f"\nSTAR-RB BUMP: {len(pairs)} lead-back pairs, star = "
          f">={thresh:.0f} rush yds/gm (75th pct). Fitted extra share for "
          f"stars: {bump:+.3f} (base slope {coef[0]:.2f})")
    save_coefs("star_rb", {"bump": round(bump, 4),
                           "ypg_thresh": round(thresh, 1)})


if __name__ == "__main__":
    build()
