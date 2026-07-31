"""Coach database: HC per team-season from CFBD + OC/DC from Wikipedia scrape.

Output (one row per team-season): hc, new_hc, hc_tenure, mid_season_change,
oc, oc_change, dc, dc_change — keyed by school name and ESPN team_id.
"""
import pandas as pd

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR


def hc_table() -> pd.DataFrame:
    """Explode CFBD coaches into (school, season) -> head coach rows."""
    c = pd.read_parquet(CFBD_PARQUET_DIR / "coaches.parquet")
    rows = []
    for r in c.itertuples():
        for s in r.seasons:
            rows.append({
                "coach": f"{r.firstName} {r.lastName}",
                "school": s["school"], "season": s["year"],
                "games": s["games"],
            })
    df = pd.DataFrame(rows)
    # HC of record = most games that season; >1 coach with games -> turmoil flag
    df = df.sort_values("games", ascending=False)
    top = df.drop_duplicates(["school", "season"]).rename(columns={"coach": "hc"})
    n = df[df.games > 0].groupby(["school", "season"]).coach.nunique()
    top = top.merge(n.rename("n_hcs"), on=["school", "season"], how="left")
    top["mid_season_change"] = (top.n_hcs.fillna(1) > 1).astype(int)
    return top[["school", "season", "hc", "mid_season_change"]]


def build() -> pd.DataFrame:
    hc = hc_table().sort_values(["school", "season"])
    hc["prev_hc"] = hc.groupby("school").hc.shift(1)
    hc["new_hc"] = (hc.hc != hc.prev_hc).astype(int)
    hc.loc[hc.prev_hc.isna(), "new_hc"] = 0  # unknown history, assume incumbent

    # consecutive seasons with the same HC at the school
    tenure, run = [], {}
    for r in hc.itertuples():
        run[r.school] = run.get(r.school, 0) + 1 if not r.new_hc else 1
        tenure.append(run[r.school])
    hc["hc_tenure"] = tenure

    staff = pd.read_parquet(PARQUET_DIR / "wiki_staff.parquet")
    staff = staff.sort_values(["school", "season"])
    for role in ("oc", "dc"):
        staff[f"prev_{role}"] = staff.groupby("school")[role].shift(1)
        known = staff[role].notna() & staff[f"prev_{role}"].notna()
        staff[f"{role}_change"] = (
            (staff[role] != staff[f"prev_{role}"]) & known).astype(int)

    out = hc.merge(
        staff[["school", "season", "team_id", "oc", "dc",
               "oc_change", "dc_change"]],
        on=["school", "season"], how="outer")
    dest = PARQUET_DIR / "coach_db.parquet"
    out.to_parquet(dest, index=False)
    print(f"coach_db: {len(out)} team-seasons -> {dest}")
    print("seasons with OC data:", sorted(out[out.oc.notna()].season.unique()))
    return out


if __name__ == "__main__":
    build()
