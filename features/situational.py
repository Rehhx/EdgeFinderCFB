"""Situational context: rest, travel, altitude, time zones, dome.

Classic handicapping factors we had never derived, all from data already on
disk (CFBD games + venues):
  rest_days      days since that team's previous game (bye = 13+)
  travel_km      great-circle distance from the team's home venue to the site
  tz_shift       time-zone hours crossed (body-clock effect)
  elev_gain_m    altitude gain vs the team's home venue (Wyoming/Air Force)
  dome_change    playing indoors when normally outdoors, or vice versa

Exposed as home-minus-away differentials per game for modelling.

  python -m features.situational
"""
import numpy as np
import pandas as pd

from ingestion.config import CFBD_PARQUET_DIR, PARQUET_DIR

SEASONS = range(2018, 2027)
EARTH_KM = 6371.0


def _haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def _tz_offset(tz: str) -> float:
    """Rough UTC offset by US time zone name (enough for shift deltas)."""
    return {"America/New_York": -5, "America/Detroit": -5,
            "America/Indiana/Indianapolis": -5, "America/Kentucky/Louisville": -5,
            "America/Chicago": -6, "America/Denver": -7, "America/Boise": -7,
            "America/Phoenix": -7, "America/Los_Angeles": -8,
            "Pacific/Honolulu": -10, "America/Anchorage": -9}.get(tz, -6)


def build() -> pd.DataFrame:
    ven = pd.read_parquet(CFBD_PARQUET_DIR / "venues.parquet")
    ven = ven.dropna(subset=["latitude", "longitude"]).set_index("id")

    games = []
    for s in SEASONS:
        try:
            g = pd.read_parquet(CFBD_PARQUET_DIR / f"games_{s}.parquet")
        except FileNotFoundError:
            continue
        g["season"] = s
        games.append(g[["id", "season", "week", "startDate", "venueId",
                        "neutralSite", "homeId", "awayId"]])
    g = pd.concat(games, ignore_index=True).dropna(subset=["homeId", "awayId"])
    g["kick"] = pd.to_datetime(g.startDate, format="mixed", utc=True)
    g = g.sort_values("kick")

    # each team's home venue = its most common venue as the home team
    reg = g.dropna(subset=["venueId"])
    reg = reg[~reg.neutralSite.fillna(False).to_numpy()]
    home_ven = reg.groupby("homeId").venueId.agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)

    # long format: one row per team-game, to compute rest
    long = pd.concat([
        g.assign(team_id=g.homeId, is_home=True),
        g.assign(team_id=g.awayId, is_home=False)])
    long = long.sort_values(["team_id", "kick"])
    long["rest_days"] = (long.groupby("team_id").kick.diff()
                         .dt.total_seconds() / 86400)
    # cap: anything beyond ~3 weeks is the offseason gap, not real "rest".
    # Season openers get the cap value (all teams equally rested).
    long["rest_days"] = long.rest_days.fillna(21).clip(upper=21)
    long["bye"] = ((long.rest_days >= 13) & (long.rest_days < 21)).astype(float)

    # travel / altitude / timezone vs the team's own home venue
    site = long.venueId.map(ven.latitude), long.venueId.map(ven.longitude)
    hv = long.team_id.map(home_ven)
    hlat, hlon = hv.map(ven.latitude), hv.map(ven.longitude)
    long["travel_km"] = _haversine(hlat, hlon, site[0], site[1]).fillna(0)
    long.loc[long.is_home & ~long.neutralSite.fillna(False), "travel_km"] = 0.0
    long["elev_gain_m"] = (long.venueId.map(ven.elevation).astype(float)
                           - hv.map(ven.elevation).astype(float)).fillna(0)
    long["tz_shift"] = (long.venueId.map(ven.timezone).map(_tz_offset)
                        - hv.map(ven.timezone).map(_tz_offset)).fillna(0)
    site_dome = long.venueId.map(ven.dome).fillna(False).astype(bool)
    home_dome = hv.map(ven.dome).fillna(False).astype(bool)
    long["dome_change"] = (site_dome != home_dome).astype(float)

    cols = ["rest_days", "bye", "travel_km", "elev_gain_m", "tz_shift",
            "dome_change"]
    h = long[long.is_home].set_index("id")[cols].add_prefix("home_")
    a = long[~long.is_home].set_index("id")[cols].add_prefix("away_")
    out = g.set_index("id")[["season", "week"]].join(h).join(a).reset_index()
    for c in cols:  # home-minus-away differentials
        out[f"d_{c}"] = out[f"home_{c}"] - out[f"away_{c}"]
    dest = PARQUET_DIR / "situational.parquet"
    out.to_parquet(dest, index=False)
    print(f"situational: {len(out):,} games -> {dest.name}")
    print(out[["d_rest_days", "d_travel_km", "d_tz_shift",
               "d_elev_gain_m"]].describe().loc[["mean", "std", "max"]]
          .round(1).to_string())
    return out


if __name__ == "__main__":
    build()
