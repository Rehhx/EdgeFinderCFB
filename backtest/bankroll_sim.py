"""Season bankroll projection from a $1,000 start — block-bootstrapped.

Every play we bet is assembled from its OWN backtest parquet (real prices, real
grades), de-duplicated for overlap, then whole WEEKS are resampled with
replacement to build synthetic seasons. Resampling weeks rather than individual
bets preserves the correlation that matters: same-game plays (Q1 vs 1H vs the
full-game spread all ride the same underdog) and same-week clustering.

Assuming independence here would materially overstate growth and understate
drawdown, because the book is far more concentrated than the bet count implies.

  python -m backtest.bankroll_sim
"""
import numpy as np
import pandas as pd

from ingestion.config import PARQUET_DIR

BANKROLL = 1000.0
SEASONS = (2023, 2024, 2025)
N_SIMS = 20000
# 1u as a fraction of the starting bankroll
UNIT_PCTS = (0.005, 0.01, 0.02, 0.03)
EXCLUDE_STATS = ("rec_yds",)


def q1_bets() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_q1_spreads.parquet")
    # Q1 PREMIUM (|spread| >= 25) DROPPED 2026-08-06. On the de-contaminated
    # sample it is the only Q1 slice that fails outright: 56.0% / +8.0% pooled
    # but bootstrap CI [-7.3, +23.8] and 2025 NEGATIVE, and it graded -0.7
    # u/szn in the complete-book seasons while consuming 49 bets/szn at 2u.
    # Excluded HERE, before dedupe(), so those >=25 games flow to BIG-DOG 1H
    # instead of disappearing from the book.
    d = d[d.covered.notna() & (d.n_books >= 2)
          & d.dog_full_line.between(17, 25, inclusive="left")].copy()
    d["play"] = "Q1 STANDARD"
    d["units"] = 1.0
    d["pnl"] = d.pnl_med
    return d[["season", "week", "game_id", "play", "units", "pnl",
              "dog_full_line"]]


def h1_bets() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_h1_saturation.parquet")
    d = d[d.covered.notna() & (d.n_books >= 2) & (d.model_on_dog >= 6)].copy()
    keep = (((d.dog_full_line >= 17) & d.week.between(1, 5))
            | ((d.dog_full_line >= 21) & (d.week >= 6)))
    d = d[keep].copy()
    # mirror picks/first_half_picks.BIGDOG_UNITS
    from picks.first_half_picks import BIGDOG_UNITS
    d["play"], d["units"], d["pnl"] = "BIG-DOG 1H", BIGDOG_UNITS, d.pnl_med
    return d[["season", "week", "game_id", "play", "units", "pnl",
              "dog_full_line"]]


def spread_bets() -> pd.DataFrame:
    """Full-game big-dog spread, weeks 1-5 (the validated tiering)."""
    d = pd.read_parquet(PARQUET_DIR / "backtest_spread_history.parquet")
    d = d[d.season.isin(SEASONS) & d.week.between(1, 5) & d.won.notna()].copy()
    d["dog_full_line"] = d.book_spread.abs()
    d["model_on_dog"] = np.where(d.book_spread > 0, d.edge, -d.edge)
    # SPREAD PREMIUM (|spread| >= 25) REMOVED 2026-08-06 after a 22-variant
    # rescue attempt on 8 clean seasons in which NOTHING cleared the bar
    # (positive every season AND bootstrap CI excluding zero):
    #   shipped dog edge>=6   48.6%  -7.0%  t=-1.81  2/8
    #   blind dog, no model   49.9%  -4.6%          3/8   <- blind BEATS us
    #   fade the favourite    51.4%  -1.9%          4/8   <- still under 52.4% BE
    #   all weeks 1-15        47.4%  -9.3%  t=-2.87 2/8   <- significantly NEGATIVE
    # and raising the edge threshold made it WORSE (>=12 -> -10.5%): on
    # |spread|>=25 the bulk of the sample (dog_edge>=12, n=477) covers 46.8%
    # while low-edge games sit near 53%. The model's confidence is
    # anti-predictive here, so there is no threshold to tune. One slice looked
    # good (HOME dogs 65.1%, +24.3%) on n=43 with a CI spanning zero — that is
    # the 1-in-22 you expect to find by looking 22 times.
    d = d[(d.dog_full_line >= 17) & (d.dog_full_line < 25)
          & (d.model_on_dog >= 6)].copy()
    d["play"] = "SPREAD STANDARD"
    # Flat 1u, matching picks/ml_spread_picks.py. The PREMIUM tier was staked
    # at 2u on the strength of a 62.3% / +18.8% / 8-of-8 claim measured on the
    # contaminated sample; on the clean universe it is NEGATIVE in every
    # specification tested (ML 48.9%/-6.5%, linear control 47.5%/-9.1%), so
    # doubling it doubled a loss. The sim must stake what the pick modules
    # stake, or the bankroll forecast describes a book nobody is betting.
    d["units"] = 1.0
    d["pnl"] = np.where(d.won == 1, 100 / 110, -1.0)
    d["game_id"] = d.game_id.astype("int64")
    return d[["season", "week", "game_id", "play", "units", "pnl",
              "dog_full_line"]]


def prop_bets() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_props_vs_book.parquet")
    d = d[d.season.isin([2024, 2025]) & (~d.stat.isin(EXCLUDE_STATS))
          & (d.n_books >= 2) & d.bet_won_bl.notna()
          & (d.bet_ev_bl >= 0.04)].copy()
    # Shipped gates (picks/prop_picks.py, both re-derived 2026-08-04 on
    # sign-bug-fixed data): EV floor 0.04 with PRIME band 0.04-0.08, and the
    # season arc — nothing before wk5, STANDARD only from wk9. The 0.04-0.05
    # slice grades +21.4% (t=+2.32, both seasons +), better than the band
    # above it; 0.03-0.04 fails 2025, which is where the floor stops. That
    # slice is its own PROBE tier at 1u, not folded into PRIME at 2u: at 1u the
    # change buys +$46 of median for +0.9pp P(losing season), at 2u a further
    # +$45 for +1.3pp — a worse trade, on the least-validated evidence.
    below_std = d.bet_ev_bl < 0.08          # PROBE + PRIME both start wk5
    d = d[(d.week >= 5) & ((d.week >= 9) | below_std)].copy()
    probe = d.bet_ev_bl.between(0.04, 0.05, inclusive="left")
    prime = d.bet_ev_bl.between(0.05, 0.08, inclusive="left")
    d["play"] = np.where(probe, "PROPS PROBE",
                         np.where(prime, "PROPS PRIME", "PROPS STANDARD"))
    # Mirror picks/prop_picks.py::STAT_STAKE_MULT — the sim must stake what the
    # pick modules stake or the forecast describes a book nobody is betting.
    from picks.prop_picks import STAT_STAKE_MULT
    from picks.prop_picks import (PRIME_UNITS, PROBE_UNITS, STANDARD_UNITS)
    d["units"] = (np.where(probe, PROBE_UNITS,
                           np.where(prime, PRIME_UNITS, STANDARD_UNITS))
                  * d.stat.map(STAT_STAKE_MULT).fillna(1.0).values)
    d["pnl"] = d.pnl_bl
    d["game_id"] = -1                             # props don't share the dog
    d["dog_full_line"] = np.nan
    return d[["season", "week", "game_id", "play", "units", "pnl",
              "dog_full_line"]]


# 2-team 10pt teasers on BIG-DOG 1H legs (backtest/teasers.py: 83.8% true hit
# on real same-week pairs, +30.3% EV at -180, t=+4.69, 80/82/89% by season).
TEASE_PTS, TEASER_PRICE = 10.0, -180.0
# "off"      1H legs are bet as singles only (the pre-2026-08-04 book)
# "replace"  legs that form a teaser are bet ONLY as the teaser
# "add"      bet the teaser IN ADDITION to the singles on those legs
#
# MEASURED, correcting the assumption this was written with. "replace" looked
# like the disciplined choice and is in fact the WORST of the three:
#     off      +103.5 u/szn   median $1,814   P(loss) 2.8%
#     replace   +97.6 u/szn   median $1,748   P(loss) 3.0%
#     add      +110.5 u/szn   median $1,883   P(loss) 2.1%
# The reason is capital efficiency, not edge: a 1u teaser covers TWO legs, so
# it deploys ~0.13 u/leg against the 2u single's 0.51 u/leg. Similar ROI%, a
# quarter of the capital — replacing singles with teasers shrinks the book.
# "add" carries real extra exposure on those games (2u single + a share of the
# teaser), and the week-block bootstrap prices that correlation in because both
# outcomes sit in the same resampled week. It still wins, because an 83.8%
# bet added to the book lifts the median AND the 5th percentile.
# ⚠️ SET TO "off" 2026-08-06. The reasoning below was measured on the
# contaminated sample where the teaser leg hit 91.5% (quoted as "83.8%", which
# was actually the JOINT rate). On the clean sample the leg is 83.6% and the
# JOINT is 69.9% -> EV +8.8% at -180, and in the two complete-book seasons the
# realised contribution is **-0.0 u/szn** (2024 +2.9, 2025 -2.9). It also
# stacks correlated exposure on games we already bet as 1H singles. Research
# retained in backtest/teasers.py; not in the book.
TEASER_MODE = "off"
# Modelled optimum is higher — return rises and P(loss) FALLS monotonically to
# at least 3u (+124.6 u/szn, median $2,024, P(loss) 1.4%). Shipped at 1u
# anyway, on the same principle as the PROPS PROBE tier: this is a new play on
# n=80 tickets whose PRICE (-180) and AVAILABILITY are both still unconfirmed.
# Revisit after a season of real results, not before.
TEASER_UNITS = 1.0


def teaser_bets(rng) -> tuple[pd.DataFrame, set]:
    """Pair BIG-DOG 1H legs into 2-team teasers; return tickets + legs used.

    Teasers are built from EXACTLY the legs the 1H single play would take, so
    the two overlap — the second return value is the set of (season, game_id)
    the tickets consumed, so the caller can drop those singles under "replace".
    Betting both IS extra exposure on the same games; the measurement above
    says it is still the best mode, because a 1u teaser deploys only ~0.13
    u/leg while the single deploys 0.51, so "replace" throws away capital.
    """
    d = pd.read_parquet(PARQUET_DIR / "backtest_h1_saturation.parquet")
    d = d[d.covered.notna() & (d.n_books >= 2) & (d.model_on_dog >= 6)].copy()
    keep = (((d.dog_full_line >= 17) & d.week.between(1, 5))
            | ((d.dog_full_line >= 21) & (d.week >= 6)))
    d = d[keep].copy()
    d["tc"] = d.dog_h1_margin + d.h1_line + TEASE_PTS
    d = d[d.tc != 0].copy()
    d["won"] = (d.tc > 0).astype(float)
    pay = 100 / abs(TEASER_PRICE)
    rows, used = [], set()
    for _, g in d.groupby(["season", "week"]):
        g = g.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        for i in range(0, len(g) - 1, 2):
            a, b = g.iloc[i], g.iloc[i + 1]
            if a.game_id == b.game_id:
                continue
            hit = min(a.won, b.won)
            rows.append({"season": a.season, "week": a.week, "game_id": -50,
                         "play": "TEASER 1H", "units": TEASER_UNITS,
                         "pnl": pay if hit else -1.0, "dog_full_line": np.nan})
            used |= {(a.season, a.game_id), (b.season, b.game_id)}
    return pd.DataFrame(rows), used


def dedupe(book: pd.DataFrame) -> pd.DataFrame:
    """One derived-market bet per game.

    Q1, 1H and the full-game spread all ride the SAME underdog. Q1 and 1H
    outcomes agree 76% (r=+0.51), so stacking them is a bigger single position,
    not diversification. Keep the best play per game by the validated
    head-to-head: spread >=25 -> Q1, 21-25 -> 1H, else 1H then Q1 then spread.
    """
    derived = book[book.game_id > 0].copy()
    props = book[book.game_id < 0]
    # Q1 PREMIUM and SPREAD PREMIUM are gone (see build_book / q1_bets), so
    # |spread| >= 25 now flows to BIG-DOG 1H, which is the best surviving
    # derived play (3/3 seasons positive, model filter helps).
    order = {"BIG-DOG 1H": 0, "Q1 STANDARD": 1, "SPREAD STANDARD": 2}
    derived["rank"] = derived.play.map(order)
    derived = derived.sort_values("rank").drop_duplicates(
        subset=["season", "game_id"], keep="first")
    return pd.concat([derived.drop(columns="rank"), props], ignore_index=True)


def build_book(teaser_mode: str | None = None) -> pd.DataFrame:
    mode = TEASER_MODE if teaser_mode is None else teaser_mode
    book = pd.concat([q1_bets(), h1_bets(), spread_bets(), prop_bets()],
                     ignore_index=True)
    print(f"raw bets across all plays: {len(book):,}")
    book = dedupe(book)
    print(f"after de-duplicating one derived bet per game: {len(book):,}")
    if mode == "off":
        return book
    tickets, used = teaser_bets(np.random.default_rng(2026))
    if tickets.empty:
        return book
    if mode == "replace":
        # a leg consumed by a teaser is no longer available as a single
        key = list(zip(book.season, book.game_id))
        drop = np.array([k in used for k in key]) & (book.play == "BIG-DOG 1H")
        print(f"teasers: {len(tickets)} tickets, replacing {int(drop.sum())} "
              "1H singles whose legs they consume")
        book = book[~drop]
    else:
        print(f"teasers: {len(tickets)} tickets IN ADDITION to the 1H singles "
              "— extra exposure on those games, priced in by the week blocks")
    return pd.concat([book, tickets], ignore_index=True)


# Q1 spreads on 25+ point mismatches are low-limit markets. Modelling them at
# the same stake as a main-market spread is the single most optimistic
# assumption in this file, so cap them in dollars.
Q1_MAX_STAKE = 50.0


# Props carry MORE forward uncertainty than the derived-line plays, so they get
# an extra haircut on top of the base. This is not conservatism for its own
# sake — it is measured (2026-08-04):
#   Q1/1H/spread  backtest/play_stability.py — threshold sweeps are PLATEAUS,
#                 survive median pricing and leave-one-season-out. Q1 PREMIUM
#                 90% CI [+11.0, +49.1] u/szn.
#   props         backtest/props_pricing_stability.py — the pricing calibration
#                 alone contributes sd 15.6 u/szn (the volume model: 3.5), and
#                 tripling the calibration data cuts that only 10%, so it is
#                 not a sample-size problem. backtest/props_blend_sweep.py then
#                 showed why: moving the market-blend weight by ONE grid step
#                 swings the season +0.4 -> +30.0 -> +54.2 u. The fitted 0.30
#                 sits directly beside a near-zero outcome.
# Requiring bets to clear EV>=5% across the whole w band was tested and does
# NOT fix it — the intersection is just the most conservative w, and that book
# is +0.4 u/szn. So the exposure is real and is priced in here instead.
# ⚠️ LOWERED 0.20 -> 0.00 on 2026-08-06. The extra penalty was set when props
# were judged "measurably less certain" than the derived-line plays. The audit
# INVERTED that: props are the only play positive in both OOS seasons with all
# controls passed, while every derived play now sits inside its own confidence
# interval (see HANDOFF §10). Penalising the best-evidenced play 20pp harder
# than the unproven ones had the ranking backwards.
# NOTE this changes what we EXPECT, not what we bet — no selection changes.
PROPS_EXTRA_HAIRCUT = 0.00


def _blocks(book: pd.DataFrame, unit: float, haircut: float):
    """Per-week arrays of (dollar stake, pnl), with the edge haircut applied.

    haircut shifts each outcome down by h * (that play's mean ROI), which cuts
    the edge while preserving the variance — the right way to ask "what if the
    true edge is smaller than the backtest measured?" PROPS_EXTRA_HAIRCUT is
    now 0: props are the best-evidenced play in the book, not the weakest.
    """
    b = book.copy()
    roi = b.groupby("play").pnl.transform("mean")
    h = np.where(b.play.str.startswith("PROPS"),
                 min(haircut + PROPS_EXTRA_HAIRCUT, 0.95), haircut)
    b["pnl_adj"] = b.pnl - h * roi
    b["stake"] = b.units * unit
    is_q1 = b.play.str.startswith("Q1")
    b.loc[is_q1, "stake"] = b.loc[is_q1, "stake"].clip(upper=Q1_MAX_STAKE)
    b = b[b.season.isin(full_book_seasons(book))]
    return [g[["stake", "pnl_adj"]].values
            for _, g in b.groupby(["season", "week"])]


def full_book_seasons(book: pd.DataFrame) -> list[int]:
    """Seasons in which EVERY play was actually priced.

    The bootstrap resamples (season, week) blocks from a pooled set, so a
    simulated season inherits the average play mix of the pool — not the mix
    of a real season. Props were only bet out-of-sample in 2024-25 (2023 is
    their calibration season), so pooling all three gave a simulated season
    just 2/3 of a real season's props:

        play              actual u/szn   simulated   ratio
        PROPS PRIME             +37.4       +25.5     68%
        PROPS PROBE              +8.4        +5.7     68%
        PROPS STANDARD          +11.1        +7.6     68%
        everything else            --          --    102%
        TOTAL                   +72.6       +54.9     76%

    So `main()` printed a +72.6 u/szn book while the bankroll distribution
    underneath it was drawn from a +54.9 one — a 24% disagreement, on what is
    now the play carrying the book.

    Computed rather than hard-coded so it self-corrects as seasons are added:
    a season counts only if every play in the book appears in it.
    """
    plays = set(book.play.unique())
    ok = [int(s) for s, g in book.groupby("season")
          if set(g.play.unique()) == plays]
    if not ok:                      # no season has the complete book
        print("  [!] no season contains every play — bootstrapping the full "
              "pool; per-play volumes will not match the table above")
        return sorted(book.season.unique().tolist())
    return ok


def simulate(book: pd.DataFrame, unit_pct: float, rng,
             haircut: float = 0.0) -> np.ndarray:
    """Block bootstrap: resample WEEKS with replacement, keeping bets together."""
    blocks = _blocks(book, BANKROLL * unit_pct, haircut)
    # divide by the seasons the blocks actually came from, not every season
    n_weeks = int(round(len(blocks) / len(full_book_seasons(book))))
    out = np.empty(N_SIMS)
    for i in range(N_SIMS):
        idx = rng.integers(0, len(blocks), n_weeks)
        out[i] = BANKROLL + sum(
            (blocks[j][:, 0] * blocks[j][:, 1]).sum() for j in idx)
    return out


def drawdown(book: pd.DataFrame, unit_pct: float, rng, haircut: float = 0.0,
             n=2000) -> np.ndarray:
    blocks = _blocks(book, BANKROLL * unit_pct, haircut)
    # divide by the seasons the blocks actually came from, not every season
    n_weeks = int(round(len(blocks) / len(full_book_seasons(book))))
    worst = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(blocks), n_weeks)
        eq = BANKROLL + np.cumsum(
            [(blocks[j][:, 0] * blocks[j][:, 1]).sum() for j in idx])
        eq = np.concatenate([[BANKROLL], eq])
        worst[i] = ((eq - np.maximum.accumulate(eq))
                    / np.maximum.accumulate(eq)).min()
    return worst


def main() -> None:
    full = build_book()

    # The per-season view FIRST, because it is the most decision-relevant fact
    # in this file: every derived-line play peaks in 2023 and decays, while
    # props are the only stable one.
    full["_u"] = full.units * full.pnl
    print("\n=== units by play and SEASON (is the edge decaying?) ===")
    piv = full.pivot_table(index="play", columns="season", values="_u",
                           aggfunc="sum").round(1)
    print(piv.to_string())
    print("season totals:",
          {int(k): round(v, 1) for k, v in full.groupby("season")._u.sum().items()})

    # Table and simulation MUST describe the same population. The bootstrap
    # resamples (season, week) blocks, so it can only reproduce a real season's
    # play mix using seasons where every play was priced — props were bet
    # out-of-sample only in 2024-25. Reporting per-play means over 3 seasons
    # while simulating from a 2-season-props pool is what made the printed
    # book (+72.6 u) disagree with the simulated one (+54.9 u) by 24%.
    fs = full_book_seasons(full)
    book = full[full.season.isin(fs)]
    print(f"\n=== the season book, per play "
          f"(seasons with the COMPLETE book: {fs}) ===")
    g = book.groupby("play").agg(
        bets=("pnl", "size"), units=("units", "mean"),
        roi=("pnl", "mean"), won=("pnl", lambda s: (s > 0).mean()))
    g["bets_per_szn"] = (g.bets / len(fs)).round(0)
    g["u_per_szn"] = [book[book.play == p]._u.sum() / len(fs) for p in g.index]
    print(g.round(3).to_string())
    total_u = g.u_per_szn.sum()
    total_b = g.bets_per_szn.sum()
    print(f"\nTOTAL: {total_b:.0f} bets/season, {total_u:+.1f} units/season")

    rng = np.random.default_rng(2026)
    print(f"\n=== ${BANKROLL:,.0f} BANKROLL, END OF SEASON ({N_SIMS:,} sims) ===")
    print(f"Q1 stakes capped at ${Q1_MAX_STAKE:.0f} (low-limit market).")
    print("\nHAIRCUT = how much of the backtested edge we assume is NOT real.")
    print("0% takes the backtest at face value — that is the optimistic case and")
    print("its P(loss)=0 is an artifact of treating a measured edge as certain.")
    print("30-50% is the standard forward haircut for overfitting, line")
    print("movement, limits and edge decay. Read the 50% row as the base case.\n")
    print(f"{'haircut':>8} {'1u':>6} {'stake':>7} {'median':>9} {'5th %ile':>9} "
          f"{'95th %ile':>10} {'P(loss)':>8} {'P(2x)':>7} {'med DD':>8}")
    for haircut in (0.0, 0.30, 0.50, 0.70):
        for pct in UNIT_PCTS:
            fin = simulate(book, pct, rng, haircut)
            dd = drawdown(book, pct, rng, haircut)
            print(f"{haircut:7.0%} {pct:6.1%} {BANKROLL*pct:6.0f}$ "
                  f"{np.median(fin):9,.0f} {np.percentile(fin, 5):9,.0f} "
                  f"{np.percentile(fin, 95):10,.0f} "
                  f"{(fin < BANKROLL).mean():7.1%} "
                  f"{(fin >= 2*BANKROLL).mean():6.1%} {np.median(dd):7.1%}")
        print()

    print("per-season reality check at 1u = 2% = "
          f"${BANKROLL*0.02:.0f}, NO haircut (optimistic):")
    for s in SEASONS:
        b = book[book.season == s]
        if len(b):
            u = (b.units * b.pnl).sum()
            note = "  <- props start in 2024, so 2023 is not comparable" \
                if s == 2023 else ""
            print(f"  {s}: {len(b):4d} bets  {u:+7.1f}u  "
                  f"${BANKROLL:,.0f} -> ${BANKROLL + u*BANKROLL*0.02:,.0f}{note}")
    print("\nNOTE: SPREAD STANDARD contributes ~0 units — after de-duplication")
    print("Q1/1H absorb nearly all of those games and grade them better. It is")
    print("kept in the book only for games where no derived line is posted.")


if __name__ == "__main__":
    main()
