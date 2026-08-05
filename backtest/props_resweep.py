"""Re-derive every fitted props threshold. Run after ANY change to the data.

Three separate filters turned out to have been fitted on the sign-bug-corrupted
region on 2026-08-04 (the week gate, PRIME-year-round, the EV floor), which is
why discipline rule 16 exists: **any threshold tuned on a dataset you have
since changed is unverified until re-swept.** Widening the prop-line pull on
2026-08-05 changed the dataset, so all of them are due again.

Everything here is decided by the SAME two rules that survived last time:
  1. a cut must be positive in BOTH out-of-sample seasons, not just pooled
  2. prefer SEASON BALANCE over the biggest total — the old week gate drew 82%
     of its units from one season and was worse for it

  python -m backtest.props_resweep
"""
import numpy as np
import pandas as pd

from backtest.props_vs_book import EXCLUDE_STATS
from ingestion.config import PARQUET_DIR

OOS = (2024, 2025)
MIN_N = 40


def load() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET_DIR / "backtest_props_vs_book.parquet")
    return d[(~d.stat.isin(EXCLUDE_STATS)) & d.bet_won_bl.notna()
             & d.season.isin(OOS)].copy()


def _units(s: pd.DataFrame, probe_lo, probe_hi, prime_hi):
    probe = s.bet_ev_bl.between(probe_lo, probe_hi, inclusive="left")
    prime = s.bet_ev_bl.between(probe_hi, prime_hi, inclusive="left")
    return np.where(probe, 1.0, np.where(prime, 2.0, 1.0))


def book(d, ev_min, probe_hi, prime_hi, wk_prime, wk_std, min_books=2):
    s = d[(d.n_books >= min_books) & (d.bet_ev_bl >= ev_min)].copy()
    below_std = s.bet_ev_bl < prime_hi
    s = s[(s.week >= wk_prime) & ((s.week >= wk_std) | below_std)]
    if s.empty:
        return None
    u = _units(s, ev_min, probe_hi, prime_hi)
    per = {}
    for season in OOS:
        m = (s.season == season).values
        per[season] = float((u[m] * s[m].pnl_bl).sum())
    return {"n": len(s) // len(OOS), "roi": float(s.pnl_bl.mean()),
            "u": float((u * s.pnl_bl).sum() / len(OOS)), "per": per,
            "both_pos": all(v > 0 for v in per.values())}


def show(label, r, mark=""):
    if r is None:
        print(f"  {label:<30} (empty)")
        return
    flag = "" if r["both_pos"] else "  <-- FAILS a season"
    per = "  ".join(f"{k}:{v:+.1f}" for k, v in r["per"].items())
    print(f"  {label:<30} {r['n']:4d}/szn {r['roi']:+7.1%} {r['u']:+7.1f}u   "
          f"[{per}]{flag}{mark}")


SHIPPED = dict(ev_min=0.04, probe_hi=0.05, prime_hi=0.08, wk_prime=5, wk_std=9)


def main() -> None:
    d = load()
    print(f"graded OOS rows: {len(d):,}  "
          f"(matched lines/szn: {len(d)//len(OOS):,})")
    base = book(d, **SHIPPED)
    print("\n=== SHIPPED CONFIG ===")
    show("EV>=.04 P<.05 W5/W9", base, "  <-- current")

    print("\n=== 1. EV FLOOR (probe band moves with it) ===")
    for ev in (0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06):
        cfg = SHIPPED | {"ev_min": ev, "probe_hi": max(ev + 0.01, 0.05)}
        show(f"ev_min={ev:.3f}", book(d, **cfg))

    print("\n=== 2. PROBE/PRIME boundary (ev_min fixed) ===")
    for hi in (0.045, 0.05, 0.055, 0.06):
        show(f"probe_hi={hi:.3f}", book(d, **(SHIPPED | {"probe_hi": hi})))

    print("\n=== 3. PRIME/STANDARD boundary ===")
    for hi in (0.07, 0.08, 0.09, 0.10, 0.12):
        show(f"prime_hi={hi:.2f}", book(d, **(SHIPPED | {"prime_hi": hi})))

    print("\n=== 4. WEEK GATES ===")
    for wp, ws in ((1, 5), (1, 9), (5, 5), (5, 9), (5, 11), (6, 9), (7, 9)):
        show(f"PRIME wk{wp}+, STD wk{ws}+",
             book(d, **(SHIPPED | {"wk_prime": wp, "wk_std": ws})))

    print("\n=== 5. MIN BOOKS ===")
    for mb in (1, 2, 3, 4):
        show(f"n_books>={mb}", book(d, **(SHIPPED | {"min_books": mb})))

    print("\n=== 6. RAW EV BANDS (is the non-monotonicity still there?) ===")
    s = d[(d.n_books >= 2) & (d.week >= 5)]
    for lo, hi in [(0.02, 0.03), (0.03, 0.04), (0.04, 0.05), (0.05, 0.06),
                   (0.06, 0.08), (0.08, 0.12), (0.12, 9.0)]:
        b = s[s.bet_ev_bl.between(lo, hi, inclusive="left")]
        if len(b) < MIN_N:
            continue
        per = b.groupby("season").pnl_bl.mean()
        t = b.pnl_bl.mean() / (b.pnl_bl.std() / np.sqrt(len(b)))
        ok = "" if (per > 0).all() else "  <-- fails a season"
        print(f"  EV {lo:.2f}-{min(hi,9):.2f}: {len(b):4d} bets {b.pnl_bl.mean():+6.1%} "
              f"t={t:+5.2f}  " + " ".join(f"{k}:{v:+.1%}" for k, v in per.items())
              + ok)

    print("\nREAD: adopt a change only if it beats SHIPPED on units AND is")
    print("positive in BOTH seasons. Prefer season balance over the max — the")
    print("old gate drew 82% of its units from one season and was worse for it.")


if __name__ == "__main__":
    main()
