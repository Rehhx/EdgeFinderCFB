"""Deep learning player-prop model: conditional DISTRIBUTIONS, notpoint estimates.

Why DL here (the honest case): our loss investigation showed better central
projections do NOT beat the book — bigger model-vs-line disagreements win no
more often. But we price every player's over/under with ONE fixed sigma per
stat. A boom/bust deep threat and a possession back have very different
shapes, and P(over) depends on the shape. So this net predicts the whole
conditional distribution per player-game.

Architecture: player embedding + position embedding + usage/context features
-> MLP -> (mu, log_dispersion). Trained by NEGATIVE LOG-LIKELIHOOD of a
Gamma (continuous yards) or Negative Binomial (receptions) — so the model
learns heteroscedastic variance, not just the mean.

Walk-forward: train on seasons < S, predict season S. Compared against the
production recalibrated-normal model on identical rows.

  python -m models.dl_props
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ingestion.config import PARQUET_DIR

torch.manual_seed(0)

STATS = {"rush_yds": "gamma", "rec_yds": "gamma", "pass_yds": "gamma",
         "receptions": "nb"}
FEATS = ["proj", "trail_share", "exp_vol", "trail_eff", "prior_games",
         "team_spread", "opp_def", "week"]
EPOCHS, BATCH, LR, EMB = 12, 512, 3e-3, 8


class DistNet(nn.Module):
    """Predicts (mu, dispersion) per player-game."""

    def __init__(self, n_players: int, n_feats: int):
        super().__init__()
        self.emb = nn.Embedding(n_players + 1, EMB)
        nn.init.normal_(self.emb.weight, std=0.05)
        self.net = nn.Sequential(
            nn.Linear(n_feats + EMB, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2))

    def forward(self, x, pid):
        h = torch.cat([x, self.emb(pid)], dim=1)
        out = self.net(h)
        mu = torch.nn.functional.softplus(out[:, 0]) + 0.05
        disp = torch.nn.functional.softplus(out[:, 1]) + 0.05
        return mu, disp


def nll(y, mu, disp, kind: str):
    """Negative log-likelihood: Gamma for yards, NegBinom for counts."""
    if kind == "gamma":
        # disp = shape k; mean mu -> rate = k/mu
        k, y = disp, torch.clamp(y, min=0.1)
        return (torch.lgamma(k) - k * torch.log(k / mu)
                - (k - 1) * torch.log(y) + (k / mu) * y).mean()
    r, y = disp, torch.clamp(y, min=0.0)
    p = r / (r + mu)
    return -(torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
             + r * torch.log(p) + y * torch.log(1 - p)).mean()


def sf(kind: str, line, mu, disp):
    """P(stat > line) under the predicted distribution."""
    from scipy.stats import gamma, nbinom
    if kind == "gamma":
        return gamma.sf(np.maximum(line, 0.01), a=disp, scale=mu / disp)
    return nbinom.sf(np.floor(line), disp, disp / (disp + mu))


def build_table(stat: str) -> pd.DataFrame:
    """One row per player-game with the features the net sees."""
    d = pd.read_parquet(PARQUET_DIR / "prop_projections.parquet")
    share = {"rush_yds": "trail_rush_share", "rec_yds": "trail_tgt_share",
             "receptions": "trail_tgt_share", "pass_yds": "trail_qb_share"}[stat]
    vol = ("exp_team_rush_att" if stat == "rush_yds" else "exp_team_pass_att")
    eff = {"rush_yds": "trail_ypc", "rec_yds": "trail_ypt",
           "receptions": "trail_catch", "pass_yds": "trail_ypa"}[stat]
    opp = ("trail_ypc_allowed" if stat == "rush_yds" else "trail_ypa_allowed")
    t = pd.DataFrame({
        "season": d.season, "week": d.week, "player": d.player,
        "y": d[stat], "proj": d[f"proj_{stat}"],
        "trail_share": d[share], "exp_vol": d[vol], "trail_eff": d[eff],
        "prior_games": d.prior_games, "team_spread": d.team_spread,
        "opp_def": d[opp]})
    return t[t.proj.notna() & t.y.notna() & (t.proj > 0)].copy()


def run_stat(stat: str, kind: str) -> pd.DataFrame:
    t = build_table(stat)
    for c in FEATS:
        t[c] = t[c].astype(float).fillna(t[c].astype(float).median())
    codes = {p: i for i, p in enumerate(t.player.unique())}
    t["pid"] = t.player.map(codes)
    mu_f, sd_f = t[FEATS].mean(), t[FEATS].std().replace(0, 1)

    preds = []
    for season in (2024, 2025):
        tr, te = t[t.season < season], t[t.season == season]
        if len(tr) < 2000 or te.empty:
            continue
        Xtr = torch.tensor(((tr[FEATS] - mu_f) / sd_f).values, dtype=torch.float32)
        ytr = torch.tensor(tr.y.values, dtype=torch.float32)
        ptr = torch.tensor(tr.pid.values, dtype=torch.long)
        net = DistNet(len(codes), len(FEATS))
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
        n = len(Xtr)
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                mu, disp = net(Xtr[idx], ptr[idx])
                loss = nll(ytr[idx], mu, disp, kind)
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            Xte = torch.tensor(((te[FEATS] - mu_f) / sd_f).values,
                               dtype=torch.float32)
            pte = torch.tensor(te.pid.values, dtype=torch.long)
            mu, disp = net(Xte, pte)
        out = te[["season", "week", "player", "y", "proj"]].copy()
        out["dl_mu"] = mu.numpy()
        out["dl_disp"] = disp.numpy()
        out["stat"] = stat
        preds.append(out)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def main() -> None:
    all_preds = []
    for stat, kind in STATS.items():
        p = run_stat(stat, kind)
        if p.empty:
            continue
        mae_dl = (p.dl_mu - p.y).abs().mean()
        mae_pr = (p.proj - p.y).abs().mean()
        print(f"{stat:11s} n={len(p):6d}  DL MAE {mae_dl:6.2f} | "
              f"production MAE {mae_pr:6.2f} | "
              f"disp range {p.dl_disp.min():.1f}-{p.dl_disp.max():.1f}")
        all_preds.append(p)
    df = pd.concat(all_preds, ignore_index=True)
    dest = PARQUET_DIR / "dl_prop_preds.parquet"
    df.to_parquet(dest, index=False)
    print(f"\nsaved {len(df):,} DL predictions -> {dest.name}")


if __name__ == "__main__":
    main()
