import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

REGION = "SA"
MAXLAG = 20
ROOT = Path(__file__).resolve().parent.parent

# daily series saved by fig11_volatility.py
df = pd.read_csv(ROOT / "data" / "processed" / f"daily_{REGION}.csv", index_col=0, parse_dates=True)
r, rv = df["r"], df["rv_bc"]
T = len(df)

# ---- 1. cross-correlations corr(RV_t, r_{t-k}) ----
# k > 0: past shocks -> today's volatility (what leverage is about)
# k < 0: future shocks (a control)
lags = np.arange(-MAXLAG, MAXLAG + 1)
xcorr = np.array([rv.corr(r.shift(k)) for k in lags])
band = 1.96 / np.sqrt(T)

# ---- 2. empirical News Impact Curve: RV_t vs r_{t-1} ----
nic = pd.DataFrame({"shock": r.shift(1), "rv": rv}).dropna()
fit = lowess(np.log(nic["rv"]), nic["shock"], frac=0.3)          # smooth on log scale (RV is very skewed)

# ---- simple asymmetry numbers for the slide ----
q = nic["shock"].abs().quantile(0.80)                             # "big" shocks = top 20% by size
up = nic.loc[nic["shock"] > q, "rv"]
down = nic.loc[nic["shock"] < -q, "rv"]
print(f"T = {T}")
print(f"corr(RV_t, r_t-1) = {xcorr[lags == 1][0]:+.3f}   (stocks: negative; inverse leverage: positive)")
print(f"Big shocks (|r| > {q:.2f}): mean RV next day after UP = {up.mean():.3f} (n={len(up)}), "
      f"after DOWN = {down.mean():.3f} (n={len(down)}), ratio up/down = {up.mean() / down.mean():.2f}")
print(f"Median RV next day: after UP = {up.median():.3f}, after DOWN = {down.median():.3f}")

# ---- figure ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1, 1.1]})

a1.axhspan(-band, band, color="#e1e0d9", alpha=0.6, lw=0, zorder=0)
a1.bar(lags, xcorr, color=np.where(lags > 0, "#2a78d6", "#c3c2b7"), width=0.7, zorder=2)
a1.axhline(0, color="grey", lw=0.6)
a1.axvline(0, color="grey", lw=0.6, ls=":")
a1.set_title("Cross-correlation: RV$_t$ vs r$_{t-k}$", loc="left", fontweight="bold")
a1.set_xlabel("k (days)   ← future shocks  |  past shocks →")
a1.spines[["top", "right"]].set_visible(False)

a2.scatter(nic["shock"], nic["rv"], s=8, color="#2a78d6", alpha=0.25, lw=0)
a2.plot(fit[:, 0], np.exp(fit[:, 1]), color="#eb6834", lw=2.5, label="Smoothed fit (lowess)")
a2.set_yscale("log")
a2.axvline(0, color="grey", lw=0.6)
a2.set_title("Empirical News Impact Curve", loc="left", fontweight="bold")
a2.set_xlabel("Yesterday's shock r$_{t-1}$   (← price fell  |  price rose →)")
a2.set_ylabel("Today's realized variance (log scale)")
a2.legend(frameon=False, loc="upper left")
a2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig12_leverage.png", dpi=220, bbox_inches="tight")
print("saved", ROOT / "figures" / "fig12_leverage.png")