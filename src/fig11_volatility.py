import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from features import daily_prices, asinh_returns, realized_variance

REGION, START, END = "SA", "2021-10", "2026-06"
NLAGS = 40
ROOT = Path(__file__).resolve().parent.parent

r = asinh_returns(daily_prices(REGION, START, END))
rv = realized_variance(REGION, START, END)
df = pd.concat({"r": r, "rv": rv}, axis=1).dropna()
df["r2"] = df["r"] ** 2

# Bias correction (L3): rescale RV so it has the same mean as r^2
k = df["r2"].mean() / df["rv"].mean()
df["rv_bc"] = df["rv"] * k

print(f"T = {len(df)} days | RV scale factor k = {k:.3f}")
for col, label in [("r2", "r²"), ("rv_bc", "RV")]:
    lb = acorr_ljungbox(df[col], lags=[10, 20])["lb_pvalue"]
    print(f"{label:>3}: ACF lag1 = {acf(df[col], nlags=1)[1]:.2f} | "
          f"LB(10) p = {lb.iloc[0]:.2g} | LB(20) p = {lb.iloc[1]:.2g}")
print(f"corr(r², RV) = {df['r2'].corr(df['rv_bc']):.2f}")

# save the daily series so later slides (GARCH, model comparison) can reuse it
(ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
df[["r", "r2", "rv", "rv_bc"]].to_csv(ROOT / "data" / "processed" / f"daily_{REGION}.csv")

# ---- figure: RV over time (top), ACFs of r² and RV (bottom) ----
fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.45, wspace=0.15)
ax_ts = fig.add_subplot(gs[0, :])
ax_ts.plot(df.index, df["rv_bc"], color="#2a78d6", lw=1)
ax_ts.set_yscale("log")
ax_ts.set_title(f"{REGION}: daily realized variance from 288 five-minute intervals (log scale)",
                loc="left", fontweight="bold")
ax_ts.spines[["top", "right"]].set_visible(False)

band = 1.96 / np.sqrt(len(df))
lags = np.arange(1, NLAGS + 1)
for j, (col, title, colr) in enumerate([("r2", "ACF of r²", "#898781"),
                                         ("rv_bc", "ACF of realized variance", "#2a78d6")]):
    ax = fig.add_subplot(gs[1, j])
    ax.axhspan(-band, band, color="#e1e0d9", alpha=0.6, lw=0, zorder=0)   # ±1.96/√T bands
    ax.bar(lags, acf(df[col], nlags=NLAGS)[1:], color=colr, width=0.7, zorder=2)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Lag (days)")
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig(ROOT / "figures" / "fig11_volatility.png", dpi=220, bbox_inches="tight")
print("saved", ROOT / "figures" / "fig11_volatility.png")