import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

REGION = "SA"
AR_LAGS = 7           # AR order for the mean; check the PACF of r and adjust if needed
ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data" / "processed" / f"daily_{REGION}.csv", index_col=0, parse_dates=True)
y = df["r"]
MEAN_ON = "levels"    # "levels" = AR on the transformed price; "changes" = AR on r_t
if MEAN_ON == "levels":
    from features import daily_prices, scale_params
    P = daily_prices(REGION)
    m, s = scale_params(P)
    y = np.arcsinh((P - m) / s).loc[df.index].rename("y")

# day-of-week dummies for the mean (Monday = baseline)
dow = pd.get_dummies(y.index.dayofweek, prefix="dow", drop_first=True, dtype=float)
dow.index = y.index

# ---- the three models: same mean (AR + day-of-week), same Student-t errors ----
specs = {
    "GARCH":  dict(vol="GARCH",  p=1, o=0, q=1),
    "GJR":    dict(vol="GARCH",  p=1, o=1, q=1),   # o=1 adds the asymmetry term
    "EGARCH": dict(vol="EGARCH", p=1, o=1, q=1),
}
fits = {}
for name, s in specs.items():
    am = arch_model(y, x=dow, mean="ARX", lags=AR_LAGS, dist="t", **s)
    fits[name] = am.fit(disp="off")
    print(f"\n==================== {name} ====================")
    print(fits[name].summary().tables[2])        # variance-equation table only

# ---- results table ----
rows = {}
for name, f in fits.items():
    p, pv = f.params, f.pvalues
    a, b = p.get("alpha[1]", np.nan), p.get("beta[1]", np.nan)
    g = p.get("gamma[1]", np.nan)
    if name == "GARCH":
        persist = a + b
    elif name == "GJR":
        persist = a + b + g / 2                    # stationarity: < 1 (L14)
    else:
        persist = b                                # EGARCH: |beta| < 1
    sr = f.std_resid.dropna()
    h = f.conditional_volatility ** 2
    both = pd.concat({"h": h, "rv": df["rv_bc"]}, axis=1).dropna()
    rows[name] = {
        "a0 (omega)": p["omega"], "a1 (alpha)": a, "b1 (beta)": b,
        "c1 (gamma)": g, "c1 p-value": pv.get("gamma[1]", np.nan),
        "t d.o.f. (nu)": p["nu"],
        "Persistence": persist,
        "LB(10) p, z": acorr_ljungbox(sr, lags=[10])["lb_pvalue"].iloc[0],
        "LB(10) p, z²": acorr_ljungbox(sr**2, lags=[10])["lb_pvalue"].iloc[0],
        "max |ACF| z, lags 1-10": np.abs(acf(sr, nlags=10)[1:]).max(),
        "Log-lik": f.loglikelihood, "AIC": f.aic, "BIC": f.bic,
        "MSE vs RV": ((both["h"] - both["rv"]) ** 2).mean(),
        "Corr with RV": both["h"].corr(both["rv"]),
        # r^2 is a noisy but conditionally unbiased proxy (L2, L7); MSE and QLIKE still rank
        # models correctly with such a proxy (Patton 2011, cited in L2)
        "MSE vs r²": ((h - f.resid ** 2) ** 2).mean(),
        "QLIKE vs r²": (np.log(h) + f.resid ** 2 / h).mean(),
        "QLIKE vs RV": (np.log(both["h"]) + both["rv"] / both["h"]).mean(),
    }
table = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("\n", table)
table.to_csv(ROOT / "figures" / "fig13_garch_results.csv")


# likelihood-ratio tests: does adding the asymmetry term c1 improve on plain GARCH? (1 extra parameter)
from scipy import stats
for name in ("GJR", "EGARCH"):
    lr = 2 * (fits[name].loglikelihood - fits["GARCH"].loglikelihood)
    print(f"LR test {name} vs GARCH: LR = {lr:.2f}, p = {stats.chi2.sf(lr, 1):.2g}")
hl = {n: np.log(0.5) / np.log(table.loc["Persistence", n]) for n in table.columns}
print("Volatility half-life (days): " + ", ".join(f"{n} {v:.1f}" for n, v in hl.items()))
print("\nHow to read c1 (gamma):")
print("  GJR:    c1 < 0  -> positive shocks raise volatility more (INVERSE leverage); c1 > 0 -> stock-style")
print("  EGARCH: c1 > 0  -> positive shocks raise volatility more (INVERSE leverage); c1 < 0 -> stock-style")

# ---- fitted (theoretical) News Impact Curves, L14 style ----
e = np.linspace(-4, 4, 401)
fig, ax = plt.subplots(figsize=(8, 5))
cols = {"GARCH": "#898781", "GJR": "#2a78d6", "EGARCH": "#eb6834"}
for name, f in fits.items():
    p = f.params
    hbar = float((f.conditional_volatility ** 2).mean())
    if name == "EGARCH":
        z = e / np.sqrt(hbar)
        logh = (p["omega"] + p["alpha[1]"] * (np.abs(z) - np.sqrt(2 / np.pi))
                + p["gamma[1]"] * z + p["beta[1]"] * np.log(hbar))
        nic = np.exp(logh)
    else:
        g = p.get("gamma[1]", 0.0)
        nic = p["omega"] + (p["alpha[1]"] + g * (e < 0)) * e**2 + p["beta[1]"] * hbar
    ax.plot(e, nic, color=cols[name], lw=2.5, label=name)
ax.axvline(0, color="grey", lw=0.6)
ax.set_title("Fitted News Impact Curves", loc="left", fontweight="bold")
ax.set_xlabel("Yesterday's shock u$_{t-1}$   (← price fell  |  price rose →)")
ax.set_ylabel("Today's conditional variance h$_t^2$")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(ROOT / "figures" / "fig13_fitted_nic.png", dpi=220, bbox_inches="tight")

# ---- fitted volatility vs realized variance over time ----
best = table.loc["AIC"].idxmin()
h = fits[best].conditional_volatility ** 2
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(df.index, df["rv_bc"], color="#c3c2b7", lw=0.8, label="Realized variance (rescaled)")
ax.plot(h.index, h, color="#2a78d6", lw=1.4, label=f"{best} conditional variance")
ax.set_yscale("log")
ax.set_title(f"{REGION}: model volatility vs realized variance", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper right", ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(ROOT / "figures" / "fig13_fitted_vs_rv.png", dpi=220, bbox_inches="tight")
print(f"\nBest by AIC: {best}. Saved results CSV and two figures to {ROOT / 'figures'}")