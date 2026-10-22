import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from features import daily_prices, asinh_returns

# ---- settings ----
START, END = "2021-10", "2026-06"
LB_LAG = 10                        # Ljung-Box lag k
TICKERS = {                        # Yahoo Finance tickers to compare against
    "ASX 200": "^AXJO",
    "AGL Energy": "AGL.AX",
    "US natural gas": "NG=F",      # swap for "Gold": "GC=F" if you prefer
}

# ---- prices and returns ----
prices = {
    "SA electricity": daily_prices("SA", START, END),
    "VIC electricity": daily_prices("VIC", START, END),
}
series = {name: asinh_returns(p) for name, p in prices.items()}   # scaled-asinh changes

end_date = (pd.Period(END, "M").end_time + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
for name, tkr in TICKERS.items():
    px = yf.download(tkr, start=f"{START}-01", end=end_date, auto_adjust=True, progress=False)["Close"].squeeze()
    px = px.dropna()
    prices[name] = px
    series[name] = np.log(px).diff().dropna()                      # log returns


def describe(r: pd.Series, p: pd.Series) -> dict:
    dp = p.diff().dropna()                                         # raw price changes
    return {
        "T": len(r),
        "Mean": r.mean(),
        "SD": r.std(),
        "Skew": stats.skew(r),
        "Kurt": stats.kurtosis(r, fisher=False),                   # raw kurtosis (normal = 3), as in L6
        "ρ̂(1)": acf(r, nlags=1)[1],
        "JB p": stats.jarque_bera(r).pvalue,
        f"LB({LB_LAG}) p, r": acorr_ljungbox(r, lags=[LB_LAG])["lb_pvalue"].iloc[0],
        f"LB({LB_LAG}) p, r²": acorr_ljungbox(r**2, lags=[LB_LAG])["lb_pvalue"].iloc[0],
        "Skew (raw ΔP)": stats.skew(dp),
        "Kurt (raw ΔP)": stats.kurtosis(dp, fisher=False),
    }


table = pd.DataFrame({name: describe(series[name], prices[name]) for name in series})
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
pd.set_option("display.width", 200)
print(table)

out = Path(__file__).resolve().parent.parent / "figures"
out.mkdir(exist_ok=True)
table.to_csv(out / "fig09_stylized_facts.csv")

# ---- table as an image (to paste straight onto the slide) ----
fmt = table.copy().astype(object)
for row in fmt.index:
    for col in fmt.columns:
        v = table.loc[row, col]
        if row == "T":
            fmt.loc[row, col] = f"{v:.0f}"
        elif " p" in row:
            fmt.loc[row, col] = "<0.001" if v < 0.001 else f"{v:.3f}"
        elif row in ("Mean", "SD"):
            fmt.loc[row, col] = f"{v:.4f}"
        elif row == "Kurt (raw ΔP)":
            fmt.loc[row, col] = f"{v:.1f}"
        else:
            fmt.loc[row, col] = f"{v:.2f}"

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.axis("off")
tbl = ax.table(cellText=fmt.values, rowLabels=fmt.index, colLabels=fmt.columns,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#e1e0d9")
    if r == 0:
        cell.set_text_props(fontweight="bold")
    if c in (0, 1) and r > 0:                                      # shade the electricity columns
        cell.set_facecolor("#eaf2fc")
fig.savefig(out / "fig09_stylized_facts_table.png", dpi=220, bbox_inches="tight")

# ---- histograms of RAW daily price changes, standardised, with N(0,1) overlaid ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
x = np.linspace(-6, 6, 400)
for ax, name, col in [(axes[0], "SA electricity", "#2a78d6"), (axes[1], "ASX 200", "#eb6834")]:
    dp = prices[name].diff().dropna()
    z = (dp - dp.mean()) / dp.std()
    outside = (z.abs() > 6).mean() * 100
    ax.hist(z.clip(-6, 6), bins=80, range=(-6, 6), density=True, color=col, alpha=0.6)
    ax.plot(x, stats.norm.pdf(x), color="#0b0b0b", lw=1.5, label="Normal")
    ax.set_title(f"{name} (kurtosis {table.loc['Kurt (raw ΔP)', name]:.1f})", loc="left", fontweight="bold")
    ax.set_xlabel("Standardised daily price change")
    ax.text(0.98, 0.6, f"{outside:.1f}% of days\nbeyond ±6 SD", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Density")
axes[0].legend(frameon=False, loc="upper left")
fig.savefig(out / "fig09_histograms.png", dpi=220, bbox_inches="tight")
print("saved table (CSV + PNG) and histograms to", out)