import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
from data_fetch import fetch_range

# ---- settings ----
REGION = "SA"
START = pd.Timestamp("2026-06-16")
DAYS = 7
TARIFF = 300          # $/MWh (30c/kWh) - check a real SA retail rate before presenting
ZOOM = (-250, 700)    # y-range for the zoomed panel

# ---- data ----
end = START + pd.Timedelta(days=DAYS)
df = fetch_range(REGION, START.strftime("%Y-%m"), (end - pd.Timedelta(minutes=5)).strftime("%Y-%m"))
df = df[(df["settlement_date"] > START) & (df["settlement_date"] <= end)]
t, p = df["settlement_date"], df["rrp"]
print(f"min {p.min():,.0f}  max {p.max():,.0f}  negative {100*(p<0).mean():.1f}%")

# ---- plot ----
fig, (top, bot) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True,
                               gridspec_kw={"height_ratios": [1.6, 1]})
for ax in (top, bot):
    ax.plot(t, p, color="#2a78d6", lw=1.2)
    ax.axhline(TARIFF, color="#eb6834", lw=2, ls="--")
    ax.axhline(0, color="grey", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9")

top.set_title("South Australia: wholesale price every 5 minutes vs a flat retail tariff",
              loc="left", fontsize=15, fontweight="bold")
top.set_ylabel("$/MWh")
i = p.idxmax()
top.annotate(f"Peak ${p[i]:,.0f}/MWh\n{t[i]:%a %d %b %H:%M}", xy=(t[i], p[i]),
             xytext=(-10, -5), textcoords="offset points", ha="right", va="top")

bot.set_ylim(*ZOOM)
bot.set_ylabel("$/MWh (zoomed)")
bot.fill_between(t, p, 0, where=p < 0, color="#1baf7a", alpha=0.35, interpolate=True)
bot.text(0.005, 0.05, f"Negative prices: {100*(p<0).mean():.0f}% of intervals this week",
         transform=bot.transAxes)
bot.text(1.01, TARIFF, f"What you pay\n≈ ${TARIFF}/MWh", color="#eb6834",
         fontweight="bold", va="center", transform=bot.get_yaxis_transform())
bot.xaxis.set_major_locator(mdates.DayLocator())
bot.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
bot.set_xlim(START, end)

out = Path(__file__).resolve().parent.parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "fig01_spot_vs_retail.png", dpi=220, bbox_inches="tight")
print("saved", out / "fig01_spot_vs_retail.png")