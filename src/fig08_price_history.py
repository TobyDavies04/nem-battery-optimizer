import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from data_fetch import fetch_range

# ---- settings ----
REGION = "SA"
START, END = "2021-10", "2026-06"   # months, inclusive (Oct 2021 = start of 5-min settlement)
N_LABELS = 4                        # how many spike days to annotate
YMAX = None                         # e.g. 1500 to zoom in if one day squashes the rest

# ---- data: 5-min prices -> daily average ----
df = fetch_range(REGION, START, END)
# SETTLEMENTDATE is the END of each interval, so 00:00 belongs to the previous day
day = (df["settlement_date"] - pd.Timedelta(minutes=5)).dt.normalize()
daily = df.groupby(day)["rrp"].mean()
typical = daily.rolling(30, center=True, min_periods=15).median()

print(f"{len(daily)} days | mean ${daily.mean():.0f} | median ${daily.median():.0f} | "
      f"max ${daily.max():,.0f} on {daily.idxmax():%d %b %Y} | "
      f"negative-average days: {(daily < 0).sum()}")

# biggest spike days, at least 30 days apart so labels don't pile up
labels = []
for d, v in daily.sort_values(ascending=False).items():
    if all(abs((d - l).days) > 30 for l, _ in labels):
        labels.append((d, v))
    if len(labels) == N_LABELS:
        break

# ---- plot ----
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(daily.index, daily, color="#2a78d6", lw=1, label="Daily average price")
ax.plot(typical.index, typical, color="#eb6834", lw=2, label="30-day rolling median (typical level)")
ax.axhline(0, color="grey", lw=0.8)

for d, v in labels:
    if YMAX is None or v <= YMAX:
        ax.annotate(f"${v:,.0f}\n{d:%d %b %Y}", xy=(d, v), xytext=(8, -2),
                    textcoords="offset points", va="top", fontsize=10, color="#52514e")

if YMAX is not None:
    ax.set_ylim(top=YMAX)

ax.set_title("South Australia: daily average wholesale price, Oct 2021 – Jun 2026",
             loc="left", fontsize=15, fontweight="bold", pad=30)
ax.set_ylabel("$/MWh")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e1e0d9")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
ax.set_xlim(daily.index.min(), daily.index.max())
fig.text(0, -0.02, "Source: AEMO 5-minute regional reference prices, averaged by day.",
         fontsize=10, color="#52514e")

out = Path(__file__).resolve().parent.parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "fig08_price_history.png", dpi=220, bbox_inches="tight")
print("saved", out / "fig08_price_history.png")