import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from features import daily_prices, asinh_returns

p = daily_prices("SA", "2021-10", "2026-06")
dp = p.diff().dropna()
r = asinh_returns(p)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
a1.plot(dp.index, dp, color="#2a78d6", lw=0.8)
a1.set_title("Before: raw daily price change ($/MWh)", loc="left", fontweight="bold")
a2.plot(r.index, r, color="#eb6834", lw=0.8)
a2.set_title("After: change in scaled asinh price", loc="left", fontweight="bold")
for ax in (a1, a2):
    ax.axhline(0, color="grey", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9")
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "figures"
fig.savefig(out / "fig10_transform.png", dpi=220, bbox_inches="tight")
print(f"median ${p.median():.1f}, MAD ${(p - p.median()).abs().median():.1f}")