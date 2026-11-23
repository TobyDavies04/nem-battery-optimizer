"""
battery_lp.py

Phase 1: battery arbitrage LP under perfect foresight.

Given a day's worth of known NEM prices (5-minute RRP), solves for the
charge/discharge schedule that maximizes revenue, subject to:
    - power limits (P_max)
    - state-of-charge bounds (soc_min, soc_max)
    - an energy balance with round-trip efficiency losses
    - a day-boundary condition forcing soc at the end of the day back to
      soc_init, so the optimizer can't "bank" free energy at the edge of
      the solve window

Solved independently per calendar day (not as one multi-year LP) — this is
what "perfect foresight" is meant to represent: if you'd known today's
prices in full, what's the best you could have done today. It also keeps
each day directly comparable to Phase 2's rolling-horizon results on the
same days.

See docs/battery_lp_study_guide.docx for the full derivation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

INTERVAL_MINUTES = 5
DT_HOURS = INTERVAL_MINUTES / 60.0
EXPECTED_INTERVALS_PER_DAY = 24 * 60 // INTERVAL_MINUTES  # 288

EPS = 1e-6  # tolerance for "is this variable effectively zero"


@dataclass
class BatterySpec:
    p_max_mw: float = 1.0       # max charge/discharge power
    e_max_mwh: float = 2.0      # usable energy capacity
    soc_min_mwh: float = 0.0
    soc_init_frac: float = 0.5  # starting SOC as a fraction of e_max_mwh
    eta_rt: float = 0.90        # round-trip efficiency

    @property
    def soc_max_mwh(self) -> float:
        return self.e_max_mwh

    @property
    def soc_init_mwh(self) -> float:
        return self.soc_init_frac * self.e_max_mwh

    @property
    def eta_leg(self) -> float:
        """Efficiency applied on each leg (charge or discharge) so that
        the product of both legs equals the round-trip efficiency."""
        return math.sqrt(self.eta_rt)


def solve_day(day_prices: pd.DataFrame, spec: BatterySpec, solver_msg: bool = False) -> pd.DataFrame | None:
    """
    Solve the perfect-foresight LP for a single day.

    day_prices: DataFrame with columns ['settlement_date', 'rrp'], already
        filtered to one calendar day, sorted by time.

    Returns a DataFrame with one row per interval (charge, discharge, soc,
    price, net_revenue), or None if the day doesn't have a usable number of
    intervals (e.g. a gappy day) and was skipped.
    """
    n = len(day_prices)
    if n < EXPECTED_INTERVALS_PER_DAY * 0.95:
        # Gappy/partial day (missing data, DST edge case, etc.) — skip
        # rather than silently optimizing against an incomplete price
        # series, which would understate or overstate achievable revenue.
        return None

    prices = day_prices["rrp"].to_numpy()
    times = day_prices["settlement_date"].to_numpy()
    idx = range(n)

    model = pulp.LpProblem("battery_arbitrage_day", pulp.LpMaximize)

    # NOTE: PuLP 4.x (Rust-core rewrite) creates variables via the LpProblem
    # instance rather than pulp.LpVariable.dicts(...) directly — the old
    # classmethod no longer exists.
    charge = model.add_variable_dicts("charge", idx, lowBound=0, upBound=spec.p_max_mw)
    discharge = model.add_variable_dicts("discharge", idx, lowBound=0, upBound=spec.p_max_mw)
    soc = model.add_variable_dicts("soc", idx, lowBound=spec.soc_min_mwh, upBound=spec.soc_max_mwh)

    # Objective: revenue from discharging minus cost of charging.
    model += pulp.lpSum(
        prices[t] * (discharge[t] - charge[t]) * DT_HOURS for t in idx
    )

    # Energy balance, with soc[-1] := soc_init.
    eta_leg = spec.eta_leg
    for t in idx:
        prev_soc = spec.soc_init_mwh if t == 0 else soc[t - 1]
        model += (
            soc[t]
            == prev_soc
            + charge[t] * eta_leg * DT_HOURS
            - discharge[t] / eta_leg * DT_HOURS
        )

    # Day-boundary closure: end the day where it started, so no day can
    # "bank" free stored energy across the solve-window edge.
    model += soc[n - 1] == spec.soc_init_mwh

    # NOTE: PuLP 4.x no longer bundles CBC. HiGHS (installed via `pip install
    # highspy`) is the solver backend here — open-source (MIT), and solve
    # status is now returned by .solve() rather than read off model.status.
    solver = pulp.getSolver("HiGHS", msg=solver_msg)
    result = model.solve(solver)

    if result.status != pulp.LpSolveStatus.Optimal:
        raise RuntimeError(f"LP did not solve to optimality: status={result.status.name}")

    charge_vals = np.array([charge[t].value() for t in idx])
    discharge_vals = np.array([discharge[t].value() for t in idx])
    soc_vals = np.array([soc[t].value() for t in idx])

    # Sanity check: the LP should never choose to charge and discharge in
    # the same interval simultaneously (it's always strictly dominated,
    # due to round-trip losses) — this confirms that empirically rather
    # than just asserting it from theory.
    both_active = (charge_vals > EPS) & (discharge_vals > EPS)
    if both_active.any():
        raise AssertionError(
            f"Solver chose simultaneous charge+discharge in {both_active.sum()} "
            "interval(s) — investigate before trusting results."
        )

    net_revenue = prices * (discharge_vals - charge_vals) * DT_HOURS

    return pd.DataFrame(
        {
            "settlement_date": times,
            "price": prices,
            "charge_mw": charge_vals,
            "discharge_mw": discharge_vals,
            "soc_mwh": soc_vals,
            "net_revenue": net_revenue,
        }
    )


def naive_baseline_day(day_prices: pd.DataFrame, spec: BatterySpec,
                        charge_window=(0, 6), discharge_window=(17, 21)) -> pd.DataFrame:
    """
    A simple fixed-schedule baseline for comparison: charge at full power
    during `charge_window` (local hour range) while there's room, discharge
    at full power during `discharge_window` while there's energy available.
    Run forward under the *same* physical constraints as the LP (power
    limits, SOC bounds, efficiency) so the comparison is apples-to-apples.
    """
    n = len(day_prices)
    prices = day_prices["rrp"].to_numpy()
    times = pd.to_datetime(day_prices["settlement_date"])
    hours = times.dt.hour.to_numpy()

    soc = spec.soc_init_mwh
    eta_leg = spec.eta_leg

    charge_vals = np.zeros(n)
    discharge_vals = np.zeros(n)
    soc_vals = np.zeros(n)

    for t in range(n):
        h = hours[t]
        if charge_window[0] <= h < charge_window[1]:
            # Charge at full power, capped by remaining headroom.
            max_charge_for_soc = (spec.soc_max_mwh - soc) / (eta_leg * DT_HOURS)
            c = min(spec.p_max_mw, max(max_charge_for_soc, 0))
            d = 0.0
        elif discharge_window[0] <= h < discharge_window[1]:
            max_discharge_for_soc = (soc - spec.soc_min_mwh) * eta_leg / DT_HOURS
            d = min(spec.p_max_mw, max(max_discharge_for_soc, 0))
            c = 0.0
        else:
            c = 0.0
            d = 0.0

        soc = soc + c * eta_leg * DT_HOURS - d / eta_leg * DT_HOURS
        charge_vals[t] = c
        discharge_vals[t] = d
        soc_vals[t] = soc

    net_revenue = prices * (discharge_vals - charge_vals) * DT_HOURS

    return pd.DataFrame(
        {
            "settlement_date": times.to_numpy(),
            "price": prices,
            "charge_mw": charge_vals,
            "discharge_mw": discharge_vals,
            "soc_mwh": soc_vals,
            "net_revenue": net_revenue,
        }
    )


def run_backtest(prices_df: pd.DataFrame, spec: BatterySpec,
                  include_baseline: bool = True) -> dict[str, pd.DataFrame]:
    """
    Loop the LP (and optionally the naive baseline) over every calendar day
    present in `prices_df`. Returns a dict with 'lp' and optionally
    'baseline' keys, each a concatenated multi-day results DataFrame, plus
    'skipped_days' listing any days that didn't have enough intervals.
    """
    df = prices_df.copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df["day"] = df["settlement_date"].dt.date

    lp_frames = []
    baseline_frames = []
    skipped_days = []

    for day, day_df in df.groupby("day"):
        day_df = day_df.sort_values("settlement_date")
        lp_result = solve_day(day_df, spec)
        if lp_result is None:
            skipped_days.append(day)
            continue
        lp_result["day"] = day
        lp_frames.append(lp_result)

        if include_baseline:
            baseline_result = naive_baseline_day(day_df, spec)
            baseline_result["day"] = day
            baseline_frames.append(baseline_result)

    results = {
        "lp": pd.concat(lp_frames, ignore_index=True) if lp_frames else pd.DataFrame(),
        "skipped_days": skipped_days,
    }
    if include_baseline:
        results["baseline"] = (
            pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()
        )
    return results


def summarize_backtest(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily revenue totals for LP vs baseline, plus the uplift from optimizing."""
    lp_daily = results["lp"].groupby("day")["net_revenue"].sum().rename("lp_revenue")
    summary = lp_daily.to_frame()

    if "baseline" in results and not results["baseline"].empty:
        base_daily = results["baseline"].groupby("day")["net_revenue"].sum().rename("baseline_revenue")
        summary = summary.join(base_daily)
        summary["uplift"] = summary["lp_revenue"] - summary["baseline_revenue"]

    return summary.reset_index()


if __name__ == "__main__":
    # Smoke test with synthetic data — this is NOT real NEM data, it's
    # purely to verify the LP mechanics (energy balance, boundary closure,
    # no-simultaneous-charge/discharge) run correctly before pointing the
    # module at real cached SA prices.
    rng = np.random.default_rng(42)
    n_days = 3
    timestamps = pd.date_range("2026-01-01", periods=n_days * EXPECTED_INTERVALS_PER_DAY, freq="5min")

    # Rough synthetic NEM-like price shape: a daily sinusoid (cheap at
    # night, expensive evening peak) plus noise plus occasional spikes.
    hours = timestamps.hour + timestamps.minute / 60
    base = 40 + 30 * np.sin((hours - 6) / 24 * 2 * np.pi)
    noise = rng.normal(0, 10, size=len(timestamps))
    spikes = rng.choice([0, 0, 0, 0, 300], size=len(timestamps), p=[0.985, 0.005, 0.005, 0.0025, 0.0025])
    synthetic_prices = base + noise + spikes

    synthetic_df = pd.DataFrame({"settlement_date": timestamps, "rrp": synthetic_prices})

    spec = BatterySpec()
    results = run_backtest(synthetic_df, spec)
    summary = summarize_backtest(results)

    print("Smoke test (synthetic data) — daily summary:")
    print(summary)
    print(f"\nSkipped days: {results['skipped_days']}")
    print(f"\nTotal LP revenue: ${summary['lp_revenue'].sum():.2f}")
    if "baseline_revenue" in summary.columns:
        print(f"Total baseline revenue: ${summary['baseline_revenue'].sum():.2f}")
        print(f"Total uplift from optimization: ${summary['uplift'].sum():.2f}")
