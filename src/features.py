"""Shared data prep used by every Part 2 slide (and reusable in the battery project)."""
import numpy as np
import pandas as pd
from data_fetch import fetch_range


def _day_index(ts: pd.Series) -> pd.Series:
    # SETTLEMENTDATE marks the END of each 5-min interval, so 00:00 belongs to the previous day
    return (ts - pd.Timedelta(minutes=5)).dt.normalize()


def daily_prices(region: str, start: str = "2021-10", end: str = "2026-06") -> pd.Series:
    """Daily average spot price ($/MWh)."""
    df = fetch_range(region, start, end)
    return df.groupby(_day_index(df["settlement_date"]))["rrp"].mean().rename(region)


def scale_params(daily: pd.Series) -> tuple[float, float]:
    """Median and MAD of the daily price series: the (m, s) in asinh((P - m) / s)."""
    m = daily.median()
    return m, (daily - m).abs().median()


def asinh_returns(prices: pd.Series) -> pd.Series:
    """Change in scaled asinh price: asinh((P - m) / s). Handles negative prices;
    typical prices stay ~linear, spikes are log-compressed (Uniejewski, Weron & Ziel 2018)."""
    m, s = scale_params(prices)
    return np.arcsinh((prices - m) / s).diff().dropna()


def realized_variance(region: str, start: str = "2021-10", end: str = "2026-06") -> pd.Series:
    """Daily realized variance: sum of squared 5-min changes in the scaled asinh price,
    using the same (m, s) as the daily series so both are on the same transform.
    The NEM trades 24/7, so there is no overnight gap to correct for."""
    df = fetch_range(region, start, end)
    m, s = scale_params(daily_prices(region, start, end))
    y = np.arcsinh((df["rrp"] - m) / s)
    dy2 = y.diff() ** 2
    rv = dy2.groupby(_day_index(df["settlement_date"])).sum(min_count=1)
    return rv.iloc[1:].rename(f"RV_{region}")   # drop day 1 (its first diff is missing)