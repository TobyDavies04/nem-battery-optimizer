"""
data_fetch.py
 
Pulls historical NEM regional price & demand data (5-minute dispatch prices,
aggregated by AEMO into monthly CSVs per region) and caches it locally as
parquet so repeated runs don't re-download.
 
Data source: AEMO's public "Aggregated price and demand data" files.
    https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data
 
URL pattern (one file per region per month):
    https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{YYYYMM}_{REGION}1.csv
 
NOTE: AEMO has been migrating some NEMweb-adjacent base URLs in 2026. If
BASE_URL below starts returning 404s, check the aggregated-data page (link
above) for the current path and update BASE_URL — nothing else needs to change.
 
Each monthly CSV has columns:
    REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE
 
RRP = Regional Reference Price ($/MWh), the wholesale spot price, in 30-minute
intervals for older data and 5-minute intervals for more recent data
(post the 5-minute settlement rule change).
"""
 
from __future__ import annotations
 
import io
import time
from dataclasses import dataclass
from pathlib import Path
 
import pandas as pd
import requests
 
BASE_URL = "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{yyyymm}_{region}1.csv"
 
VALID_REGIONS = {"NSW", "QLD", "VIC", "SA", "TAS"}
 
HEADERS = {
    # AEMO's server has been known to reject requests with no user-agent.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
 
DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
 
 
@dataclass
class FetchResult:
    region: str
    year: int
    month: int
    n_rows: int
    from_cache: bool
 
 
def _month_str(year: int, month: int) -> str:
    return f"{year:04d}{month:02d}"
 
 
def _cache_path(region: str, year: int, month: int, raw_dir: Path) -> Path:
    return raw_dir / f"{region}_{_month_str(year, month)}.parquet"
 
 
def fetch_month(
    region: str,
    year: int,
    month: int,
    raw_dir: Path = DEFAULT_RAW_DIR,
    force_refresh: bool = False,
    request_timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch one region-month of NEM price/demand data, caching to parquet
    locally. Returns a DataFrame with columns:
        region, settlement_date, total_demand_mw, rrp, period_type
    """
    region = region.upper()
    if region not in VALID_REGIONS:
        raise ValueError(f"region must be one of {VALID_REGIONS}, got {region!r}")
 
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(region, year, month, raw_dir)
 
    if cache_file.exists() and not force_refresh:
        return pd.read_parquet(cache_file)
 
    url = BASE_URL.format(yyyymm=_month_str(year, month), region=region)
    resp = requests.get(url, headers=HEADERS, timeout=request_timeout)
 
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch {url} (status {resp.status_code}). "
            "AEMO may have changed the URL scheme — check the aggregated-data "
            "page and update BASE_URL in data_fetch.py."
        )
 
    df = pd.read_csv(io.StringIO(resp.text))
    df = df.rename(
        columns={
            "REGION": "region",
            "SETTLEMENTDATE": "settlement_date",
            "TOTALDEMAND": "total_demand_mw",
            "RRP": "rrp",
            "PERIODTYPE": "period_type",
        }
    )
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df = df.sort_values("settlement_date").reset_index(drop=True)
 
    df.to_parquet(cache_file, index=False)
    return df
 
 
def fetch_range(
    region: str,
    start: str,
    end: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    force_refresh: bool = False,
    sleep_between_requests: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch and concatenate all months of data for `region` between `start`
    and `end` (inclusive), given as 'YYYY-MM' strings, e.g. '2023-01'.
 
    Only hits the network for months not already cached (unless
    force_refresh=True), and sleeps briefly between real network requests
    to be polite to AEMO's servers.
    """
    months = pd.period_range(start=start, end=end, freq="M")
    frames = []
 
    for period in months:
        year, month = period.year, period.month
        cache_file = _cache_path(region, year, month, raw_dir)
        already_cached = cache_file.exists() and not force_refresh
 
        df = fetch_month(region, year, month, raw_dir=raw_dir, force_refresh=force_refresh)
        frames.append(df)
 
        if not already_cached:
            time.sleep(sleep_between_requests)
 
    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset="settlement_date").sort_values("settlement_date")
    return full.reset_index(drop=True)
 
 
if __name__ == "__main__":
    # Smoke test: pull a single recent month for one region and print a
    # summary. Run this first (`python src/data_fetch.py`) to confirm the
    # AEMO URL scheme still works before doing a bigger historical pull.
    import datetime
 
    today = datetime.date.today()
    # Aggregated files typically lag by a few days/weeks, so go back two
    # months to be safe on the very first test.
    test_month = today.month - 2
    test_year = today.year
    if test_month <= 0:
        test_month += 12
        test_year -= 1
 
    print(f"Smoke-testing fetch for VIC {test_year}-{test_month:02d} ...")
    result = fetch_month("VIC", test_year, test_month)
    print(f"Got {len(result)} rows.")
    print(result.head())
    print(result.describe())
 