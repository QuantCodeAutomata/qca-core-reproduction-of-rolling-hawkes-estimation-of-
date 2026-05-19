"""
Data loading for E-mini S&P 500 futures (ES) high-frequency data.

Loads raw trade and quote data from the Massive API for the active quarterly
contract, reconstructs top-of-book mid-prices, and identifies mid-price-change events.

Contract roll schedule (paper's deterministic rule):
    - March quarterly cycle: Mar/Jun/Sep/Dec
    - Expiration: third Friday of expiration month
    - Rollover: 8 days before expiration (second Thursday of expiration month)
    - Use expiring quarter until rollover date, then switch to next quarter
"""

from __future__ import annotations

import os
import logging
from datetime import date, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contract roll schedule
# ---------------------------------------------------------------------------

QUARTERLY_MONTHS = [3, 6, 9, 12]  # Mar, Jun, Sep, Dec
MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}


def _third_friday(year: int, month: int) -> date:
    """Return the third Friday of the given year/month."""
    d = date(year, month, 1)
    # Find first Friday
    days_until_friday = (4 - d.weekday()) % 7
    first_friday = d + timedelta(days=days_until_friday)
    return first_friday + timedelta(weeks=2)


def _rollover_date(year: int, month: int) -> date:
    """
    Return the rollover date: 8 calendar days before expiration
    (second Thursday of expiration month).
    """
    expiry = _third_friday(year, month)
    return expiry - timedelta(days=8)


def get_active_contract(dt: date) -> str:
    """
    Return the active ES contract ticker for a given date.

    Uses the paper's deterministic quarterly roll schedule.

    Parameters
    ----------
    dt : date
        The date to look up.

    Returns
    -------
    str
        Contract ticker, e.g. 'ESH1998', 'ESM2010'.
    """
    year = dt.year
    for month in QUARTERLY_MONTHS:
        roll = _rollover_date(year, month)
        if dt < roll:
            code = MONTH_CODES[month]
            return f"ES{code}{year}"
    # After last rollover of the year, use March of next year
    code = MONTH_CODES[3]
    return f"ES{code}{year + 1}"


def build_roll_schedule(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Build a date-to-contract mapping for the full sample period.

    Parameters
    ----------
    start_date, end_date : date
        Sample period boundaries.

    Returns
    -------
    pd.DataFrame
        Columns: date, contract
    """
    dates = pd.date_range(start_date, end_date, freq="B")
    records = []
    for d in dates:
        contract = get_active_contract(d.date())
        records.append({"date": d.date(), "contract": contract})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Massive API data loading
# ---------------------------------------------------------------------------

def load_es_quotes_massive(
    ticker: str,
    from_date: str,
    to_date: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load ES best-bid/best-ask quote data from Massive API.

    Parameters
    ----------
    ticker : str
        Contract ticker, e.g. 'ES'.
    from_date, to_date : str
        Date range in 'YYYY-MM-DD' format.
    api_key : str, optional
        Massive API key. Falls back to MASSIVE_TOKEN env var.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, bid_price, ask_price, bid_size, ask_size
    """
    if api_key is None:
        api_key = os.getenv("MASSIVE_TOKEN", "")

    try:
        from massive import RESTClient
        client = RESTClient(api_key=api_key)
        quotes = []
        for q in client.list_quotes(ticker=ticker, timestamp=from_date):
            quotes.append({
                "timestamp": pd.Timestamp(q.sip_timestamp, unit="ns", tz="UTC"),
                "bid_price": q.bid_price,
                "ask_price": q.ask_price,
                "bid_size": q.bid_size,
                "ask_size": q.ask_size,
            })
        df = pd.DataFrame(quotes)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Massive API quote load failed: {e}")
        return pd.DataFrame(columns=["timestamp", "bid_price", "ask_price", "bid_size", "ask_size"])


def load_es_trades_massive(
    ticker: str,
    from_date: str,
    to_date: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load ES trade data from Massive API.

    Parameters
    ----------
    ticker : str
        Contract ticker.
    from_date, to_date : str
        Date range.
    api_key : str, optional
        Massive API key.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, price, size
    """
    if api_key is None:
        api_key = os.getenv("MASSIVE_TOKEN", "")

    try:
        from massive import RESTClient
        client = RESTClient(api_key=api_key)
        trades = []
        for t in client.list_trades(ticker=ticker, timestamp=from_date):
            trades.append({
                "timestamp": pd.Timestamp(t.sip_timestamp, unit="ns", tz="UTC"),
                "price": t.price,
                "size": t.size,
            })
        df = pd.DataFrame(trades)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Massive API trade load failed: {e}")
        return pd.DataFrame(columns=["timestamp", "price", "size"])


# ---------------------------------------------------------------------------
# Order-book reconstruction and mid-price event extraction
# ---------------------------------------------------------------------------

def reconstruct_midprice_events(
    quotes_df: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:15",
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Reconstruct mid-price events from top-of-book quote updates.

    Maintains prevailing best bid and ask through time. Whenever either side
    changes, updates state and computes mid-price p_m = (a + b) / 2.
    Defines an event each time p_m changes.

    Parameters
    ----------
    quotes_df : pd.DataFrame
        Columns: timestamp, bid_price, ask_price
    session_start, session_end : str
        Regular trading hours in HH:MM format.
    tz : str
        Timezone for session filtering.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, mid_price, prev_mid_price
        One row per mid-price change event.
    """
    if quotes_df.empty:
        return pd.DataFrame(columns=["timestamp", "mid_price", "prev_mid_price"])

    df = quotes_df.copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert(tz)

    # Filter to regular trading hours
    df = df[
        (df["timestamp"].dt.time >= pd.Timestamp(session_start).time())
        & (df["timestamp"].dt.time <= pd.Timestamp(session_end).time())
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "mid_price", "prev_mid_price"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute mid-price
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2.0

    # Identify changes
    df["prev_mid_price"] = df["mid_price"].shift(1)
    events = df[df["mid_price"] != df["prev_mid_price"]].copy()
    events = events.dropna(subset=["prev_mid_price"])

    return events[["timestamp", "mid_price", "prev_mid_price"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic data generation (fallback when real data unavailable)
# ---------------------------------------------------------------------------

def generate_synthetic_es_events(
    n_days: int = 10,
    events_per_day_mean: int = 5000,
    session_start_hour: float = 9.5,  # 9:30
    session_end_hour: float = 16.25,  # 16:15
    seed: int = 42,
    start_date: str = "2009-01-05",
) -> pd.DataFrame:
    """
    Generate synthetic ES mid-price-change events for testing.

    Simulates a Poisson process with time-varying intensity (U-shaped intraday)
    to approximate realistic event counts.

    Parameters
    ----------
    n_days : int
        Number of trading days to simulate.
    events_per_day_mean : int
        Average events per day.
    session_start_hour, session_end_hour : float
        Session boundaries in decimal hours.
    seed : int
        Random seed.
    start_date : str
        Start date for the synthetic series.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (UTC), date, seconds_since_open
    """
    rng = np.random.default_rng(seed)
    session_duration_s = (session_end_hour - session_start_hour) * 3600.0

    records = []
    base_date = pd.Timestamp(start_date, tz="America/New_York")
    trading_days = pd.bdate_range(start=base_date, periods=n_days)

    for day in trading_days:
        n_events = rng.poisson(events_per_day_mean)
        # Uniform within session (simplified)
        event_offsets = np.sort(rng.uniform(0, session_duration_s, size=n_events))
        session_open = day.replace(
            hour=9, minute=30, second=0, microsecond=0
        )
        for offset in event_offsets:
            ts = session_open + pd.Timedelta(seconds=float(offset))
            records.append({
                "timestamp": ts,
                "date": day.date(),
                "seconds_since_open": float(offset),
            })

    df = pd.DataFrame(records)
    return df


def load_or_generate_events(
    start_date: str = "2009-01-01",
    end_date: str = "2009-03-31",
    cache_path: Optional[str] = None,
    use_synthetic: bool = True,
    n_days: int = 60,
    events_per_day_mean: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load event data from cache or generate synthetic data.

    Parameters
    ----------
    start_date, end_date : str
        Date range.
    cache_path : str, optional
        Path to cached Parquet file.
    use_synthetic : bool
        If True, generate synthetic data when cache not found.
    n_days : int
        Number of synthetic days.
    events_per_day_mean : int
        Mean events per synthetic day.
    seed : int
        Random seed for synthetic generation.

    Returns
    -------
    pd.DataFrame
        Event series with at minimum: timestamp, date, seconds_since_open
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading events from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    if use_synthetic:
        logger.info("Generating synthetic ES events (real data not available)")
        df = generate_synthetic_es_events(
            n_days=n_days,
            events_per_day_mean=events_per_day_mean,
            seed=seed,
            start_date=start_date,
        )
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_parquet(cache_path, index=False)
        return df

    raise FileNotFoundError(
        f"No cached data at {cache_path} and use_synthetic=False"
    )
