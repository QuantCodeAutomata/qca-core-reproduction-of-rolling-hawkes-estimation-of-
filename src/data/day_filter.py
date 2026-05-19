"""
Day-level filtering for ES mid-price-change event series.

Implements the paper's exclusion criteria:
    1. Non-trading days (weekends, holidays)
    2. Early-close days (session ends before 16:15)
    3. Days with nonactive trading (large intraday gaps)
    4. Days with daily volume below the year-specific 5th percentile

Implementation choices are documented explicitly as required by the paper.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# US market holidays (approximate list for 1998-2010)
# Source: NYSE holiday schedule
US_MARKET_HOLIDAYS = {
    # New Year's Day
    date_str
    for year in range(1998, 2011)
    for date_str in [
        f"{year}-01-01",
        f"{year}-07-04",
        f"{year}-12-25",
    ]
}

# Known early-close days (day after Thanksgiving, Christmas Eve, etc.)
# Implementation choice: we use a gap-based heuristic to detect early closes
# rather than a hardcoded list, as the paper does not specify the exact list.


def filter_nontrading_days(
    event_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Remove non-trading days (days with zero events).

    Parameters
    ----------
    event_df : pd.DataFrame
        Event series with a date column.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Filtered event series.
    """
    daily_counts = event_df.groupby(date_col).size()
    active_dates = daily_counts[daily_counts > 0].index
    return event_df[event_df[date_col].isin(active_dates)].copy()


def filter_early_close_days(
    event_df: pd.DataFrame,
    date_col: str = "date",
    timestamp_col: str = "timestamp",
    session_end_hour: float = 16.25,  # 16:15
    min_session_end_hour: float = 16.20,  # Must trade until at least 16:12
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Remove days where trading ended before the regular session close.

    Implementation choice: a day is considered early-close if the last event
    occurs before min_session_end_hour (16:12 NY time).

    Parameters
    ----------
    event_df : pd.DataFrame
        Event series.
    date_col : str
        Date column name.
    timestamp_col : str
        Timestamp column name.
    session_end_hour : float
        Regular session end (16.25 = 16:15).
    min_session_end_hour : float
        Minimum acceptable last-event time.
    tz : str
        Timezone.

    Returns
    -------
    pd.DataFrame
        Filtered event series.
    """
    df = event_df.copy()
    if df[timestamp_col].dt.tz is None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")
    df["_ts_local"] = df[timestamp_col].dt.tz_convert(tz)
    df["_hour_local"] = df["_ts_local"].dt.hour + df["_ts_local"].dt.minute / 60.0

    last_event_hour = df.groupby(date_col)["_hour_local"].max()
    full_days = last_event_hour[last_event_hour >= min_session_end_hour].index

    df = df[df[date_col].isin(full_days)].copy()
    df = df.drop(columns=["_ts_local", "_hour_local"])
    return df


def filter_nonactive_trading_days(
    event_df: pd.DataFrame,
    date_col: str = "date",
    timestamp_col: str = "timestamp",
    max_gap_minutes: float = 30.0,
) -> pd.DataFrame:
    """
    Remove days with large intraday gaps indicating nonactive trading.

    Implementation choice: a day is flagged as nonactive if any consecutive
    event gap exceeds max_gap_minutes during the regular session.

    Parameters
    ----------
    event_df : pd.DataFrame
        Event series.
    date_col : str
        Date column name.
    timestamp_col : str
        Timestamp column name.
    max_gap_minutes : float
        Maximum allowed gap in minutes. Default 30.

    Returns
    -------
    pd.DataFrame
        Filtered event series.
    """
    max_gap_seconds = max_gap_minutes * 60.0
    df = event_df.copy()
    df = df.sort_values([date_col, timestamp_col])

    active_dates = []
    for day, group in df.groupby(date_col):
        ts = group[timestamp_col].values.astype("datetime64[ns]").astype(np.float64) / 1e9
        if len(ts) < 2:
            continue
        gaps = np.diff(ts)
        if np.max(gaps) <= max_gap_seconds:
            active_dates.append(day)

    return df[df[date_col].isin(active_dates)].copy()


def filter_low_volume_days(
    event_df: pd.DataFrame,
    volume_df: Optional[pd.DataFrame] = None,
    date_col: str = "date",
    volume_col: str = "volume",
    quantile: float = 0.05,
) -> pd.DataFrame:
    """
    Remove days with daily volume below the year-specific 5th percentile.

    If volume_df is not provided, uses daily event count as a proxy for volume.

    Parameters
    ----------
    event_df : pd.DataFrame
        Event series.
    volume_df : pd.DataFrame, optional
        Daily volume data with columns [date, volume]. If None, uses event count.
    date_col : str
        Date column name.
    volume_col : str
        Volume column name in volume_df.
    quantile : float
        Quantile threshold. Paper uses 5th percentile.

    Returns
    -------
    pd.DataFrame
        Filtered event series.
    """
    df = event_df.copy()

    if volume_df is not None:
        daily_vol = volume_df.set_index(date_col)[volume_col]
    else:
        # Use daily event count as proxy
        daily_vol = df.groupby(date_col).size().rename("volume")

    # Compute year-specific 5th percentile
    vol_df = daily_vol.reset_index()
    vol_df.columns = ["date", "volume"]
    vol_df["year"] = pd.to_datetime(vol_df["date"]).dt.year
    year_quantiles = vol_df.groupby("year")["volume"].quantile(quantile)

    accepted_dates = []
    for _, row in vol_df.iterrows():
        threshold = year_quantiles.get(row["year"], 0)
        if row["volume"] >= threshold:
            accepted_dates.append(row["date"])

    return df[df[date_col].isin(accepted_dates)].copy()


def apply_all_filters(
    event_df: pd.DataFrame,
    date_col: str = "date",
    timestamp_col: str = "timestamp",
    volume_df: Optional[pd.DataFrame] = None,
    max_gap_minutes: float = 30.0,
    volume_quantile: float = 0.05,
) -> pd.DataFrame:
    """
    Apply all day-level filters in sequence.

    Order: nontrading → early-close → nonactive → low-volume

    Parameters
    ----------
    event_df : pd.DataFrame
        Raw event series.
    date_col : str
        Date column name.
    timestamp_col : str
        Timestamp column name.
    volume_df : pd.DataFrame, optional
        Daily volume data.
    max_gap_minutes : float
        Max intraday gap for nonactive detection.
    volume_quantile : float
        Volume quantile threshold.

    Returns
    -------
    pd.DataFrame
        Filtered event series.
    """
    n_before = len(event_df[date_col].unique()) if date_col in event_df.columns else 0
    logger.info(f"Day filtering: starting with {n_before} unique dates")

    df = filter_nontrading_days(event_df, date_col)
    n1 = len(df[date_col].unique())
    logger.info(f"After nontrading filter: {n1} dates")

    if timestamp_col in df.columns:
        df = filter_early_close_days(df, date_col, timestamp_col)
        n2 = len(df[date_col].unique())
        logger.info(f"After early-close filter: {n2} dates")

        df = filter_nonactive_trading_days(df, date_col, timestamp_col, max_gap_minutes)
        n3 = len(df[date_col].unique())
        logger.info(f"After nonactive filter: {n3} dates")

    df = filter_low_volume_days(df, volume_df, date_col, quantile=volume_quantile)
    n4 = len(df[date_col].unique())
    logger.info(f"After low-volume filter: {n4} dates")

    return df
