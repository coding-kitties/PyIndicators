"""
EMA Trend Ribbon Indicator

Uses a set of 9 Exponential Moving Averages with increasing periods to
visualise trend strength and direction.  When the majority of EMAs are
rising the trend is bullish; when most are falling it is bearish.
"""
from typing import Union, List, Optional

import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl

from pyindicators.exceptions import PyIndicatorException


def _calc_ema(src: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA over a numpy array."""
    n = len(src)
    alpha = 2.0 / (period + 1)
    out = np.empty(n)
    out[0] = src[0]

    for i in range(1, n):
        out[i] = src[i] * alpha + out[i - 1] * (1.0 - alpha)

    return out


def ema_trend_ribbon(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str = 'Close',
    ema_lengths: Optional[List[int]] = None,
    smoothing_period: int = 2,
    threshold: int = 7,
    trend_column: str = 'ema_ribbon_trend',
    bullish_count_column: str = 'ema_ribbon_bullish_count',
    bearish_count_column: str = 'ema_ribbon_bearish_count',
    ema_column_prefix: str = 'ema_ribbon',
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the EMA Trend Ribbon indicator.

    Computes 9 EMAs with increasing periods and determines the overall
    trend by counting how many EMAs are rising vs falling over a
    smoothing period.  When *threshold* or more EMAs agree on
    direction the trend is classified as bullish (1) or bearish (-1);
    otherwise neutral (0).

    Calculation:
        - For each EMA period, compute EMA(source, period)
        - An EMA is "rising" when EMA[t] >= EMA[t - smoothing_period]
        - bullish_count = number of rising EMAs
        - bearish_count = number of falling EMAs
        - trend = 1 if bullish_count >= threshold,
                 -1 if bearish_count >= threshold, else 0

    Args:
        data: pandas or polars DataFrame with price data.
        source_column: Column name for the source prices
            (default: 'Close').
        ema_lengths: List of EMA periods (default: [8, 14, 20, 26,
            32, 38, 44, 50, 60]).
        smoothing_period: Number of bars to look back when
            determining EMA slope direction (default: 2).
        threshold: Minimum number of EMAs that must agree for a
            bullish or bearish classification (default: 7, out of 9).
        trend_column: Result column name for the trend state
            (default: 'ema_ribbon_trend').
            1 = bullish, -1 = bearish, 0 = neutral.
        bullish_count_column: Result column name for the count of
            rising EMAs (default: 'ema_ribbon_bullish_count').
        bearish_count_column: Result column name for the count of
            falling EMAs (default: 'ema_ribbon_bearish_count').
        ema_column_prefix: Prefix for individual EMA result columns
            (default: 'ema_ribbon'). Each EMA is stored as
            ``{prefix}_{period}``.

    Returns:
        DataFrame with added columns:
            - {ema_column_prefix}_{period}: Each individual EMA line
            - {bullish_count_column}: Number of rising EMAs at each bar
            - {bearish_count_column}: Number of falling EMAs at each bar
            - {trend_column}: Trend state (1 / -1 / 0)

    Example:
        >>> import pandas as pd
        >>> from pyindicators import ema_trend_ribbon
        >>> df = pd.DataFrame({
        ...     'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        ... })
        >>> result = ema_trend_ribbon(df, smoothing_period=1, threshold=5)
        >>> print(result[['ema_ribbon_trend']].tail())
    """
    if ema_lengths is None:
        ema_lengths = [8, 14, 20, 26, 32, 38, 44, 50, 60]

    # --- Validation -------------------------------------------------------
    if smoothing_period < 1:
        raise PyIndicatorException(
            "Smoothing period must be at least 1"
        )

    if threshold < 1:
        raise PyIndicatorException("Threshold must be at least 1")

    if threshold > len(ema_lengths):
        raise PyIndicatorException(
            f"Threshold ({threshold}) cannot exceed the number of "
            f"EMAs ({len(ema_lengths)})"
        )

    if len(ema_lengths) < 2:
        raise PyIndicatorException(
            "At least 2 EMA lengths are required"
        )

    for length in ema_lengths:
        if length < 1:
            raise PyIndicatorException(
                f"All EMA lengths must be at least 1, got {length}"
            )

    if source_column not in data.columns:
        raise PyIndicatorException(
            f"The column '{source_column}' does not exist "
            "in the DataFrame."
        )

    is_polars = isinstance(data, PlDataFrame)

    # --- Extract source array ---------------------------------------------
    if is_polars:
        src = data[source_column].to_numpy().astype(float)
    else:
        src = data[source_column].values.astype(float)

    n = len(src)

    # --- Compute all EMAs -------------------------------------------------
    ema_arrays = []

    for length in ema_lengths:
        ema_arrays.append(_calc_ema(src, length))

    # --- Slope counting ---------------------------------------------------
    bullish_count = np.zeros(n, dtype=int)
    bearish_count = np.zeros(n, dtype=int)

    for ema_vals in ema_arrays:
        for t in range(n):
            prev_idx = t - smoothing_period

            if prev_idx < 0:
                # Not enough history; treat as neutral
                continue

            if ema_vals[t] >= ema_vals[prev_idx]:
                bullish_count[t] += 1
            else:
                bearish_count[t] += 1

    # --- Trend classification ---------------------------------------------
    trend = np.zeros(n, dtype=int)

    for t in range(n):
        if bullish_count[t] >= threshold:
            trend[t] = 1
        elif bearish_count[t] >= threshold:
            trend[t] = -1

    # --- Write results back -----------------------------------------------
    if is_polars:
        new_cols = []

        for length, ema_vals in zip(ema_lengths, ema_arrays):
            col_name = f"{ema_column_prefix}_{length}"
            new_cols.append(pl.Series(name=col_name, values=ema_vals))

        new_cols.append(
            pl.Series(name=bullish_count_column,
                      values=bullish_count.tolist())
        )
        new_cols.append(
            pl.Series(name=bearish_count_column,
                      values=bearish_count.tolist())
        )
        new_cols.append(
            pl.Series(name=trend_column, values=trend.tolist())
        )
        data = data.with_columns(new_cols)
    else:
        for length, ema_vals in zip(ema_lengths, ema_arrays):
            col_name = f"{ema_column_prefix}_{length}"
            data[col_name] = ema_vals

        data[bullish_count_column] = bullish_count
        data[bearish_count_column] = bearish_count
        data[trend_column] = trend

    return data
