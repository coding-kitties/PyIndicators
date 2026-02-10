from typing import Union
import math
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
from pyindicators.exceptions import PyIndicatorException


def _gauss(x: float, h: float) -> float:
    """Gaussian kernel function."""
    return math.exp(-(x * x) / (h * h * 2))


def nadaraya_watson_envelope(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str = 'Close',
    bandwidth: float = 8.0,
    mult: float = 3.0,
    lookback: int = 500,
    upper_column: str = 'nwe_upper',
    lower_column: str = 'nwe_lower',
    middle_column: str = 'nwe_middle',
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the Nadaraya-Watson Envelope indicator.

    Uses Gaussian kernel regression to create a smoothed price estimate,
    then adds an envelope based on the mean absolute error (MAE) scaled
    by a multiplier. This is a non-repainting (endpoint) implementation.

    Based on the TradingView "Nadaraya-Watson Envelope [LuxAlgo]" indicator.

    Calculation:
        - Kernel weights: w(i) = exp(-i² / (2 * h²)) for i = 0..lookback-1
        - Smoothed value: sum(src[t-i] * w(i)) / sum(w(i))
        - MAE: SMA of |src - smoothed| over lookback period
        - Upper: smoothed + mult * MAE
        - Lower: smoothed - mult * MAE

    Args:
        data: pandas or polars DataFrame with price data
        source_column: Column name for the source prices
            (default: 'Close')
        bandwidth: Gaussian kernel bandwidth / smoothing factor
            (default: 8.0). Higher values produce smoother curves.
        mult: Multiplier for the MAE envelope width (default: 3.0)
        lookback: Number of bars to use for kernel regression
            (default: 500)
        upper_column: Result column name for upper envelope
            (default: 'nwe_upper')
        lower_column: Result column name for lower envelope
            (default: 'nwe_lower')
        middle_column: Result column name for the smoothed line
            (default: 'nwe_middle')

    Returns:
        DataFrame with added columns:
            - {upper_column}: Upper boundary of the envelope
            - {lower_column}: Lower boundary of the envelope
            - {middle_column}: Nadaraya-Watson smoothed estimate

    Example:
        >>> import pandas as pd
        >>> from pyindicators import nadaraya_watson_envelope
        >>> df = pd.DataFrame({
        ...     'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        ... })
        >>> result = nadaraya_watson_envelope(df, bandwidth=3, lookback=5)
        >>> print(result[['nwe_upper', 'nwe_middle', 'nwe_lower']].tail())
    """
    if bandwidth <= 0:
        raise PyIndicatorException("Bandwidth must be greater than 0")

    if mult < 0:
        raise PyIndicatorException("Multiplier must be non-negative")

    if lookback < 1:
        raise PyIndicatorException("Lookback must be at least 1")

    if source_column not in data.columns:
        raise PyIndicatorException(
            f"The source column '{source_column}' does not "
            "exist in the DataFrame."
        )

    is_polars = isinstance(data, PlDataFrame)

    if is_polars:
        src = data[source_column].to_numpy().astype(float)
    else:
        src = data[source_column].values.astype(float)

    n = len(src)

    # Pre-compute Gaussian kernel weights
    max_k = min(lookback, n)
    weights = np.array([_gauss(i, bandwidth) for i in range(max_k)])

    # Compute the smoothed (endpoint) estimate for each bar
    smoothed = np.full(n, np.nan)

    for t in range(n):
        # Number of available past bars (including current)
        available = min(t + 1, max_k)
        w = weights[:available]
        s = src[t - available + 1: t + 1][::-1]  # most recent first
        smoothed[t] = np.nansum(s * w) / np.nansum(w)

    # Compute MAE (mean absolute error) using a rolling window
    abs_err = np.abs(src - smoothed)
    mae = np.full(n, np.nan)

    for t in range(n):
        window = min(t + 1, max_k)
        mae[t] = np.nanmean(abs_err[t - window + 1: t + 1])

    upper = smoothed + mult * mae
    lower = smoothed - mult * mae

    if is_polars:
        data = data.with_columns([
            pl.Series(name=middle_column, values=smoothed),
            pl.Series(name=upper_column, values=upper),
            pl.Series(name=lower_column, values=lower),
        ])
    else:
        data[middle_column] = smoothed
        data[upper_column] = upper
        data[lower_column] = lower

    return data
