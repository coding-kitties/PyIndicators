import pandas as pd
import polars as pl
from typing import Union
from pyindicators.exceptions import PyIndicatorException

from .utils import pad_zero_values_pandas, pad_zero_values_polars


def adx(
    data: Union[pd.DataFrame, pl.DataFrame],
    period=14,
    high_column="High",
    low_column="Low",
    close_column="Close",
    result_adx_column="ADX",
    result_pdi_column="+DI",
    result_ndi_column="-DI",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Calculate the Average Directional Index (ADX) for a given DataFrame.

    Args:
        data (Union[pd.DataFrame, pl.DataFrame]): Input data containing
            the price series.
        period (int, optional): Period for the ADX calculation (default: 14).
        high_column (str, optional): Column name for the high price series.
        low_column (str, optional): Column name for the low price series.
        close_column (str, optional): Column name for the close price series.
        result_adx_column (str, optional): Column name to store the ADX.
        result_pdi_column (str, optional): Column name to store the +DI.
        result_ndi_column (str, optional): Column name to store the -DI.

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: DataFrame with ADX, +DI, and -DI.
    """

    # Check if the high, low, and close columns are in the DataFrame
    if high_column not in data.columns:
        raise PyIndicatorException(
            f"Column '{high_column}' not found in DataFrame"
        )

    if low_column not in data.columns:
        raise PyIndicatorException(
            f"Column '{low_column}' not found in DataFrame"
        )

    if close_column not in data.columns:
        raise PyIndicatorException(
            f"Column '{close_column}' not found in DataFrame"
        )

    if isinstance(data, pd.DataFrame):
        # Pandas version of the ADX calculation
        high = data[high_column]
        low = data[low_column]
        close = data[close_column]

        # Calculate True Range (TR)
        tr = pd.DataFrame({
            'TR': pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
        })

        # Calculate Directional Movement (+DM and -DM)
        plus_dm = pd.DataFrame(
            {'+DM': (high.diff() > low.diff()).astype(int)
                * (high.diff().clip(lower=0))}
        )
        minus_dm = pd.DataFrame(
            {'-DM': (low.diff() > high.diff()).astype(int)
                * (-low.diff().clip(upper=0))}
        )

        # Smooth the TR, +DM, and -DM over the period
        tr_smooth = tr['TR'].rolling(window=period).mean()
        plus_dm_smooth = plus_dm['+DM'].rolling(window=period).mean()
        minus_dm_smooth = minus_dm['-DM'].rolling(window=period).mean()

        # Calculate +DI and -DI
        pdi = 100 * (plus_dm_smooth / tr_smooth)
        ndi = 100 * (minus_dm_smooth / tr_smooth)

        # Smooth the difference to get ADX
        adx = pd.DataFrame({
            result_adx_column: (pdi - ndi).abs().rolling(window=period).mean()
        })

        # Add columns to the original dataframe
        data[result_adx_column] = adx
        data[result_pdi_column] = pdi
        data[result_ndi_column] = ndi

        pad_zero_values_pandas(data, result_adx_column, period)
        pad_zero_values_pandas(data, result_pdi_column, period - 1)
        pad_zero_values_pandas(data, result_ndi_column, period - 1)
        return data

    elif isinstance(data, pl.DataFrame):
        # Polars version of the ADX calculation
        high = data[high_column]
        low = data[low_column]
        close = data[close_column]

        # Calculate True Range (TR)
        tr = pl.max_horizontal([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ])

        # Calculate Directional Movement (+DM and -DM)
        plus_dm = high.diff().clip_min(0)
        minus_dm = (-low.diff()).clip_min(0).abs()

        # Smooth the TR, +DM, and -DM over the period
        # (use rolling sum, not mean)
        tr_smooth = tr.rolling_sum(window_size=period, min_periods=1)
        plus_dm_smooth = plus_dm.rolling_sum(window_size=period, min_periods=1)
        minus_dm_smooth = minus_dm.rolling_sum(
            window_size=period, min_periods=1
        )

        # Calculate +DI and -DI
        pdi = 100 * (plus_dm_smooth / tr_smooth)
        ndi = 100 * (minus_dm_smooth / tr_smooth)

        # Calculate ADX (average of the absolute difference
        # between +DI and -DI)

        di_diff = (pdi - ndi).abs()
        # Smooth the difference to get ADX
        adx = di_diff.rolling_mean(window_size=period)

        # Add columns to the original dataframe
        data = data.with_columns([
            adx.alias(result_adx_column),
            pdi.alias(result_pdi_column),
            ndi.alias(result_ndi_column)
        ])

        # Pad the first `period` rows with zero values
        data = pad_zero_values_polars(data, result_adx_column, period)
        data = pad_zero_values_polars(data, result_pdi_column, period - 1)
        data = pad_zero_values_polars(data, result_ndi_column, period - 1)

        return data
    else:
        raise PyIndicatorException(
            "Input data must be either a pandas or polars DataFrame."
        )


def adx_v2(
    data: Union[pd.DataFrame, pl.DataFrame],
    period=14,
    high_column="High",
    low_column="Low",
    close_column="Close",
    result_adx_column="ADX",
    result_pdi_column="+DI",
    result_ndi_column="-DI",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Calculate the Average Directional Index (ADX) using Wilder's smoothing.
    Matches Tulipy's ADX calculation.

    Args:
        data: Input DataFrame (Pandas or Polars).
        period: Period for the ADX calculation (default: 14).
        high_column, low_column, close_column: Column names for price data.
            result_adx_column, result_pdi_column,
            result_ndi_column: Output column names.

    Returns:
        DataFrame with ADX, +DI, and -DI.
    """
    if high_column not in data.columns \
            or low_column not in data.columns \
            or close_column not in data.columns:
        raise PyIndicatorException(
            "High, Low, or Close column not found in DataFrame."
        )

    if isinstance(data, pd.DataFrame):
        # Pandas version
        high, low, close = data[high_column], data[low_column], \
            data[close_column]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)

        # Wilder’s smoothing with EMA
        tr_smooth = tr.ewm(span=period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()

        pdi = 100 * (plus_dm_smooth / tr_smooth)
        ndi = 100 * (minus_dm_smooth / tr_smooth)
        adx = (100 * (pdi - ndi).abs().ewm(span=period, adjust=False).mean())

        # Add results to DataFrame
        data[result_adx_column] = adx
        data[result_pdi_column] = pdi
        data[result_ndi_column] = ndi

        # Pad with zeros
        pad_zero_values_pandas(data, result_adx_column, period)
        pad_zero_values_pandas(data, result_pdi_column, period - 1)
        pad_zero_values_pandas(data, result_ndi_column, period - 1)

        return data

    elif isinstance(data, pl.DataFrame):
        # Polars version
        high, low, close = data[high_column], data[low_column], \
            data[close_column]

        tr = pl.max_horizontal([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ])

        plus_dm = high.diff().clip_min(0)
        minus_dm = (-low.diff()).clip_min(0).abs()

        # Wilder’s smoothing (manual EMA for Polars)
        def wilder_ema(series, period):
            alpha = 1 / period
            return series.cumsum() * alpha

        tr_smooth = wilder_ema(tr, period)
        plus_dm_smooth = wilder_ema(plus_dm, period)
        minus_dm_smooth = wilder_ema(minus_dm, period)

        pdi = 100 * (plus_dm_smooth / tr_smooth)
        ndi = 100 * (minus_dm_smooth / tr_smooth)
        adx = (100 * (pdi - ndi).abs()).cumsum() / period

        # Add results to DataFrame
        data = data.with_columns([
            adx.alias(result_adx_column),
            pdi.alias(result_pdi_column),
            ndi.alias(result_ndi_column)
        ])

        # Pad with zeros
        data = pad_zero_values_polars(data, result_adx_column, period)
        data = pad_zero_values_polars(data, result_pdi_column, period - 1)
        data = pad_zero_values_polars(data, result_ndi_column, period - 1)

        return data

    else:
        raise PyIndicatorException(
            "Input data must be either a pandas or polars DataFrame."
        )


def di(
    data: Union[pd.DataFrame, pl.DataFrame],
    period=14,
    high_column="High",
    low_column="Low",
    close_column="Close",
    result_pdi_column="+DI",
    result_ndi_column="-DI",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Calculate the +DI and -DI indicators exactly like Tulipy,
        supporting both Pandas and Polars.

    Args:
        data (Union[pd.DataFrame, pl.DataFrame]): Input data
            containing the price series.
        period (int, optional): Period for the DI calculation (default: 14).
        high_column (str, optional): Column name for the high price series.
        low_column (str, optional): Column name for the low price series.
        close_column (str, optional): Column name for the close price series.
        result_pdi_column (str, optional): Column name to store the +DI.
        result_ndi_column (str, optional): Column name to store the -DI.

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: DataFrame with +DI and -DI.
    """

    if isinstance(data, pd.DataFrame):
        high = data[high_column]
        low = data[low_column]
        close = data[close_column]

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = (
            (high.diff() > low.shift(1) - low) & (high.diff() > 0)
        ) * high.diff()
        minus_dm = (
            (low.shift(1) - low > high.diff()) & (low.shift(1) - low > 0)
        ) * (low.shift(1) - low)

        # Smoothed values
        tr_smooth = tr.rolling(window=period).sum()
        plus_dm_smooth = plus_dm.rolling(window=period).sum()
        minus_dm_smooth = minus_dm.rolling(window=period).sum()

        # Calculate +DI and -DI
        pdi = 100 * (plus_dm_smooth / tr_smooth)
        ndi = 100 * (minus_dm_smooth / tr_smooth)

        # Add to DataFrame
        data[result_pdi_column] = pdi
        data[result_ndi_column] = ndi

        # Pad initial values with zero
        # (replace NaN values for first `period-1` rows)
        data[result_pdi_column].iloc[:period-1] = 0
        data[result_ndi_column].iloc[:period-1] = 0

        return data

    elif isinstance(data, pl.DataFrame):
        high = data[high_column]
        low = data[low_column]
        close = data[close_column]

        # True Range
        tr = pl.max_horizontal([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ])

        # Directional Movement
        plus_dm = (high.diff() > low.shift(1) - low) & (high.diff() > 0)
        plus_dm = plus_dm * high.diff()

        minus_dm = (
            low.shift(1) - low > high.diff()
        ) & (low.shift(1) - low > 0)
        minus_dm = minus_dm * (low.shift(1) - low)

        # Smoothed values
        tr_smooth = tr.rolling_sum(window_size=period)
        plus_dm_smooth = plus_dm.rolling_sum(window_size=period)
        minus_dm_smooth = minus_dm.rolling_sum(window_size=period)

        # Calculate +DI and -DI
        pdi = 100 * (plus_dm_smooth / tr_smooth)
        ndi = 100 * (minus_dm_smooth / tr_smooth)

        # Add to DataFrame
        data = data.with_columns([
            pdi.alias(result_pdi_column),
            ndi.alias(result_ndi_column)
        ])

        # Pad initial values with zero
        # (replace NaN values for first `period-1` rows)
        data = data.with_columns([
            pl.when(pl.col(result_pdi_column).is_null()).then(0)
            .otherwise(pl.col(result_pdi_column)).alias(result_pdi_column),
            pl.when(pl.col(result_ndi_column).is_null()).then(0)
            .otherwise(pl.col(result_ndi_column)).alias(result_ndi_column)
        ])

        return data

    else:
        raise ValueError(
            "Input data must be either a pandas or polars DataFrame."
        )
