from typing import Union, List, Optional
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
from pyindicators.exceptions import PyIndicatorException


# Standard Fibonacci retracement levels
FIBONACCI_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


def fibonacci_retracement(
    data: Union[PdDataFrame, PlDataFrame],
    high_column: str = 'High',
    low_column: str = 'Low',
    levels: Optional[List[float]] = None,
    lookback_period: Optional[int] = None,
    swing_high: Optional[float] = None,
    swing_low: Optional[float] = None,
    result_prefix: str = 'fib'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate Fibonacci retracement levels for a price series.

    Fibonacci retracement levels are horizontal lines that indicate where
    support and resistance are likely to occur. They are based on Fibonacci
    numbers and are drawn between a swing high and swing low.

    The standard levels are:
        - 0.0 (0%) - Swing High
        - 0.236 (23.6%)
        - 0.382 (38.2%)
        - 0.5 (50%)
        - 0.618 (61.8%) - Golden Ratio
        - 0.786 (78.6%)
        - 1.0 (100%) - Swing Low

    Calculation:
        Level Price = Swing High - (Swing High - Swing Low) × Fibonacci Ratio

    Args:
        data: pandas or polars DataFrame with price data
        high_column: Column name for high prices (default: 'High')
        low_column: Column name for low prices (default: 'Low')
        levels: Custom Fibonacci levels (default: standard levels)
        lookback_period: Period to look back for swing high/low.
            If None, uses entire dataset.
        swing_high: Manual swing high value.
            If provided, overrides calculation.
        swing_low: Manual swing low value.
            If provided, overrides calculation.
        result_prefix: Prefix for result columns (default: 'fib')

    Returns:
        DataFrame with added columns for each Fibonacci level.
        Column names follow pattern: {prefix}_{level}
        (e.g., fib_0.0, fib_0.618)
            (e.g., fib_0.0, fib_0.618)
    """
    if levels is None:
        levels = FIBONACCI_LEVELS

    if isinstance(data, PdDataFrame):
        # Determine swing high and low
        if swing_high is None or swing_low is None:
            if lookback_period is not None:
                high_val = data[high_column].iloc[-lookback_period:].max()
                low_val = data[low_column].iloc[-lookback_period:].min()
            else:
                high_val = data[high_column].max()
                low_val = data[low_column].min()

            if swing_high is None:
                swing_high = high_val
            if swing_low is None:
                swing_low = low_val

        # Calculate range
        price_range = swing_high - swing_low

        # Calculate each Fibonacci level
        for level in levels:
            level_price = swing_high - (price_range * level)
            col_name = f"{result_prefix}_{level}"
            data[col_name] = level_price

        # Also store the swing high and low used
        data[f"{result_prefix}_swing_high"] = swing_high
        data[f"{result_prefix}_swing_low"] = swing_low

        return data

    elif isinstance(data, PlDataFrame):
        # Determine swing high and low
        if swing_high is None or swing_low is None:
            if lookback_period is not None:
                high_val = data[high_column].tail(lookback_period).max()
                low_val = data[low_column].tail(lookback_period).min()
            else:
                high_val = data[high_column].max()
                low_val = data[low_column].min()

            if swing_high is None:
                swing_high = high_val
            if swing_low is None:
                swing_low = low_val

        # Calculate range
        price_range = swing_high - swing_low

        # Build column expressions
        new_columns = []
        for level in levels:
            level_price = swing_high - (price_range * level)
            col_name = f"{result_prefix}_{level}"
            new_columns.append(pl.lit(level_price).alias(col_name))

        # Add swing high and low columns
        new_columns.append(
            pl.lit(swing_high).alias(f"{result_prefix}_swing_high")
        )
        new_columns.append(
            pl.lit(swing_low).alias(f"{result_prefix}_swing_low")
        )

        return data.with_columns(new_columns)

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def fibonacci_retracement_levels(
    swing_high: float,
    swing_low: float,
    levels: Optional[List[float]] = None
) -> dict:
    """
    Calculate Fibonacci retracement price levels given swing high and low.

    This is a utility function that returns just the level prices without
    modifying a DataFrame.

    Args:
        swing_high: The swing high price
        swing_low: The swing low price
        levels: Custom Fibonacci levels (default: standard levels)

    Returns:
        Dictionary mapping level ratios to price levels.
        Example: {0.0: 5698.75, 0.236: 5130.15, ..., 1.0: 3613.15}
    """
    if levels is None:
        levels = FIBONACCI_LEVELS

    price_range = swing_high - swing_low

    return {
        level: swing_high - (price_range * level)
        for level in levels
    }


def fibonacci_extension(
    data: Union[PdDataFrame, PlDataFrame],
    high_column: str = 'High',
    low_column: str = 'Low',
    levels: Optional[List[float]] = None,
    lookback_period: Optional[int] = None,
    swing_high: Optional[float] = None,
    swing_low: Optional[float] = None,
    result_prefix: str = 'fib_ext'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate Fibonacci extension levels for a price series.

    Fibonacci extensions are used to determine potential price targets
    beyond the original move. They project levels above the swing high
    (for uptrends) or below the swing low (for downtrends).

    Common extension levels:
        - 1.0 (100%) - Full retracement
        - 1.272 (127.2%)
        - 1.414 (141.4%)
        - 1.618 (161.8%) - Golden Ratio Extension
        - 2.0 (200%)
        - 2.618 (261.8%)

    Args:
        data: pandas or polars DataFrame with price data
        high_column: Column name for high prices (default: 'High')
        low_column: Column name for low prices (default: 'Low')
        levels: Custom extension levels (default: common extension levels)
        lookback_period: Period to look back for swing high/low.
        swing_high: Manual swing high value.
        swing_low: Manual swing low value.
        result_prefix: Prefix for result columns (default: 'fib_ext')

    Returns:
        DataFrame with added columns for each Fibonacci extension level.
    """
    if levels is None:
        levels = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]

    if isinstance(data, PdDataFrame):
        if swing_high is None or swing_low is None:
            if lookback_period is not None:
                high_val = data[high_column].iloc[-lookback_period:].max()
                low_val = data[low_column].iloc[-lookback_period:].min()
            else:
                high_val = data[high_column].max()
                low_val = data[low_column].min()

            if swing_high is None:
                swing_high = high_val
            if swing_low is None:
                swing_low = low_val

        price_range = swing_high - swing_low

        for level in levels:
            # Extension above swing high
            level_price = swing_high + (price_range * (level - 1.0))
            col_name = f"{result_prefix}_{level}"
            data[col_name] = level_price

        return data

    elif isinstance(data, PlDataFrame):
        if swing_high is None or swing_low is None:
            if lookback_period is not None:
                high_val = data[high_column].tail(lookback_period).max()
                low_val = data[low_column].tail(lookback_period).min()
            else:
                high_val = data[high_column].max()
                low_val = data[low_column].min()

            if swing_high is None:
                swing_high = high_val
            if swing_low is None:
                swing_low = low_val

        price_range = swing_high - swing_low

        new_columns = []
        for level in levels:
            level_price = swing_high + (price_range * (level - 1.0))
            col_name = f"{result_prefix}_{level}"
            new_columns.append(pl.lit(level_price).alias(col_name))

        return data.with_columns(new_columns)

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )
