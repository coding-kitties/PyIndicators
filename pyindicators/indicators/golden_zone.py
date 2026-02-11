from typing import Union
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
from pyindicators.exceptions import PyIndicatorException


def golden_zone(
    data: Union[PdDataFrame, PlDataFrame],
    high_column: str = 'High',
    low_column: str = 'Low',
    length: int = 60,
    retracement_level_1: float = 0.5,
    retracement_level_2: float = 0.618,
    upper_column: str = 'golden_zone_upper',
    lower_column: str = 'golden_zone_lower',
    hh_column: str = 'golden_zone_hh',
    ll_column: str = 'golden_zone_ll'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate Fibonacci Golden Zone levels based on the highest high
    and lowest low over a specified rolling period.

    The Golden Zone is the area between two Fibonacci retracement levels
    (typically 50% and 61.8%), which is often considered a key area for
    potential price reversals or continuations. This indicator plots
    dynamic support/resistance levels that update with each bar.

    Calculation:
        - Highest High (HH): Rolling maximum of high prices over `length` bars
        - Lowest Low (LL): Rolling minimum of low prices over `length` bars
        - Diff: HH - LL
        - Upper Level: HH - (Diff × retracement_level_1)
        - Lower Level: HH - (Diff × retracement_level_2)

    Args:
        data: pandas or polars DataFrame with OHLC price data
        high_column: Column name for high prices (default: 'High')
        low_column: Column name for low prices (default: 'Low')
        length: Number of bars to look back for highest high and
            lowest low (default: 60)
        retracement_level_1: First retracement level as decimal
            (default: 0.5 for 50%)
        retracement_level_2: Second retracement level as decimal
            (default: 0.618 for 61.8%)
        upper_column: Result column name for upper boundary of
            golden zone (default: 'golden_zone_upper')
        lower_column: Result column name for lower boundary of
            golden zone (default: 'golden_zone_lower')
        hh_column: Result column name for highest high
            (default: 'golden_zone_hh')
        ll_column: Result column name for lowest low
            (default: 'golden_zone_ll')

    Returns:
        DataFrame with added columns:
            - {upper_column}: Upper boundary of the golden zone
            - {lower_column}: Lower boundary of the golden zone
            - {hh_column}: Highest high over the rolling period
            - {ll_column}: Lowest low over the rolling period

    Example:
        >>> import pandas as pd
        >>> from pyindicators import golden_zone
        >>> df = pd.DataFrame({
        ...     'High': [105, 110, 108, 112, 115, 113, 117, 120, 118, 116],
        ...     'Low': [100, 105, 103, 107, 110, 108, 112, 115, 113, 111]
        ... })
        >>> result = golden_zone(df, length=5)
        >>> print(result[['golden_zone_upper', 'golden_zone_lower']].tail())
    """
    if length < 1:
        raise PyIndicatorException("Length must be at least 1")

    # Ensure upper level is the smaller retracement (closer to HH)
    level_upper = min(retracement_level_1, retracement_level_2)
    level_lower = max(retracement_level_1, retracement_level_2)

    if isinstance(data, PdDataFrame):
        # Calculate rolling highest high and lowest low
        hh = data[high_column].rolling(window=length, min_periods=1).max()
        ll = data[low_column].rolling(window=length, min_periods=1).min()

        # Calculate the difference
        diff = hh - ll

        # Calculate Fibonacci retracement levels
        fib_upper = hh - diff * level_upper
        fib_lower = hh - diff * level_lower

        # Add columns to dataframe
        data[upper_column] = fib_upper
        data[lower_column] = fib_lower
        data[hh_column] = hh
        data[ll_column] = ll

        return data

    elif isinstance(data, PlDataFrame):
        # Calculate rolling highest high and lowest low
        hh = pl.col(high_column).rolling_max(window_size=length, min_samples=1)
        ll = pl.col(low_column).rolling_min(window_size=length, min_samples=1)

        # Calculate the difference and Fibonacci levels
        diff = hh - ll
        fib_upper = hh - diff * level_upper
        fib_lower = hh - diff * level_lower

        return data.with_columns([
            fib_upper.alias(upper_column),
            fib_lower.alias(lower_column),
            hh.alias(hh_column),
            ll.alias(ll_column)
        ])

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def golden_zone_signal(
    data: Union[PdDataFrame, PlDataFrame],
    close_column: str = 'Close',
    upper_column: str = 'golden_zone_upper',
    lower_column: str = 'golden_zone_lower',
    signal_column: str = 'golden_zone_signal'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate signals based on price position relative to the Golden Zone.

    This function identifies when price enters, exits, or is within the
    Golden Zone, which can be used for trading signals.

    Signal values:
        - 1: Price is within the Golden Zone (potential support/resistance)
        - 0: Price is outside the Golden Zone

    Args:
        data: pandas or polars DataFrame with golden zone columns
            (must have golden_zone_upper and golden_zone_lower calculated)
        close_column: Column name for close prices (default: 'Close')
        upper_column: Column name for upper boundary
            (default: 'golden_zone_upper')
        lower_column: Column name for lower boundary
            (default: 'golden_zone_lower')
        signal_column: Result column name for signal
            (default: 'golden_zone_signal')

    Returns:
        DataFrame with added signal column indicating if price is in the zone.
    """
    if isinstance(data, PdDataFrame):
        if (upper_column not in data.columns or
                lower_column not in data.columns):
            raise PyIndicatorException(
                f"Golden zone columns '{upper_column}' and '{lower_column}' "
                "must be present. Run golden_zone() first."
            )

        close = data[close_column]
        upper = data[upper_column]
        lower = data[lower_column]

        # Price is in golden zone when between upper and lower boundaries
        data[signal_column] = ((close <= upper) & (close >= lower)).astype(int)

        return data

    elif isinstance(data, PlDataFrame):
        if (upper_column not in data.columns
                or lower_column not in data.columns):
            raise PyIndicatorException(
                f"Golden zone columns '{upper_column}' and '{lower_column}' "
                "must be present. Run golden_zone() first."
            )

        return data.with_columns([
            pl.when(
                (pl.col(close_column) <= pl.col(upper_column)) &
                (pl.col(close_column) >= pl.col(lower_column))
            ).then(1).otherwise(0).alias(signal_column)
        ])

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )
