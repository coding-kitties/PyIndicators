from typing import Union
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
import numpy as np
from pyindicators.exceptions import PyIndicatorException


def fair_value_gap(
    data: Union[PdDataFrame, PlDataFrame],
    high_column: str = 'High',
    low_column: str = 'Low',
    close_column: str = 'Close',
    threshold_pct: float = 0.0,
    bullish_fvg_column: str = 'bullish_fvg',
    bearish_fvg_column: str = 'bearish_fvg',
    bullish_fvg_top_column: str = 'bullish_fvg_top',
    bullish_fvg_bottom_column: str = 'bullish_fvg_bottom',
    bearish_fvg_top_column: str = 'bearish_fvg_top',
    bearish_fvg_bottom_column: str = 'bearish_fvg_bottom'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Identify Fair Value Gaps (FVG) in price data.

    A Fair Value Gap is a price imbalance that occurs when there's a gap
    between candlesticks, representing institutional order flow. These gaps
    often act as support/resistance zones where price tends to return.

    This implementation follows the LuxAlgo methodology which includes:
    - A threshold filter to ignore small gaps
    - Close price confirmation (close[1] must confirm the gap direction)

    **Bullish FVG (Gap Up):**
        Occurs when:
        - The low of the current candle > high of candle 2 bars ago
        - The close of the previous candle > high of candle 2 bars ago
        - The gap size exceeds the threshold percentage

    **Bearish FVG (Gap Down):**
        Occurs when:
        - The high of the current candle < low of candle 2 bars ago
        - The close of the previous candle < low of candle 2 bars ago
        - The gap size exceeds the threshold percentage

    The gap zone boundaries are:
        - Bullish FVG: Bottom = High[2 bars ago], Top = Low[current]
        - Bearish FVG: Top = Low[2 bars ago], Bottom = High[current]

    Args:
        data: pandas or polars DataFrame with OHLC price data
        high_column: Column name for high prices (default: 'High')
        low_column: Column name for low prices (default: 'Low')
        close_column: Column name for close prices (default: 'Close')
        threshold_pct: Minimum gap size as percentage (default: 0.0)
            Set to 0 to detect all gaps, or higher to filter small gaps
        bullish_fvg_column: Result column name for bullish FVG signal
            (default: 'bullish_fvg')
        bearish_fvg_column: Result column name for bearish FVG signal
            (default: 'bearish_fvg')
        bullish_fvg_top_column: Result column name for bullish FVG top
            boundary (default: 'bullish_fvg_top')
        bullish_fvg_bottom_column: Result column name for bullish FVG bottom
            boundary (default: 'bullish_fvg_bottom')
        bearish_fvg_top_column: Result column name for bearish FVG top
            boundary (default: 'bearish_fvg_top')
        bearish_fvg_bottom_column: Result column name for bearish FVG bottom
            boundary (default: 'bearish_fvg_bottom')

    Returns:
        DataFrame with added columns:
            - {bullish_fvg_column}: 1 if bullish FVG detected, 0 otherwise
            - {bearish_fvg_column}: 1 if bearish FVG detected, 0 otherwise
            - {bullish_fvg_top_column}: Top of bullish FVG zone (NaN if none)
            - {bullish_fvg_bottom_column}: Bottom of bullish FVG zone
            - {bearish_fvg_top_column}: Top of bearish FVG zone
            - {bearish_fvg_bottom_column}: Bottom of bearish FVG zone

    Example:
        >>> import pandas as pd
        >>> from pyindicators import fair_value_gap
        >>> df = pd.DataFrame({
        ...     'High': [100, 102, 106, 108, 105],
        ...     'Low': [98, 100, 104, 106, 103],
        ...     'Close': [99, 101, 105, 107, 104]
        ... })
        >>> result = fair_value_gap(df)
        >>> print(result[['bullish_fvg', 'bearish_fvg']])
    """
    threshold = threshold_pct / 100

    if isinstance(data, PdDataFrame):
        # Get shifted values
        high_2_ago = data[high_column].shift(2)
        low_2_ago = data[low_column].shift(2)
        close_1_ago = data[close_column].shift(1)
        current_high = data[high_column]
        current_low = data[low_column]

        # Bullish FVG: current low > high[2] AND close[1] > high[2]
        # AND gap size > threshold
        bull_gap_exists = (
            (current_low > high_2_ago) & (close_1_ago > high_2_ago)
        )
        bull_gap_size = (current_low - high_2_ago) / high_2_ago
        bullish_fvg = (
            bull_gap_exists & (bull_gap_size > threshold)
        ).astype(int)

        # Bearish FVG: current high < low[2] AND close[1] < low[2]
        # AND gap size > threshold
        bear_gap_exists = (
            (current_high < low_2_ago) & (close_1_ago < low_2_ago)
        )
        bear_gap_size = (low_2_ago - current_high) / current_high
        bearish_fvg = (
            bear_gap_exists & (bear_gap_size > threshold)
        ).astype(int)

        # Calculate FVG zone boundaries
        # Bullish: gap between high[2 ago] and low[current]
        bullish_top = np.where(bullish_fvg == 1, current_low, np.nan)
        bullish_bottom = np.where(bullish_fvg == 1, high_2_ago, np.nan)

        # Bearish: gap between low[2 ago] and high[current]
        bearish_top = np.where(bearish_fvg == 1, low_2_ago, np.nan)
        bearish_bottom = np.where(bearish_fvg == 1, current_high, np.nan)

        # Add columns to dataframe
        data[bullish_fvg_column] = bullish_fvg
        data[bearish_fvg_column] = bearish_fvg
        data[bullish_fvg_top_column] = bullish_top
        data[bullish_fvg_bottom_column] = bullish_bottom
        data[bearish_fvg_top_column] = bearish_top
        data[bearish_fvg_bottom_column] = bearish_bottom

        return data

    elif isinstance(data, PlDataFrame):
        # Get shifted values
        high_2_ago = pl.col(high_column).shift(2)
        low_2_ago = pl.col(low_column).shift(2)
        close_1_ago = pl.col(close_column).shift(1)
        current_high = pl.col(high_column)
        current_low = pl.col(low_column)

        # Bullish FVG condition with close confirmation and threshold
        bull_gap_exists = (
            (current_low > high_2_ago) & (close_1_ago > high_2_ago)
        )
        bull_gap_size = (current_low - high_2_ago) / high_2_ago
        bullish_cond = bull_gap_exists & (bull_gap_size > threshold)

        # Bearish FVG condition with close confirmation and threshold
        bear_gap_exists = (
            (current_high < low_2_ago) & (close_1_ago < low_2_ago)
        )
        bear_gap_size = (low_2_ago - current_high) / current_high
        bearish_cond = bear_gap_exists & (bear_gap_size > threshold)

        return data.with_columns([
            pl.when(bullish_cond).then(1).otherwise(0)
            .alias(bullish_fvg_column),
            pl.when(bearish_cond).then(1).otherwise(0)
            .alias(bearish_fvg_column),
            pl.when(bullish_cond).then(current_low).otherwise(None)
            .alias(bullish_fvg_top_column),
            pl.when(bullish_cond).then(high_2_ago).otherwise(None)
            .alias(bullish_fvg_bottom_column),
            pl.when(bearish_cond).then(low_2_ago).otherwise(None)
            .alias(bearish_fvg_top_column),
            pl.when(bearish_cond).then(current_high).otherwise(None)
            .alias(bearish_fvg_bottom_column),
        ])

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def fvg_signal(
    data: Union[PdDataFrame, PlDataFrame],
    close_column: str = 'Close',
    bullish_fvg_top_column: str = 'bullish_fvg_top',
    bullish_fvg_bottom_column: str = 'bullish_fvg_bottom',
    bearish_fvg_top_column: str = 'bearish_fvg_top',
    bearish_fvg_bottom_column: str = 'bearish_fvg_bottom',
    signal_column: str = 'fvg_signal'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate signals when price enters a Fair Value Gap zone.

    This function checks if the current close price is within any
    previously identified FVG zone. Price returning to an FVG zone
    is often considered a potential entry opportunity.

    Signal values:
        - 1: Price is within a bullish FVG zone (potential long entry)
        - -1: Price is within a bearish FVG zone (potential short entry)
        - 0: Price is not within any FVG zone

    Note: This function uses the most recent FVG zones. For tracking
    multiple unfilled FVGs, use fvg_zones() instead.

    Args:
        data: pandas or polars DataFrame with FVG columns calculated
        close_column: Column name for close prices (default: 'Close')
        bullish_fvg_top_column: Column name for bullish FVG top
            (default: 'bullish_fvg_top')
        bullish_fvg_bottom_column: Column name for bullish FVG bottom
            (default: 'bullish_fvg_bottom')
        bearish_fvg_top_column: Column name for bearish FVG top
            (default: 'bearish_fvg_top')
        bearish_fvg_bottom_column: Column name for bearish FVG bottom
            (default: 'bearish_fvg_bottom')
        signal_column: Result column name for signal (default: 'fvg_signal')

    Returns:
        DataFrame with added signal column

    Example:
        >>> import pandas as pd
        >>> from pyindicators import fair_value_gap, fvg_signal
        >>> df = pd.DataFrame({
        ...     'High': [100, 102, 106, 108, 105],
        ...     'Low': [98, 100, 104, 106, 103],
        ...     'Close': [99, 101, 105, 107, 104]
        ... })
        >>> df = fair_value_gap(df)
        >>> result = fvg_signal(df)
    """
    # Validate required columns exist
    required_columns = [
        bullish_fvg_top_column, bullish_fvg_bottom_column,
        bearish_fvg_top_column, bearish_fvg_bottom_column
    ]

    if isinstance(data, PdDataFrame):
        for col in required_columns:
            if col not in data.columns:
                raise PyIndicatorException(
                    f"Required column '{col}' not found. "
                    "Run fair_value_gap() first."
                )

        # Forward fill FVG zones to track them until filled
        bullish_top_ff = data[bullish_fvg_top_column].ffill()
        bullish_bottom_ff = data[bullish_fvg_bottom_column].ffill()
        bearish_top_ff = data[bearish_fvg_top_column].ffill()
        bearish_bottom_ff = data[bearish_fvg_bottom_column].ffill()

        close = data[close_column]

        # Check if price is within bullish FVG zone
        in_bullish_fvg = (
            (close >= bullish_bottom_ff) &
            (close <= bullish_top_ff)
        )

        # Check if price is within bearish FVG zone
        in_bearish_fvg = (
            (close >= bearish_bottom_ff) &
            (close <= bearish_top_ff)
        )

        # Generate signals
        signal = np.where(
            in_bullish_fvg, 1,
            np.where(in_bearish_fvg, -1, 0)
        )

        data[signal_column] = signal
        return data

    elif isinstance(data, PlDataFrame):
        for col in required_columns:
            if col not in data.columns:
                raise PyIndicatorException(
                    f"Required column '{col}' not found. "
                    "Run fair_value_gap() first."
                )

        close = pl.col(close_column)

        # Forward fill FVG zones
        bullish_top_ff = pl.col(bullish_fvg_top_column).forward_fill()
        bullish_bottom_ff = pl.col(bullish_fvg_bottom_column).forward_fill()
        bearish_top_ff = pl.col(bearish_fvg_top_column).forward_fill()
        bearish_bottom_ff = pl.col(bearish_fvg_bottom_column).forward_fill()

        # Check if price is within FVG zones
        in_bullish_fvg = (
            (close >= bullish_bottom_ff) & (close <= bullish_top_ff)
        )
        in_bearish_fvg = (
            (close >= bearish_bottom_ff) & (close <= bearish_top_ff)
        )

        return data.with_columns([
            pl.when(in_bullish_fvg).then(1)
            .when(in_bearish_fvg).then(-1)
            .otherwise(0)
            .alias(signal_column)
        ])

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def fvg_filled(
    data: Union[PdDataFrame, PlDataFrame],
    high_column: str = 'High',
    low_column: str = 'Low',
    bullish_fvg_column: str = 'bullish_fvg',
    bearish_fvg_column: str = 'bearish_fvg',
    bullish_fvg_top_column: str = 'bullish_fvg_top',
    bullish_fvg_bottom_column: str = 'bullish_fvg_bottom',
    bearish_fvg_top_column: str = 'bearish_fvg_top',
    bearish_fvg_bottom_column: str = 'bearish_fvg_bottom',
    bullish_filled_column: str = 'bullish_fvg_filled',
    bearish_filled_column: str = 'bearish_fvg_filled'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect when Fair Value Gaps have been filled (mitigated).

    An FVG is considered "filled" or "mitigated" when price returns to
    completely fill the gap:
        - Bullish FVG filled: Price drops to reach the bottom of the gap
        - Bearish FVG filled: Price rises to reach the top of the gap

    Args:
        data: pandas or polars DataFrame with FVG columns calculated
        high_column: Column name for high prices (default: 'High')
        low_column: Column name for low prices (default: 'Low')
        bullish_fvg_column: Column name for bullish FVG signal
        bearish_fvg_column: Column name for bearish FVG signal
        bullish_fvg_top_column: Column name for bullish FVG top
        bullish_fvg_bottom_column: Column name for bullish FVG bottom
        bearish_fvg_top_column: Column name for bearish FVG top
        bearish_fvg_bottom_column: Column name for bearish FVG bottom
        bullish_filled_column: Result column for bullish FVG filled signal
        bearish_filled_column: Result column for bearish FVG filled signal

    Returns:
        DataFrame with added columns indicating when FVGs are filled

    Example:
        >>> import pandas as pd
        >>> from pyindicators import fair_value_gap, fvg_filled
        >>> df = pd.DataFrame({
        ...     'High': [100, 102, 106, 108, 105, 104],
        ...     'Low': [98, 100, 104, 106, 103, 101]
        ... })
        >>> df = fair_value_gap(df)
        >>> result = fvg_filled(df)
    """
    if isinstance(data, PdDataFrame):
        # Forward fill FVG zones
        bullish_bottom_ff = data[bullish_fvg_bottom_column].ffill()
        bearish_top_ff = data[bearish_fvg_top_column].ffill()

        # Bullish FVG filled when low reaches the bottom of the gap
        bullish_filled = (data[low_column] <= bullish_bottom_ff).astype(int)

        # Bearish FVG filled when high reaches the top of the gap
        bearish_filled = (data[high_column] >= bearish_top_ff).astype(int)

        data[bullish_filled_column] = bullish_filled
        data[bearish_filled_column] = bearish_filled

        return data

    elif isinstance(data, PlDataFrame):
        # Forward fill FVG zones
        bullish_bottom_ff = pl.col(bullish_fvg_bottom_column).forward_fill()
        bearish_top_ff = pl.col(bearish_fvg_top_column).forward_fill()

        # Check if FVGs are filled
        bullish_filled = pl.col(low_column) <= bullish_bottom_ff
        bearish_filled = pl.col(high_column) >= bearish_top_ff

        return data.with_columns([
            pl.when(bullish_filled).then(1).otherwise(0)
            .alias(bullish_filled_column),
            pl.when(bearish_filled).then(1).otherwise(0)
            .alias(bearish_filled_column)
        ])

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )
