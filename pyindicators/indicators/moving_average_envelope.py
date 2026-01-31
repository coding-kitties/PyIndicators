from typing import Union
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
from pyindicators.exceptions import PyIndicatorException


def moving_average_envelope(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str = 'Close',
    period: int = 20,
    percentage: float = 2.5,
    ma_type: str = 'sma',
    middle_column: str = 'ma_envelope_middle',
    upper_column: str = 'ma_envelope_upper',
    lower_column: str = 'ma_envelope_lower'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate Moving Average Envelope for a price series.

    Moving Average Envelopes are percentage-based envelopes set above and
    below a moving average. The moving average forms the base, and the
    envelopes are set at a fixed percentage above and below.

    This indicator is useful for:
        - Identifying overbought/oversold conditions
        - Spotting trend direction (price following the envelope)
        - Finding support and resistance levels

    Calculation:
        - Middle Band: Moving Average (SMA or EMA)
        - Upper Band: Middle × (1 + percentage/100)
        - Lower Band: Middle × (1 - percentage/100)

    Args:
        data: pandas or polars DataFrame with price data
        source_column: Column name for source prices (default: 'Close')
        period: Moving average period (default: 20)
        percentage: Envelope percentage above/below MA (default: 2.5%)
        ma_type: Type of moving average - 'sma' or 'ema' (default: 'sma')
        middle_column: Result column name for middle band
        upper_column: Result column name for upper band
        lower_column: Result column name for lower band

    Returns:
        DataFrame with added columns for middle, upper, and lower
        envelope bands.
    """
    envelope_mult_upper = 1 + (percentage / 100)
    envelope_mult_lower = 1 - (percentage / 100)

    if isinstance(data, PdDataFrame):
        if ma_type.lower() == 'sma':
            ma = data[source_column].rolling(window=period).mean()
        elif ma_type.lower() == 'ema':
            ma = data[source_column].ewm(span=period, adjust=False).mean()
        else:
            raise PyIndicatorException(
                f"Invalid ma_type '{ma_type}'. Use 'sma' or 'ema'."
            )

        data[middle_column] = ma
        data[upper_column] = ma * envelope_mult_upper
        data[lower_column] = ma * envelope_mult_lower

        return data

    elif isinstance(data, PlDataFrame):
        if ma_type.lower() == 'sma':
            ma = pl.col(source_column).rolling_mean(window_size=period)
        elif ma_type.lower() == 'ema':
            ma = pl.col(source_column).ewm_mean(span=period, adjust=False)
        else:
            raise PyIndicatorException(
                f"Invalid ma_type '{ma_type}'. Use 'sma' or 'ema'."
            )

        return data.with_columns([
            ma.alias(middle_column),
            (ma * envelope_mult_upper).alias(upper_column),
            (ma * envelope_mult_lower).alias(lower_column)
        ])

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def sma_envelope(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str = 'Close',
    period: int = 20,
    percentage: float = 2.5,
    middle_column: str = 'sma_envelope_middle',
    upper_column: str = 'sma_envelope_upper',
    lower_column: str = 'sma_envelope_lower'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate SMA Envelope (Simple Moving Average Envelope).

    Convenience function that calls moving_average_envelope with ma_type='sma'.

    Args:
        data: pandas or polars DataFrame with price data
        source_column: Column name for source prices (default: 'Close')
        period: SMA period (default: 20)
        percentage: Envelope percentage above/below SMA (default: 2.5%)
        middle_column: Result column name for middle band
        upper_column: Result column name for upper band
        lower_column: Result column name for lower band

    Returns:
        DataFrame with added envelope columns.
    """
    return moving_average_envelope(
        data=data,
        source_column=source_column,
        period=period,
        percentage=percentage,
        ma_type='sma',
        middle_column=middle_column,
        upper_column=upper_column,
        lower_column=lower_column
    )


def ema_envelope(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str = 'Close',
    period: int = 20,
    percentage: float = 2.5,
    middle_column: str = 'ema_envelope_middle',
    upper_column: str = 'ema_envelope_upper',
    lower_column: str = 'ema_envelope_lower'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate EMA Envelope (Exponential Moving Average Envelope).

    Convenience function that calls moving_average_envelope with ma_type='ema'.

    Args:
        data: pandas or polars DataFrame with price data
        source_column: Column name for source prices (default: 'Close')
        period: EMA period (default: 20)
        percentage: Envelope percentage above/below EMA (default: 2.5%)
        middle_column: Result column name for middle band
        upper_column: Result column name for upper band
        lower_column: Result column name for lower band

    Returns:
        DataFrame with added envelope columns.
    """
    return moving_average_envelope(
        data=data,
        source_column=source_column,
        period=period,
        percentage=percentage,
        ma_type='ema',
        middle_column=middle_column,
        upper_column=upper_column,
        lower_column=lower_column
    )
