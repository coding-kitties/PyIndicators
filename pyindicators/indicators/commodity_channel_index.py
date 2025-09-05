from typing import Union
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
from pyindicators.exceptions import PyIndicatorException


def cci(
    data: Union[PdDataFrame, PlDataFrame],
    high_column='High',
    low_column='Low',
    close_column='Close',
    period=20,
    result_column='CCI'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the Commodity Channel Index (CCI) for a price series.

    Args:
        data: Input DataFrame (pandas or polars).
        high_column: Name of the column with high prices.
        low_column: Name of the column with low prices.
        close_column: Name of the column with close prices.
        period: Lookback period for CCI calculation.
        result_column: Name of the result column to store CCI values.

    Returns the original DataFrame with a new column for CCI.
    """
    if isinstance(data, PdDataFrame):
        # Calculate CCI for pandas DataFrame
        typical_price = (data[high_column] +
                         data[low_column] + data[close_column]) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = (typical_price - sma).abs().rolling(window=period).mean()
        data[result_column] = (typical_price - sma) / (0.015 * mad)
        return data

    elif isinstance(data, PlDataFrame):
        # Calculate CCI for polars DataFrame
        typical_price = (pl.col(high_column)
                         + pl.col(low_column)
                         + pl.col(close_column)) / 3
        sma = typical_price.rolling_mean(window_size=period)
        mad = (typical_price - sma).abs().rolling_mean(window_size=period)
        return data.with_columns(
            (typical_price - sma) / (0.015 * mad).alias(result_column)
        )

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )
