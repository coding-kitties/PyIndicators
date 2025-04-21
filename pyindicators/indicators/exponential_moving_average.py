from typing import Union
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
from pyindicators.exceptions import PyIndicatorException


def ema(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str,
    period: int,
    result_column: str = None,
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Function to calculate the Exponential Moving Average (EMA) of a series.

    Args:
        data (Union[PdDataFrame, PlDataFrame]): The input data.
        source_column (str): The name of the series.
        period (int): The period for the exponential moving average.
        result_column (str, optional): The name of the column to store the
            exponential moving average. Defaults to None.

    Returns:
        Union[PdDataFrame, PlDataFrame]: Returns a DataFrame with
            the EMA of the series.
    """

    if len(data) < period:
        raise PyIndicatorException(
            "The data must be larger than the period " +
            f"{period} to calculate the EMA. The data " +
            f"only contains {len(data)} data points."
        )

    # Check if source_column exists in the DataFrame
    if source_column not in data.columns:
        raise PyIndicatorException(
            f"The source column '{source_column}' does not "
            "exist in the DataFrame."
        )

    if result_column is None:
        result_column = f"ema_{source_column}_{period}"

    if isinstance(data, PdDataFrame):
        data[result_column] = data[source_column]\
            .ewm(span=period, adjust=False).mean()
    else:
        # Polars does not have a direct EWM function,
        # so we implement it manually
        alpha = 2 / (period + 1)
        ema_values = []
        ema_prev = data[source_column][0]  # Initialize with the first value

        for price in data[source_column]:
            ema_current = (price * alpha) + (ema_prev * (1 - alpha))
            ema_values.append(ema_current)
            ema_prev = ema_current

        data = data.with_columns(pl.Series(result_column, ema_values))

    return data
