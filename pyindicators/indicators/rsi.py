from typing import Union
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException

def rsi(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str,
    period: int,
    result_column: str = None,
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Function to calculate the RSI of a series.

    Args:
        data (Union[PdDataFrame, PlDataFrame]): The input data.
        source_column (str): The name of the series.
        period (int): The period for the exponential moving average.
        result_column (str, optional): The name of the column to store the
            exponential moving average. Defaults to None.

    Returns:
        Union[PdDataFrame, PlDataFrame]: Returns a DataFrame with
            the RSI of the series.
    """

    if result_column is None:
        result_column = f"RSI_{period}"

    if source_column not in data.columns:
        raise PyIndicatorException(
            f"The column {source_column} does not exist in the DataFrame."
        )

    if isinstance(data, PdDataFrame):
        # Compute price changes
        delta = data[source_column].diff()

        # Compute gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Compute the rolling average of gains and losses
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        # Compute RSI
        rs = avg_gain / avg_loss
        data[result_column] = 100 - (100 / (1 + rs))

    elif isinstance(data, PlDataFrame):
        # Compute price changes
        delta = data[source_column].diff().fill_null(0)

        # Compute gains and losses
        gain = delta.clip_min(0)
        loss = (-delta).clip_min(0)

        # Compute rolling averages of gains and losses
        avg_gain = gain.rolling_mean(window_size=period)
        avg_loss = loss.rolling_mean(window_size=period)

        # Compute RSI
        rs = avg_gain / avg_loss
        rsi_values = 100 - (100 / (1 + rs))

        # Add column to DataFrame
        data = data.with_columns(rsi_values.alias(result_column))

    else:
        raise TypeError("Input data must be a pandas or polars DataFrame.")

    return data
