from typing import Union
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
from pyindicators.exceptions import PyIndicatorException


def sma(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str,
    period: int,
    result_column: str = None,
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Function to calculate the simple moving average of a series.

    Args:
        data (Union[PdDataFrame, PlDataFrame]): The input data.
        source_column (str): The name of the series.
        period (int): The period for the simple moving average.
        result_column (str, optional): The name of the column to store the
            simple moving average. Defaults to None.

    Returns:
        Union[PdDataFrame, PlDataFrame]: Returns a DataFrame
            with the simple moving average of the series.
    """

    if len(data) < period:
        raise PyIndicatorException(
            "The data must be larger than the period " +
            f"{period} to calculate the SMA. The data " +
            f"only contains {len(data)} data points."
        )

    if result_column is None:
        result_column = f"sma_{source_column}_{period}"

    if isinstance(data, PdDataFrame):
        data[result_column] = data[source_column].rolling(window=period).mean()
    else:
        data = data.with_column(
            data[source_column].rolling(window=period).mean(),
            result_column
        )

    return data
