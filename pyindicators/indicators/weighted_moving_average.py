from typing import Union, Optional

import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import pandas as pd
import polars as pl

from pyindicators.exceptions import PyIndicatorException


def wma(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str,
    period: int,
    result_column: Optional[str] = None,
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Function to calculate the weighted moving average of a series.

    Args:
        data (Union[PdDataFrame, PlDataFrame]): The input data.
        source_column (str): The name of the series.
        period (int): The period for the simple moving average.
        result_column (str, optional): The name of the column to store the
            simple moving average. Defaults to None.

    Returns:
        Union[PdDataFrame, PlDataFrame]: Returns a DataFrame
            with the weighted moving average of the series.
    """
    if len(data) < period:
        raise PyIndicatorException(
            "The data must be larger than the period " +
            f"{period} to calculate the WMA. The data " +
            f"only contains {len(data)} data points."
        )
    if result_column is None:
        result_column = f"wma_{period}"

    weights = np.arange(1, period + 1)

    if isinstance(data, pd.DataFrame):
        if source_column not in data.columns:
            raise PyIndicatorException(
                f"Column '{source_column}' not found in DataFrame"
            )

        data[result_column] = (
            data[source_column]
            .rolling(window=period)
            .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        )
        return data

    elif isinstance(data, pl.DataFrame):
        if source_column not in data.columns:
            raise PyIndicatorException(
                f"Column '{source_column}' not found in DataFrame"
            )

        wma_values = (
            data[source_column]
            .rolling_mean(window_size=period, weights=weights.tolist())
        )

        data = data.with_columns(pl.Series(result_column, wma_values))
        return data

    else:
        raise PyIndicatorException(
            "Unsupported DataFrame type. Use Pandas or Polars."
        )
