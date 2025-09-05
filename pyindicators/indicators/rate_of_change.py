from typing import Union
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
from pyindicators.exceptions import PyIndicatorException


def roc(
    data: Union[PdDataFrame, PlDataFrame],
    source_column='Close',
    period=14,
    result_column='ROC'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the Rate of Change (ROC) for a price series.

    Args:
        data: Input DataFrame (pandas or polars).
        source_column: Name of the column with price data.
        period: Lookback period for ROC calculation.
        result_column: Name of the result column to store ROC values.

    Returns:
        DataFrame with a new column for ROC.
    """
    if isinstance(data, PdDataFrame):
        # Calculate ROC for pandas DataFrame
        data[result_column] = data[source_column]\
            .pct_change(periods=period) * 100
        return data

    elif isinstance(data, PlDataFrame):
        # Calculate ROC for polars DataFrame
        return data.with_columns(
            (
                pl.col(source_column).pct_change(periods=period) * 100
            ).alias(result_column)
        )

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )
