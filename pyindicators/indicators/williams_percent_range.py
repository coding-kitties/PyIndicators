from typing import Union
import pandas as pd
import polars as pl
from pyindicators.exceptions import PyIndicatorException


def willr(
    data: Union[pd.DataFrame, pl.DataFrame],
    period: int = 14,
    result_column: str = None,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close"
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Function to calculate the Williams %R indicator of a series.

    Args:
        data (Union[pd.DataFrame, pl.DataFrame]): The input data.
        source_column (str): The name of the series.
        period (int): The period for the Williams %R calculation.
        result_column (str, optional): The name of the column to store
        the Williams %R values. Defaults to None, which means it will
        be named "WilliamsR_{period}".

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: The DataFrame with
        the Williams %R column added.
    """

    # Check if the high and low columns are present
    if high_column not in data.columns:
        raise PyIndicatorException(
            f"Column '{high_column}' not found in DataFrame"
        )

    if low_column not in data.columns:
        raise PyIndicatorException(
            f"Column '{low_column}' not found in DataFrame"
        )

    if isinstance(data, pd.DataFrame):
        data["high_n"] = data[high_column]\
            .rolling(window=period, min_periods=1).max()
        data["low_n"] = data[low_column]\
            .rolling(window=period, min_periods=1).min()
        data[result_column] = ((data["high_n"] - data[close_column]) /
                               (data["high_n"] - data["low_n"])) * -100
        return data.drop(columns=["high_n", "low_n"])

    elif isinstance(data, pl.DataFrame):
        high_n = data.select(pl.col(high_column).rolling_max(period).alias("high_n"))
        low_n = data.select(pl.col(low_column).rolling_min(period).alias("low_n"))

        data = data.with_columns([
            high_n["high_n"],
            low_n["low_n"]
        ])
        data = data.with_columns(
            ((pl.col("high_n") - pl.col(close_column)) /
             (pl.col("high_n") - pl.col("low_n")) * -100).alias(result_column)
        )
        return data.drop(["high_n", "low_n"])

    else:
        raise PyIndicatorException(
            "Unsupported data type. Must be pandas or polars DataFrame."
        )
