from typing import Union
import pandas as pd
import polars as pl
from pyindicators.exceptions import PyIndicatorException


def willr(
    data: Union[pd.DataFrame, pl.DataFrame],
    period: int = 14,
    result_column: str = "willr",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close"
) -> Union[pd.DataFrame, pl.DataFrame]:

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

        data[result_column] = (
            (data["high_n"].squeeze() - data[close_column].squeeze()) /
            (data["high_n"].squeeze() - data["low_n"].squeeze())
        ).squeeze() * -100

        # Set the first `period` rows to 0 using .iloc
        if not data.empty:
            data.iloc[:period - 1, data.columns.get_loc(result_column)] = 0

        return data.drop(columns=["high_n", "low_n"])

    elif isinstance(data, pl.DataFrame):
        high_n = data.select(
            pl.col(high_column).rolling_max(period).alias("high_n")
        )
        low_n = data.select(
            pl.col(low_column).rolling_min(period).alias("low_n")
        )

        data = data.with_columns([
            high_n["high_n"],
            low_n["low_n"]
        ])

        data = data.with_columns(
            ((pl.col("high_n") - pl.col(close_column))
                / (pl.col("high_n") - pl.col("low_n")) * -100)
            .alias(result_column)
        )

        # Set the first `period` rows of result_column to 0 directly in Polars
        if data.height > 0:
            zero_values = [0] * (period - 1) \
                + data[result_column].to_list()[period - 1:]
            data = data.with_columns(
                pl.Series(result_column, zero_values, dtype=pl.Float64)
            )

        return data.drop(["high_n", "low_n"])
    else:
        raise PyIndicatorException(
            "Unsupported data type. Must be pandas or polars DataFrame."
        )
