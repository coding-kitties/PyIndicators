from typing import Union

from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
import pandas as pd

from pyindicators.exceptions import PyIndicatorException


def atr(
    data: Union[PdDataFrame, PlDataFrame],
    source_column="Close",
    period=14,
    result_column="ATR"
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the Average True Range (ATR) for a given dataset.

    Parameters:
        data (Union[PdDataFrame, PlDataFrame]): The input data
            containing OHLC prices.
        source_column (str): The column to use as the source
            for ATR calculation.
        period (int): The number of periods to use for the ATR calculation.

    Returns:
        Union[PdDataFrame, PlDataFrame]: The calculated ATR values
            contained in a DataFrame.
    """
    if isinstance(data, PdDataFrame):
        high = data['High']
        low = data['Low']
        close = data[source_column]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period, min_periods=1).mean()
        data[result_column] = atr
        return data

    elif isinstance(data, PlDataFrame):
        # Polars version
        df = data.with_columns([
            (pl.col("High") - pl.col("Low")).alias("H_L"),
            (pl.col("High") -
                pl.col(source_column).shift(1)).abs().alias("H_Cp"),
            (pl.col("Low") -
                pl.col(source_column).shift(1)).abs().alias("L_Cp"),
        ])

        # True Range = max of H-L, H-Cprev, L-Cprev
        df = df.with_columns([
            pl.max_horizontal(["H_L", "H_Cp", "L_Cp"]).alias("TR")
        ])

        # Rolling mean ATR
        df = df.with_columns([
            pl.col("TR").rolling_mean(window_size=period).alias(result_column)
        ])

        return df.drop(["H_L", "H_Cp", "L_Cp"])  # optional cleanup

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )
