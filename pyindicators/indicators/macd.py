from typing import Union

from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl

from pyindicators.exceptions import PyIndicatorException
from pyindicators.indicators import ema


def macd(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str,
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
    macd_column: str = "macd",
    signal_column: str = "macd_signal",
    histogram_column: str = "macd_histogram"
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the MACD (Moving Average Convergence Divergence) for
    a given DataFrame.

    Args:
        data (Union[pd.DataFrame, pl.DataFrame]): Input data containing
            the price series.
        source_column (str): Column name for the price series.
        short_period (int, optional): Period for the short-term EMA
            (default: 12).
        long_period (int, optional): Period for the long-term EMA
            (default: 26).
        signal_period (int, optional): Period for the Signal Line
            EMA (default: 9).
        macd_column (str, optional): Column name to store the MACD line.
        signal_column (str, optional): Column name to store the Signal line.
        histogram_column (str, optional): Column name to store the
            MACD histogram.

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: DataFrame with MACD, Signal
        Line, and Histogram.
    """
    if source_column not in data.columns:
        raise PyIndicatorException(
            f"Column '{source_column}' not found in DataFrame"
        )

    if isinstance(data, PdDataFrame):
        # Calculate the short-term and long-term EMAs
        data = ema(
            data, source_column, short_period, f"EMA_MACD_TEMP_{short_period}"
        )
        data = ema(
            data, source_column, long_period, f"EMA_MACD_TEMP_{long_period}"
        )

        # Calculate the MACD line
        data[macd_column] = \
            data[
                f"EMA_MACD_TEMP_{short_period}"
            ] - data[f"EMA_MACD_TEMP_{long_period}"]

        # Calculate the Signal Line
        data = ema(data, macd_column, signal_period, signal_column)

        # Calculate the MACD Histogram
        data[histogram_column] = data[macd_column] - data[signal_column]

        # Delete the temporary EMA columns
        data = data.drop(
            columns=[
                f"EMA_MACD_TEMP_{short_period}",
                f"EMA_MACD_TEMP_{long_period}"
            ]
        )
        return data
    elif isinstance(data, pl.DataFrame):
        # Polars implementation
        data = data.with_columns([
            ema(
                data,
                source_column,
                short_period,
                f"EMA_MACD_TEMP_{short_period}"
            )[f"EMA_MACD_TEMP_{short_period}"],
            ema(
                data,
                source_column,
                long_period,
                f"EMA_MACD_TEMP_{long_period}"
            )[f"EMA_MACD_TEMP_{long_period}"]
        ])

        data = data.with_columns(
            (
                pl.col(
                    f"EMA_MACD_TEMP_{short_period}"
                ) - pl.col(f"EMA_MACD_TEMP_{long_period}")
            ).alias(macd_column)
        )

        data = data.with_columns(
            ema(data, macd_column, signal_period, signal_column)[signal_column]
        )

        data = data.with_columns(
            (
                pl.col(macd_column) - pl.col(signal_column)
            ).alias(histogram_column)
        )

        # Delete the temporary EMA columns
        data = data.drop(
            [
                f"EMA_MACD_TEMP_{short_period}",
                f"EMA_MACD_TEMP_{long_period}"
            ]
        )
        return data
    else:
        raise PyIndicatorException(
            "Unsupported DataFrame type. Use Pandas or Polars."
        )
