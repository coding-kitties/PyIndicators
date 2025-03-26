from typing import Union
import pandas as pd
import polars as pl


def rsi(
    data: Union[pd.DataFrame, pl.DataFrame],
    source_column: str,
    period: int = 14,
    result_column: str = None,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Function to calculate the RSI (Relative Strength Index) of a series.

    Args:
        data (Union[pd.DataFrame, pl.DataFrame]): The input data.
        source_column (str): The name of the series.
        period (int): The period for the RSI calculation.
        result_column (str, optional): The name of the column to store
        the RSI values. Defaults to None, which means it will
        be named "RSI_{period}".

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: The DataFrame with the RSI
        column added.
    """

    if result_column is None:
        result_column = f"rsi_{period}"

    if isinstance(data, pd.DataFrame):
        # Compute price changes
        delta = data[source_column].diff()

        # Compute gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Compute the rolling average of gains and losses
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Compute RSI
        rs = avg_gain / avg_loss
        rsi_values = 100 - (100 / (1 + rs))

        # Ensure first `period` rows are NaN
        rsi_values[:period] = pd.NA

        # Assign to DataFrame
        data[result_column] = rsi_values

    elif isinstance(data, pl.DataFrame):
        # Compute price changes
        delta = data[source_column].diff().fill_null(0)

        # Compute gains and losses
        gain = delta.clip(0)
        loss = (-delta).clip(0)

        # Compute rolling averages of gains and losses
        avg_gain = gain.rolling_mean(window_size=period, min_periods=period)
        avg_loss = loss.rolling_mean(window_size=period, min_periods=period)

        # Compute RSI
        rs = avg_gain / avg_loss
        rsi_values = 100 - (100 / (1 + rs))

        # Replace first `period` values with nulls (polars uses `None`)
        rsi_values = rsi_values.scatter(list(range(period)), None)

        # Add column to DataFrame
        data = data.with_columns(rsi_values.alias(result_column))

    else:
        raise TypeError("Input data must be a pandas or polars DataFrame.")

    return data


def wilders_rsi(
    data: Union[pd.DataFrame, pl.DataFrame],
    source_column: str,
    period: int = 14,
    result_column: str = None,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Compute RSI using wilders method (Wilder’s Smoothing).

    Args:
        data (Union[pd.DataFrame, pl.DataFrame]): Input DataFrame.
        source_column (str): Name of the column with price data.
        period (int): RSI period (e.g., 14).
        result_column (str, optional): Name for the output column.

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: DataFrame with RSI values.
    """

    if result_column is None:
        result_column = f"rsi_{period}"

    if isinstance(data, pd.DataFrame):
        delta = data[source_column].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Compute the initial SMA (first `period` rows)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Apply Wilder's Smoothing for the remaining values
        for i in range(period, len(data)):
            avg_gain.iloc[i] = (
                avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]
            ) / period
            avg_loss.iloc[i] = (
                avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]
            ) / period

        rs = avg_gain / avg_loss
        data[result_column] = 100 - (100 / (1 + rs))

        # Ensure first `period` rows are NaN
        data.iloc[:period, data.columns.get_loc(result_column)] = pd.NA

    elif isinstance(data, pl.DataFrame):
        delta = data[source_column].diff().fill_null(0)
        gain = delta.clip(0)
        loss = (-delta).clip(0)

        # Compute initial SMA (first `period` rows)
        avg_gain = gain.rolling_mean(window_size=period, min_periods=period)
        avg_loss = loss.rolling_mean(window_size=period, min_periods=period)

        # Initialize smoothed gains/losses with the first SMA values
        smoothed_gain = avg_gain[:period].to_list()
        smoothed_loss = avg_loss[:period].to_list()

        # Apply Wilder's Smoothing
        for i in range(period, len(data)):
            smoothed_gain.append(
                (smoothed_gain[-1] * (period - 1) + gain[i]) / period
            )
            smoothed_loss.append(
                (smoothed_loss[-1] * (period - 1) + loss[i]) / period
            )

        # Compute RSI
        rs = pl.Series(smoothed_gain) / pl.Series(smoothed_loss)
        rsi_values = 100 - (100 / (1 + rs))

        # Replace first `period` values with None
        rsi_values = rsi_values.scatter(list(range(period)), None)

        # Add column to DataFrame
        data = data.with_columns(rsi_values.alias(result_column))

    else:
        raise TypeError("Input data must be a pandas or polars DataFrame.")

    return data
