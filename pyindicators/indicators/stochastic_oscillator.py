from typing import Union, Optional
import pandas as pd
import polars as pl


def stochastic_oscillator(
    data: Union[pd.DataFrame, pl.DataFrame],
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    k_period: int = 14,
    k_slowing: int = 3,
    d_period: int = 3,
    result_column: Optional[str] = None
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Calculate the Stochastic Oscillator (%K and %D) for a given DataFrame.
    The Stochastic Oscillator is a momentum indicator comparing
    a particular closing price of a security to a range of its
    prices over a certain period.

    Args:
        data (Union[pd.DataFrame, pl.DataFrame]): Input DataFrame
            containing the high, low, and close prices.
        high_column (str): Name of the column containing high prices.
        low_column (str): Name of the column containing low prices.
        close_column (str): Name of the column containing close prices.
        k_period (int): The period for %K calculation.
        d_period (int): The period for %D calculation.
        k_slowing (int): The period for smoothing the %K line.
        result_column (Optional[str]): Optional prefix for result
            columns. If None, defaults to "%K" and "%D".

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: DataFrame
            with %K and %D columns.
    """

    k_col = f"{result_column}_%K" if result_column else "%K"
    d_col = f"{result_column}_%D" if result_column else "%D"

    if isinstance(data, pd.DataFrame):
        # Fast %K
        low_min = data[low_column].rolling(
            window=k_period, min_periods=k_period
        ).min()
        high_max = data[high_column].rolling(
            window=k_period, min_periods=k_period
        ).max()

        fast_k = 100 * (data[close_column] - low_min) / (high_max - low_min)

        # Slow %K (smoothed Fast %K)
        slow_k = fast_k.rolling(window=k_slowing, min_periods=k_slowing).mean()

        # %D (smoothed Slow %K)
        d = slow_k.rolling(window=d_period, min_periods=d_period).mean()

        data[k_col] = slow_k
        data[d_col] = d

        return data

    elif isinstance(data, pl.DataFrame):
        # Compute Fast %K
        low_min = pl.col(low_column).rolling_min(k_period)
        high_max = pl.col(high_column).rolling_max(k_period)

        fast_k_expr = ((pl.col(close_column) - low_min) /
                       (high_max - low_min)) * 100

        # Compute Slow %K and %D in steps
        df = data.with_columns([
            fast_k_expr.alias("_fast_k")
        ])

        df = df.with_columns([
            pl.col("_fast_k").rolling_mean(k_slowing).alias(k_col)
        ])

        df = df.with_columns([
            pl.col(k_col).rolling_mean(d_period).alias(d_col)
        ])

        return df.drop("_fast_k")

    else:
        raise TypeError("Input data must be a pandas or polars DataFrame.")
