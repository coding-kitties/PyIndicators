from typing import Union
import numpy as np

from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
import pandas as pd

from pyindicators.exceptions import PyIndicatorException
from pyindicators.indicators.utils import pad_zero_values_pandas


def polars_ewm_mean_via_pandas(column: pl.Series, alpha: float) -> pl.Series:
    pd_series = pd.Series(column.to_numpy())
    ewm_result = pd_series.ewm(alpha=alpha, adjust=False).mean()
    return pl.Series(name=column.name, values=ewm_result.to_numpy())


def calculate_adx_pandas(
    data: PdDataFrame,
    period: int,
    adx_result_column: str = "ADX",
    di_plus_result_column: str = "+DI",
    di_minus_result_column: str = "-DI",
) -> PdDataFrame:
    alpha = 1/period
    copy_df = data.copy()

    # TR
    copy_df['H-L'] = copy_df['High'] - copy_df['Low']
    copy_df['H-C'] = np.abs(copy_df['High'] - copy_df['Close'].shift(1))
    copy_df['L-C'] = np.abs(copy_df['Low'] - copy_df['Close'].shift(1))
    copy_df['TR'] = copy_df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del copy_df['H-L'], copy_df['H-C'], copy_df['L-C']

    # Average True Range (ATR)
    copy_df['ATR'] = copy_df['TR'].ewm(alpha=alpha, adjust=False).mean()

    # +-DX calculation
    copy_df['H-pH'] = copy_df['High'] - copy_df['High'].shift(1)
    copy_df['pL-L'] = copy_df['Low'].shift(1) - copy_df['Low']
    copy_df['+DI'] = np.where(
        (copy_df['H-pH'] > copy_df['pL-L']) & (copy_df['H-pH'] > 0),
        copy_df['H-pH'],
        0.0
    )
    copy_df['-DI'] = np.where(
        (copy_df['H-pH'] < copy_df['pL-L']) & (copy_df['pL-L'] > 0),
        copy_df['pL-L'],
        0.0
    )
    del copy_df['H-pH'], copy_df['pL-L']

    # +- DMI
    copy_df['S+DM'] = copy_df['+DI'].ewm(alpha=alpha, adjust=False).mean()
    copy_df['S-DM'] = copy_df['-DI'].ewm(alpha=alpha, adjust=False).mean()
    copy_df['+DMI'] = (copy_df['S+DM']/copy_df['ATR'])*100
    copy_df['-DMI'] = (copy_df['S-DM']/copy_df['ATR'])*100
    del copy_df['S+DM'], copy_df['S-DM']

    # ADX
    copy_df['DX'] = (
        np.abs(copy_df['+DMI'] - copy_df['-DMI'])/(copy_df['+DMI']
                                                   + copy_df['-DMI'])
    )*100
    copy_df['ADX'] = copy_df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del copy_df['DX'], copy_df['ATR'], copy_df['TR']
    copy_df = pad_zero_values_pandas(
        data=copy_df,
        column="+DMI",
        period=period,
    )

    copy_df = pad_zero_values_pandas(
        data=copy_df,
        column="-DMI",
        period=period,
    )
    copy_df = pad_zero_values_pandas(
        data=copy_df,
        column="ADX",
        period=period,
    )

    # Add the ADX column to the original DataFrame
    data[adx_result_column] = copy_df['ADX']
    data[di_plus_result_column] = copy_df['+DMI']
    data[di_minus_result_column] = copy_df['-DMI']
    return data


def adx(
    data: Union[PdDataFrame, PlDataFrame],
    period=14,
    adx_result_column="ADX",
    di_plus_result_column="+DI",
    di_minus_result_column="-DI",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the Average Directional Index (ADX) along with the
    +DI and -DI indicators.
    The ADX is a trend strength indicator that ranges from 0 to 100.
    The +DI and -DI indicators are used to identify the direction
    of the trend.

    The following columns are required in the input DataFrame:
    - 'High': High prices
    - 'Low': Low prices
    - 'Close': Close prices
    The output DataFrame will contain the following columns:
    - 'ADX': Average Directional Index
    - '+DI': Positive Directional Indicator
    - '-DI': Negative Directional Indicator

    Args:
        data: Pandas or Polars DataFrame
        period (int, optional): Period for the ADX calculation (default: 14).
        adx_result_column (str, optional): Name of the column to store
            the ADX values (default: "ADX").
        di_plus_result_column (str, optional): Name of the column to
            store the +DI values (default: "+DI").
        di_minus_result_column (str, optional): Name of the column to
            store the -DI values (default: "-DI").

    Returns:
        Union[Pandas Dataframe, Polars Dataframe]: DataFrame with ADX,
        +DI, and -DI columns.
    """

    # Check if the input DataFrame has the required columns
    required_columns = ['High', 'Low', 'Close']

    if not all(col in data.columns for col in required_columns):
        raise PyIndicatorException(
            "Input DataFrame must contain the " +
            f"following columns: {required_columns}"
        )

    if len(data) < period:
        raise PyIndicatorException(
            "The data must be larger than the period " +
            f"{period} to calculate the EMA. The data " +
            f"only contains {len(data)} data points."
        )

    # Pandas implementation
    if isinstance(data, PdDataFrame):
        data = calculate_adx_pandas(
            data=data,
            period=period,
            adx_result_column=adx_result_column,
            di_plus_result_column=di_plus_result_column,
            di_minus_result_column=di_minus_result_column
        )
        return data
    else:

        # The following code is commented out because it is does
        # not give the same result as the Pandas implementation.
        # Therefore for now we only use the Pandas implementation.

        # convert the Polars DataFrame to a Pandas DataFrame
        data = data.to_pandas()
        data = calculate_adx_pandas(
            data=data,
            period=period,
            adx_result_column=adx_result_column,
            di_plus_result_column=di_plus_result_column,
            di_minus_result_column=di_minus_result_column
        )

        # Convert the Pandas DataFrame back to a Polars DataFrame
        data = pl.from_pandas(data)
        return data
