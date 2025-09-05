from typing import Union
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
from pyindicators.exceptions import PyIndicatorException


def bollinger_bands(
    data: Union[PdDataFrame, PlDataFrame],
    source_column='Close',
    period=20,
    std_dev=2,
    middle_band_column_result_column='bollinger_middle',
    upper_band_column_result_column='bollinger_upper',
    lower_band_column_result_column='bollinger_lower'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate Bollinger Bands for a price series.

    Returns the original DataFrame with added columns for
    middle, upper, and lower bands.
    """
    if isinstance(data, PdDataFrame):
        mb = data[source_column].rolling(period).mean()
        std = data[source_column].rolling(period).std()

        data[middle_band_column_result_column] = mb
        data[upper_band_column_result_column] = mb + std_dev * std
        data[lower_band_column_result_column] = mb - std_dev * std
        return data

    elif isinstance(data, PlDataFrame):
        df = data
        mb = pl.col(source_column).rolling_mean(window_size=period)
        std = pl.col(source_column).rolling_std(window_size=period)

        return df.with_columns([
            mb.alias(middle_band_column_result_column),
            (mb + std_dev * std).alias(upper_band_column_result_column),
            (mb - std_dev * std).alias(lower_band_column_result_column)
        ])

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def bollinger_width(
    data: Union[PdDataFrame, PlDataFrame],
    source_column='Close',
    period=20,
    std_dev=2,
    result_column='Bollinger_Width'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate Bollinger Band width for a price series.

    Returns the original DataFrame with a new column for width.
    """
    # First calculate the bands
    data = bollinger_bands(
        data,
        source_column=source_column,
        period=period,
        std_dev=std_dev,
        middle_band_column_result_column='BB_middle_temp',
        upper_band_column_result_column='BB_upper_temp',
        lower_band_column_result_column='BB_lower_temp'
    )

    if isinstance(data, PdDataFrame):
        data[result_column] = data['BB_upper_temp'] - data['BB_lower_temp']
        # Drop temporary columns
        data = data.drop(
            columns=['BB_middle_temp', 'BB_upper_temp', 'BB_lower_temp']
        )
        return data

    elif isinstance(data, PlDataFrame):
        return data.with_columns(
            (pl.col('BB_upper_temp') -
             pl.col('BB_lower_temp')).alias(result_column)
        ).drop(['BB_middle_temp', 'BB_upper_temp', 'BB_lower_temp'])

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )
