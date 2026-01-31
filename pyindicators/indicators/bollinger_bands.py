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


def bollinger_overshoot(
    data: Union[PdDataFrame, PlDataFrame],
    source_column='Close',
    period=20,
    std_dev=2,
    result_column='bollinger_overshoot'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate Bollinger Band overshoot percentage for a price series.

    Measures how far the price has exceeded the upper or lower band,
    expressed as a percentage of the half-band width (distance from
    middle to upper/lower band).

    Calculation:
    - When price > upper band (bullish overshoot):
        Overshoot % = ((Price - Upper Band) / (Upper Band - Middle Band)) × 100
    - When price < lower band (bearish overshoot):
        Overshoot % = ((Price - Lower Band) / (Middle Band - Lower Band)) × 100
    - When price is within bands: 0%

    Example interpretation:
    - A 40% overshoot means the price is 40% of the band width beyond the band
    - Positive values indicate overbought conditions (above upper band)
    - Negative values indicate oversold conditions (below lower band)
    - High overshoots (e.g., 40% for silver) indicate risk of mean reversion

    Args:
        data: pandas or polars DataFrame with price data
        source_column: Column name containing the price data (default: 'Close')
        period: Rolling window period for calculation (default: 20)
        std_dev: Number of standard deviations for bands (default: 2)
        result_column: Name for the result column
            (default: 'bollinger_overshoot')

    Returns:
        DataFrame with added overshoot percentage column
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
        import numpy as np

        price = data[source_column]
        upper = data['BB_upper_temp']
        middle = data['BB_middle_temp']
        lower = data['BB_lower_temp']

        # Calculate half-band width (same for upper and
        # lower with symmetric bands)
        half_band_width = upper - middle

        # Calculate overshoot
        # Above upper band: positive overshoot
        # Below lower band: negative overshoot
        # Within bands: 0
        overshoot = np.where(
            price > upper,
            ((price - upper) / half_band_width) * 100,
            np.where(
                price < lower,
                ((price - lower) / half_band_width) * 100,
                0.0
            )
        )

        data[result_column] = overshoot

        # Drop temporary columns
        data = data.drop(
            columns=['BB_middle_temp', 'BB_upper_temp', 'BB_lower_temp']
        )
        return data

    elif isinstance(data, PlDataFrame):
        half_band_width = pl.col('BB_upper_temp') - pl.col('BB_middle_temp')

        overshoot = pl.when(pl.col(source_column) >
                            pl.col('BB_upper_temp')).then(
            ((pl.col(source_column) -
              pl.col('BB_upper_temp')) / half_band_width) * 100
        ).when(pl.col(source_column) < pl.col('BB_lower_temp')).then(
            ((pl.col(source_column) -
              pl.col('BB_lower_temp')) / half_band_width) * 100
        ).otherwise(0.0)

        return data.with_columns(
            overshoot.alias(result_column)
        ).drop(['BB_middle_temp', 'BB_upper_temp', 'BB_lower_temp'])

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
