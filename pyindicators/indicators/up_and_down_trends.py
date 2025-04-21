from typing import Union, List
from datetime import timedelta

from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import pandas as pd

from .exponential_moving_average import ema
from .utils import is_above
from pyindicators.date_range import DateRange
from pyindicators.exceptions import PyIndicatorException


def up_and_downtrends(
    data: Union[PdDataFrame, PlDataFrame]
) -> List[DateRange]:
    """
    Function to get the up and down trends of a pandas dataframe.

    Params:
        data: pd.Dataframe - instance of pandas Dateframe
        containing OHLCV data.

    Returns:
        List of date ranges that with up_trend and down_trend
        flags specified.
    """

    # Check if the data is larger then 200 data points
    if len(data) < 200:
        raise PyIndicatorException(
            "The data must be larger than 200 data " +
            "points to determine up and down trends."
        )

    if isinstance(data, PlDataFrame):
        # Convert Polars DataFrame to Pandas DataFrame
        data = data.to_pandas()

    selection = data.copy()
    selection = ema(
        selection,
        source_column="Close",
        period=50,
        result_column="SMA_Close_50"
    )
    selection = ema(
        selection,
        source_column="Close",
        period=200,
        result_column="SMA_Close_200"
    )

    # Make selections based on the trend
    current_trend = None
    start_date_range = selection.index[0]
    date_ranges = []

    for idx, row in enumerate(selection.itertuples(index=True), start=1):
        selected_rows = selection.iloc[:idx]

        # Check if last row is null for the SMA_50 and SMA_200
        if pd.isnull(selected_rows["SMA_Close_50"].iloc[-1]) \
                or pd.isnull(selected_rows["SMA_Close_200"].iloc[-1]):
            continue

        if is_above(
            selected_rows,
            first_column="SMA_Close_50",
            second_column="SMA_Close_200"
        ):
            if current_trend != 'Up':

                if current_trend is not None:
                    end_date = selection.loc[
                        row.Index - timedelta(days=1)
                    ].name
                    date_ranges.append(
                        DateRange(
                            start_date=start_date_range,
                            end_date=end_date,
                            name=current_trend,
                            down_trend=True
                        )
                    )
                    start_date_range = row.Index
                    current_trend = 'Up'
                else:
                    current_trend = 'Up'
                    start_date_range = row.Index
        else:

            if current_trend != 'Down':

                if current_trend is not None:
                    end_date = selection.loc[
                        row.Index - timedelta(days=1)
                    ].name
                    date_ranges.append(
                        DateRange(
                            start_date=start_date_range,
                            end_date=end_date,
                            name=current_trend,
                            up_trend=True
                        )
                    )
                    start_date_range = row.Index
                    current_trend = 'Down'
                else:
                    current_trend = 'Down'
                    start_date_range = row.Index

    if current_trend is not None:
        end_date = selection.index[-1]

        if current_trend == 'Up':
            date_ranges.append(
                DateRange(
                    start_date=start_date_range,
                    end_date=end_date,
                    name=current_trend,
                    up_trend=True
                )
            )
        else:
            date_ranges.append(
                DateRange(
                    start_date=start_date_range,
                    end_date=end_date,
                    name=current_trend,
                    down_trend=True
                )
            )

    return date_ranges
