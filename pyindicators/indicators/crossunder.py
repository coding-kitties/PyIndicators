from typing import Union

from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
from pyindicators.exceptions import PyIndicatorException


def crossunder(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str,
    second_column: str,
    result_column="crossunder",
    number_of_data_points: int = None,
    strict: bool = True,
) -> Union[PdDataFrame, PlDataFrame]:

    if number_of_data_points is not None:

        if number_of_data_points < 2:
            raise PyIndicatorException(
                "The number of data points must be greater or equal than 2 for"
                " crossunder detection."
            )

        if isinstance(data, PdDataFrame):
            data = data.tail(number_of_data_points).copy()
        else:
            data = data.slice(-number_of_data_points)

    if isinstance(data, PdDataFrame):
        col1, col2 = data[first_column], data[second_column]
        prev_col1, prev_col2 = col1.shift(1), col2.shift(1)

        if strict:
            crossunder_mask = (prev_col1 > prev_col2) & (col1 < col2)
        else:
            crossunder_mask = (col1 > col2) & (prev_col1 <= prev_col2) \
                | (col1 >= col2) & (prev_col1 < prev_col2)

        data.loc[:, result_column] = crossunder_mask.astype(int)

    elif isinstance(data, PlDataFrame):
        col1, col2 = data[first_column], data[second_column]
        prev_col1, prev_col2 = col1.shift(1), col2.shift(1)

        if strict:
            crossunder_mask = (prev_col1 > prev_col2) & (col1 < col2)
        else:
            crossunder_mask = (col1 > col2) & (prev_col1 <= prev_col2) \
                | (col1 >= col2) & (prev_col1 < prev_col2)

        data = data.with_columns(
            pl.when(crossunder_mask).then(1).otherwise(0).alias(result_column)
        )

    return data


def is_crossunder(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str = None,
    second_column: str = None,
    crossunder_column: str = None,
    number_of_data_points: int = None,
    strict: bool = True,
) -> bool:

    if len(data) < 2:
        return False

    if number_of_data_points is None:
        number_of_data_points = len(data)
    elif number_of_data_points < 2:
        raise PyIndicatorException(
            "The number of data points must be greater or equal than 2 for"
            " crossover detection."
        )

    if crossunder_column is None:
        crossunder_column = f"{first_column}_crossunder_{second_column}"
        data = crossunder(
            data=data,
            first_column=first_column,
            second_column=second_column,
            result_column=crossunder_column,
            number_of_data_points=number_of_data_points,
            strict=strict
        )

    if isinstance(data, PdDataFrame):
        return data[crossunder_column].tail(number_of_data_points)\
            .eq(1).any()
    elif isinstance(data, pl.DataFrame):
        return data[crossunder_column][-number_of_data_points:]\
            .to_list().count(1) > 0

    raise PyIndicatorException(
        "Data type not supported. Please provide a Pandas or Polars DataFrame."
    )
