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
    """
    Identifies crossunder points where `first_column` crosses below
    or below `second_column`.

    Args:
        data: Pandas or Polars DataFrame
        first_column: Name of the first column
        second_column: Name of the second column
        result_column (optional): Name of the column to
            store the crossunder points
        number_of_data_points (optional):
            Number of recent rows to consider
        strict (optional): If True, requires exact crossunders; otherwise,
            detects when one surpasses the other.

    Returns:
        A DataFrame with crossunder points marked.
    """

    # Restrict data to the last `data_points` rows if specified
    if number_of_data_points is not None:
        data = data.tail(number_of_data_points) \
            if isinstance(data, PdDataFrame) \
            else data.slice(-number_of_data_points)

    # Pandas Implementation
    if isinstance(data, PdDataFrame):
        col1, col2 = data[first_column], data[second_column]
        prev_col1, prev_col2 = col1.shift(1), col2.shift(1)

        if strict:
            crossunder_mask = (
                (prev_col1 > prev_col2) & (col1 < col2)
            )
        else:
            crossunder_mask = (col1 > col2) & (prev_col1 <= prev_col2)

        data[result_column] = crossunder_mask.astype(int)

    # Polars Implementation
    elif isinstance(data, PlDataFrame):
        col1, col2 = data[first_column], data[second_column]
        prev_col1, prev_col2 = col1.shift(1), col2.shift(1)

        if strict:
            crossunder_mask = ((prev_col1 > prev_col2) & (col1 < col2))
        else:
            crossunder_mask = (col1 > col2) & (prev_col1 <= prev_col2)

        # Convert boolean mask to 1s and 0s
        data = data.with_columns(pl.when(crossunder_mask).then(1)
                                 .otherwise(0).alias(result_column))

    return data


def is_crossunder(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str = None,
    second_column: str = None,
    crossunder_column: str = None,
    number_of_data_points: int = None,
    strict: bool = True,
) -> bool:
    """
    Returns a boolean when the first series crosses below the second series
    at any point or within the last n data points.

    Args:
        data (Union[pd.DataFrame, pl.DataFrame]): The input data.
        first_column (str): The name of the first series.
        second_column (str): The name of the second series.
        crossunder_column (str) (optional):
            The name of the column to store the crossunder points.
        number_of_data_points (int) (optional):
            The number of data points to consider. Defaults to None.
        strict (bool) (optional): If True, requires a strict
            crossunder. Defaults to True.

    Returns:
        bool: True if a crossunder occurs, False otherwise.
    """

    if len(data) < 2:
        return False

    if crossunder_column is None:
        crossunder_column = f"{first_column}_crossunder_{second_column}"
        data = crossunder(
            data=data,
            first_column=first_column,
            second_column=second_column,
            result_column=crossunder_column,
            number_of_data_points=number_of_data_points,
            strics=strict
        )

    # If crossunder_column is set, check for a value of 1
    # in the last data points
    if isinstance(data, PdDataFrame):
        return data[crossunder_column].tail(number_of_data_points).eq(1).any()
    elif isinstance(data, pl.DataFrame):
        return data[crossunder_column][-number_of_data_points:]\
            .to_list().count(1) > 0

    raise PyIndicatorException(
        "Data type not supported. Please provide a Pandas or Polars DataFrame."
    )
