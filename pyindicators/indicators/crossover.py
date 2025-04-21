from typing import Union

from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
from pyindicators.exceptions import PyIndicatorException


def crossover(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str,
    second_column: str,
    result_column="crossover",
    number_of_data_points: int = None,
    strict: bool = True,
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Identifies crossover points where `first_column` crosses above
    or below `second_column`.

    Args:
        data: Pandas or Polars DataFrame
        first_column: Name of the first column
        second_column: Name of the second column
        result_column (optional): Name of the column to
            store the crossover points
        number_of_data_points (optional):
            Number of recent rows to consider (optional)
        strict (optional): If True, requires exact crossovers; otherwise,
            detects when one surpasses the other.

    Returns:
        A DataFrame with crossover points marked.
    """

    # Restrict data to the last `data_points` rows if specified
    if number_of_data_points is not None:

        if number_of_data_points < 2:
            raise PyIndicatorException(
                "The number of data points must be greater or equal than 2 for"
                " crossover detection."
            )

        data = data.tail(number_of_data_points) \
            if isinstance(data, PdDataFrame) \
            else data.slice(-number_of_data_points)

    # Pandas Implementation
    if isinstance(data, PdDataFrame):
        col1, col2 = data[first_column], data[second_column]
        prev_col1, prev_col2 = col1.shift(1), col2.shift(1)

        if strict:
            crossover_mask = ((prev_col1 < prev_col2) & (col1 > col2))
        else:
            crossover_mask = (col1 > col2) & (prev_col1 <= prev_col2)

        data = data.copy()
        data.loc[:, result_column] = crossover_mask.astype(int)

    # Polars Implementation
    elif isinstance(data, PlDataFrame):
        col1, col2 = data[first_column], data[second_column]
        prev_col1, prev_col2 = col1.shift(1), col2.shift(1)

        if strict:
            crossover_mask = ((prev_col1 < prev_col2) & (col1 > col2))
        else:
            crossover_mask = (col1 > col2) & (prev_col1 <= prev_col2)

        # Convert boolean mask to 1s and 0s
        data = data.clone()
        data = data.with_columns(pl.when(crossover_mask).then(1)
                                 .otherwise(0).alias(result_column))

    return data


def is_crossover(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str = None,
    second_column: str = None,
    crossover_column: str = None,
    number_of_data_points: int = None,
    strict=True,
) -> bool:
    """
    Returns a boolean when the first series crosses above the second
        series at any point or within the last n data points.

    Args:
        data (Union[PdDataFrame, PlDataFrame]): The input data.
        first_column (str): The name of the first series.
        second_column (str): The name of the second series.
        crossover_column (str, optional): The name of the column to store
            the crossover points. Defaults to None.
        number_of_data_points (int, optional): The number of data points
            to consider. Defaults to None.
        strict (bool, optional): If True, the first series must
            be strictly greater than the second series. If False,
            the first series must be greater than or equal
            to the second series. Defaults to True.

    Returns:
        bool: Returns True if the first series crosses above the
            second series at any point or within the last n data points.
    """
    if len(data) < 2:
        return False

    if number_of_data_points is None:
        number_of_data_points = len(data)
    elif number_of_data_points < 2:
        raise PyIndicatorException(
            "The number of data points must be greater or equal than 2 for"
            " crossover detection."
        )

    if crossover_column is None:
        crossover_column = f"{first_column}_crossover_{second_column}"
        data = crossover(
            data=data,
            first_column=first_column,
            second_column=second_column,
            result_column=crossover_column,
            number_of_data_points=number_of_data_points,
            strict=strict
        )

    # If crossunder_column is set, check for a value of 1
    # in the last data points
    if isinstance(data, PdDataFrame):
        return data[crossover_column].tail(number_of_data_points).eq(1).any()
    elif isinstance(data, pl.DataFrame):
        return data[crossover_column][-number_of_data_points:]\
            .to_list().count(1) > 0

    raise PyIndicatorException(
        "Data type not supported. Please provide a Pandas or Polars DataFrame."
    )
