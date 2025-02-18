from typing import Union

from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl


def crossover(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str,
    second_column: str,
    result_column="crossover",
    data_points: int = None,
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
        data_points (optional): Number of recent rows to consider (optional)
        strict (optional): If True, requires exact crossovers; otherwise,
            detects when one surpasses the other.

    Returns:
        A DataFrame with crossover points marked.
    """

    # Restrict data to the last `data_points` rows if specified
    if data_points is not None:
        data = data.tail(data_points) if isinstance(data, PdDataFrame) \
            else data.slice(-data_points)

    # Pandas Implementation
    if isinstance(data, PdDataFrame):
        col1, col2 = data[first_column], data[second_column]
        prev_col1, prev_col2 = col1.shift(1), col2.shift(1)

        if strict:
            crossover_mask = (
                (prev_col1 < prev_col2)
                & (col1 > col2)) | ((prev_col1 > prev_col2) & (col1 < col2))
        else:
            crossover_mask = (col1 > col2) | (col1 < col2)

        data[result_column] = crossover_mask.astype(int)

    # Polars Implementation
    elif isinstance(data, PlDataFrame):
        col1, col2 = data[first_column], data[second_column]
        prev_col1, prev_col2 = col1.shift(1), col2.shift(1)

        if strict:
            crossover_mask = ((prev_col1 < prev_col2) & (col1 > col2)) | \
                ((prev_col1 > prev_col2) & (col1 < col2))
        else:
            crossover_mask = (col1 > col2) | (col1 < col2)

        # Convert boolean mask to 1s and 0s
        data = data.with_columns(pl.when(crossover_mask).then(1)
                                 .otherwise(0).alias(result_column))

    return data


def is_crossover(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str,
    second_column: str,
    data_points: int = None,
    strict=True,
) -> bool:
    """
    Returns a boolean when the first series crosses above the second
        series at any point or within the last n data points.

    Args:
        data (Union[PdDataFrame, PlDataFrame]): The input data.
        first_column (str): The name of the first series.
        second_column (str): The name of the second series.
        data_points (int, optional): The number of data points
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

    if data_points is None:
        data_points = len(data) - 1

    if isinstance(data, PdDataFrame):

        # Loop through the data points and check if the first key
        # is greater than the second key
        for i in range(data_points, 0, -1):

            if strict:
                if data[first_column].iloc[-(i + 1)] \
                        < data[second_column].iloc[-(i + 1)] \
                        and data[first_column].iloc[-i] \
                        > data[second_column].iloc[-i]:
                    return True
            else:
                if data[first_column].iloc[-(i + 1)] \
                        <= data[second_column].iloc[-(i + 1)]  \
                        and data[first_column].iloc[-i] >= \
                        data[second_column].iloc[-i]:
                    return True

    else:
        # Loop through the data points and check if the first key
        # is greater than the second key
        for i in range(data_points, 0, -1):

            if strict:
                if data[first_column][-i - 1] \
                        < data[second_column][-i - 1] \
                        and data[first_column][-i] \
                        > data[second_column][-i]:
                    return True
            else:
                if data[first_column][-i - 1] \
                        <= data[second_column][-i - 1]  \
                        and data[first_column][-i] >= \
                        data[second_column][-i]:
                    return True

    return False
