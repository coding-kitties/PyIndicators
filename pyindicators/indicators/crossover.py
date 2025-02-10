from typing import Union
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame


def is_crossover(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str,
    second_column: str,
    data_points: int = None,
    strict=True,
) -> bool:
    """
    Returns a boolean when the first series crosses above the second series at any point or within the last n data points.

    Args:
        data (Union[PdDataFrame, PlDataFrame]): The input data.
        first_column (str): The name of the first series.
        second_column (str): The name of the second series.
        data_points (int, optional): The number of data points to consider. Defaults to None.
        strict (bool, optional): If True, the first series must be strictly greater than the second series. If False, the first series must be greater than or equal to the second series. Defaults to True.

    Returns:
        bool: Returns True if the first series crosses above the second series at any point or within the last n data points.
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
