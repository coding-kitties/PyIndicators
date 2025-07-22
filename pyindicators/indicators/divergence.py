from typing import Union
from collections import deque

import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
import polars as pl

from pyindicators.exceptions import PyIndicatorException


def _to_numpy(data: Union[np.ndarray, pd.Series, pl.Series]) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, pl.Series):
        return data.to_numpy()
    else:
        raise TypeError(
            "Input must be a NumPy array, pandas Series, or polars Series."
        )


def get_higher_lows(data, order=5, K=2):
    data = _to_numpy(data)
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    extrema, ex_deque = [], deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0 or lows[i] < lows[i - 1]:
            ex_deque.clear()
        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())
    return extrema


def get_lower_highs(data, order=5, K=2):
    data = _to_numpy(data)
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    extrema, ex_deque = [], deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0 or highs[i] > highs[i - 1]:
            ex_deque.clear()
        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())
    return extrema


def get_higher_highs(data, order=5, K=2):
    data = _to_numpy(data)
    high_idx = argrelextrema(data, np.greater_equal, order=order)[0]
    highs = data[high_idx]
    extrema, ex_deque = [], deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0 or highs[i] < highs[i - 1]:
            ex_deque.clear()
        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())
    return extrema


def get_lower_lows(data, order=5, K=2):
    data = _to_numpy(data)
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    extrema, ex_deque = [], deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0 or lows[i] > lows[i - 1]:
            ex_deque.clear()
        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())
    return extrema


def get_higher_high_index(data, order=5, K=2):
    extrema = get_higher_highs(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[idx < len(_to_numpy(data))]


def get_lower_highs_index(data, order=5, K=2):
    extrema = get_lower_highs(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[idx < len(_to_numpy(data))]


def get_lower_lows_index(data, order=5, K=2):
    extrema = get_lower_lows(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[idx < len(_to_numpy(data))]


def get_higher_lows_index(data, order=5, K=2):
    extrema = get_higher_lows(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[idx < len(_to_numpy(data))]


def detect_peaks(
    data: Union[pd.DataFrame, pl.DataFrame],
    source_column: str,
    number_of_neighbors_to_compare: int = 5,
    min_consecutive: int = 2
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Detects local peak structures in a time series column
    using trend-based logic.

    This function identifies local highs and lows based
    on comparisons over a rolling window.
    It marks the following in the output DataFrame:
        - Higher Highs (1) and Lower Highs (-1) in
            a "{column}_highs" column.
        - Lower Lows (1) and Higher Lows (-1) in
            a "{column}_lows" column.

    Args:
        data (Union[pd.DataFrame, pl.DataFrame]): Input DataFrame
            containing the time series data.
        number_of_neighbors_to_compare (int, optional): Number of
            neighboring points to compare on each side to
            determine local peaks.
        min_consecutive (int, optional): Minimum number of
            consecutive peaks required to confirm a peak structure.

    Returns:
        Union[pl.DataFrame, pd.DataFrame]: DataFrame with detected
            peaks and troughs.
    """
    values = data[source_column] \
        if isinstance(data, pd.DataFrame) else data[source_column].to_numpy()

    hh_idx = get_higher_high_index(
        values, number_of_neighbors_to_compare, min_consecutive
    )
    lh_idx = get_lower_highs_index(
        values, number_of_neighbors_to_compare, min_consecutive
    )
    ll_idx = get_lower_lows_index(
        values, number_of_neighbors_to_compare, min_consecutive
    )
    hl_idx = get_higher_lows_index(
        values, number_of_neighbors_to_compare, min_consecutive
    )

    if isinstance(data, pd.DataFrame):
        data[f"{source_column}_highs"] = np.nan
        data[f"{source_column}_lows"] = np.nan

        data.loc[data.index[hh_idx], f"{source_column}_highs"] = 1
        data.loc[data.index[lh_idx], f"{source_column}_highs"] = -1
        data.loc[data.index[ll_idx], f"{source_column}_lows"] = 1
        data.loc[data.index[hl_idx], f"{source_column}_lows"] = -1
        return data

    elif isinstance(data, pl.DataFrame):
        highs_col = np.full(len(data), np.nan)
        lows_col = np.full(len(data), np.nan)

        highs_col[hh_idx] = 1
        highs_col[lh_idx] = -1
        lows_col[ll_idx] = 1
        lows_col[hl_idx] = -1

        data = data.with_columns([
            pl.Series(f"{source_column}_highs", highs_col),
            pl.Series(f"{source_column}_lows", lows_col),
        ])
        return data

    else:
        raise TypeError("df must be a pandas or polars DataFrame")


def check_divergence_pattern(series_a, series_b, target_a=-1, target_b=1):
    """
    Check for bullish divergence pattern:
    - series_a must contain `target_a` (e.g., -1)
    - series_b must contain `target_b` (e.g., 1) *after* the target_a,
      and must not contain another `-1` before that point

    Returns:
        bool: True if pattern is found, False otherwise
    """
    try:
        # Find the first index of `target_a` (e.g., -1 in the indicator)
        a_index = next(i for i, val in enumerate(series_a) if val == target_a)
    except StopIteration:
        return False

    # From that point forward, check if series_b has a target_b
    for j in range(a_index, len(series_b)):
        if series_b[j] == -1:
            return False  # Higher low before lower low — invalid
        if series_b[j] == target_b:
            return True  # Valid divergence pattern
    return False


def bullish_divergence(
    data: Union[pd.DataFrame, pl.DataFrame],
    first_column: str,
    second_column: str,
    window_size=1,
    result_column: str = "bullish_divergence",
    number_of_neighbors_to_compare: int = 5,
    min_consecutive: int = 2
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Check for bullish divergence between two columns in a DataFrame.
    Given two columns in a DataFrame with peaks and lows,
    check if there is a bullish divergence. Peaks and lows are
    calculated using the get_peaks function. Usually the first column
    is the indicator column and the second column is the price column.

    Make sure that before calling this function, you have already
    identified the peaks and lows in both columns by using the
    `detect_peaks` function. If this is not the case, the function
    will automatically call the `detect_peaks` function to calculate
    the peaks and lows in the columns.

    The peaks are indicated as follows:
        * Higher Low = -1
        * Lower Low = 1
        * Higher High = 1
        * Lower High = -1

    Given that the low columns are selected for both columns; For
    a bullish divergence:
        * First Column: Look for a higher low (-1) within the window.
        * Second Column: Look for a lower low (1) within the window.

    Args:
        data (DataFrame): The data to check for bullish divergence.
        first_column (str): The column to check for divergence.
        second_column (str): The column to check for divergence.
        window_size (int): The windows size represent the
          total search space when checking for divergence. For example,
          if the window_size is 1, the function will consider only the
          current two data points, e.g. this will be true [1] and [-1]
          and false [0] and [-1]. If the window_size is 2,
            the function will consider the current and previous data point,
            e.g. this will be true [1, 0] and [0, -1]
            and false [0, 0] and [0, -1].
        number_of_data_points (int): The number of data points
            to consider when using a sliding windows size when checking for
          divergence. For example, if the number_of_data_points
          is 1, the function will consider only the current two data points.
          If the number_of_data_points is 4 and the window size is 2,
          the function will consider the current and previous 3 data
          points when checking for divergence. Then the function will
          slide the window by 1 and check the next 2 data points until
          the end of the data.
        result_column (str): The name of the column to store
            the bullish divergence results. Defaults to "bullish_divergence".
        number_of_neighbors_to_compare (int): The number of neighboring
            points to compare on each side to determine local peaks. This
            param is only used when the peaks and lows are not
            already calculated in the columns. If no peaks are detected,
            the function will use the `detect_peaks` function to
            calculate the peaks and lows in the columns.
        min_consecutive (int): Minimum number of consecutive peaks required
            to confirm a peak structure. This
            param is only used when the peaks and lows are not
            already calculated in the columns. If no peaks are detected,
            the function will use the `detect_peaks` function to
            calculate the peaks and lows in the columns.

    Returns:
        Boolean: True if there is a bullish divergence, False otherwise.
    """
    is_polars = isinstance(data, pl.DataFrame)
    df = data.to_pandas() if is_polars else data.copy()

    # Check if the two columns are in the data
    if first_column not in data.columns or second_column not in data.columns:
        raise PyIndicatorException(
            f"{first_column} and {second_column} columns "
            "are required in the data"
        )

    if window_size < 1:
        raise PyIndicatorException("Window size must be greater than 0")

    if len(data) < window_size:
        raise PyIndicatorException(
            f"Data must have at least {window_size} data points." +
            f"It currently has {len(data)} data points"
        )

    # Check if the highs and lows columns are present
    first_column_lows = f"{first_column}_lows"
    second_column_lows = f"{second_column}_lows"

    if first_column_lows not in data.columns:
        data = detect_peaks(
            data,
            source_column=first_column,
            number_of_neighbors_to_compare=number_of_neighbors_to_compare,
            min_consecutive=min_consecutive
        )

    if second_column_lows not in data.columns:
        data = detect_peaks(
            data,
            source_column=second_column,
            number_of_neighbors_to_compare=number_of_neighbors_to_compare,
            min_consecutive=min_consecutive
        )

    indicator_lows = df[f"{first_column}_lows"].values
    price_lows = df[f"{second_column}_lows"].values
    result = [False] * len(df)

    i = window_size - 1
    while i < len(df):
        window_a = indicator_lows[i - window_size + 1:i + 1]
        window_b = price_lows[i - window_size + 1:i + 1]

        if check_divergence_pattern(window_a, window_b, target_a=-1, target_b=1):
            result[i] = True
            i += window_size  # Skip forward to avoid repeated triggers in same window
        else:
            i += 1

    df[result_column] = result
    return pl.DataFrame(df) if is_polars else df


def bearish_divergence(
    data: Union[pd.DataFrame, pl.DataFrame],
    first_column: str,
    second_column: str,
    window_size=1,
    result_column: str = "bearish_divergence",
    number_of_neighbors_to_compare: int = 5,
    min_consecutive: int = 2
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Check for bearish divergence between two columns in a DataFrame.
    Given two columns in a DataFrame with peaks and lows,
    check if there is a bearish divergence. Usually the first column
    is the indicator column and the second column is the price column.

    Make sure that before calling this function, you have already
    identified the peaks and lows in both columns by using the
    `detect_peaks` function. If this is not the case, the function
    will automatically call the `detect_peaks` function to calculate
    the peaks and lows in the columns.

    The peaks are indicated as follows:
        * Higher Low = -1
        * Lower Low = 1
        * Higher High = 1
        * Lower High = -1

    Given that the highs columns are selected for both columns; For
    a bearish divergence:
        * First Column: Look for a lower high (-1) within the window.
        * Second Column: Look for a higher high (1) within the window.

    Args:
        data (DataFrame): The data to check for bearish divergence.
        first_column (str): The column to check for divergence.
        second_column (str): The column to check for divergence.
        window_size (int): The windows size represent the
          total search space when checking for divergence. For example,
          if the window_size is 1, the function will consider only the
          current two data points, e.g. this will be true [1] and [-1]
          and false [0] and [-1]. If the window_size is 2,
            the function will consider the current and previous data point,
            e.g. this will be true [1, 0] and [0, -1]
            and false [0, 0] and [0, -1].
        number_of_data_points (int): The number of data points
            to consider when using a sliding windows size when checking for
          divergence. For example, if the number_of_data_points
          is 1, the function will consider only the current two data points.
          If the number_of_data_points is 4 and the window size is 2,
          the function will consider the current and previous 3 data
          points when checking for divergence. Then the function will
          slide the window by 1 and check the next 2 data points until
          the end of the data.
        result_column (str): The name of the column to store
            the bearish divergence results. Defaults to "bearish_divergence".
        number_of_neighbors_to_compare (int): The number of neighboring
            points to compare on each side to determine local peaks. This
            param is only used when the peaks and lows are not
            already calculated in the columns. If no peaks are detected,
            the function will use the `detect_peaks` function to
            calculate the peaks and lows in the columns.
        min_consecutive (int): Minimum number of consecutive peaks required
            to confirm a peak structure. This
            param is only used when the peaks and lows are not
            already calculated in the columns. If no peaks are detected,
            the function will use the `detect_peaks` function to
            calculate the peaks and lows in the columns.

    Returns:
        Boolean: True if there is a bearish divergence, False otherwise.
    """
    is_polars = isinstance(data, pl.DataFrame)
    df = data.to_pandas() if is_polars else data.copy()

    # Check if the two columns are in the data
    if first_column not in data.columns or second_column not in data.columns:
        raise PyIndicatorException(
            f"{first_column} and {second_column} columns "
            "are required in the data"
        )

    if window_size < 1:
        raise PyIndicatorException("Window size must be greater than 0")

    if len(data) < window_size:
        raise PyIndicatorException(
            f"Data must have at least {window_size} data points." +
            f"It currently has {len(data)} data points"
        )

    # Check if the highs and lows columns are present
    first_column_highs = f"{first_column}_highs"
    second_column_highs = f"{second_column}_highs"

    if first_column_highs not in data.columns:
        data = detect_peaks(
            data,
            source_column=first_column,
            number_of_neighbors_to_compare=number_of_neighbors_to_compare,
            min_consecutive=min_consecutive
        )
    if second_column_highs not in data.columns:
        data = detect_peaks(
            data,
            source_column=second_column,
            number_of_neighbors_to_compare=number_of_neighbors_to_compare,
            min_consecutive=min_consecutive
        )

    indicator_highs = df[f"{first_column}_highs"].values
    price_highs = df[f"{second_column}_highs"].values
    result = [False] * len(df)

    i = window_size - 1
    while i < len(df):
        window_a = indicator_highs[i - window_size + 1:i + 1]
        window_b = price_highs[i - window_size + 1:i + 1]

        if check_divergence_pattern(window_a, window_b):
            result[i] = True
            i += window_size
        else:
            i += 1

    df[result_column] = result
    return pl.DataFrame(df) if is_polars else df
