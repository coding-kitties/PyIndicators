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

        # Fix: Filter indices to ensure they're within bounds and
        # convert to list
        valid_hh_idx = [i for i in hh_idx if 0 <= i < len(data)]
        valid_lh_idx = [i for i in lh_idx if 0 <= i < len(data)]
        valid_ll_idx = [i for i in ll_idx if 0 <= i < len(data)]
        valid_hl_idx = [i for i in hl_idx if 0 <= i < len(data)]

        # Use iloc for integer-based indexing instead of loc with index arrays
        if len(valid_hh_idx) > 0:
            data.iloc[valid_hh_idx, data.columns.get_loc(
                f"{source_column}_highs")] = 1
        if len(valid_lh_idx) > 0:
            data.iloc[valid_lh_idx, data.columns.get_loc(
                f"{source_column}_highs")] = -1
        if len(valid_ll_idx) > 0:
            data.iloc[valid_ll_idx, data.columns.get_loc(
                f"{source_column}_lows")] = 1
        if len(valid_hl_idx) > 0:
            data.iloc[valid_hl_idx, data.columns.get_loc(
                f"{source_column}_lows")] = -1

        return data

    elif isinstance(data, pl.DataFrame):
        highs_col = np.full(len(data), np.nan)
        lows_col = np.full(len(data), np.nan)

        # Filter indices for polars as well
        valid_hh_idx = [i for i in hh_idx if 0 <= i < len(data)]
        valid_lh_idx = [i for i in lh_idx if 0 <= i < len(data)]
        valid_ll_idx = [i for i in ll_idx if 0 <= i < len(data)]
        valid_hl_idx = [i for i in hl_idx if 0 <= i < len(data)]

        if len(valid_hh_idx) > 0:
            highs_col[valid_hh_idx] = 1
        if len(valid_lh_idx) > 0:
            highs_col[valid_lh_idx] = -1
        if len(valid_ll_idx) > 0:
            lows_col[valid_ll_idx] = 1
        if len(valid_hl_idx) > 0:
            lows_col[valid_hl_idx] = -1

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
    # Convert to flat numpy arrays for consistent integer indexing
    # This handles pandas Series with DatetimeIndex correctly
    if hasattr(series_a, 'values'):
        series_a = series_a.values.flatten()
    elif hasattr(series_a, 'flatten'):
        series_a = series_a.flatten()
    if hasattr(series_b, 'values'):
        series_b = series_b.values.flatten()
    elif hasattr(series_b, 'flatten'):
        series_b = series_b.flatten()

    try:
        # Find the first index of `target_a` (e.g., -1 in the indicator)
        a_index = next(i for i, val in enumerate(series_a) if val == target_a)
    except StopIteration:
        return False

    # From that point forward, check if series_b has a target_b
    for j in range(a_index, len(series_b)):
        val = series_b[j]
        # Handle numpy scalar comparison explicitly
        if hasattr(val, 'item'):
            val = val.item()
        if val == -1:
            return False  # Higher low before lower low — invalid
        if val == target_b:
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

    # Check if the highs and lows columns are present
    first_column_lows = f"{first_column}_lows"
    second_column_lows = f"{second_column}_lows"

    if first_column_lows not in data.columns \
            or second_column_lows not in data.columns:

        # Check if the two columns are in the data
        if first_column not in data.columns \
                or second_column not in data.columns:
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

    # Skip forward to avoid repeated triggers in same window
    i = window_size - 1
    while i < len(df):
        window_a = indicator_lows[i - window_size + 1:i + 1]
        window_b = price_lows[i - window_size + 1:i + 1]

        if check_divergence_pattern(
            window_a, window_b, target_a=-1, target_b=1
        ):
            result[i] = True
            i += window_size
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

    # Check if the highs and lows columns are present
    first_column_highs = f"{first_column}_highs"
    second_column_highs = f"{second_column}_highs"

    if first_column_highs not in data.columns \
            or second_column_highs not in data.columns:

        # Check if the two columns are in the data
        if first_column not in data.columns \
                or second_column not in data.columns:
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


def bearish_divergence_multi_dataframe(
    first_df: Union[pd.DataFrame, pl.DataFrame],
    second_df: Union[pd.DataFrame, pl.DataFrame],
    result_df: Union[pd.DataFrame, pl.DataFrame],
    first_column: str,
    second_column: str,
    window_size: int = 1,
    result_column: str = "bearish_divergence",
    number_of_neighbors_to_compare: int = 5,
    min_consecutive: int = 2
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Detect bearish divergence between two different DataFrames.

    Given that the highs columns are selected for both columns; For
    a bearish divergence:
        * First Column of the first dataframe: Look for a lower
        high (-1) within the window.
        * Second Column of the second dataframe: Look for a higher
        high (1) within the window.

    Args:
        first_df: DataFrame containing the indicator data (e.g., RSI).
        second_df: DataFrame containing the price data.
        result_df: DataFrame used to store results. Must be aligned in time.
        first_column: Column in first_df (e.g., RSI).
        second_column: Column in second_df (e.g., price).
        window_size: Number of bars to consider for pattern.
        result_column: Output column name.
        number_of_neighbors_to_compare: For peak detection.
        min_consecutive: Minimum consecutive peaks required.

    Returns:
        A DataFrame with a new column indicating bearish divergence.
    """
    is_polars = isinstance(first_df, pl.DataFrame)

    if is_polars:
        first_df = first_df.to_pandas()
        second_df = second_df.to_pandas()
        result_df = result_df.to_pandas()

    # Validate columns
    for df, col, label in [
        (first_df, first_column, "first_df"),
        (second_df, second_column, "second_df")
    ]:
        high_column = f"{col}_highs"

        if high_column not in df.columns and col not in df.columns:
            raise PyIndicatorException(f"{col} column is missing in {label}")

    # Determine which df has more granular datetime index
    first_freq = first_df.index.to_series().diff().median()
    second_freq = second_df.index.to_series().diff().median()

    if first_freq < second_freq:
        align_index = first_df.index
    else:
        align_index = second_df.index

    if len(result_df) != len(align_index):
        raise PyIndicatorException(
            "result_df must have the same length as the aligned index"
        )

    # Reindex all DataFrames to the most granular one
    first_df = first_df.reindex(align_index)
    second_df = second_df.reindex(align_index)

    # Peak detection
    first_highs_col = f"{first_column}_highs"
    second_highs_col = f"{second_column}_highs"

    if first_highs_col not in first_df.columns:
        first_df = detect_peaks(
            first_df,
            source_column=first_column,
            number_of_neighbors_to_compare=number_of_neighbors_to_compare,
            min_consecutive=min_consecutive
        )

    if second_highs_col not in second_df.columns:
        second_df = detect_peaks(
            second_df,
            source_column=second_column,
            number_of_neighbors_to_compare=number_of_neighbors_to_compare,
            min_consecutive=min_consecutive
        )

    # Now align and merge
    merged_df = pd.concat([
        first_df[[first_highs_col]],
        second_df[[second_highs_col]],
        result_df.copy()
    ], axis=1, join='inner')

    # Validate enough data
    if len(merged_df) < window_size:
        raise PyIndicatorException(
            f"Not enough data points (need at least {window_size}, "
            f"got {len(merged_df)})"
        )

    # Apply divergence detection
    indicator_highs = merged_df[first_highs_col].values
    price_highs = merged_df[second_highs_col].values
    result = [False] * len(merged_df)

    i = window_size - 1
    while i < len(merged_df):
        win_a = indicator_highs[i - window_size + 1:i + 1]
        win_b = price_highs[i - window_size + 1:i + 1]

        if check_divergence_pattern(win_a, win_b):
            result[i] = True
            i += window_size  # Skip forward to avoid overlap
        else:
            i += 1

    merged_df[result_column] = result

    # Merge back result column to result_df using the original index
    result_df[result_column] = merged_df[result_column]

    return pl.DataFrame(result_df) if is_polars else result_df


def bullish_divergence_multi_dataframe(
    first_df: Union[pd.DataFrame, pl.DataFrame],
    second_df: Union[pd.DataFrame, pl.DataFrame],
    result_df: Union[pd.DataFrame, pl.DataFrame],
    first_column: str,
    second_column: str,
    window_size: int = 1,
    result_column: str = "bearish_divergence",
    number_of_neighbors_to_compare: int = 5,
    min_consecutive: int = 2
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Detect bullish divergence between two different DataFrames.

    Given that the low columns are selected for both columns; For
    a bullish divergence:
        * First Column: Look for a higher low (-1) within the window.
        * Second Column: Look for a lower low (1) within the window.

    Args:
        first_df: DataFrame containing the indicator data (e.g., RSI).
        second_df: DataFrame containing the price data.
        result_df: DataFrame used to store results. Must be aligned in time.
        first_column: Column in first_df (e.g., RSI).
        second_column: Column in second_df (e.g., price).
        window_size: Number of bars to consider for pattern.
        result_column: Output column name.
        number_of_neighbors_to_compare: For peak detection.
        min_consecutive: Minimum consecutive peaks required.

    Returns:
        A DataFrame with a new column indicating bullish divergence.
    """
    is_polars = isinstance(first_df, pl.DataFrame)

    if is_polars:
        first_df = first_df.to_pandas()
        second_df = second_df.to_pandas()
        result_df = result_df.to_pandas()

    # Validate columns
    for df, col, label in [
        (first_df, first_column, "first_df"),
        (second_df, second_column, "second_df")
    ]:
        lows_column = f"{col}_lows"

        if lows_column not in df.columns and col not in df.columns:
            raise PyIndicatorException(f"{col} column is missing in {label}")

    # Determine which df has more granular datetime index
    first_freq = first_df.index.to_series().diff().median()
    second_freq = second_df.index.to_series().diff().median()

    if first_freq < second_freq:
        align_index = first_df.index
    else:
        align_index = second_df.index

    if len(result_df) != len(align_index):
        raise PyIndicatorException(
            "result_df must have the same length as the aligned index"
        )

    # Reindex all DataFrames to the most granular one
    first_df = first_df.reindex(align_index)
    second_df = second_df.reindex(align_index)

    # Peak detection
    first_lows_col = f"{first_column}_lows"
    second_lows_col = f"{second_column}_lows"

    if first_lows_col not in first_df.columns:
        first_df = detect_peaks(
            first_df,
            source_column=first_column,
            number_of_neighbors_to_compare=number_of_neighbors_to_compare,
            min_consecutive=min_consecutive
        )

    if second_lows_col not in second_df.columns:
        second_df = detect_peaks(
            second_df,
            source_column=second_column,
            number_of_neighbors_to_compare=number_of_neighbors_to_compare,
            min_consecutive=min_consecutive
        )

    # Now align and merge
    merged_df = pd.concat([
        first_df[[first_lows_col]],
        second_df[[second_lows_col]],
        result_df.copy()
    ], axis=1, join='inner')

    # Validate enough data
    if len(merged_df) < window_size:
        raise PyIndicatorException(
            f"Not enough data points (need at least {window_size}, "
            f"got {len(merged_df)})"
        )

    # Apply divergence detection
    indicator_lows = merged_df[first_lows_col].values
    price_lows = merged_df[second_lows_col].values
    result = [False] * len(merged_df)

    i = window_size - 1
    while i < len(merged_df):
        win_a = indicator_lows[i - window_size + 1:i + 1]
        win_b = price_lows[i - window_size + 1:i + 1]

        if check_divergence_pattern(win_a, win_b):
            result[i] = True
            i += window_size  # Skip forward to avoid overlap
        else:
            i += 1

    merged_df[result_column] = result

    # Merge back result column to result_df using the original index
    result_df[result_column] = merged_df[result_column]

    return pl.DataFrame(result_df) if is_polars else result_df
