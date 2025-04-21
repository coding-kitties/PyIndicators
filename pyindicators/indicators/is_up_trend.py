from typing import Union

from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from .exponential_moving_average import ema


def is_up_trend(
    data: Union[PdDataFrame, PlDataFrame],
    use_golden_cross: bool = True,
) -> bool:
    """
    Check if the market is in a uptrend using various indicators.

    The function decides whether it is a uptrend based on a set of
    weighted indicators.

    If the value of the sum of the indicators is greater than 0.5,
    it is considered a uptrend.

    Args:
        data: Pandas or Polars DataFrame
        use_golden_cross: If True, use the golden cross indicator
            to check for a golden cross

    Returns:
        bool: True if the market is in a uptrend, False otherwise
    """

    weights = {
        "golden_cross": {}
    }

    source_data = data.copy()

    if use_golden_cross:
        # Check for a golden cross
        source_data = ema(
            source_data,
            source_column="Close",
            period=50,
            result_column="EMA_CLOSE_50"
        )
        source_data = ema(
            source_data,
            source_column="Close",
            period=200,
            result_column="EMA_CLOSE_200"
        )
        golden_cross = source_data["EMA_CLOSE_50"].iloc[-1] \
            < source_data["EMA_CLOSE_200"].iloc[-1]

        if golden_cross:
            weights["golden_cross"]["value"] = 1
        else:
            weights["golden_cross"]["value"] = 0

        weights["golden_cross"]["weight"] = 1

    # Calculate the weighted sum of the indicators
    weighted_sum = 0

    for indicator, values in weights.items():
        weighted_sum += values["value"] * values["weight"]

    # Check if the weighted sum is greater than 0.5
    if weighted_sum > 0.5:
        return True
    else:
        return False
