"""
Accumulation & Distribution Zones Indicator

Detects Wyckoff-style **Accumulation** and **Distribution** zone
patterns by identifying converging swing structures — alternating
pivot highs and lows that form a narrowing price range (triangle /
wedge).

Based on the "Price Action Concepts [RUDYINDICATOR]" methodology:

- **Accumulation Zone (bullish):**  A sequence of alternating swing
  lows and highs where lows are *rising* (higher lows) and highs
  are *falling* (lower highs).  This converging pattern signals
  that smart money is accumulating positions and a bullish breakout
  is likely.

- **Distribution Zone (bearish):**  A sequence of alternating swing
  highs and lows where highs are *falling* (lower highs) and lows
  are *rising* (higher lows).  This converging pattern signals
  that smart money is distributing positions and a bearish breakdown
  is likely.

The two patterns are distinguished by the *starting* swing
direction, reflecting their Wyckoff context:

    - Accumulation starts from a swing low (buying from the bottom)
    - Distribution starts from a swing high (selling from the top)

Two detection modes are supported:

    - **Fast** – requires 4 alternating swing points (2 highs +
      2 lows) to confirm the pattern.
    - **Slow** – requires 6 alternating swing points (3 highs +
      3 lows) for higher-conviction confirmation.
"""
from typing import Union, Dict
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


# -------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------

def accumulation_distribution_zones(
    data: Union[PdDataFrame, PlDataFrame],
    pivot_length: int = 5,
    mode: str = "fast",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    accumulation_column: str = "adz_accumulation",
    distribution_column: str = "adz_distribution",
    zone_top_column: str = "adz_zone_top",
    zone_bottom_column: str = "adz_zone_bottom",
    zone_left_column: str = "adz_zone_left",
    zone_right_column: str = "adz_zone_right",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect Accumulation and Distribution zones on OHLC data.

    Identifies converging swing structures where alternating pivot
    highs are falling and pivot lows are rising, forming a
    narrowing price range.

    Args:
        data: pandas or polars DataFrame with OHLC data.
        pivot_length: Left and right lookback for pivot detection
            (default: 5).  A pivot high at bar *i* requires
            ``high[i]`` ≥ all highs in ``[i-pivot_length …
            i+pivot_length]``.
        mode: Detection mode — ``"fast"`` requires 4 alternating
            swing points, ``"slow"`` requires 6 (default:
            ``"fast"``).
        high_column: Column name for highs (default: ``"High"``).
        low_column: Column name for lows (default: ``"Low"``).
        close_column: Column name for closes (default: ``"Close"``).
        accumulation_column: Output column – 1 on the bar where an
            accumulation zone is confirmed (default:
            ``"adz_accumulation"``).
        distribution_column: Output column – 1 on the bar where a
            distribution zone is confirmed (default:
            ``"adz_distribution"``).
        zone_top_column: Output column – top price of the zone
            bounding box (default: ``"adz_zone_top"``).
        zone_bottom_column: Output column – bottom price of the zone
            bounding box (default: ``"adz_zone_bottom"``).
        zone_left_column: Output column – bar index of the oldest
            swing point in the zone (default: ``"adz_zone_left"``).
        zone_right_column: Output column – bar index of the most
            recent swing point in the zone (default:
            ``"adz_zone_right"``).

    Returns:
        DataFrame with added columns:

        - ``{accumulation_column}``: 1 when an accumulation zone is
          detected, else 0.
        - ``{distribution_column}``: 1 when a distribution zone is
          detected, else 0.
        - ``{zone_top_column}``: Top price boundary of the
          detected zone (NaN otherwise).
        - ``{zone_bottom_column}``: Bottom price boundary of the
          detected zone (NaN otherwise).
        - ``{zone_left_column}``: Integer bar index of the leftmost
          (oldest) swing point in the zone (NaN otherwise).
        - ``{zone_right_column}``: Integer bar index of the
          rightmost (most recent) swing point in the zone (NaN
          otherwise).

    Example:
        >>> import pandas as pd
        >>> from pyindicators import accumulation_distribution_zones
        >>> df = pd.DataFrame({
        ...     'High': [...],
        ...     'Low': [...],
        ...     'Close': [...]
        ... })
        >>> result = accumulation_distribution_zones(df, pivot_length=5)
    """
    if mode not in ("fast", "slow"):
        raise PyIndicatorException(
            "mode must be 'fast' or 'slow'."
        )

    if isinstance(data, PdDataFrame):
        return _adz_pandas(
            data,
            pivot_length=pivot_length,
            mode=mode,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            accumulation_column=accumulation_column,
            distribution_column=distribution_column,
            zone_top_column=zone_top_column,
            zone_bottom_column=zone_bottom_column,
            zone_left_column=zone_left_column,
            zone_right_column=zone_right_column,
        )
    elif isinstance(data, PlDataFrame):
        pd_data = data.to_pandas()
        result = _adz_pandas(
            pd_data,
            pivot_length=pivot_length,
            mode=mode,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            accumulation_column=accumulation_column,
            distribution_column=distribution_column,
            zone_top_column=zone_top_column,
            zone_bottom_column=zone_bottom_column,
            zone_left_column=zone_left_column,
            zone_right_column=zone_right_column,
        )
        import polars as pl
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def accumulation_distribution_zones_signal(
    data: Union[PdDataFrame, PlDataFrame],
    accumulation_column: str = "adz_accumulation",
    distribution_column: str = "adz_distribution",
    signal_column: str = "adz_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a combined signal column from Accumulation/Distribution
    zone results.

    Args:
        data: DataFrame containing accumulation/distribution zone
            columns (output of
            :func:`accumulation_distribution_zones`).
        accumulation_column: Column with accumulation flags.
        distribution_column: Column with distribution flags.
        signal_column: Output column name (default:
            ``"adz_signal"``).

    Returns:
        DataFrame with ``{signal_column}`` added:

        - ``1``  – Accumulation zone detected (bullish)
        - ``-1`` – Distribution zone detected (bearish)
        - ``0``  – no signal
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        signal = np.where(
            data[accumulation_column] == 1,
            1,
            np.where(data[distribution_column] == 1, -1, 0),
        )
        data[signal_column] = signal
        return data
    elif isinstance(data, PlDataFrame):
        import polars as pl
        return data.with_columns(
            pl.when(pl.col(accumulation_column) == 1)
            .then(1)
            .when(pl.col(distribution_column) == 1)
            .then(-1)
            .otherwise(0)
            .alias(signal_column)
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def get_accumulation_distribution_zones_stats(
    data: Union[PdDataFrame, PlDataFrame],
    accumulation_column: str = "adz_accumulation",
    distribution_column: str = "adz_distribution",
) -> Dict:
    """
    Return summary statistics for detected Accumulation/Distribution
    zones.

    Args:
        data: DataFrame containing accumulation/distribution zone
            columns (output of
            :func:`accumulation_distribution_zones`).
        accumulation_column: Column with accumulation flags.
        distribution_column: Column with distribution flags.

    Returns:
        Dictionary with keys:

        - ``total_accumulation``
        - ``total_distribution``
        - ``total``
        - ``accumulation_ratio`` (0–1)
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    acc = int(data[accumulation_column].sum())
    dist = int(data[distribution_column].sum())
    total = acc + dist
    return {
        "total_accumulation": acc,
        "total_distribution": dist,
        "total": total,
        "accumulation_ratio": round(acc / total, 4) if total > 0 else 0.0,
    }


# -------------------------------------------------------------------
#  Internal helpers
# -------------------------------------------------------------------

def _detect_pivot_highs(high: np.ndarray, length: int) -> np.ndarray:
    """
    Detect pivot highs.  A pivot high at index *i* means
    ``high[i]`` ≥ all highs in ``[i-length … i+length]``.

    Returns an array where confirmed pivot positions contain the
    pivot price and all others are ``NaN``.
    """
    n = len(high)
    pivots = np.full(n, np.nan)

    for i in range(length, n - length):
        window = high[i - length: i + length + 1]
        if high[i] == np.max(window):
            pivots[i] = high[i]

    return pivots


def _detect_pivot_lows(low: np.ndarray, length: int) -> np.ndarray:
    """Detect pivot lows (mirror of ``_detect_pivot_highs``)."""
    n = len(low)
    pivots = np.full(n, np.nan)

    for i in range(length, n - length):
        window = low[i - length: i + length + 1]
        if low[i] == np.min(window):
            pivots[i] = low[i]

    return pivots


def _adz_pandas(
    data: PdDataFrame,
    pivot_length: int,
    mode: str,
    high_column: str,
    low_column: str,
    close_column: str,
    accumulation_column: str,
    distribution_column: str,
    zone_top_column: str,
    zone_bottom_column: str,
    zone_left_column: str,
    zone_right_column: str,
) -> PdDataFrame:
    """Core pandas implementation of Accumulation/Distribution Zones."""
    data = data.copy()
    high = data[high_column].values.astype(float)
    low = data[low_column].values.astype(float)
    n = len(data)

    # Detect pivots
    pivot_highs = _detect_pivot_highs(high, pivot_length)
    pivot_lows = _detect_pivot_lows(low, pivot_length)

    # Output arrays
    acc_arr = np.zeros(n, dtype=int)
    dist_arr = np.zeros(n, dtype=int)
    zone_top = np.full(n, np.nan)
    zone_bottom = np.full(n, np.nan)
    zone_left_arr = np.full(n, np.nan)
    zone_right_arr = np.full(n, np.nan)

    # Required number of alternating points
    required = 6 if mode == "slow" else 4

    # Track alternating swing points as (price, bar_index, type)
    # type: 1 = pivot high, -1 = pivot low
    # We store them most-recent-first (like Pine Script's unshift)
    swing_points: list = []  # [(price, bar_idx, swing_type), ...]

    for i in range(n):
        is_ph = not np.isnan(pivot_highs[i])
        is_pl = not np.isnan(pivot_lows[i])

        # If both a high and low occur at same bar, process the high
        # first then the low (arbitrary but consistent)
        new_swings = []
        if is_ph:
            new_swings.append((pivot_highs[i], i, 1))
        if is_pl:
            new_swings.append((pivot_lows[i], i, -1))

        for price, idx, stype in new_swings:
            # Only add if it alternates with the previous swing
            if len(swing_points) == 0 or swing_points[0][2] != stype:
                swing_points.insert(0, (price, idx, stype))
            else:
                # Same type as the last one — replace it if this is
                # a more extreme value (higher high or lower low)
                if stype == 1 and price >= swing_points[0][0]:
                    swing_points[0] = (price, idx, stype)
                elif stype == -1 and price <= swing_points[0][0]:
                    swing_points[0] = (price, idx, stype)

            # If we have two consecutive same-type swings (shouldn't
            # happen after the above check, but guard), clear
            if len(swing_points) > 1:
                if swing_points[0][2] == swing_points[1][2]:
                    swing_points.clear()
                    continue

            # Check for pattern when we have enough points
            if len(swing_points) >= required:
                # Confirmation bar is the current bar (where the
                # most recent pivot is confirmed)
                confirm_bar = i + pivot_length
                if confirm_bar >= n:
                    confirm_bar = n - 1

                found_acc, found_dist = _check_pattern(
                    swing_points, required
                )

                if found_acc:
                    acc_arr[confirm_bar] = 1
                    # Zone boundaries: from oldest to newest point
                    oldest_idx = required - 1
                    zone_top[confirm_bar] = swing_points[oldest_idx][0]
                    zone_bottom[confirm_bar] = swing_points[oldest_idx - 1][0]
                    zone_left_arr[confirm_bar] = float(
                        swing_points[oldest_idx][1]
                    )
                    zone_right_arr[confirm_bar] = float(
                        swing_points[0][1]
                    )
                    swing_points.clear()

                elif found_dist:
                    dist_arr[confirm_bar] = 1
                    oldest_idx = required - 1
                    zone_top[confirm_bar] = swing_points[oldest_idx - 1][0]
                    zone_bottom[confirm_bar] = swing_points[oldest_idx][0]
                    zone_left_arr[confirm_bar] = float(
                        swing_points[oldest_idx][1]
                    )
                    zone_right_arr[confirm_bar] = float(
                        swing_points[0][1]
                    )
                    swing_points.clear()

        # Keep the list from growing unbounded
        if len(swing_points) > required + 2:
            swing_points[:] = swing_points[:required + 2]

    # Assign results
    data[accumulation_column] = acc_arr
    data[distribution_column] = dist_arr
    data[zone_top_column] = zone_top
    data[zone_bottom_column] = zone_bottom
    data[zone_left_column] = zone_left_arr
    data[zone_right_column] = zone_right_arr

    return data


def _check_pattern(
    swings: list,
    required: int,
) -> tuple:
    """
    Check whether the most recent *required* swing points form an
    accumulation or distribution pattern.

    Accumulation (fast, required=4):
        Points [0..3] most-recent-first:
        - [0] = low, [1] = high, [2] = low, [3] = high
        - [0].price > [2].price  (higher lows)
        - [1].price < [3].price  (lower highs)

    Distribution (fast, required=4):
        Points [0..3] most-recent-first:
        - [0] = high, [1] = low, [2] = high, [3] = low
        - [0].price < [2].price  (lower highs)
        - [1].price > [3].price  (higher lows)

    Slow mode (required=6) extends the same pattern to 6 points
    with 3 consecutive comparisons.

    Returns:
        (is_accumulation, is_distribution) booleans.
    """
    if len(swings) < required:
        return False, False

    # Extract the relevant points
    pts = swings[:required]

    if required == 4:
        # Fast mode
        # Accumulation: low, high, low, high (most recent first)
        if (pts[0][2] == -1 and pts[1][2] == 1 and
                pts[2][2] == -1 and pts[3][2] == 1):
            # Higher lows and lower highs
            if pts[0][0] > pts[2][0] and pts[1][0] < pts[3][0]:
                return True, False

        # Distribution: high, low, high, low (most recent first)
        if (pts[0][2] == 1 and pts[1][2] == -1 and
                pts[2][2] == 1 and pts[3][2] == -1):
            # Lower highs and higher lows
            if pts[0][0] < pts[2][0] and pts[1][0] > pts[3][0]:
                return False, True

    elif required == 6:
        # Slow mode
        # Accumulation: low, high, low, high, low, high
        if (pts[0][2] == -1 and pts[1][2] == 1 and
                pts[2][2] == -1 and pts[3][2] == 1 and
                pts[4][2] == -1 and pts[5][2] == 1):
            # Successively higher lows and lower highs
            if (pts[0][0] > pts[2][0] and pts[2][0] > pts[4][0] and
                    pts[1][0] < pts[3][0] and pts[3][0] < pts[5][0]):
                return True, False

        # Distribution: high, low, high, low, high, low
        if (pts[0][2] == 1 and pts[1][2] == -1 and
                pts[2][2] == 1 and pts[3][2] == -1 and
                pts[4][2] == 1 and pts[5][2] == -1):
            # Successively lower highs and higher lows
            if (pts[0][0] < pts[2][0] and pts[2][0] < pts[4][0] and
                    pts[1][0] > pts[3][0] and pts[3][0] > pts[5][0]):
                return False, True

    return False, False
