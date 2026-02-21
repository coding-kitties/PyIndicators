"""
Swing Structure Scanner

Detects swing highs and swing lows in price data and classifies
the resulting market structure.

**Swing High:**
    Confirmed when the high at bar ``i`` is strictly greater than
    the highest high in the ``swing_length`` bars on **both** sides:
    i.e. ``high[i] > max(high[i-swing_length:i])`` and
    ``high[i] > max(high[i+1:i+swing_length+1])``.
    The point is only confirmed ``swing_length`` bars after the
    actual extremum (right side must be lower).

**Swing Low:**
    Mirror of swing high using the low array.

**Structure Classification:**
    Each swing high is classified as a *Higher High* (``HH``) or
    *Lower High* (``LH``) relative to the previous swing high.
    Each swing low is classified as a *Higher Low* (``HL``) or
    *Lower Low* (``LL``) relative to the previous swing low.

**Trend Direction:**
    * ``1``  – bullish (last swing was a higher low *or* higher
      high making HH/HL sequence)
    * ``-1`` – bearish (last swing was a lower high *or* lower low
      making LH/LL sequence)
    * ``0``  – undetermined

The core ``get_data`` logic:
    - Use rolling highest/lowest over ``swing_length`` bars.
    - A swing high is confirmed when ``high[swing_length]`` exceeds
      the rolling highest of the following ``swing_length`` bars.
    - Direction flips accordingly.
"""
from typing import Union, Dict
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


def swing_structure(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 5,
    lookback: int = 3,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    swing_high_column: str = "swing_high",
    swing_low_column: str = "swing_low",
    swing_high_price_column: str = "swing_high_price",
    swing_low_price_column: str = "swing_low_price",
    structure_column: str = "swing_structure",
    direction_column: str = "swing_direction",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect Swing Highs / Lows and classify the resulting market
    structure (HH, HL, LH, LL).

    The indicator identifies confirmed swing
    points and tracks whether the market is making *Higher Highs
    / Higher Lows* (bullish) or *Lower Highs / Lower Lows*
    (bearish).

    Args:
        data: pandas or polars DataFrame with OHLC price data.
        swing_length: Number of bars on each side required to
            confirm a swing point (default: 5).
        lookback: Number of most-recent swing highs / lows to
            track for structure classification (default: 3).
        high_column: Column name for highs (default: ``"High"``).
        low_column: Column name for lows (default: ``"Low"``).
        close_column: Column name for closes (default: ``"Close"``).
        swing_high_column: Output – ``1`` on the bar where a swing
            high is confirmed, else ``0`` (default:
            ``"swing_high"``).
        swing_low_column: Output – ``1`` on the bar where a swing
            low is confirmed, else ``0`` (default:
            ``"swing_low"``).
        swing_high_price_column: Output – the price of the
            confirmed swing high (``NaN`` otherwise)
            (default: ``"swing_high_price"``).
        swing_low_price_column: Output – the price of the
            confirmed swing low (``NaN`` otherwise)
            (default: ``"swing_low_price"``).
        structure_column: Output – string label for each
            confirmed swing: ``"HH"``, ``"HL"``, ``"LH"``,
            ``"LL"`` or ``""`` (default: ``"swing_structure"``).
        direction_column: Output – running trend direction:
            ``1`` (bullish), ``-1`` (bearish), ``0``
            (undetermined) (default: ``"swing_direction"``).

    Returns:
        DataFrame with added columns:

        - ``{swing_high_column}``: 1 when a swing high is
          confirmed, else 0.
        - ``{swing_low_column}``: 1 when a swing low is confirmed,
          else 0.
        - ``{swing_high_price_column}``: The high price of the
          swing point (NaN otherwise).
        - ``{swing_low_price_column}``: The low price of the swing
          point (NaN otherwise).
        - ``{structure_column}``: Structure label (``"HH"``,
          ``"HL"``, ``"LH"``, ``"LL"``, or ``""``).
        - ``{direction_column}``: Running trend direction (1 / -1
          / 0).

    Example:
        >>> import pandas as pd
        >>> from pyindicators import swing_structure
        >>> df = pd.DataFrame({
        ...     'High': [...],
        ...     'Low': [...],
        ...     'Close': [...]
        ... })
        >>> result = swing_structure(df, swing_length=5)
    """
    if isinstance(data, PdDataFrame):
        return _swing_structure_pandas(
            data,
            swing_length=swing_length,
            lookback=lookback,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            swing_high_column=swing_high_column,
            swing_low_column=swing_low_column,
            swing_high_price_column=swing_high_price_column,
            swing_low_price_column=swing_low_price_column,
            structure_column=structure_column,
            direction_column=direction_column,
        )
    elif isinstance(data, PlDataFrame):
        pd_data = data.to_pandas()
        result = _swing_structure_pandas(
            pd_data,
            swing_length=swing_length,
            lookback=lookback,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            swing_high_column=swing_high_column,
            swing_low_column=swing_low_column,
            swing_high_price_column=swing_high_price_column,
            swing_low_price_column=swing_low_price_column,
            structure_column=structure_column,
            direction_column=direction_column,
        )
        import polars as pl
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def swing_structure_signal(
    data: Union[PdDataFrame, PlDataFrame],
    swing_high_column: str = "swing_high",
    swing_low_column: str = "swing_low",
    structure_column: str = "swing_structure",
    signal_column: str = "swing_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a combined trading signal from swing structure results.

    Args:
        data: DataFrame containing swing structure columns (output
            of :func:`swing_structure`).
        swing_high_column: Column with swing high flags.
        swing_low_column: Column with swing low flags.
        structure_column: Column with structure labels.
        signal_column: Output column name (default:
            ``"swing_signal"``).

    Returns:
        DataFrame with ``{signal_column}`` added:

        - ``1`` – bullish structure (HL or HH detected)
        - ``-1`` – bearish structure (LH or LL detected)
        - ``0`` – no signal
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        signal = np.where(
            data[structure_column].isin(["HL", "HH"]),
            1,
            np.where(
                data[structure_column].isin(["LH", "LL"]),
                -1,
                0,
            ),
        )
        data[signal_column] = signal.astype(int)
        return data
    elif isinstance(data, PlDataFrame):
        import polars as pl
        return data.with_columns(
            pl.when(pl.col(structure_column).is_in(["HL", "HH"]))
            .then(1)
            .when(pl.col(structure_column).is_in(["LH", "LL"]))
            .then(-1)
            .otherwise(0)
            .alias(signal_column)
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def get_swing_structure_stats(
    data: Union[PdDataFrame, PlDataFrame],
    swing_high_column: str = "swing_high",
    swing_low_column: str = "swing_low",
    structure_column: str = "swing_structure",
) -> Dict:
    """
    Return summary statistics for swing structure results.

    Args:
        data: DataFrame containing swing structure columns (output
            of :func:`swing_structure`).
        swing_high_column: Column with swing high flags.
        swing_low_column: Column with swing low flags.
        structure_column: Column with structure labels.

    Returns:
        Dictionary with keys:

        - ``total_swing_highs``
        - ``total_swing_lows``
        - ``total_swings``
        - ``higher_highs``
        - ``lower_highs``
        - ``higher_lows``
        - ``lower_lows``
        - ``bullish_ratio`` (0-1, proportion of HH + HL among
          classified swings)
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    total_sh = int(data[swing_high_column].sum())
    total_sl = int(data[swing_low_column].sum())
    hh = int((data[structure_column] == "HH").sum())
    lh = int((data[structure_column] == "LH").sum())
    hl = int((data[structure_column] == "HL").sum())
    ll = int((data[structure_column] == "LL").sum())
    classified = hh + lh + hl + ll
    return {
        "total_swing_highs": total_sh,
        "total_swing_lows": total_sl,
        "total_swings": total_sh + total_sl,
        "higher_highs": hh,
        "lower_highs": lh,
        "higher_lows": hl,
        "lower_lows": ll,
        "bullish_ratio": (
            round((hh + hl) / classified, 4) if classified > 0 else 0.0
        ),
    }


# -------------------------------------------------------------------
#  Internal implementation
# -------------------------------------------------------------------

def _swing_structure_pandas(
    data: PdDataFrame,
    swing_length: int,
    lookback: int,
    high_column: str,
    low_column: str,
    close_column: str,
    swing_high_column: str,
    swing_low_column: str,
    swing_high_price_column: str,
    swing_low_price_column: str,
    structure_column: str,
    direction_column: str,
) -> PdDataFrame:
    """Core pandas implementation of swing structure detection.

    1.  For each bar *i*, compute the rolling highest high and
        lowest low of the **following** ``swing_length`` bars
        (bars ``i-swing_length+1 … i``).
    2.  Look back ``swing_length`` bars: if
        ``high[i - swing_length]`` > rolling highest → swing high.
    3.  Similarly for swing lows.
    4.  Classify each swing relative to the previous swing of the
        same type (HH / LH / HL / LL).
    5.  Update the running direction.
    """
    data = data.copy()
    high = data[high_column].values.astype(float)
    low = data[low_column].values.astype(float)
    n = len(data)

    # Output arrays
    swing_high_arr = np.zeros(n, dtype=int)
    swing_low_arr = np.zeros(n, dtype=int)
    swing_high_price = np.full(n, np.nan)
    swing_low_price = np.full(n, np.nan)
    structure_arr = np.full(n, "", dtype=object)
    direction_arr = np.zeros(n, dtype=int)

    # Detect raw pivot positions
    pivot_high_positions = _detect_pivots_high(high, swing_length)
    pivot_low_positions = _detect_pivots_low(low, swing_length)

    # State tracking
    direction = 0  # 0 = undetermined, 1 = bullish, -1 = bearish
    prev_swing_high_price = np.nan
    prev_swing_low_price = np.nan

    # Track last `lookback` swings for structure
    recent_highs = []  # list of (bar_index, price)
    recent_lows = []   # list of (bar_index, price)

    for i in range(n):
        # The confirmation bar for a pivot at position p is
        # p + swing_length.  So on bar i we check p = i - swing_length.
        p = i - swing_length
        if p < swing_length:
            direction_arr[i] = direction
            continue

        confirmed_high = pivot_high_positions[p]
        confirmed_low = pivot_low_positions[p]

        # --- Swing High confirmed ---
        if confirmed_high:
            sh_price = high[p]
            swing_high_arr[i] = 1
            swing_high_price[i] = sh_price

            # Classify structure
            if not np.isnan(prev_swing_high_price):
                if sh_price > prev_swing_high_price:
                    structure_arr[i] = "HH"
                else:
                    structure_arr[i] = "LH"

            prev_swing_high_price = sh_price
            recent_highs.append((i, sh_price))
            if len(recent_highs) > lookback:
                recent_highs.pop(0)

        # --- Swing Low confirmed ---
        if confirmed_low:
            sl_price = low[p]
            swing_low_arr[i] = 1
            swing_low_price[i] = sl_price

            # Classify structure
            if not np.isnan(prev_swing_low_price):
                if sl_price > prev_swing_low_price:
                    structure_arr[i] = "HL"
                else:
                    structure_arr[i] = "LL"

            prev_swing_low_price = sl_price
            recent_lows.append((i, sl_price))
            if len(recent_lows) > lookback:
                recent_lows.pop(0)

        # Update direction based on structure
        label = structure_arr[i]
        if label in ("HH", "HL"):
            direction = 1
        elif label in ("LH", "LL"):
            direction = -1

        direction_arr[i] = direction

    # Assign results
    data[swing_high_column] = swing_high_arr
    data[swing_low_column] = swing_low_arr
    data[swing_high_price_column] = swing_high_price
    data[swing_low_price_column] = swing_low_price
    data[structure_column] = structure_arr
    data[direction_column] = direction_arr

    return data


def _detect_pivots_high(
    high: np.ndarray, length: int
) -> np.ndarray:
    """
    Return a boolean array where ``True`` at position *p* means
    ``high[p]`` is greater than or equal to all highs in
    ``[p - length, p + length]`` (a swing high).
    """
    n = len(high)
    result = np.zeros(n, dtype=bool)

    for i in range(length, n - length):
        window = high[i - length: i + length + 1]
        if high[i] >= np.max(window):
            result[i] = True

    return result


def _detect_pivots_low(
    low: np.ndarray, length: int
) -> np.ndarray:
    """
    Return a boolean array where ``True`` at position *p* means
    ``low[p]`` is less than or equal to all lows in
    ``[p - length, p + length]`` (a swing low).
    """
    n = len(low)
    result = np.zeros(n, dtype=bool)

    for i in range(length, n - length):
        window = low[i - length: i + length + 1]
        if low[i] <= np.min(window):
            result[i] = True

    return result
