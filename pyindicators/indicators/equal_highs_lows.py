"""
Equal Highs & Lows (EQH/EQL) Indicator

Identifies Equal Highs and Equal Lows — consecutive swing pivots
that form at approximately the same price level.  These are
important in Smart Money Concepts because they signal resting
liquidity:

- **Equal Highs (EQH):**  Two (or more) consecutive pivot highs
  within an ATR-based threshold of each other.  They mark a
  resistance level where sell-side liquidity clusters above.
- **Equal Lows (EQL):**  Two (or more) consecutive pivot lows
  within an ATR-based threshold.  They mark a support level where
  buy-side liquidity clusters below.

Methodology (based on "Equal Highs/Lows [UAlgo]"):
    1.  Detect pivot highs and pivot lows using a configurable
        lookback/look-ahead window (``pivot_length``).
    2.  Compute ATR over a long period (``atr_length``, default 200)
        to normalise the threshold.
    3.  For each new pivot high, compare with the *previous* pivot
        high.  If the absolute difference between the two is ≤
        ``atr * threshold``, flag both as forming an **Equal High**.
    4.  Same logic for pivot lows → **Equal Low**.
    5.  When ``wait_for_confirmation=True``, pivots are only
        confirmed *pivot_length* bars after the actual extremum
        (both left and right bars must be lower for a high, higher
        for a low).
"""
from typing import Union, Dict
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


def equal_highs_lows(
    data: Union[PdDataFrame, PlDataFrame],
    pivot_length: int = 4,
    atr_length: int = 200,
    threshold: float = 0.10,
    wait_for_confirmation: bool = True,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    eqh_column: str = "eqh",
    eql_column: str = "eql",
    eqh_price_column: str = "eqh_price",
    eql_price_column: str = "eql_price",
    eqh_prev_idx_column: str = "eqh_prev_bar",
    eqh_curr_idx_column: str = "eqh_curr_bar",
    eql_prev_idx_column: str = "eql_prev_bar",
    eql_curr_idx_column: str = "eql_curr_bar",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect Equal Highs and Equal Lows on OHLC data.

    Two consecutive pivot highs (or lows) are considered **equal**
    when the absolute difference between them is ≤ ATR × threshold.

    Args:
        data: pandas or polars DataFrame with OHLC data.
        pivot_length: Left and right lookback for pivot detection
            (default: 4).  A pivot high at bar *i* requires
            ``high[i]`` ≥ all highs in ``[i-pivot_length …
            i+pivot_length]``.
        atr_length: Period for the ATR used to normalise the
            threshold (default: 200).
        threshold: Fraction of ATR within which two pivots are
            considered "equal" (default: 0.10).
        wait_for_confirmation: When ``True``, pivots are only
            confirmed ``pivot_length`` bars after the extremum
            (default: ``True``).
        high_column: Column name for highs (default: ``"High"``).
        low_column: Column name for lows (default: ``"Low"``).
        close_column: Column name for closes (default: ``"Close"``).
        eqh_column: Output column – 1 on the bar where an EQH is
            confirmed (default: ``"eqh"``).
        eql_column: Output column – 1 on the bar where an EQL is
            confirmed (default: ``"eql"``).
        eqh_price_column: Output column – the price level of the
            equal high (the average of the two pivots)
            (default: ``"eqh_price"``).
        eql_price_column: Output column – the price level of the
            equal low (the average of the two pivots)
            (default: ``"eql_price"``).
        eqh_prev_idx_column: Output column – positional index of the
            *previous* pivot high that forms the EQH pair
            (default: ``"eqh_prev_bar"``).
        eqh_curr_idx_column: Output column – positional index of the
            *current* pivot high that forms the EQH pair
            (default: ``"eqh_curr_bar"``).
        eql_prev_idx_column: Output column – positional index of the
            *previous* pivot low that forms the EQL pair
            (default: ``"eql_prev_bar"``).
        eql_curr_idx_column: Output column – positional index of the
            *current* pivot low that forms the EQL pair
            (default: ``"eql_curr_bar"``).

    Returns:
        DataFrame with added columns:

        - ``{eqh_column}``: 1 when an Equal High is detected, else 0.
        - ``{eql_column}``: 1 when an Equal Low is detected, else 0.
        - ``{eqh_price_column}``: Average price of the two equal
          highs (NaN otherwise).
        - ``{eql_price_column}``: Average price of the two equal
          lows (NaN otherwise).
        - ``{eqh_prev_idx_column}``: Integer position of the previous
          pivot high in the EQH pair (NaN otherwise).
        - ``{eqh_curr_idx_column}``: Integer position of the current
          pivot high in the EQH pair (NaN otherwise).
        - ``{eql_prev_idx_column}``: Integer position of the previous
          pivot low in the EQL pair (NaN otherwise).
        - ``{eql_curr_idx_column}``: Integer position of the current
          pivot low in the EQL pair (NaN otherwise).

    Example:
        >>> import pandas as pd
        >>> from pyindicators import equal_highs_lows
        >>> df = pd.DataFrame({
        ...     'High': [...],
        ...     'Low': [...],
        ...     'Close': [...]
        ... })
        >>> result = equal_highs_lows(df, pivot_length=4)
    """
    if isinstance(data, PdDataFrame):
        return _equal_highs_lows_pandas(
            data,
            pivot_length=pivot_length,
            atr_length=atr_length,
            threshold=threshold,
            wait_for_confirmation=wait_for_confirmation,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            eqh_column=eqh_column,
            eql_column=eql_column,
            eqh_price_column=eqh_price_column,
            eql_price_column=eql_price_column,
            eqh_prev_idx_column=eqh_prev_idx_column,
            eqh_curr_idx_column=eqh_curr_idx_column,
            eql_prev_idx_column=eql_prev_idx_column,
            eql_curr_idx_column=eql_curr_idx_column,
        )
    elif isinstance(data, PlDataFrame):
        pd_data = data.to_pandas()
        result = _equal_highs_lows_pandas(
            pd_data,
            pivot_length=pivot_length,
            atr_length=atr_length,
            threshold=threshold,
            wait_for_confirmation=wait_for_confirmation,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            eqh_column=eqh_column,
            eql_column=eql_column,
            eqh_price_column=eqh_price_column,
            eql_price_column=eql_price_column,
            eqh_prev_idx_column=eqh_prev_idx_column,
            eqh_curr_idx_column=eqh_curr_idx_column,
            eql_prev_idx_column=eql_prev_idx_column,
            eql_curr_idx_column=eql_curr_idx_column,
        )
        import polars as pl
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def equal_highs_lows_signal(
    data: Union[PdDataFrame, PlDataFrame],
    eqh_column: str = "eqh",
    eql_column: str = "eql",
    signal_column: str = "eqhl_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a combined signal column from Equal Highs/Lows results.

    Args:
        data: DataFrame containing equal highs/lows columns
            (output of :func:`equal_highs_lows`).
        eqh_column: Column with EQH flags.
        eql_column: Column with EQL flags.
        signal_column: Output column name (default:
            ``"eqhl_signal"``).

    Returns:
        DataFrame with ``{signal_column}`` added:

        - ``1``  – Equal Low detected (potential support / bullish)
        - ``-1`` – Equal High detected (potential resistance / bearish)
        - ``0``  – no signal
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        signal = np.where(
            data[eql_column] == 1,
            1,
            np.where(data[eqh_column] == 1, -1, 0),
        )
        data[signal_column] = signal
        return data
    elif isinstance(data, PlDataFrame):
        import polars as pl
        return data.with_columns(
            pl.when(pl.col(eql_column) == 1)
            .then(1)
            .when(pl.col(eqh_column) == 1)
            .then(-1)
            .otherwise(0)
            .alias(signal_column)
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def get_equal_highs_lows_stats(
    data: Union[PdDataFrame, PlDataFrame],
    eqh_column: str = "eqh",
    eql_column: str = "eql",
) -> Dict:
    """
    Return summary statistics for detected Equal Highs/Lows.

    Args:
        data: DataFrame containing equal highs/lows columns
            (output of :func:`equal_highs_lows`).
        eqh_column: Column with EQH flags.
        eql_column: Column with EQL flags.

    Returns:
        Dictionary with keys:

        - ``total_equal_highs``
        - ``total_equal_lows``
        - ``total``
        - ``eqh_ratio`` (0–1)
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    eqh = int(data[eqh_column].sum())
    eql = int(data[eql_column].sum())
    total = eqh + eql
    return {
        "total_equal_highs": eqh,
        "total_equal_lows": eql,
        "total": total,
        "eqh_ratio": round(eqh / total, 4) if total > 0 else 0.0,
    }


# -------------------------------------------------------------------
#  Internal helpers
# -------------------------------------------------------------------

def _compute_atr(high: np.ndarray, low: np.ndarray,
                 close: np.ndarray, period: int) -> np.ndarray:
    """Compute ATR as a simple rolling mean of True Range."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Rolling mean with min_periods=1
    atr_vals = np.empty(n)
    cumsum = 0.0
    for i in range(n):
        cumsum += tr[i]
        if i < period:
            atr_vals[i] = cumsum / (i + 1)
        else:
            cumsum -= tr[i - period]
            atr_vals[i] = cumsum / period

    return atr_vals


def _detect_pivot_highs(high: np.ndarray,
                        length: int) -> np.ndarray:
    """
    Detect pivot highs.  A pivot high at index *i* means
    ``high[i]`` ≥ all highs in ``[i-length … i+length]``.

    Returns an array the same length as *high* where confirmed
    pivot positions contain the pivot price and all other positions
    are ``NaN``.
    """
    n = len(high)
    pivots = np.full(n, np.nan)

    for i in range(length, n - length):
        window = high[i - length: i + length + 1]
        if high[i] == np.max(window):
            pivots[i] = high[i]

    return pivots


def _detect_pivot_lows(low: np.ndarray,
                       length: int) -> np.ndarray:
    """Detect pivot lows (mirror of ``_detect_pivot_highs``)."""
    n = len(low)
    pivots = np.full(n, np.nan)

    for i in range(length, n - length):
        window = low[i - length: i + length + 1]
        if low[i] == np.min(window):
            pivots[i] = low[i]

    return pivots


def _equal_highs_lows_pandas(
    data: PdDataFrame,
    pivot_length: int,
    atr_length: int,
    threshold: float,
    wait_for_confirmation: bool,
    high_column: str,
    low_column: str,
    close_column: str,
    eqh_column: str,
    eql_column: str,
    eqh_price_column: str,
    eql_price_column: str,
    eqh_prev_idx_column: str,
    eqh_curr_idx_column: str,
    eql_prev_idx_column: str,
    eql_curr_idx_column: str,
) -> PdDataFrame:
    """Core pandas implementation of Equal Highs/Lows."""
    data = data.copy()
    high = data[high_column].values.astype(float)
    low = data[low_column].values.astype(float)
    close = data[close_column].values.astype(float)
    n = len(data)

    # Compute ATR for normalisation
    atr_vals = _compute_atr(high, low, close, atr_length)

    # Detect pivots
    pivot_highs = _detect_pivot_highs(high, pivot_length)
    pivot_lows = _detect_pivot_lows(low, pivot_length)

    # Output arrays
    eqh_arr = np.zeros(n, dtype=int)
    eql_arr = np.zeros(n, dtype=int)
    eqh_price = np.full(n, np.nan)
    eql_price = np.full(n, np.nan)
    eqh_prev = np.full(n, np.nan)
    eqh_curr = np.full(n, np.nan)
    eql_prev_arr = np.full(n, np.nan)
    eql_curr_arr = np.full(n, np.nan)

    # Confirmation offset: when wait_for_confirmation is True, a pivot
    # at bar *i* is only confirmed at bar *i + pivot_length*.  The
    confirm_offset = pivot_length if wait_for_confirmation else 0

    # Track the previous pivot high/low position & price
    prev_ph_idx = -1
    prev_ph_price = np.nan
    prev_pl_idx = -1
    prev_pl_price = np.nan

    for i in range(n):
        # The confirmation bar for a pivot at position *p* is
        # *p + confirm_offset*.  So on bar *i* we check whether
        # there was a pivot at *i - confirm_offset*.
        p = i - confirm_offset
        if p < 0:
            continue

        # --- Pivot Highs ------------------------------------------------
        if not np.isnan(pivot_highs[p]):
            current_ph_price = pivot_highs[p]

            if prev_ph_idx >= 0 and not np.isnan(prev_ph_price):
                hi = max(current_ph_price, prev_ph_price)
                lo = min(current_ph_price, prev_ph_price)
                atr_at_bar = atr_vals[i] if i < n else atr_vals[-1]

                if hi < lo + atr_at_bar * threshold:
                    # "Equal" — both pivots are at roughly same level
                    eqh_arr[i] = 1
                    eqh_price[i] = (current_ph_price + prev_ph_price) / 2
                    eqh_prev[i] = prev_ph_idx
                    eqh_curr[i] = p

            prev_ph_idx = p
            prev_ph_price = current_ph_price

        # --- Pivot Lows -------------------------------------------------
        if not np.isnan(pivot_lows[p]):
            current_pl_price = pivot_lows[p]

            if prev_pl_idx >= 0 and not np.isnan(prev_pl_price):
                hi = max(current_pl_price, prev_pl_price)
                lo = min(current_pl_price, prev_pl_price)
                atr_at_bar = atr_vals[i] if i < n else atr_vals[-1]

                if hi < lo + atr_at_bar * threshold:
                    eql_arr[i] = 1
                    eql_price[i] = (current_pl_price + prev_pl_price) / 2
                    eql_prev_arr[i] = prev_pl_idx
                    eql_curr_arr[i] = p

            prev_pl_idx = p
            prev_pl_price = current_pl_price

    # Assign results
    data[eqh_column] = eqh_arr
    data[eql_column] = eql_arr
    data[eqh_price_column] = eqh_price
    data[eql_price_column] = eql_price
    data[eqh_prev_idx_column] = eqh_prev
    data[eqh_curr_idx_column] = eqh_curr
    data[eql_prev_idx_column] = eql_prev_arr
    data[eql_curr_idx_column] = eql_curr_arr

    return data
