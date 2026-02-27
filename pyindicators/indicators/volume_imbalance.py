"""
Volume Imbalance (VI) Indicator

Identifies Volume Imbalances — a gap between consecutive candle bodies
(open/close range) where no trading overlap occurred.  Unlike a Fair
Value Gap which spans three bars, a Volume Imbalance only needs two
adjacent bars.

Based on the "Price Action Concepts [RUDYINDICATOR]" methodology:

- **Bullish VI:**  The higher body edge of bar *i* (``max(open, close)``)
  is *above* the lower body edge of bar *i-1* (``min(open, close)``),
  creating a gap between the two bodies.  Additionally:
    - The high of bar *i-1* must be below the lower body of bar *i*
      (ensuring no wick fills the gap)
    - Bar *i* closes above the previous close and opens above the
      previous open
    - The preceding bar's high stays below the body of bar *i*

- **Bearish VI:**  The mirror: the lower body of bar *i* is *below*
  the upper body of bar *i-1*, with matching wick / directional
  constraints.

The imbalance zone boundaries are:
    - Bullish VI : Top = min(open, close) of current bar,
                   Bottom = max(open, close) of previous bar
    - Bearish VI : Top = min(open, close) of previous bar,
                   Bottom = max(open, close) of current bar

Three exported functions follow the library convention:
    - ``volume_imbalance()``       — core detector
    - ``volume_imbalance_signal()`` — directional signal
    - ``get_volume_imbalance_stats()`` — summary statistics
"""
from typing import Union, Dict
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


# -------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------

def volume_imbalance(
    data: Union[PdDataFrame, PlDataFrame],
    open_column: str = "Open",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    bullish_vi_column: str = "bullish_vi",
    bearish_vi_column: str = "bearish_vi",
    bullish_vi_top_column: str = "bullish_vi_top",
    bullish_vi_bottom_column: str = "bullish_vi_bottom",
    bearish_vi_top_column: str = "bearish_vi_top",
    bearish_vi_bottom_column: str = "bearish_vi_bottom",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect Volume Imbalances on OHLC data.

    A Volume Imbalance is a gap between consecutive candle bodies
    where no trading occurred, indicating a sudden shift in
    supply / demand.

    **Bullish VI:**
        - ``open_i > close_{i-1}`` and ``high_{i-1} < min(open_i, close_i)``
        - The current bar moves up aggressively relative to the prior bar
        - Zone: [max(open, close) of bar i-1, min(open, close) of bar i]

    **Bearish VI:**
        - ``open_i < close_{i-1}`` and ``low_{i-1} > max(open_i, close_i)``
        - The current bar moves down aggressively
        - Zone: [max(open, close) of bar i, min(open, close) of bar i-1]

    Args:
        data: pandas or polars DataFrame with OHLC data.
        open_column: Column name for open prices (default: ``"Open"``).
        high_column: Column name for high prices (default: ``"High"``).
        low_column: Column name for low prices (default: ``"Low"``).
        close_column: Column name for close prices (default: ``"Close"``).
        bullish_vi_column: Output column — 1 on bullish VI bar
            (default: ``"bullish_vi"``).
        bearish_vi_column: Output column — 1 on bearish VI bar
            (default: ``"bearish_vi"``).
        bullish_vi_top_column: Top of bullish VI zone
            (default: ``"bullish_vi_top"``).
        bullish_vi_bottom_column: Bottom of bullish VI zone
            (default: ``"bullish_vi_bottom"``).
        bearish_vi_top_column: Top of bearish VI zone
            (default: ``"bearish_vi_top"``).
        bearish_vi_bottom_column: Bottom of bearish VI zone
            (default: ``"bearish_vi_bottom"``).

    Returns:
        DataFrame with added columns for VI detection and zone
        boundaries.

    Example:
        >>> import pandas as pd
        >>> from pyindicators import volume_imbalance
        >>> df = pd.DataFrame({
        ...     'Open':  [100, 103, 110, 108, 105],
        ...     'High':  [102, 105, 112, 110, 107],
        ...     'Low':   [ 98, 101, 108, 106, 103],
        ...     'Close': [101, 104, 111, 107, 104],
        ... })
        >>> result = volume_imbalance(df)
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        pdf = _compute_vi(
            pdf, open_column, high_column, low_column, close_column,
            bullish_vi_column, bearish_vi_column,
            bullish_vi_top_column, bullish_vi_bottom_column,
            bearish_vi_top_column, bearish_vi_bottom_column,
        )
        import polars as pl
        return pl.from_pandas(pdf)

    if isinstance(data, PdDataFrame):
        return _compute_vi(
            data, open_column, high_column, low_column, close_column,
            bullish_vi_column, bearish_vi_column,
            bullish_vi_top_column, bullish_vi_bottom_column,
            bearish_vi_top_column, bearish_vi_bottom_column,
        )

    raise PyIndicatorException(
        "Input data must be a pandas or polars DataFrame."
    )


def volume_imbalance_signal(
    data: Union[PdDataFrame, PlDataFrame],
    bullish_vi_column: str = "bullish_vi",
    bearish_vi_column: str = "bearish_vi",
    signal_column: str = "vi_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a directional signal from Volume Imbalance detection.

    Signal values:
        -  1 : bullish Volume Imbalance detected
        - -1 : bearish Volume Imbalance detected
        -  0 : no imbalance

    If both a bullish and bearish VI fire on the same bar the
    bullish signal takes priority.

    Args:
        data: DataFrame with VI columns already computed.
        bullish_vi_column: Column with bullish VI flags.
        bearish_vi_column: Column with bearish VI flags.
        signal_column: Name of the output signal column.

    Returns:
        DataFrame with added signal column.
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        pdf = _compute_signal(
            pdf, bullish_vi_column, bearish_vi_column, signal_column
        )
        import polars as pl
        return pl.from_pandas(pdf)

    if isinstance(data, PdDataFrame):
        return _compute_signal(
            data, bullish_vi_column, bearish_vi_column, signal_column
        )

    raise PyIndicatorException(
        "Input data must be a pandas or polars DataFrame."
    )


def get_volume_imbalance_stats(
    data: Union[PdDataFrame, PlDataFrame],
    bullish_vi_column: str = "bullish_vi",
    bearish_vi_column: str = "bearish_vi",
) -> Dict[str, object]:
    """
    Compute summary statistics for Volume Imbalance detections.

    Returns:
        dict with keys:
            - total_bullish : int
            - total_bearish : int
            - total         : int
            - bullish_pct   : float (0-100)
            - bearish_pct   : float (0-100)
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    if not isinstance(data, PdDataFrame):
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )

    bull = int(data[bullish_vi_column].sum())
    bear = int(data[bearish_vi_column].sum())
    total = bull + bear
    return {
        "total_bullish": bull,
        "total_bearish": bear,
        "total": total,
        "bullish_pct": round(bull / total * 100, 2) if total else 0.0,
        "bearish_pct": round(bear / total * 100, 2) if total else 0.0,
    }


# -------------------------------------------------------------------
#  Internal helpers
# -------------------------------------------------------------------

def _compute_vi(
    df: PdDataFrame,
    open_col: str, high_col: str, low_col: str, close_col: str,
    bull_col: str, bear_col: str,
    bull_top: str, bull_bot: str,
    bear_top: str, bear_bot: str,
) -> PdDataFrame:
    """Core numpy-based Volume Imbalance detection (pandas only)."""
    n = len(df)
    o = df[open_col].values.astype(float)
    h = df[high_col].values.astype(float)
    low = df[low_col].values.astype(float)
    c = df[close_col].values.astype(float)

    bull_flag = np.zeros(n, dtype=int)
    bear_flag = np.zeros(n, dtype=int)
    b_top = np.full(n, np.nan)
    b_bot = np.full(n, np.nan)
    s_top = np.full(n, np.nan)
    s_bot = np.full(n, np.nan)

    for i in range(1, n):
        # Body edges
        cur_body_hi = max(o[i], c[i])
        cur_body_lo = min(o[i], c[i])
        prev_body_hi = max(o[i - 1], c[i - 1])
        prev_body_lo = min(o[i - 1], c[i - 1])

        # Bullish VI: gap between prev body top and current body bottom
        # Previous bar's body top < current bar's body bottom
        # AND previous bar high < current bar body bottom
        # AND current bar is moving up (close > prev close, open > prev open)
        if (prev_body_hi < cur_body_lo
                and h[i - 1] < cur_body_lo
                and c[i] > c[i - 1]
                and o[i] > o[i - 1]):
            bull_flag[i] = 1
            b_top[i] = cur_body_lo
            b_bot[i] = prev_body_hi

        # Bearish VI: gap between current body top and prev body bottom
        # Current bar's body top < previous bar's body bottom
        # AND previous bar low > current bar body top
        # AND current bar is moving down (close < prev close, open < prev open)
        if (cur_body_hi < prev_body_lo
                and low[i - 1] > cur_body_hi
                and c[i] < c[i - 1]
                and o[i] < o[i - 1]):
            bear_flag[i] = 1
            s_top[i] = prev_body_lo
            s_bot[i] = cur_body_hi

    df[bull_col] = bull_flag
    df[bear_col] = bear_flag
    df[bull_top] = b_top
    df[bull_bot] = b_bot
    df[bear_top] = s_top
    df[bear_bot] = s_bot
    return df


def _compute_signal(
    df: PdDataFrame,
    bull_col: str, bear_col: str, sig_col: str,
) -> PdDataFrame:
    bull = df[bull_col].values
    bear = df[bear_col].values
    sig = np.where(bull == 1, 1, np.where(bear == 1, -1, 0))
    df[sig_col] = sig.astype(int)
    return df
