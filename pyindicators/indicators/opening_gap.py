"""
Opening Gap (OG) Indicator

Identifies Opening Gaps — a gap between two non-adjacent candles
(spanning three bars) similar to a Fair Value Gap but measured from
the *close* side rather than the wick side.

Based on the "Price Action Concepts [RUDYINDICATOR]" methodology:

- **Bullish OG:**  The low of the current bar is *above* the high
  of bar *i-2*, creating an opening gap visible across three bars.
  This mirrors the standard FVG definition but uses the alternate
  reference point from the Pine Script OG mode.

- **Bearish OG:**  The high of the current bar is *below* the low
  of bar *i-1* (shifted reference), spanning three bars in the
  opposite direction.

The gap zone boundaries are:
    - Bullish OG : Top = Low of current bar, Bottom = High of bar 2 ago
    - Bearish OG : Top = Low of bar 1 ago, Bottom = High of current bar

Three exported functions follow the library convention:
    - ``opening_gap()``       — core detector
    - ``opening_gap_signal()`` — directional signal
    - ``get_opening_gap_stats()`` — summary statistics
"""
from typing import Union, Dict
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


# -------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------

def opening_gap(
    data: Union[PdDataFrame, PlDataFrame],
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    bullish_og_column: str = "bullish_og",
    bearish_og_column: str = "bearish_og",
    bullish_og_top_column: str = "bullish_og_top",
    bullish_og_bottom_column: str = "bullish_og_bottom",
    bearish_og_top_column: str = "bearish_og_top",
    bearish_og_bottom_column: str = "bearish_og_bottom",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect Opening Gaps on OHLC data.

    An Opening Gap is a three-bar price gap — conceptually similar
    to a Fair Value Gap, but measured from different anchor points
    per the RUDYINDICATOR Pine Script OG variant.

    **Bullish OG (Gap Up):**
        Occurs when the low of bar *i* > high of bar *i-2*,
        and bar *i-1*'s high also sits between those levels.
        Zone: [High_{i-2}, Low_i]

    **Bearish OG (Gap Down):**
        Occurs when the high of bar *i* < low of bar *i-1* (shifted),
        with bar *i-1*'s low used as the reference.
        Zone: [High_i, Low_{i-1}]

    Args:
        data: pandas or polars DataFrame with OHLC data.
        high_column: Column name for highs (default: ``"High"``).
        low_column: Column name for lows (default: ``"Low"``).
        close_column: Column name for closes (default: ``"Close"``).
        bullish_og_column: Output column — 1 on bullish OG bar
            (default: ``"bullish_og"``).
        bearish_og_column: Output column — 1 on bearish OG bar
            (default: ``"bearish_og"``).
        bullish_og_top_column: Top of bullish OG zone
            (default: ``"bullish_og_top"``).
        bullish_og_bottom_column: Bottom of bullish OG zone
            (default: ``"bullish_og_bottom"``).
        bearish_og_top_column: Top of bearish OG zone
            (default: ``"bearish_og_top"``).
        bearish_og_bottom_column: Bottom of bearish OG zone
            (default: ``"bearish_og_bottom"``).

    Returns:
        DataFrame with added columns for OG detection and zone
        boundaries.

    Example:
        >>> import pandas as pd
        >>> from pyindicators import opening_gap
        >>> df = pd.DataFrame({
        ...     'High':  [100, 105, 115, 120, 118],
        ...     'Low':   [ 95, 102, 108, 116, 114],
        ...     'Close': [ 98, 104, 112, 118, 116],
        ... })
        >>> result = opening_gap(df)
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        pdf = _compute_og(
            pdf, high_column, low_column, close_column,
            bullish_og_column, bearish_og_column,
            bullish_og_top_column, bullish_og_bottom_column,
            bearish_og_top_column, bearish_og_bottom_column,
        )
        import polars as pl
        return pl.from_pandas(pdf)

    if isinstance(data, PdDataFrame):
        return _compute_og(
            data, high_column, low_column, close_column,
            bullish_og_column, bearish_og_column,
            bullish_og_top_column, bullish_og_bottom_column,
            bearish_og_top_column, bearish_og_bottom_column,
        )

    raise PyIndicatorException(
        "Input data must be a pandas or polars DataFrame."
    )


def opening_gap_signal(
    data: Union[PdDataFrame, PlDataFrame],
    bullish_og_column: str = "bullish_og",
    bearish_og_column: str = "bearish_og",
    signal_column: str = "og_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a directional signal from Opening Gap detection.

    Signal values:
        -  1 : bullish Opening Gap detected
        - -1 : bearish Opening Gap detected
        -  0 : no gap

    Args:
        data: DataFrame with OG columns already computed.
        bullish_og_column: Column with bullish OG flags.
        bearish_og_column: Column with bearish OG flags.
        signal_column: Name of the output signal column.

    Returns:
        DataFrame with added signal column.
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        pdf = _compute_signal(
            pdf, bullish_og_column, bearish_og_column, signal_column
        )
        import polars as pl
        return pl.from_pandas(pdf)

    if isinstance(data, PdDataFrame):
        return _compute_signal(
            data, bullish_og_column, bearish_og_column, signal_column
        )

    raise PyIndicatorException(
        "Input data must be a pandas or polars DataFrame."
    )


def get_opening_gap_stats(
    data: Union[PdDataFrame, PlDataFrame],
    bullish_og_column: str = "bullish_og",
    bearish_og_column: str = "bearish_og",
) -> Dict[str, object]:
    """
    Compute summary statistics for Opening Gap detections.

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

    bull = int(data[bullish_og_column].sum())
    bear = int(data[bearish_og_column].sum())
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

def _compute_og(
    df: PdDataFrame,
    high_col: str, low_col: str, close_col: str,
    bull_col: str, bear_col: str,
    bull_top: str, bull_bot: str,
    bear_top: str, bear_bot: str,
) -> PdDataFrame:
    """Core numpy-based Opening Gap detection."""
    n = len(df)
    h = df[high_col].values.astype(float)
    low = df[low_col].values.astype(float)

    bull_flag = np.zeros(n, dtype=int)
    bear_flag = np.zeros(n, dtype=int)
    b_top = np.full(n, np.nan)
    b_bot = np.full(n, np.nan)
    s_top = np.full(n, np.nan)
    s_bot = np.full(n, np.nan)

    for i in range(2, n):
        # Bullish OG: low[i] > high[i-2]
        # The three-bar gap up spans bar i-2, i-1, i
        if low[i] > h[i - 2]:
            bull_flag[i] = 1
            b_top[i] = low[i]       # top of the gap = low of current bar
            b_bot[i] = h[i - 2]     # bottom of gap = high of bar 2 ago

        # Bearish OG: high[i] < low[i-1] (shifted reference per RUDY)
        # Pine Script OG bearish uses low[i-1] as reference
        if h[i] < low[i - 1]:
            bear_flag[i] = 1
            s_top[i] = low[i - 1]   # top = low of previous bar
            s_bot[i] = h[i]         # bottom = high of current bar

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
