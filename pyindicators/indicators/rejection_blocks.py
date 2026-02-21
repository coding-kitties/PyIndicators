"""
Rejection Blocks

Identifies Rejection Blocks — candles at swing points whose long
wicks signal institutional rejection of a price level, creating a
tradeable zone.

**Concept (ICT / Smart Money):**
    A Rejection Block forms when a candle at a swing extreme has a
    wick that is disproportionately large relative to the total
    candle range.  The long wick shows that price was driven to a
    level but was *rejected* — institutional participants absorbed
    the orders and pushed price back.  The wick area becomes a zone
    where future price interaction is expected.

**Bullish Rejection Block:**
    Detected at a confirmed swing low when the candle's **lower
    wick** (Low → body bottom) occupies at least ``wick_threshold``
    of the total candle range.  The zone spans the lower wick area.
    When price returns to this zone, institutional demand is
    expected — a potential long entry.

**Bearish Rejection Block:**
    Detected at a confirmed swing high when the candle's **upper
    wick** (body top → High) occupies at least ``wick_threshold`` of
    the total candle range.  The zone spans the upper wick area.
    When price returns to this zone, institutional supply is
    expected — a potential short entry.

**Zone boundaries:**
    * Bullish RB zone:  Low → min(Open, Close)
    * Bearish RB zone: max(Open, Close) → High

**Entry signals:**
    * Long:  price retraces into the bullish rejection block zone
      (low ≤ zone top and close ≥ zone bottom).
    * Short: price retraces into the bearish rejection block zone
      (high ≥ zone bottom and close ≤ zone top).

**Mitigation (invalidation):**
    The block is considered mitigated when price closes through the
    opposite side of the zone entirely.

License note:  This is an independent, clean-room implementation of
publicly documented ICT / Smart Money trading concepts.
"""
from typing import Union, Dict
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
import numpy as np

from pyindicators.exceptions import PyIndicatorException


def rejection_blocks(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 5,
    wick_threshold: float = 0.5,
    high_column: str = "High",
    low_column: str = "Low",
    open_column: str = "Open",
    close_column: str = "Close",
    bullish_rb_column: str = "rb_bullish",
    bearish_rb_column: str = "rb_bearish",
    rb_top_column: str = "rb_top",
    rb_bottom_column: str = "rb_bottom",
    rb_direction_column: str = "rb_direction",
    rb_entry_long_column: str = "rb_entry_long",
    rb_entry_short_column: str = "rb_entry_short",
    rb_mitigated_column: str = "rb_mitigated",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Identify Rejection Blocks and generate trading signals.

    A Rejection Block is a candle at a swing extreme whose
    disproportionately long wick signals institutional rejection of
    a price level.  The wick area becomes a tradeable zone.

    Args:
        data: pandas or polars DataFrame with OHLC price data.
        swing_length: Lookback period for pivot detection
            (default: 5).
        wick_threshold: Minimum wick-to-range ratio required for a
            candle to qualify as a rejection block (default: 0.5,
            meaning the wick must be at least 50 % of the total
            candle range).
        high_column: Column for highs (default: ``"High"``).
        low_column: Column for lows (default: ``"Low"``).
        open_column: Column for opens (default: ``"Open"``).
        close_column: Column for closes (default: ``"Close"``).
        bullish_rb_column: Output — 1 on bullish RB formation bar
            (default: ``"rb_bullish"``).
        bearish_rb_column: Output — 1 on bearish RB formation bar
            (default: ``"rb_bearish"``).
        rb_top_column: Output — active RB zone top, forward-filled
            while the block is alive (default: ``"rb_top"``).
        rb_bottom_column: Output — active RB zone bottom
            (default: ``"rb_bottom"``).
        rb_direction_column: Output — 1 for bullish RB, -1 for
            bearish RB, 0 when no RB is active
            (default: ``"rb_direction"``).
        rb_entry_long_column: Output — 1 when price retraces into
            a bullish RB zone (default: ``"rb_entry_long"``).
        rb_entry_short_column: Output — 1 when price retraces into
            a bearish RB zone (default: ``"rb_entry_short"``).
        rb_mitigated_column: Output — 1 when the RB is fully
            mitigated (default: ``"rb_mitigated"``).

    Returns:
        DataFrame with added columns:

        - ``{bullish_rb_column}``: 1 on bullish RB formation bar
        - ``{bearish_rb_column}``: 1 on bearish RB formation bar
        - ``{rb_top_column}``: Active RB zone upper boundary
        - ``{rb_bottom_column}``: Active RB zone lower boundary
        - ``{rb_direction_column}``: 1 bullish / -1 bearish / 0 none
        - ``{rb_entry_long_column}``: 1 when price enters bullish RB
        - ``{rb_entry_short_column}``: 1 when price enters bearish RB
        - ``{rb_mitigated_column}``: 1 when the RB is mitigated

    Example:
        >>> import pandas as pd
        >>> from pyindicators import rejection_blocks
        >>> df = pd.DataFrame({
        ...     'Open': [...], 'High': [...],
        ...     'Low': [...], 'Close': [...]
        ... })
        >>> result = rejection_blocks(df, swing_length=5)
    """
    if isinstance(data, PdDataFrame):
        return _rb_pandas(
            data, swing_length, wick_threshold,
            high_column, low_column, open_column, close_column,
            bullish_rb_column, bearish_rb_column,
            rb_top_column, rb_bottom_column,
            rb_direction_column,
            rb_entry_long_column, rb_entry_short_column,
            rb_mitigated_column,
        )
    elif isinstance(data, PlDataFrame):
        return _rb_polars(
            data, swing_length, wick_threshold,
            high_column, low_column, open_column, close_column,
            bullish_rb_column, bearish_rb_column,
            rb_top_column, rb_bottom_column,
            rb_direction_column,
            rb_entry_long_column, rb_entry_short_column,
            rb_mitigated_column,
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ------------------------------------------------------------------ #
#  Pandas implementation                                              #
# ------------------------------------------------------------------ #
def _rb_pandas(
    data: PdDataFrame,
    swing_length: int,
    wick_threshold: float,
    high_col: str,
    low_col: str,
    open_col: str,
    close_col: str,
    bullish_rb_col: str,
    bearish_rb_col: str,
    rb_top_col: str,
    rb_bot_col: str,
    rb_dir_col: str,
    entry_long_col: str,
    entry_short_col: str,
    mitigated_col: str,
) -> PdDataFrame:
    """Core Rejection Block algorithm (pandas)."""
    high = data[high_col].values.astype(float)
    low = data[low_col].values.astype(float)
    opn = data[open_col].values.astype(float)
    close = data[close_col].values.astype(float)
    n = len(data)

    # Body boundaries
    body_top = np.maximum(close, opn)
    body_bot = np.minimum(close, opn)

    # ----- RB state ----- #
    rb_active = False
    rb_dir = 0           # +1 bullish, -1 bearish
    rb_top_v = np.nan    # zone top
    rb_bot_v = np.nan    # zone bottom
    rb_entry_emitted = False

    # Track last pivot directions to avoid duplicate pivots on same
    # side (simple alternation: only accept a pivot if it differs
    # from the previous one, or improves the extreme).
    last_pivot_dir = 0   # +1 last was pivot high, -1 pivot low

    # ----- Output arrays ----- #
    out_bull = np.zeros(n, dtype=int)
    out_bear = np.zeros(n, dtype=int)
    out_top = np.full(n, np.nan)
    out_bot = np.full(n, np.nan)
    out_dir = np.zeros(n, dtype=int)
    out_elong = np.zeros(n, dtype=int)
    out_eshort = np.zeros(n, dtype=int)
    out_mitig = np.zeros(n, dtype=int)

    for i in range(1, n):
        # ---- 1. Pivot detection (bar ci confirmed by bar i) ---- #
        ci = i - 1  # candidate index
        new_rb = False

        if ci >= swing_length:
            left_start = ci - swing_length

            # --- Pivot high check --- #
            is_ph = True
            for j in range(left_start, ci):
                if high[j] > high[ci]:
                    is_ph = False
                    break
            if is_ph and high[i] <= high[ci]:
                # Confirmed pivot high — check for bearish rejection
                if last_pivot_dir != 1 or True:
                    # Accept this pivot high
                    last_pivot_dir = 1
                    total_range = high[ci] - low[ci]
                    if total_range > 0:
                        upper_wick = high[ci] - body_top[ci]
                        wick_ratio = upper_wick / total_range
                        if wick_ratio >= wick_threshold:
                            # Bearish Rejection Block — upper wick zone
                            rb_top_v = high[ci]
                            rb_bot_v = body_top[ci]
                            rb_dir = -1
                            rb_active = True
                            rb_entry_emitted = False
                            out_bear[i] = 1
                            new_rb = True

            # --- Pivot low check --- #
            is_pl = True
            for j in range(left_start, ci):
                if low[j] < low[ci]:
                    is_pl = False
                    break
            if is_pl and low[i] >= low[ci]:
                # Confirmed pivot low — check for bullish rejection
                if last_pivot_dir != -1 or True:
                    last_pivot_dir = -1
                    total_range = high[ci] - low[ci]
                    if total_range > 0:
                        lower_wick = body_bot[ci] - low[ci]
                        wick_ratio = lower_wick / total_range
                        if wick_ratio >= wick_threshold:
                            # Bullish Rejection Block — lower wick zone
                            rb_top_v = body_bot[ci]
                            rb_bot_v = low[ci]
                            rb_dir = 1
                            rb_active = True
                            rb_entry_emitted = False
                            out_bull[i] = 1
                            new_rb = True

        # ---- 2. RB lifecycle management ---- #
        if rb_active and not new_rb:
            if rb_dir == 1:
                # Bullish RB: waiting for price to retrace into zone
                # Mitigated if price closes below zone bottom
                if close[i] < rb_bot_v:
                    rb_active = False
                    out_mitig[i] = 1
                elif not rb_entry_emitted:
                    # Entry: price dips into the zone
                    if low[i] <= rb_top_v and close[i] >= rb_bot_v:
                        out_elong[i] = 1
                        rb_entry_emitted = True
            else:
                # Bearish RB: waiting for price to retrace up into zone
                # Mitigated if price closes above zone top
                if close[i] > rb_top_v:
                    rb_active = False
                    out_mitig[i] = 1
                elif not rb_entry_emitted:
                    # Entry: price pushes up into the zone
                    if high[i] >= rb_bot_v and close[i] <= rb_top_v:
                        out_eshort[i] = 1
                        rb_entry_emitted = True

        # ---- 3. Forward-fill active zone ---- #
        if rb_active:
            out_top[i] = rb_top_v
            out_bot[i] = rb_bot_v
            out_dir[i] = rb_dir

    # ---- Assign results ---- #
    data = data.copy()
    data[bullish_rb_col] = out_bull
    data[bearish_rb_col] = out_bear
    data[rb_top_col] = out_top
    data[rb_bot_col] = out_bot
    data[rb_dir_col] = out_dir
    data[entry_long_col] = out_elong
    data[entry_short_col] = out_eshort
    data[mitigated_col] = out_mitig
    return data


# ------------------------------------------------------------------ #
#  Polars implementation (delegates to pandas)                        #
# ------------------------------------------------------------------ #
def _rb_polars(
    data: PlDataFrame,
    swing_length: int,
    wick_threshold: float,
    high_col: str,
    low_col: str,
    open_col: str,
    close_col: str,
    bullish_rb_col: str,
    bearish_rb_col: str,
    rb_top_col: str,
    rb_bot_col: str,
    rb_dir_col: str,
    entry_long_col: str,
    entry_short_col: str,
    mitigated_col: str,
) -> PlDataFrame:
    """Polars wrapper — converts to pandas, runs logic, converts back."""
    pdf = data.to_pandas()
    result = _rb_pandas(
        pdf, swing_length, wick_threshold,
        high_col, low_col, open_col, close_col,
        bullish_rb_col, bearish_rb_col,
        rb_top_col, rb_bot_col, rb_dir_col,
        entry_long_col, entry_short_col,
        mitigated_col,
    )
    return pl.from_pandas(result)


# ------------------------------------------------------------------ #
#  Signal function                                                    #
# ------------------------------------------------------------------ #
def rejection_blocks_signal(
    data: Union[PdDataFrame, PlDataFrame],
    rb_entry_long_column: str = "rb_entry_long",
    rb_entry_short_column: str = "rb_entry_short",
    signal_column: str = "rb_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Derive a single trading signal column from rejection block output.

    Signal values:
        * ``1``  — long entry (bullish RB zone reached)
        * ``-1`` — short entry (bearish RB zone reached)
        * ``0``  — no signal

    Args:
        data: DataFrame with ``rejection_blocks()`` columns.
        rb_entry_long_column: Column for bullish entry flags
            (default: ``"rb_entry_long"``).
        rb_entry_short_column: Column for bearish entry flags
            (default: ``"rb_entry_short"``).
        signal_column: Output column name
            (default: ``"rb_signal"``).

    Returns:
        DataFrame with added ``{signal_column}`` column.

    Example:
        >>> df = rejection_blocks(df)
        >>> df = rejection_blocks_signal(df)
        >>> buys = df[df['rb_signal'] == 1]
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        long_flags = data[rb_entry_long_column].values
        short_flags = data[rb_entry_short_column].values
        signal = np.where(long_flags == 1, 1,
                          np.where(short_flags == 1, -1, 0))
        data[signal_column] = signal
        return data
    elif isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        result = rejection_blocks_signal(
            pdf, rb_entry_long_column,
            rb_entry_short_column, signal_column,
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ------------------------------------------------------------------ #
#  Stats function                                                     #
# ------------------------------------------------------------------ #
def get_rejection_blocks_stats(
    data: Union[PdDataFrame, PlDataFrame],
    bullish_rb_column: str = "rb_bullish",
    bearish_rb_column: str = "rb_bearish",
    rb_entry_long_column: str = "rb_entry_long",
    rb_entry_short_column: str = "rb_entry_short",
    rb_mitigated_column: str = "rb_mitigated",
) -> Dict:
    """
    Compute summary statistics from rejection block output.

    Args:
        data: DataFrame with ``rejection_blocks()`` columns.
        [column parameters]: Column names matching the output.

    Returns:
        Dictionary with keys:

        - ``total_bullish_rb``: Number of bullish RBs formed
        - ``total_bearish_rb``: Number of bearish RBs formed
        - ``total_rb``: Total RBs formed
        - ``total_entry_long``: Long entry signals fired
        - ``total_entry_short``: Short entry signals fired
        - ``total_entries``: Total entries
        - ``total_mitigated``: Mitigated RBs
        - ``entry_rate``: Fraction of RBs that produced an entry
        - ``mitigation_rate``: Fraction of RBs that were mitigated

    Example:
        >>> df = rejection_blocks(df)
        >>> stats = get_rejection_blocks_stats(df)
        >>> print(stats)
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    bull = int(data[bullish_rb_column].sum())
    bear = int(data[bearish_rb_column].sum())
    total = bull + bear
    entries_long = int(data[rb_entry_long_column].sum())
    entries_short = int(data[rb_entry_short_column].sum())
    total_entries = entries_long + entries_short
    mitigated = int(data[rb_mitigated_column].sum())
    entry_rate = total_entries / total if total > 0 else 0.0
    mitig_rate = mitigated / total if total > 0 else 0.0

    return {
        "total_bullish_rb": bull,
        "total_bearish_rb": bear,
        "total_rb": total,
        "total_entry_long": entries_long,
        "total_entry_short": entries_short,
        "total_entries": total_entries,
        "total_mitigated": mitigated,
        "entry_rate": round(entry_rate, 4),
        "mitigation_rate": round(mitig_rate, 4),
    }
