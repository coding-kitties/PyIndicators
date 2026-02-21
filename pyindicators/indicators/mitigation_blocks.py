"""
Mitigation Blocks

Identifies Mitigation Blocks — the **first candle that initiates** an
impulsive move leading to a Market Structure Shift (MSS).

**Concept (ICT / Smart Money):**
    While an *Order Block* is the last **opposing** candle before an
    impulse move, a *Mitigation Block* is the first **same-direction**
    candle that *starts* the move.  When price returns to this zone,
    institutional traders are "mitigating" (closing or adjusting)
    positions opened at the origin of the impulse — hence the name.

**Bullish Mitigation Block:**
    Detected when a bullish MSS occurs (close breaks above the most
    recent swing high after a confirmed Lower-Low pattern).  The
    block is the **first bullish candle** (close > open) after the
    preceding swing low that kicked off the upward impulse.

**Bearish Mitigation Block:**
    Detected when a bearish MSS occurs (close breaks below the most
    recent swing low after a confirmed Higher-High pattern).  The
    block is the **first bearish candle** (close < open) after the
    preceding swing high that kicked off the downward impulse.

**Zone boundaries:**
    By default the zone spans the full candle (high → low).  With
    ``use_body=True`` the zone is restricted to the body (open/close).

**Entry signals:**
    * Long:  price retraces into the bullish mitigation block zone
      (low ≤ zone top and close ≥ zone bottom).
    * Short: price retraces into the bearish mitigation block zone
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


def mitigation_blocks(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 5,
    use_body: bool = False,
    high_column: str = "High",
    low_column: str = "Low",
    open_column: str = "Open",
    close_column: str = "Close",
    bullish_mb_column: str = "mb_bullish",
    bearish_mb_column: str = "mb_bearish",
    mb_top_column: str = "mb_top",
    mb_bottom_column: str = "mb_bottom",
    mb_direction_column: str = "mb_direction",
    mb_entry_long_column: str = "mb_entry_long",
    mb_entry_short_column: str = "mb_entry_short",
    mb_mitigated_column: str = "mb_mitigated",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Identify Mitigation Blocks and generate trading signals.

    A Mitigation Block is the first candle that initiates an impulsive
    move which leads to a Market Structure Shift.  This function
    detects MB zones, tracks their lifecycle, and emits entry /
    mitigation signals.

    Args:
        data: pandas or polars DataFrame with OHLC price data.
        swing_length: Lookback period for pivot detection
            (default: 5).
        use_body: If ``True``, use candle body (open/close) instead
            of wicks (high/low) for zone boundaries
            (default: ``False``).
        high_column: Column for highs (default: ``"High"``).
        low_column: Column for lows (default: ``"Low"``).
        open_column: Column for opens (default: ``"Open"``).
        close_column: Column for closes (default: ``"Close"``).
        bullish_mb_column: Output — 1 on bullish MB formation bar
            (default: ``"mb_bullish"``).
        bearish_mb_column: Output — 1 on bearish MB formation bar
            (default: ``"mb_bearish"``).
        mb_top_column: Output — active MB zone top, forward-filled
            while the block is alive (default: ``"mb_top"``).
        mb_bottom_column: Output — active MB zone bottom
            (default: ``"mb_bottom"``).
        mb_direction_column: Output — 1 for bullish MB, -1 for
            bearish MB, 0 when no MB is active
            (default: ``"mb_direction"``).
        mb_entry_long_column: Output — 1 when price retraces into
            a bullish MB zone (default: ``"mb_entry_long"``).
        mb_entry_short_column: Output — 1 when price retraces into
            a bearish MB zone (default: ``"mb_entry_short"``).
        mb_mitigated_column: Output — 1 when the MB is fully
            mitigated (default: ``"mb_mitigated"``).

    Returns:
        DataFrame with added columns:

        - ``{bullish_mb_column}``: 1 on bullish MB formation bar
        - ``{bearish_mb_column}``: 1 on bearish MB formation bar
        - ``{mb_top_column}``: Active MB zone upper boundary
        - ``{mb_bottom_column}``: Active MB zone lower boundary
        - ``{mb_direction_column}``: 1 bullish / -1 bearish / 0 none
        - ``{mb_entry_long_column}``: 1 when price enters bullish MB
        - ``{mb_entry_short_column}``: 1 when price enters bearish MB
        - ``{mb_mitigated_column}``: 1 when the MB is mitigated

    Example:
        >>> import pandas as pd
        >>> from pyindicators import mitigation_blocks
        >>> df = pd.DataFrame({
        ...     'Open': [...], 'High': [...],
        ...     'Low': [...], 'Close': [...]
        ... })
        >>> result = mitigation_blocks(df, swing_length=5)
    """
    if isinstance(data, PdDataFrame):
        return _mb_pandas(
            data, swing_length, use_body,
            high_column, low_column, open_column, close_column,
            bullish_mb_column, bearish_mb_column,
            mb_top_column, mb_bottom_column,
            mb_direction_column,
            mb_entry_long_column, mb_entry_short_column,
            mb_mitigated_column,
        )
    elif isinstance(data, PlDataFrame):
        return _mb_polars(
            data, swing_length, use_body,
            high_column, low_column, open_column, close_column,
            bullish_mb_column, bearish_mb_column,
            mb_top_column, mb_bottom_column,
            mb_direction_column,
            mb_entry_long_column, mb_entry_short_column,
            mb_mitigated_column,
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ------------------------------------------------------------------ #
#  Pandas implementation                                              #
# ------------------------------------------------------------------ #
def _mb_pandas(
    data: PdDataFrame,
    swing_length: int,
    use_body: bool,
    high_col: str,
    low_col: str,
    open_col: str,
    close_col: str,
    bullish_mb_col: str,
    bearish_mb_col: str,
    mb_top_col: str,
    mb_bot_col: str,
    mb_dir_col: str,
    entry_long_col: str,
    entry_short_col: str,
    mitigated_col: str,
) -> PdDataFrame:
    """Core Mitigation Block algorithm (pandas)."""
    high = data[high_col].values.astype(float)
    low = data[low_col].values.astype(float)
    opn = data[open_col].values.astype(float)
    close = data[close_col].values.astype(float)
    n = len(data)

    # Body boundaries
    body_top = np.maximum(close, opn)
    body_bot = np.minimum(close, opn)

    # ----- Zigzag ring buffer (most-recent-first) ----- #
    _MAX_ZZ = 50
    zz_d = [0] * _MAX_ZZ       # direction: +1 = swing high, -1 = low
    zz_x = [0] * _MAX_ZZ       # bar index
    zz_y = [np.nan] * _MAX_ZZ  # price
    zz_count = 0

    def _zz_push(d: int, x: int, y: float) -> None:
        nonlocal zz_count
        limit = min(_MAX_ZZ - 1, zz_count)
        for k in range(limit, 0, -1):
            zz_d[k] = zz_d[k - 1]
            zz_x[k] = zz_x[k - 1]
            zz_y[k] = zz_y[k - 1]
        zz_d[0] = d
        zz_x[0] = x
        zz_y[0] = y
        zz_count = min(zz_count + 1, _MAX_ZZ)

    # ----- MB state ----- #
    mss_dir = 0          # last MSS direction
    mb_active = False    # is a mitigation block active?
    mb_dir = 0           # +1 bullish, -1 bearish
    mb_top_v = np.nan    # zone top
    mb_bot_v = np.nan    # zone bottom
    mb_entry_emitted = False  # only one entry per MB

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
        if ci >= swing_length:
            left_start = ci - swing_length

            # Pivot high
            is_ph = True
            for j in range(left_start, ci):
                if high[j] > high[ci]:
                    is_ph = False
                    break
            if is_ph and high[i] <= high[ci]:
                if zz_count == 0 or zz_d[0] != 1:
                    _zz_push(1, ci, high[ci])
                elif zz_d[0] == 1 and high[ci] > zz_y[0]:
                    zz_x[0] = ci
                    zz_y[0] = high[ci]

            # Pivot low
            is_pl = True
            for j in range(left_start, ci):
                if low[j] < low[ci]:
                    is_pl = False
                    break
            if is_pl and low[i] >= low[ci]:
                if zz_count == 0 or zz_d[0] != -1:
                    _zz_push(-1, ci, low[ci])
                elif zz_d[0] == -1 and low[ci] < zz_y[0]:
                    zz_x[0] = ci
                    zz_y[0] = low[ci]

        # ---- 2. MSS detection & MB zone establishment ---- #
        mss_formed = False

        if zz_count >= 4:
            iH = 2 if (zz_count > 2 and zz_d[2] == 1) else 1
            iL = 2 if (zz_count > 2 and zz_d[2] == -1) else 1

            # --- Bullish MSS --- #
            # Structure: previous swing low (C) → swing high (D) →
            #   swing low E (LL, E_y < C_y) → close breaks above D.
            # Mitigation Block: first bullish candle after swing
            #   low E that starts the impulse up.
            if (iH < zz_count and iH + 1 < zz_count
                    and iH - 1 >= 0
                    and zz_d[iH] == 1
                    and close[i] > zz_y[iH]
                    and mss_dir < 1):
                E_x = zz_x[iH - 1]
                E_y = zz_y[iH - 1]   # most recent swing low
                C_y = zz_y[iH + 1]   # previous swing low

                if E_y < C_y:
                    # Find the first bullish candle after the swing
                    # low E that started the impulse move.
                    mb_candle = _find_initiating_candle(
                        E_x, i, close, opn, high, low,
                        body_top, body_bot, use_body,
                        bullish=True,
                    )
                    if mb_candle is not None:
                        mb_top_v, mb_bot_v = mb_candle
                        mb_dir = 1
                        mb_active = True
                        mb_entry_emitted = False
                        out_bull[i] = 1
                        mss_formed = True

                mss_dir = 1

            # --- Bearish MSS --- #
            # Structure: previous swing high (C) → swing low (D) →
            #   swing high E (HH, E_y > C_y) → close breaks below D.
            # Mitigation Block: first bearish candle after swing
            #   high E that starts the impulse down.
            elif (iL < zz_count and iL + 1 < zz_count
                    and iL - 1 >= 0
                    and zz_d[iL] == -1
                    and close[i] < zz_y[iL]
                    and mss_dir > -1):
                E_x = zz_x[iL - 1]
                E_y = zz_y[iL - 1]   # most recent swing high
                C_y = zz_y[iL + 1]   # previous swing high

                if E_y > C_y:
                    # Find the first bearish candle after the swing
                    # high E that started the impulse move.
                    mb_candle = _find_initiating_candle(
                        E_x, i, close, opn, high, low,
                        body_top, body_bot, use_body,
                        bullish=False,
                    )
                    if mb_candle is not None:
                        mb_top_v, mb_bot_v = mb_candle
                        mb_dir = -1
                        mb_active = True
                        mb_entry_emitted = False
                        out_bear[i] = 1
                        mss_formed = True

                mss_dir = -1

        # ---- 3. MB lifecycle management ---- #
        if mb_active and not mss_formed:
            if mb_dir == 1:
                # Bullish MB: waiting for price to retrace into zone
                # Mitigated if price closes below zone bottom
                if close[i] < mb_bot_v:
                    mb_active = False
                    out_mitig[i] = 1
                elif not mb_entry_emitted:
                    # Entry: price dips into the zone
                    if low[i] <= mb_top_v and close[i] >= mb_bot_v:
                        out_elong[i] = 1
                        mb_entry_emitted = True
            else:
                # Bearish MB: waiting for price to retrace up into zone
                # Mitigated if price closes above zone top
                if close[i] > mb_top_v:
                    mb_active = False
                    out_mitig[i] = 1
                elif not mb_entry_emitted:
                    # Entry: price pushes up into the zone
                    if high[i] >= mb_bot_v and close[i] <= mb_top_v:
                        out_eshort[i] = 1
                        mb_entry_emitted = True

        # ---- 4. Forward-fill active zone ---- #
        if mb_active:
            out_top[i] = mb_top_v
            out_bot[i] = mb_bot_v
            out_dir[i] = mb_dir

    # ---- Assign results ---- #
    data = data.copy()
    data[bullish_mb_col] = out_bull
    data[bearish_mb_col] = out_bear
    data[mb_top_col] = out_top
    data[mb_bot_col] = out_bot
    data[mb_dir_col] = out_dir
    data[entry_long_col] = out_elong
    data[entry_short_col] = out_eshort
    data[mitigated_col] = out_mitig
    return data


def _find_initiating_candle(
    swing_idx: int,
    mss_bar: int,
    close: np.ndarray,
    opn: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    body_top: np.ndarray,
    body_bot: np.ndarray,
    use_body: bool,
    bullish: bool,
):
    """Find the first candle that initiates the impulse move.

    For bullish: scan forward from ``swing_idx`` looking for the
    first candle where ``close > open`` (bullish candle).

    For bearish: scan forward from ``swing_idx`` looking for the
    first candle where ``close < open`` (bearish candle).

    Returns ``(top, bottom)`` of the zone or ``None`` if no
    matching candle is found.
    """
    for j in range(swing_idx, mss_bar):
        if j < 0 or j >= len(close):
            continue
        is_match = (close[j] > opn[j]) if bullish else (close[j] < opn[j])
        if is_match:
            if use_body:
                return (body_top[j], body_bot[j])
            else:
                return (high[j], low[j])
    return None


# ------------------------------------------------------------------ #
#  Polars implementation (delegates to pandas)                        #
# ------------------------------------------------------------------ #
def _mb_polars(
    data: PlDataFrame,
    swing_length: int,
    use_body: bool,
    high_col: str,
    low_col: str,
    open_col: str,
    close_col: str,
    bullish_mb_col: str,
    bearish_mb_col: str,
    mb_top_col: str,
    mb_bot_col: str,
    mb_dir_col: str,
    entry_long_col: str,
    entry_short_col: str,
    mitigated_col: str,
) -> PlDataFrame:
    """Polars wrapper — converts to pandas, runs logic, converts back."""
    pdf = data.to_pandas()
    result = _mb_pandas(
        pdf, swing_length, use_body,
        high_col, low_col, open_col, close_col,
        bullish_mb_col, bearish_mb_col,
        mb_top_col, mb_bot_col, mb_dir_col,
        entry_long_col, entry_short_col,
        mitigated_col,
    )
    return pl.from_pandas(result)


# ------------------------------------------------------------------ #
#  Signal function                                                    #
# ------------------------------------------------------------------ #
def mitigation_blocks_signal(
    data: Union[PdDataFrame, PlDataFrame],
    mb_entry_long_column: str = "mb_entry_long",
    mb_entry_short_column: str = "mb_entry_short",
    signal_column: str = "mb_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Derive a single trading signal column from mitigation block output.

    Signal values:
        * ``1``  — long entry (bullish MB zone reached)
        * ``-1`` — short entry (bearish MB zone reached)
        * ``0``  — no signal

    Args:
        data: DataFrame with ``mitigation_blocks()`` columns.
        mb_entry_long_column: Column for bullish entry flags
            (default: ``"mb_entry_long"``).
        mb_entry_short_column: Column for bearish entry flags
            (default: ``"mb_entry_short"``).
        signal_column: Output column name
            (default: ``"mb_signal"``).

    Returns:
        DataFrame with added ``{signal_column}`` column.

    Example:
        >>> df = mitigation_blocks(df)
        >>> df = mitigation_blocks_signal(df)
        >>> buys = df[df['mb_signal'] == 1]
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        long_flags = data[mb_entry_long_column].values
        short_flags = data[mb_entry_short_column].values
        signal = np.where(long_flags == 1, 1,
                          np.where(short_flags == 1, -1, 0))
        data[signal_column] = signal
        return data
    elif isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        result = mitigation_blocks_signal(
            pdf, mb_entry_long_column,
            mb_entry_short_column, signal_column,
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ------------------------------------------------------------------ #
#  Stats function                                                     #
# ------------------------------------------------------------------ #
def get_mitigation_blocks_stats(
    data: Union[PdDataFrame, PlDataFrame],
    bullish_mb_column: str = "mb_bullish",
    bearish_mb_column: str = "mb_bearish",
    mb_entry_long_column: str = "mb_entry_long",
    mb_entry_short_column: str = "mb_entry_short",
    mb_mitigated_column: str = "mb_mitigated",
) -> Dict:
    """
    Compute summary statistics from mitigation block output.

    Args:
        data: DataFrame with ``mitigation_blocks()`` columns.
        [column parameters]: Column names matching the output.

    Returns:
        Dictionary with keys:

        - ``total_bullish_mb``: Number of bullish MBs formed
        - ``total_bearish_mb``: Number of bearish MBs formed
        - ``total_mb``: Total MBs formed
        - ``total_entry_long``: Long entry signals fired
        - ``total_entry_short``: Short entry signals fired
        - ``total_entries``: Total entries
        - ``total_mitigated``: Mitigated MBs
        - ``entry_rate``: Fraction of MBs that produced an entry
        - ``mitigation_rate``: Fraction of MBs that were mitigated

    Example:
        >>> df = mitigation_blocks(df)
        >>> stats = get_mitigation_blocks_stats(df)
        >>> print(stats)
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    bull = int(data[bullish_mb_column].sum())
    bear = int(data[bearish_mb_column].sum())
    total = bull + bear
    entries_long = int(data[mb_entry_long_column].sum())
    entries_short = int(data[mb_entry_short_column].sum())
    total_entries = entries_long + entries_short
    mitigated = int(data[mb_mitigated_column].sum())
    entry_rate = total_entries / total if total > 0 else 0.0
    mitig_rate = mitigated / total if total > 0 else 0.0

    return {
        "total_bullish_mb": bull,
        "total_bearish_mb": bear,
        "total_mb": total,
        "total_entry_long": entries_long,
        "total_entry_short": entries_short,
        "total_entries": total_entries,
        "total_mitigated": mitigated,
        "entry_rate": round(entry_rate, 4),
        "mitigation_rate": round(mitig_rate, 4),
    }
