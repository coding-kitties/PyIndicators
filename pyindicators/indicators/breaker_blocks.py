"""
Breaker Blocks with Signals

Identifies Breaker Blocks -- failed Order Blocks that flip into opposite
support/resistance zones -- using Market Structure Shifts (MSS).

**Concept:**
    A Breaker Block forms when an Order Block is "broken" by a Market
    Structure Shift.  The zone that previously acted as support flips
    to resistance (or vice-versa).

**Bullish Breaker Block (+BB):**
    Detected when a bullish MSS occurs (close breaks above the most
    recent swing high) after a confirmed lower-low pattern.  The
    decisive bullish candle in the up-leg leading to the broken swing
    high becomes the breaker block zone.

**Bearish Breaker Block (-BB):**
    Detected when a bearish MSS occurs (close breaks below the most
    recent swing low) after a confirmed higher-high pattern.  The
    decisive bearish candle in the down-leg leading to the broken
    swing low becomes the breaker block zone.

**Entry Signals:**
    * Long entry (+BB):  Price opens between the center line and the
      top of the zone, then closes above the top.
    * Short entry (-BB): Price opens between the center line and the
      bottom of the zone, then closes below the bottom.

**Cancellation:**
    The BB is considered "broken" when price closes past the center
    line (the halfway point of the zone) without triggering an entry.

**Mitigation:**
    The BB is fully mitigated (invalidated) when price closes through
    the opposite side of the zone entirely.
"""
from typing import Union, Dict
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
import numpy as np

from pyindicators.exceptions import PyIndicatorException


def breaker_blocks(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 5,
    use_body: bool = False,
    use_2_candles: bool = False,
    stop_at_first_center_break: bool = True,
    high_column: str = "High",
    low_column: str = "Low",
    open_column: str = "Open",
    close_column: str = "Close",
    bullish_bb_column: str = "bb_bullish",
    bearish_bb_column: str = "bb_bearish",
    bb_top_column: str = "bb_top",
    bb_bottom_column: str = "bb_bottom",
    bb_center_column: str = "bb_center",
    bb_direction_column: str = "bb_direction",
    bb_entry_long_column: str = "bb_entry_long",
    bb_entry_short_column: str = "bb_entry_short",
    bb_cancel_column: str = "bb_cancel",
    bb_mitigated_column: str = "bb_mitigated",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Identify Breaker Blocks and generate trading signals.

    A Breaker Block is a failed Order Block that "flips" into an
    opposite-direction zone after a Market Structure Shift.  This
    function detects BB zones, tracks their lifecycle, and emits
    entry / cancellation / mitigation signals.

    Args:
        data: pandas or polars DataFrame with OHLC price data.
        swing_length: Lookback period for pivot detection
            (default: 5).  The pivot is confirmed 1 bar later,
            checking ``swing_length`` bars to the left.
        use_body: If ``True``, use candle body (open/close) instead
            of wicks (high/low) for the BB zone boundaries
            (default: ``False``).
        use_2_candles: If ``True``, expand the BB zone to include the
            preceding candle when it is in the same direction
            (default: ``False``).
        stop_at_first_center_break: If ``True`` (default), once the
            center line is broken, the BB remains invalidated.  If
            ``False``, the BB can reactivate when price returns
            through the zone.
        high_column: Column for highs (default: ``"High"``).
        low_column: Column for lows (default: ``"Low"``).
        open_column: Column for opens (default: ``"Open"``).
        close_column: Column for closes (default: ``"Close"``).
        bullish_bb_column: Output – 1 on bullish BB formation bar
            (default: ``"bb_bullish"``).
        bearish_bb_column: Output – 1 on bearish BB formation bar
            (default: ``"bb_bearish"``).
        bb_top_column: Output – active BB zone top, forward-filled
            while the BB is alive (default: ``"bb_top"``).
        bb_bottom_column: Output – active BB zone bottom
            (default: ``"bb_bottom"``).
        bb_center_column: Output – center (50 %) line of the BB
            (default: ``"bb_center"``).
        bb_direction_column: Output – 1 for bullish BB, -1 for
            bearish BB, 0 when no BB is active
            (default: ``"bb_direction"``).
        bb_entry_long_column: Output – 1 when a long entry signal
            fires (default: ``"bb_entry_long"``).
        bb_entry_short_column: Output – 1 when a short entry signal
            fires (default: ``"bb_entry_short"``).
        bb_cancel_column: Output – 1 when the BB center line is
            broken (default: ``"bb_cancel"``).
        bb_mitigated_column: Output – 1 when the BB is fully
            mitigated (default: ``"bb_mitigated"``).

    Returns:
        DataFrame with added columns (see parameter docs above).

    Example:
        >>> import pandas as pd
        >>> from pyindicators import breaker_blocks
        >>> df = pd.DataFrame({
        ...     'Open': [...], 'High': [...],
        ...     'Low': [...], 'Close': [...]
        ... })
        >>> result = breaker_blocks(df, swing_length=5)
    """
    if isinstance(data, PdDataFrame):
        return _breaker_blocks_pandas(
            data, swing_length, use_body, use_2_candles,
            stop_at_first_center_break,
            high_column, low_column, open_column, close_column,
            bullish_bb_column, bearish_bb_column,
            bb_top_column, bb_bottom_column, bb_center_column,
            bb_direction_column,
            bb_entry_long_column, bb_entry_short_column,
            bb_cancel_column, bb_mitigated_column,
        )
    elif isinstance(data, PlDataFrame):
        return _breaker_blocks_polars(
            data, swing_length, use_body, use_2_candles,
            stop_at_first_center_break,
            high_column, low_column, open_column, close_column,
            bullish_bb_column, bearish_bb_column,
            bb_top_column, bb_bottom_column, bb_center_column,
            bb_direction_column,
            bb_entry_long_column, bb_entry_short_column,
            bb_cancel_column, bb_mitigated_column,
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ------------------------------------------------------------------ #
#  Pandas implementation                                              #
# ------------------------------------------------------------------ #
def _breaker_blocks_pandas(
    data: PdDataFrame,
    swing_length: int,
    use_body: bool,
    use_2_candles: bool,
    stop_at_first_center_break: bool,
    high_col: str,
    low_col: str,
    open_col: str,
    close_col: str,
    bullish_bb_col: str,
    bearish_bb_col: str,
    bb_top_col: str,
    bb_bottom_col: str,
    bb_center_col: str,
    bb_direction_col: str,
    bb_entry_long_col: str,
    bb_entry_short_col: str,
    bb_cancel_col: str,
    bb_mitigated_col: str,
) -> PdDataFrame:
    """Core algorithm (pandas)."""
    high = data[high_col].values.astype(float)
    low = data[low_col].values.astype(float)
    opn = data[open_col].values.astype(float)
    close = data[close_col].values.astype(float)
    n = len(data)

    # Body boundaries
    body_top = np.maximum(close, opn)
    body_bot = np.minimum(close, opn)

    # ----- Zigzag state (most-recent-first ring buffer) ----- #
    _MAX_ZZ = 50
    zz_d = [0] * _MAX_ZZ       # direction: +1 = swing high, -1 = low
    zz_x = [0] * _MAX_ZZ       # bar index
    zz_y = [np.nan] * _MAX_ZZ  # price
    zz_count = 0                # valid entries

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

    # ----- MSS / BB state ----- #
    mss_dir = 0           # 0 = neutral, 1 = bullish, -1 = bearish
    bb_top_v = np.nan     # active BB zone boundaries
    bb_bot_v = np.nan
    bb_cen_v = np.nan
    bb_dir_v = 0          # 1 = bullish BB, -1 = bearish BB
    bb_broken = False     # center-line broken
    bb_mitigated = False  # fully invalidated
    bb_scalp = False      # entry signal active (unused in output

    # ----- Output arrays ----- #
    out_bull = np.zeros(n, dtype=int)
    out_bear = np.zeros(n, dtype=int)
    out_top = np.full(n, np.nan)
    out_bot = np.full(n, np.nan)
    out_cen = np.full(n, np.nan)
    out_dir = np.zeros(n, dtype=int)
    out_elong = np.zeros(n, dtype=int)
    out_eshort = np.zeros(n, dtype=int)
    out_cancel = np.zeros(n, dtype=int)
    out_mitig = np.zeros(n, dtype=int)

    for i in range(1, n):
        # ---- 1. Pivot detection (bar i-1 confirmed by bar i) ---- #
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

        # ---- 2. MSS / BB formation ---- #
        mss_formed = False  # flag to avoid signals on formation bar

        if zz_count >= 4:
            iH = 2 if (zz_count > 2 and zz_d[2] == 1) else 1
            iL = 2 if (zz_count > 2 and zz_d[2] == -1) else 1

            # --- Bullish MSS --- #
            if (iH < zz_count and iH + 1 < zz_count
                    and iH - 1 >= 0
                    and zz_d[iH] == 1
                    and close[i] > zz_y[iH]
                    and mss_dir < 1):
                _, E_y = zz_x[iH - 1], zz_y[iH - 1]
                D_x = zz_x[iH]
                C_x, C_y = zz_x[iH + 1], zz_y[iH + 1]

                if E_y < C_y and C_x != D_x:
                    found = _search_bb_candle(
                        D_x, C_x, close, opn, high, low,
                        body_top, body_bot, use_body,
                        use_2_candles, bullish=True,
                    )
                    if found is not None:
                        bb_top_v, bb_bot_v = found
                        bb_cen_v = (bb_top_v + bb_bot_v) / 2.0
                        bb_dir_v = 1
                        bb_broken = False
                        bb_mitigated = False
                        bb_scalp = False
                        out_bull[i] = 1
                        mss_formed = True

                mss_dir = 1

            # --- Bearish MSS --- #
            elif (iL < zz_count and iL + 1 < zz_count
                    and iL - 1 >= 0
                    and zz_d[iL] == -1
                    and close[i] < zz_y[iL]
                    and mss_dir > -1):
                _, E_y = zz_x[iL - 1], zz_y[iL - 1]
                D_x = zz_x[iL]
                C_x, C_y = zz_x[iL + 1], zz_y[iL + 1]

                if E_y > C_y and C_x != D_x:
                    found = _search_bb_candle(
                        D_x, C_x, close, opn, high, low,
                        body_top, body_bot, use_body,
                        use_2_candles, bullish=False,
                    )
                    if found is not None:
                        bb_top_v, bb_bot_v = found
                        bb_cen_v = (bb_top_v + bb_bot_v) / 2.0
                        bb_dir_v = -1
                        bb_broken = False
                        bb_mitigated = False
                        bb_scalp = False
                        out_bear[i] = 1
                        mss_formed = True

                mss_dir = -1

        # ---- 3. Update active BB state ---- #
        if not np.isnan(bb_top_v) and not bb_mitigated:
            if bb_dir_v == 1:
                # Bullish BB
                if close[i] < bb_bot_v:
                    bb_mitigated = True
                    out_mitig[i] = 1
                elif not mss_formed:
                    if not bb_broken:
                        if (opn[i] > bb_cen_v
                                and opn[i] < bb_top_v
                                and close[i] > bb_top_v):
                            out_elong[i] = 1
                            bb_scalp = True
                        elif (close[i] < bb_cen_v
                                and close[i] > bb_bot_v):
                            bb_broken = True
                            out_cancel[i] = 1
                    else:
                        if not stop_at_first_center_break:
                            if close[i] > bb_top_v:
                                bb_broken = False
                                bb_scalp = True
            else:
                # Bearish BB
                if close[i] > bb_top_v:
                    bb_mitigated = True
                    out_mitig[i] = 1
                elif not mss_formed:
                    if not bb_broken:
                        if (opn[i] < bb_cen_v
                                and opn[i] > bb_bot_v
                                and close[i] < bb_bot_v):
                            out_eshort[i] = 1
                            bb_scalp = True
                        elif (close[i] > bb_cen_v
                                and close[i] < bb_top_v):
                            bb_broken = True
                            out_cancel[i] = 1
                    else:
                        if not stop_at_first_center_break:
                            if close[i] < bb_bot_v:
                                bb_broken = False
                                bb_scalp = True  # noqa: F841

        # ---- 4. Forward-fill active zone ---- #
        if not np.isnan(bb_top_v) and not bb_mitigated:
            out_top[i] = bb_top_v
            out_bot[i] = bb_bot_v
            out_cen[i] = bb_cen_v
            out_dir[i] = bb_dir_v

    # ---- Assign results ---- #
    data = data.copy()
    data[bullish_bb_col] = out_bull
    data[bearish_bb_col] = out_bear
    data[bb_top_col] = out_top
    data[bb_bottom_col] = out_bot
    data[bb_center_col] = out_cen
    data[bb_direction_col] = out_dir
    data[bb_entry_long_col] = out_elong
    data[bb_entry_short_col] = out_eshort
    data[bb_cancel_col] = out_cancel
    data[bb_mitigated_col] = out_mitig
    return data


def _search_bb_candle(
    d_x: int,
    c_x: int,
    close: np.ndarray,
    opn: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    body_top: np.ndarray,
    body_bot: np.ndarray,
    use_body: bool,
    use_2_candles: bool,
    bullish: bool,
):
    """Search from swing D backward to swing C for the BB candle.

    Returns ``(top, bottom)`` of the zone or ``None`` if no
    matching candle is found.
    """
    for j in range(d_x, c_x - 1, -1):
        if j < 0:
            break
        is_match = (close[j] > opn[j]) if bullish else (close[j] < opn[j])
        if is_match:
            if use_body:
                t, b = body_top[j], body_bot[j]
            else:
                t, b = high[j], low[j]

            if use_2_candles and j - 1 >= max(c_x, 0):
                prev_match = (
                    (close[j - 1] > opn[j - 1]) if bullish
                    else (close[j - 1] < opn[j - 1])
                )
                if prev_match:
                    if use_body:
                        c2_t, c2_b = body_top[j - 1], body_bot[j - 1]
                    else:
                        c2_t, c2_b = high[j - 1], low[j - 1]
                    t = max(t, c2_t)
                    b = min(b, c2_b)

            return (t, b)
    return None


# ------------------------------------------------------------------ #
#  Polars implementation (delegates to pandas)                        #
# ------------------------------------------------------------------ #
def _breaker_blocks_polars(
    data: PlDataFrame,
    swing_length: int,
    use_body: bool,
    use_2_candles: bool,
    stop_at_first_center_break: bool,
    high_col: str,
    low_col: str,
    open_col: str,
    close_col: str,
    bullish_bb_col: str,
    bearish_bb_col: str,
    bb_top_col: str,
    bb_bottom_col: str,
    bb_center_col: str,
    bb_direction_col: str,
    bb_entry_long_col: str,
    bb_entry_short_col: str,
    bb_cancel_col: str,
    bb_mitigated_col: str,
) -> PlDataFrame:
    """Polars wrapper – converts to pandas, runs logic, converts back."""
    pdf = data.to_pandas()
    result = _breaker_blocks_pandas(
        pdf, swing_length, use_body, use_2_candles,
        stop_at_first_center_break,
        high_col, low_col, open_col, close_col,
        bullish_bb_col, bearish_bb_col,
        bb_top_col, bb_bottom_col, bb_center_col,
        bb_direction_col,
        bb_entry_long_col, bb_entry_short_col,
        bb_cancel_col, bb_mitigated_col,
    )
    return pl.from_pandas(result)


# ------------------------------------------------------------------ #
#  Signal function                                                    #
# ------------------------------------------------------------------ #
def breaker_blocks_signal(
    data: Union[PdDataFrame, PlDataFrame],
    bb_entry_long_column: str = "bb_entry_long",
    bb_entry_short_column: str = "bb_entry_short",
    signal_column: str = "bb_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Derive a single trading signal column from breaker-block output.

    Signal values:
        * ``1``  – long entry (bullish BB bounce)
        * ``-1`` – short entry (bearish BB bounce)
        * ``0``  – no signal

    Args:
        data: DataFrame with ``breaker_blocks()`` columns already
            computed.
        bb_entry_long_column: Column for bullish entry flags
            (default: ``"bb_entry_long"``).
        bb_entry_short_column: Column for bearish entry flags
            (default: ``"bb_entry_short"``).
        signal_column: Output column name
            (default: ``"bb_signal"``).

    Returns:
        DataFrame with added ``{signal_column}`` column.

    Example:
        >>> df = breaker_blocks(df)
        >>> df = breaker_blocks_signal(df)
        >>> buys = df[df['bb_signal'] == 1]
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        long_flags = data[bb_entry_long_column].values
        short_flags = data[bb_entry_short_column].values
        signal = np.where(long_flags == 1, 1,
                          np.where(short_flags == 1, -1, 0))
        data[signal_column] = signal
        return data
    elif isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        result = breaker_blocks_signal(
            pdf, bb_entry_long_column,
            bb_entry_short_column, signal_column,
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ------------------------------------------------------------------ #
#  Stats function                                                     #
# ------------------------------------------------------------------ #
def get_breaker_blocks_stats(
    data: Union[PdDataFrame, PlDataFrame],
    bullish_bb_column: str = "bb_bullish",
    bearish_bb_column: str = "bb_bearish",
    bb_entry_long_column: str = "bb_entry_long",
    bb_entry_short_column: str = "bb_entry_short",
    bb_cancel_column: str = "bb_cancel",
    bb_mitigated_column: str = "bb_mitigated",
) -> Dict:
    """
    Compute summary statistics from breaker-block output.

    Args:
        data: DataFrame with ``breaker_blocks()`` columns.
        [column parameters]: Column names matching the output of
            ``breaker_blocks()``.

    Returns:
        Dictionary with keys:

        - ``total_bullish_bb``: Number of bullish BBs formed
        - ``total_bearish_bb``: Number of bearish BBs formed
        - ``total_bb``: Total BBs formed
        - ``total_entry_long``: Long entry signals fired
        - ``total_entry_short``: Short entry signals fired
        - ``total_cancels``: Center-line breaks (cancellations)
        - ``total_mitigated``: Fully mitigated BBs
        - ``entry_rate``: Fraction of BBs that produced an entry

    Example:
        >>> df = breaker_blocks(df)
        >>> stats = get_breaker_blocks_stats(df)
        >>> print(stats)
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    bull = int(data[bullish_bb_column].sum())
    bear = int(data[bearish_bb_column].sum())
    total = bull + bear
    entries_long = int(data[bb_entry_long_column].sum())
    entries_short = int(data[bb_entry_short_column].sum())
    cancels = int(data[bb_cancel_column].sum())
    mitigated = int(data[bb_mitigated_column].sum())
    total_entries = entries_long + entries_short
    entry_rate = total_entries / total if total > 0 else 0.0

    return {
        "total_bullish_bb": bull,
        "total_bearish_bb": bear,
        "total_bb": total,
        "total_entry_long": entries_long,
        "total_entry_short": entries_short,
        "total_cancels": cancels,
        "total_mitigated": mitigated,
        "entry_rate": round(entry_rate, 4),
    }
