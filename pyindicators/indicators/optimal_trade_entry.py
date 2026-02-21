"""
Optimal Trade Entry (OTE)

Identifies ICT Optimal Trade Entry zones — Fibonacci retracement
sweet spots (61.8 %–78.6 %) of the impulse leg following a Market
Structure Shift (MSS).

**Concept (ICT / Smart Money):**
    After a Break of Structure (price closes beyond a significant
    swing high or low), the market typically retraces before
    continuing.  The *Optimal Trade Entry* is the 61.8 %–78.6 %
    Fibonacci retracement of that impulse leg — the zone where
    institutional traders are most likely to add to positions.

**Detection algorithm:**
    1.  Build a zigzag (alternating swing highs / swing lows) using
        a configurable ``swing_length``.
    2.  Detect a **Market Structure Shift** (MSS):
        * Bullish MSS — close breaks above the most recent swing
          high while the preceding swing low made a **Lower Low**
          (LL).
        * Bearish MSS — close breaks below the most recent swing
          low while the preceding swing high made a **Higher High**
          (HH).
    3.  The *impulse leg* runs from the MSS-causing swing extreme
        to the MSS bar's close.
    4.  The OTE zone is the 61.8 %–78.6 % retracement of that leg.
    5.  Additional Fibonacci levels (50 %, 100 %, extensions) are
        also emitted for confluence analysis.

**Entry signals:**
    * Bullish:  price pulls back into the OTE zone (low ≤ zone top)
      after a bullish MSS.
    * Bearish:  price pushes up into the OTE zone (high ≥ zone
      bottom) after a bearish MSS.

**Premium / Discount filter (optional):**
    When ``premium_discount_filter=True`` bullish OTEs are only
    emitted when the swing low of the impulse leg sits in the
    *discount* half of the overall range, and bearish OTEs only
    when the swing high is in the *premium* half.

License note:  This is an independent, clean-room implementation of
publicly documented ICT trading concepts.
"""
from typing import Union, Dict
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
import numpy as np

from pyindicators.exceptions import PyIndicatorException


# Standard OTE Fibonacci levels
OTE_FIB_LEVELS = {
    "fib_0": 0.0,       # Swing extreme
    "fib_236": 0.236,
    "fib_382": 0.382,
    "fib_50": 0.5,       # Equilibrium
    "fib_618": 0.618,    # OTE zone start
    "fib_705": 0.705,    # OTE mid-point
    "fib_786": 0.786,    # OTE zone end
    "fib_100": 1.0,      # Opposite extreme
}


def optimal_trade_entry(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 5,
    ote_fib_start: float = 0.618,
    ote_fib_end: float = 0.786,
    premium_discount_filter: bool = False,
    high_column: str = "High",
    low_column: str = "Low",
    open_column: str = "Open",
    close_column: str = "Close",
    ote_bullish_column: str = "ote_bullish",
    ote_bearish_column: str = "ote_bearish",
    ote_zone_top_column: str = "ote_zone_top",
    ote_zone_bottom_column: str = "ote_zone_bottom",
    ote_entry_long_column: str = "ote_entry_long",
    ote_entry_short_column: str = "ote_entry_short",
    ote_invalidated_column: str = "ote_invalidated",
    ote_direction_column: str = "ote_direction",
    impulse_high_column: str = "ote_impulse_high",
    impulse_low_column: str = "ote_impulse_low",
    fib_prefix: str = "ote",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Identify Optimal Trade Entry (OTE) zones and generate signals.

    Args:
        data: pandas or polars DataFrame with OHLC price data.
        swing_length: Lookback period for pivot detection (default: 5).
        ote_fib_start: Fibonacci level for the start (shallow end)
            of the OTE zone (default: 0.618).
        ote_fib_end: Fibonacci level for the end (deep end) of the
            OTE zone (default: 0.786).
        premium_discount_filter: If ``True``, only emit bullish OTEs
            when the impulse origin is in the discount zone and
            bearish OTEs when in the premium zone (default: ``False``).
        high_column: Column for highs (default: ``"High"``).
        low_column: Column for lows (default: ``"Low"``).
        open_column: Column for opens (default: ``"Open"``).
        close_column: Column for closes (default: ``"Close"``).
        ote_bullish_column: Output — 1 on the bar where a bullish
            OTE zone is established (default: ``"ote_bullish"``).
        ote_bearish_column: Output — 1 on the bar where a bearish
            OTE zone is established (default: ``"ote_bearish"``).
        ote_zone_top_column: Output — active OTE zone upper boundary,
            forward-filled while the zone is alive
            (default: ``"ote_zone_top"``).
        ote_zone_bottom_column: Output — active OTE zone lower boundary
            (default: ``"ote_zone_bottom"``).
        ote_entry_long_column: Output — 1 when price enters a bullish
            OTE zone (default: ``"ote_entry_long"``).
        ote_entry_short_column: Output — 1 when price enters a bearish
            OTE zone (default: ``"ote_entry_short"``).
        ote_invalidated_column: Output — 1 when the OTE zone is
            invalidated (price breaks the impulse origin)
            (default: ``"ote_invalidated"``).
        ote_direction_column: Output — 1 for active bullish OTE,
            -1 for active bearish OTE, 0 when no OTE is active
            (default: ``"ote_direction"``).
        impulse_high_column: Output — high of the impulse leg
            (default: ``"ote_impulse_high"``).
        impulse_low_column: Output — low of the impulse leg
            (default: ``"ote_impulse_low"``).
        fib_prefix: Prefix for Fibonacci level columns
            (default: ``"ote"``).

    Returns:
        DataFrame with added columns:

        - ``{ote_bullish_column}``: 1 on bullish OTE formation bar
        - ``{ote_bearish_column}``: 1 on bearish OTE formation bar
        - ``{ote_zone_top_column}``: Upper boundary of OTE zone
        - ``{ote_zone_bottom_column}``: Lower boundary of OTE zone
        - ``{ote_entry_long_column}``: 1 when price enters bullish
          OTE zone
        - ``{ote_entry_short_column}``: 1 when price enters bearish
          OTE zone
        - ``{ote_invalidated_column}``: 1 when OTE is invalidated
        - ``{ote_direction_column}``: Active OTE direction (1 / -1 / 0)
        - ``{impulse_high_column}``: Impulse leg high price
        - ``{impulse_low_column}``: Impulse leg low price
        - ``{fib_prefix}_fib_*``: Fibonacci retracement levels

    Example:
        >>> import pandas as pd
        >>> from pyindicators import optimal_trade_entry
        >>> df = pd.DataFrame({
        ...     'Open': [...], 'High': [...],
        ...     'Low': [...], 'Close': [...]
        ... })
        >>> result = optimal_trade_entry(df, swing_length=5)
    """
    if isinstance(data, PdDataFrame):
        return _ote_pandas(
            data, swing_length, ote_fib_start, ote_fib_end,
            premium_discount_filter,
            high_column, low_column, open_column, close_column,
            ote_bullish_column, ote_bearish_column,
            ote_zone_top_column, ote_zone_bottom_column,
            ote_entry_long_column, ote_entry_short_column,
            ote_invalidated_column, ote_direction_column,
            impulse_high_column, impulse_low_column,
            fib_prefix,
        )
    elif isinstance(data, PlDataFrame):
        return _ote_polars(
            data, swing_length, ote_fib_start, ote_fib_end,
            premium_discount_filter,
            high_column, low_column, open_column, close_column,
            ote_bullish_column, ote_bearish_column,
            ote_zone_top_column, ote_zone_bottom_column,
            ote_entry_long_column, ote_entry_short_column,
            ote_invalidated_column, ote_direction_column,
            impulse_high_column, impulse_low_column,
            fib_prefix,
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ------------------------------------------------------------------ #
#  Pandas implementation                                              #
# ------------------------------------------------------------------ #
def _ote_pandas(
    data: PdDataFrame,
    swing_length: int,
    ote_fib_start: float,
    ote_fib_end: float,
    pd_filter: bool,
    high_col: str,
    low_col: str,
    open_col: str,
    close_col: str,
    ote_bull_col: str,
    ote_bear_col: str,
    zone_top_col: str,
    zone_bot_col: str,
    entry_long_col: str,
    entry_short_col: str,
    invalidated_col: str,
    direction_col: str,
    imp_high_col: str,
    imp_low_col: str,
    fib_prefix: str,
) -> PdDataFrame:
    """Core OTE algorithm (pandas)."""
    high = data[high_col].values.astype(float)
    low = data[low_col].values.astype(float)
    close = data[close_col].values.astype(float)
    n = len(data)

    # ----- Zigzag ring buffer ----- #
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

    # ----- OTE state ----- #
    mss_dir = 0             # last MSS direction
    ote_active = False      # is an OTE zone active?
    ote_dir = 0             # +1 bullish, -1 bearish
    ote_ztop = np.nan       # OTE zone top
    ote_zbot = np.nan       # OTE zone bottom
    ote_imp_hi = np.nan     # impulse leg high
    ote_imp_lo = np.nan     # impulse leg low
    ote_entry_emitted = False  # only one entry per OTE
    ote_fibs = {}           # current fib levels

    # ----- Output arrays ----- #
    out_bull = np.zeros(n, dtype=int)
    out_bear = np.zeros(n, dtype=int)
    out_ztop = np.full(n, np.nan)
    out_zbot = np.full(n, np.nan)
    out_elong = np.zeros(n, dtype=int)
    out_eshort = np.zeros(n, dtype=int)
    out_inval = np.zeros(n, dtype=int)
    out_dir = np.zeros(n, dtype=int)
    out_imp_hi = np.full(n, np.nan)
    out_imp_lo = np.full(n, np.nan)

    # Fib level output arrays
    fib_arrays = {}
    for key in OTE_FIB_LEVELS:
        fib_arrays[key] = np.full(n, np.nan)

    for i in range(1, n):
        # ---- 1. Pivot detection ---- #
        ci = i - 1  # candidate
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

        # ---- 2. MSS detection & OTE zone establishment ---- #
        mss_formed = False

        if zz_count >= 4:
            iH = 2 if (zz_count > 2 and zz_d[2] == 1) else 1
            iL = 2 if (zz_count > 2 and zz_d[2] == -1) else 1

            # --- Bullish MSS --- #
            #   Structure: swing low E (LL) → swing high D → swing low C
            #   → close breaks above D.
            #   E_y < C_y (lower low confirmed).
            #   Impulse leg: swing low E (low) → MSS bar close (high).
            if (iH < zz_count and iH + 1 < zz_count
                    and iH - 1 >= 0
                    and zz_d[iH] == 1
                    and close[i] > zz_y[iH]
                    and mss_dir < 1):
                E_y = zz_y[iH - 1]      # most recent swing low
                C_y = zz_y[iH + 1]       # previous swing low

                if E_y < C_y:
                    # Impulse: from swing low E to the MSS bar
                    imp_lo = E_y
                    imp_hi = close[i]
                    leg = imp_hi - imp_lo

                    # Premium/discount filter
                    pass_filter = True
                    if pd_filter and zz_count >= 6:
                        # Use overall range from recent swings
                        recent_prices = [
                            zz_y[k] for k in range(min(zz_count, 8))
                            if not np.isnan(zz_y[k])
                        ]
                        if recent_prices:
                            range_hi = max(recent_prices)
                            range_lo = min(recent_prices)
                            equilibrium = (range_hi + range_lo) / 2
                            # Bullish OTE valid only if impulse
                            # origin is in discount zone
                            pass_filter = imp_lo < equilibrium

                    if pass_filter and leg > 0:
                        # Calculate Fibonacci retracement levels
                        # For a bullish retracement: levels are measured
                        # DOWN from the impulse high
                        ote_ztop = imp_hi - leg * ote_fib_start
                        ote_zbot = imp_hi - leg * ote_fib_end
                        ote_imp_hi = imp_hi
                        ote_imp_lo = imp_lo
                        ote_dir = 1
                        ote_active = True
                        ote_entry_emitted = False
                        out_bull[i] = 1
                        mss_formed = True

                        # Calculate all fib levels
                        ote_fibs = {}
                        for key, level in OTE_FIB_LEVELS.items():
                            ote_fibs[key] = imp_hi - leg * level

                mss_dir = 1

            # --- Bearish MSS --- #
            #   Structure: swing high E (HH) → swing low D → swing high C
            #   → close breaks below D.
            #   E_y > C_y (higher high confirmed).
            #   Impulse leg: swing high E (high) → MSS bar close (low).
            elif (iL < zz_count and iL + 1 < zz_count
                    and iL - 1 >= 0
                    and zz_d[iL] == -1
                    and close[i] < zz_y[iL]
                    and mss_dir > -1):
                E_y = zz_y[iL - 1]     # most recent swing high
                C_y = zz_y[iL + 1]     # previous swing high

                if E_y > C_y:
                    # Impulse: from swing high E to the MSS bar
                    imp_hi = E_y
                    imp_lo = close[i]
                    leg = imp_hi - imp_lo

                    # Premium/discount filter
                    pass_filter = True
                    if pd_filter and zz_count >= 6:
                        recent_prices = [
                            zz_y[k] for k in range(min(zz_count, 8))
                            if not np.isnan(zz_y[k])
                        ]
                        if recent_prices:
                            range_hi = max(recent_prices)
                            range_lo = min(recent_prices)
                            equilibrium = (range_hi + range_lo) / 2
                            # Bearish OTE valid only in premium
                            pass_filter = imp_hi > equilibrium

                    if pass_filter and leg > 0:
                        # For bearish retracement: levels measured
                        # UP from the impulse low
                        ote_ztop = imp_lo + leg * ote_fib_end
                        ote_zbot = imp_lo + leg * ote_fib_start
                        ote_imp_hi = imp_hi
                        ote_imp_lo = imp_lo
                        ote_dir = -1
                        ote_active = True
                        ote_entry_emitted = False
                        out_bear[i] = 1
                        mss_formed = True

                        # Calculate all fib levels
                        ote_fibs = {}
                        for key, level in OTE_FIB_LEVELS.items():
                            ote_fibs[key] = imp_lo + leg * level

                mss_dir = -1

        # ---- 3. OTE lifecycle management ---- #
        if ote_active and not mss_formed:
            if ote_dir == 1:
                # Bullish OTE: waiting for pullback into zone
                # Invalidated if price closes below impulse low
                if close[i] < ote_imp_lo:
                    ote_active = False
                    out_inval[i] = 1
                elif not ote_entry_emitted:
                    # Entry: price low dips into the OTE zone
                    if low[i] <= ote_ztop and low[i] >= ote_zbot:
                        out_elong[i] = 1
                        ote_entry_emitted = True
                    elif low[i] < ote_zbot and high[i] >= ote_zbot:
                        # Price wicked through zone
                        out_elong[i] = 1
                        ote_entry_emitted = True
            else:
                # Bearish OTE: waiting for retracement up into zone
                # Invalidated if price closes above impulse high
                if close[i] > ote_imp_hi:
                    ote_active = False
                    out_inval[i] = 1
                elif not ote_entry_emitted:
                    # Entry: price high pushes into the OTE zone
                    if high[i] >= ote_zbot and high[i] <= ote_ztop:
                        out_eshort[i] = 1
                        ote_entry_emitted = True
                    elif high[i] > ote_ztop and low[i] <= ote_ztop:
                        # Price wicked through zone
                        out_eshort[i] = 1
                        ote_entry_emitted = True

        # ---- 4. Forward-fill active zone ---- #
        if ote_active:
            out_ztop[i] = ote_ztop
            out_zbot[i] = ote_zbot
            out_dir[i] = ote_dir
            out_imp_hi[i] = ote_imp_hi
            out_imp_lo[i] = ote_imp_lo
            for key in OTE_FIB_LEVELS:
                if key in ote_fibs:
                    fib_arrays[key][i] = ote_fibs[key]

    # ---- Assign results ---- #
    data = data.copy()
    data[ote_bull_col] = out_bull
    data[ote_bear_col] = out_bear
    data[zone_top_col] = out_ztop
    data[zone_bot_col] = out_zbot
    data[entry_long_col] = out_elong
    data[entry_short_col] = out_eshort
    data[invalidated_col] = out_inval
    data[direction_col] = out_dir
    data[imp_high_col] = out_imp_hi
    data[imp_low_col] = out_imp_lo

    for key in OTE_FIB_LEVELS:
        col_name = f"{fib_prefix}_{key}"
        data[col_name] = fib_arrays[key]

    return data


# ------------------------------------------------------------------ #
#  Polars implementation (delegates to pandas)                        #
# ------------------------------------------------------------------ #
def _ote_polars(
    data: PlDataFrame,
    swing_length: int,
    ote_fib_start: float,
    ote_fib_end: float,
    pd_filter: bool,
    high_col: str,
    low_col: str,
    open_col: str,
    close_col: str,
    ote_bull_col: str,
    ote_bear_col: str,
    zone_top_col: str,
    zone_bot_col: str,
    entry_long_col: str,
    entry_short_col: str,
    invalidated_col: str,
    direction_col: str,
    imp_high_col: str,
    imp_low_col: str,
    fib_prefix: str,
) -> PlDataFrame:
    """Polars wrapper — converts to pandas, runs logic, converts back."""
    pdf = data.to_pandas()
    result = _ote_pandas(
        pdf, swing_length, ote_fib_start, ote_fib_end, pd_filter,
        high_col, low_col, open_col, close_col,
        ote_bull_col, ote_bear_col,
        zone_top_col, zone_bot_col,
        entry_long_col, entry_short_col,
        invalidated_col, direction_col,
        imp_high_col, imp_low_col,
        fib_prefix,
    )
    return pl.from_pandas(result)


# ------------------------------------------------------------------ #
#  Signal function                                                    #
# ------------------------------------------------------------------ #
def optimal_trade_entry_signal(
    data: Union[PdDataFrame, PlDataFrame],
    ote_entry_long_column: str = "ote_entry_long",
    ote_entry_short_column: str = "ote_entry_short",
    signal_column: str = "ote_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Derive a single trading signal column from OTE output.

    Signal values:
        * ``1``  — long entry (bullish OTE zone reached)
        * ``-1`` — short entry (bearish OTE zone reached)
        * ``0``  — no signal

    Args:
        data: DataFrame with ``optimal_trade_entry()`` columns.
        ote_entry_long_column: Column for bullish entry flags
            (default: ``"ote_entry_long"``).
        ote_entry_short_column: Column for bearish entry flags
            (default: ``"ote_entry_short"``).
        signal_column: Output column name
            (default: ``"ote_signal"``).

    Returns:
        DataFrame with added ``{signal_column}`` column.

    Example:
        >>> df = optimal_trade_entry(df)
        >>> df = optimal_trade_entry_signal(df)
        >>> buys = df[df['ote_signal'] == 1]
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        long_flags = data[ote_entry_long_column].values
        short_flags = data[ote_entry_short_column].values
        signal = np.where(long_flags == 1, 1,
                          np.where(short_flags == 1, -1, 0))
        data[signal_column] = signal
        return data
    elif isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        result = optimal_trade_entry_signal(
            pdf, ote_entry_long_column,
            ote_entry_short_column, signal_column,
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ------------------------------------------------------------------ #
#  Stats function                                                     #
# ------------------------------------------------------------------ #
def get_optimal_trade_entry_stats(
    data: Union[PdDataFrame, PlDataFrame],
    ote_bullish_column: str = "ote_bullish",
    ote_bearish_column: str = "ote_bearish",
    ote_entry_long_column: str = "ote_entry_long",
    ote_entry_short_column: str = "ote_entry_short",
    ote_invalidated_column: str = "ote_invalidated",
) -> Dict:
    """
    Compute summary statistics from OTE output.

    Args:
        data: DataFrame with ``optimal_trade_entry()`` columns.
        [column parameters]: Column names matching the output.

    Returns:
        Dictionary with keys:

        - ``total_bullish_ote``: Number of bullish OTE zones formed
        - ``total_bearish_ote``: Number of bearish OTE zones formed
        - ``total_ote``: Total OTE zones formed
        - ``total_entry_long``: Long entry signals fired
        - ``total_entry_short``: Short entry signals fired
        - ``total_entries``: Total entries
        - ``total_invalidated``: Invalidated OTE zones
        - ``entry_rate``: Fraction of OTEs that produced an entry
        - ``invalidation_rate``: Fraction of OTEs that were
          invalidated

    Example:
        >>> df = optimal_trade_entry(df)
        >>> stats = get_optimal_trade_entry_stats(df)
        >>> print(stats)
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    bull = int(data[ote_bullish_column].sum())
    bear = int(data[ote_bearish_column].sum())
    total = bull + bear
    entries_long = int(data[ote_entry_long_column].sum())
    entries_short = int(data[ote_entry_short_column].sum())
    total_entries = entries_long + entries_short
    invalidated = int(data[ote_invalidated_column].sum())
    entry_rate = total_entries / total if total > 0 else 0.0
    inval_rate = invalidated / total if total > 0 else 0.0

    return {
        "total_bullish_ote": bull,
        "total_bearish_ote": bear,
        "total_ote": total,
        "total_entry_long": entries_long,
        "total_entry_short": entries_short,
        "total_entries": total_entries,
        "total_invalidated": invalidated,
        "entry_rate": round(entry_rate, 4),
        "invalidation_rate": round(inval_rate, 4),
    }
