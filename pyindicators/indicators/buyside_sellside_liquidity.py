"""
Buyside & Sellside Liquidity Indicator

Identifies clusters of swing highs (buyside liquidity) and swing lows
(sellside liquidity) that represent pools of resting orders. Large
players tend to push price into these clusters to fill orders, making
them key reversal / continuation zones.

1. **Zigzag tracking** – maintains a rolling buffer of swing
   highs/lows identified via pivot detection.
2. **Cluster detection** – when a new pivot is found the indicator
   scans historical swings for a *cluster* of ≥ 3 points within an
   ATR-scaled margin. The average price of the cluster becomes a
   liquidity level.
3. **Breach detection** – when price breaks through a level the
   indicator records the event and the resulting *liquidity zone*.
4. **Liquidity voids** – large directional gaps (measured against a
   long ATR window) where very little two-way trading took place.
"""
from typing import Union, Dict, List
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


# ── public API ────────────────────────────────────────────────────────


def buyside_sellside_liquidity(
    data: Union[PdDataFrame, PlDataFrame],
    detection_length: int = 7,
    margin: float = 6.9,
    buyside_margin: float = 2.3,
    sellside_margin: float = 2.3,
    detect_voids: bool = False,
    atr_period: int = 10,
    atr_void_period: int = 200,
    min_cluster_count: int = 3,
    max_swings: int = 50,
    high_column: str = "High",
    low_column: str = "Low",
    open_column: str = "Open",
    close_column: str = "Close",
    # Output column names
    buyside_level_column: str = "buyside_liq_level",
    sellside_level_column: str = "sellside_liq_level",
    buyside_top_column: str = "buyside_liq_top",
    buyside_bottom_column: str = "buyside_liq_bottom",
    sellside_top_column: str = "sellside_liq_top",
    sellside_bottom_column: str = "sellside_liq_bottom",
    buyside_broken_column: str = "buyside_liq_broken",
    sellside_broken_column: str = "sellside_liq_broken",
    void_bullish_column: str = "liq_void_bullish",
    void_bearish_column: str = "liq_void_bearish",
    void_top_column: str = "liq_void_top",
    void_bottom_column: str = "liq_void_bottom",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect buyside and sellside liquidity levels, breaches, and voids.

    A *buyside liquidity level* forms when ≥ ``min_cluster_count``
    swing highs are clustered within an ATR-scaled margin. Sell-side
    is the mirror for swing lows.

    Args:
        data: pandas or polars DataFrame with OHLC data.
        detection_length: Lookback period for pivot detection
            (default: 7).
        margin: Divisor for the ATR margin band (default: 6.9).
            Effective margin = ``10 / margin * ATR(atr_period)``.
        buyside_margin: Multiplier for the breach zone around
            buyside levels (default: 2.3).
        sellside_margin: Multiplier for the breach zone around
            sellside levels (default: 2.3).
        detect_voids: Whether to detect liquidity voids
            (default: False).
        atr_period: ATR period for margin calculation
            (default: 10).
        atr_void_period: ATR period for void detection
            (default: 200).
        min_cluster_count: Minimum number of clustered swing
            points to form a liquidity level (default: 3).
        max_swings: Maximum number of swings retained in the
            zigzag buffer (default: 50).
        high_column: Column name for highs.
        low_column: Column name for lows.
        open_column: Column name for opens.
        close_column: Column name for closes.
        buyside_level_column: Output – buyside liquidity level
            price (NaN when none detected on that bar).
        sellside_level_column: Output – sellside liquidity level
            price.
        buyside_top_column: Output – top of the buyside margin
            zone.
        buyside_bottom_column: Output – bottom of the buyside
            margin zone.
        sellside_top_column: Output – top of the sellside margin
            zone.
        sellside_bottom_column: Output – bottom of the sellside
            margin zone.
        buyside_broken_column: Output – 1 on the bar where a
            buyside level is breached.
        sellside_broken_column: Output – 1 on the bar where a
            sellside level is breached.
        void_bullish_column: Output – 1 on bars with a bullish
            liquidity void.
        void_bearish_column: Output – 1 on bars with a bearish
            liquidity void.
        void_top_column: Output – top of the void zone.
        void_bottom_column: Output – bottom of the void zone.

    Returns:
        DataFrame with all output columns appended.

    Example:
        >>> import pandas as pd
        >>> from pyindicators import buyside_sellside_liquidity
        >>> df = pd.DataFrame({
        ...     'Open': [...], 'High': [...],
        ...     'Low': [...], 'Close': [...]
        ... })
        >>> result = buyside_sellside_liquidity(df)
    """
    if isinstance(data, PdDataFrame):
        return _bsl_pandas(
            data,
            detection_length=detection_length,
            margin=margin,
            buyside_margin=buyside_margin,
            sellside_margin=sellside_margin,
            detect_voids=detect_voids,
            atr_period=atr_period,
            atr_void_period=atr_void_period,
            min_cluster_count=min_cluster_count,
            max_swings=max_swings,
            high_column=high_column,
            low_column=low_column,
            open_column=open_column,
            close_column=close_column,
            buyside_level_column=buyside_level_column,
            sellside_level_column=sellside_level_column,
            buyside_top_column=buyside_top_column,
            buyside_bottom_column=buyside_bottom_column,
            sellside_top_column=sellside_top_column,
            sellside_bottom_column=sellside_bottom_column,
            buyside_broken_column=buyside_broken_column,
            sellside_broken_column=sellside_broken_column,
            void_bullish_column=void_bullish_column,
            void_bearish_column=void_bearish_column,
            void_top_column=void_top_column,
            void_bottom_column=void_bottom_column,
        )
    elif isinstance(data, PlDataFrame):
        pd_data = data.to_pandas()
        result = _bsl_pandas(
            pd_data,
            detection_length=detection_length,
            margin=margin,
            buyside_margin=buyside_margin,
            sellside_margin=sellside_margin,
            detect_voids=detect_voids,
            atr_period=atr_period,
            atr_void_period=atr_void_period,
            min_cluster_count=min_cluster_count,
            max_swings=max_swings,
            high_column=high_column,
            low_column=low_column,
            open_column=open_column,
            close_column=close_column,
            buyside_level_column=buyside_level_column,
            sellside_level_column=sellside_level_column,
            buyside_top_column=buyside_top_column,
            buyside_bottom_column=buyside_bottom_column,
            sellside_top_column=sellside_top_column,
            sellside_bottom_column=sellside_bottom_column,
            buyside_broken_column=buyside_broken_column,
            sellside_broken_column=sellside_broken_column,
            void_bullish_column=void_bullish_column,
            void_bearish_column=void_bearish_column,
            void_top_column=void_top_column,
            void_bottom_column=void_bottom_column,
        )
        import polars as pl

        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def buyside_sellside_liquidity_signal(
    data: Union[PdDataFrame, PlDataFrame],
    buyside_broken_column: str = "buyside_liq_broken",
    sellside_broken_column: str = "sellside_liq_broken",
    signal_column: str = "bsl_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a combined signal from buyside / sellside breach events.

    Args:
        data: DataFrame containing breach columns (output of
            :func:`buyside_sellside_liquidity`).
        buyside_broken_column: Column with buyside breach flags.
        sellside_broken_column: Column with sellside breach flags.
        signal_column: Output column name.

    Returns:
        DataFrame with ``{signal_column}``:

        - ``1`` – sellside liquidity breached (bearish grab → may
          reverse up)
        - ``-1`` – buyside liquidity breached (bullish grab → may
          reverse down)
        - ``0`` – no breach
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        data[signal_column] = np.where(
            data[sellside_broken_column] == 1,
            1,
            np.where(data[buyside_broken_column] == 1, -1, 0),
        )
        return data
    elif isinstance(data, PlDataFrame):
        import polars as pl

        return data.with_columns(
            pl.when(pl.col(sellside_broken_column) == 1)
            .then(1)
            .when(pl.col(buyside_broken_column) == 1)
            .then(-1)
            .otherwise(0)
            .alias(signal_column)
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def get_buyside_sellside_liquidity_stats(
    data: Union[PdDataFrame, PlDataFrame],
    buyside_level_column: str = "buyside_liq_level",
    sellside_level_column: str = "sellside_liq_level",
    buyside_broken_column: str = "buyside_liq_broken",
    sellside_broken_column: str = "sellside_liq_broken",
) -> Dict:
    """
    Return summary statistics for detected liquidity levels.

    Args:
        data: DataFrame with liquidity columns.

    Returns:
        Dictionary with counts and ratios.
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    buy_levels = int(data[buyside_level_column].notna().sum())
    sell_levels = int(data[sellside_level_column].notna().sum())
    buy_breaks = int(data[buyside_broken_column].sum())
    sell_breaks = int(data[sellside_broken_column].sum())
    total_levels = buy_levels + sell_levels
    total_breaks = buy_breaks + sell_breaks

    return {
        "buyside_levels_detected": buy_levels,
        "sellside_levels_detected": sell_levels,
        "total_levels": total_levels,
        "buyside_breaches": buy_breaks,
        "sellside_breaches": sell_breaks,
        "total_breaches": total_breaks,
        "breach_ratio": (
            round(total_breaks / total_levels, 4)
            if total_levels > 0
            else 0.0
        ),
    }


# ── helpers ───────────────────────────────────────────────────────────


def _compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """Simple ATR using a rolling mean of True Range."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr_out = np.full(n, np.nan)
    cumsum = 0.0
    for i in range(n):
        cumsum += tr[i]
        if i >= period - 1:
            if i > period - 1:
                cumsum -= tr[i - period]
            atr_out[i] = cumsum / period
    return atr_out


def _detect_pivot_highs_1(
    high: np.ndarray, length: int
) -> np.ndarray:
    """
    Pivot high with ``ta.pivothigh(length, 1)`` semantics:
    ``high[i-1]`` is the highest in ``[i-1-length … i-1]`` *and*
    ``high[i-1] >= high[i]``.  The result is written at index *i*.
    """
    n = len(high)
    pivots = np.full(n, np.nan)
    for i in range(length + 1, n):
        candidate = high[i - 1]
        is_pivot = True
        # Look back `length` bars from the candidate
        for k in range(i - 1 - length, i - 1):
            if high[k] > candidate:
                is_pivot = False
                break
        # Look forward 1 bar (i.e. current bar)
        if is_pivot and high[i] > candidate:
            is_pivot = False
        if is_pivot:
            pivots[i] = candidate
    return pivots


def _detect_pivot_lows_1(
    low: np.ndarray, length: int
) -> np.ndarray:
    """Mirror of ``_detect_pivot_highs_1`` for lows."""
    n = len(low)
    pivots = np.full(n, np.nan)
    for i in range(length + 1, n):
        candidate = low[i - 1]
        is_pivot = True
        for k in range(i - 1 - length, i - 1):
            if low[k] < candidate:
                is_pivot = False
                break
        if is_pivot and low[i] < candidate:
            is_pivot = False
        if is_pivot:
            pivots[i] = candidate
    return pivots


# ── core pandas implementation ────────────────────────────────────────


def _bsl_pandas(
    data: PdDataFrame,
    detection_length: int,
    margin: float,
    buyside_margin: float,
    sellside_margin: float,
    detect_voids: bool,
    atr_period: int,
    atr_void_period: int,
    min_cluster_count: int,
    max_swings: int,
    high_column: str,
    low_column: str,
    open_column: str,
    close_column: str,
    buyside_level_column: str,
    sellside_level_column: str,
    buyside_top_column: str,
    buyside_bottom_column: str,
    sellside_top_column: str,
    sellside_bottom_column: str,
    buyside_broken_column: str,
    sellside_broken_column: str,
    void_bullish_column: str,
    void_bearish_column: str,
    void_top_column: str,
    void_bottom_column: str,
) -> PdDataFrame:
    data = data.copy()
    high = data[high_column].values.astype(float)
    low = data[low_column].values.astype(float)
    close = data[close_column].values.astype(float)
    n = len(data)

    # ATR values
    atr_vals = _compute_atr(high, low, close, atr_period)
    atr_void = (
        _compute_atr(high, low, close, atr_void_period)
        if detect_voids
        else None
    )

    liq_mar = 10.0 / margin  # effective margin multiplier

    # Pivot detection
    pivot_highs = _detect_pivot_highs_1(high, detection_length)
    pivot_lows = _detect_pivot_lows_1(low, detection_length)

    # ── Zigzag buffer (direction, bar_index, price) ──────────────
    zz_dir: List[int] = [0] * max_swings
    zz_x: List[int] = [0] * max_swings
    zz_y: List[float] = [np.nan] * max_swings

    def zz_unshift(d: int, x: int, y: float):
        zz_dir.insert(0, d)
        zz_dir.pop()
        zz_x.insert(0, x)
        zz_x.pop()
        zz_y.insert(0, y)
        zz_y.pop()

    # ── Active liquidity levels ──────────────────────────────────
    # Each: {level, top, bottom, start_bar, broken}
    active_buyside: List[Dict] = []
    active_sellside: List[Dict] = []

    # Output arrays
    buyside_level = np.full(n, np.nan)
    sellside_level = np.full(n, np.nan)
    buyside_top = np.full(n, np.nan)
    buyside_bottom = np.full(n, np.nan)
    sellside_top = np.full(n, np.nan)
    sellside_bottom = np.full(n, np.nan)
    buyside_broken = np.zeros(n, dtype=int)
    sellside_broken = np.zeros(n, dtype=int)
    void_bull = np.zeros(n, dtype=int)
    void_bear = np.zeros(n, dtype=int)
    void_top_arr = np.full(n, np.nan)
    void_bot_arr = np.full(n, np.nan)

    for i in range(1, n):
        cur_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0
        band = cur_atr * liq_mar  # margin band half-width

        # ── Handle Pivot HIGH at bar i ───────────────────────────
        if not np.isnan(pivot_highs[i]):
            ph = pivot_highs[i]
            x2 = i - 1
            y2 = high[i - 1] if i >= 1 else ph

            # Update zigzag
            if zz_dir[0] < 1:
                zz_unshift(1, x2, y2)
            elif zz_dir[0] == 1 and ph > zz_y[0]:
                zz_x[0] = x2
                zz_y[0] = y2

            # Scan for cluster of swing highs within margin
            count = 0
            st_bar = 0
            st_price = 0.0
            min_p = 0.0
            max_p = 1e18

            for k in range(max_swings):
                if zz_dir[k] == 1:
                    if zz_y[k] > ph + band:
                        break
                    if (
                        not np.isnan(zz_y[k])
                        and zz_y[k] > ph - band
                        and zz_y[k] < ph + band
                    ):
                        count += 1
                        st_bar = zz_x[k]
                        st_price = zz_y[k]
                        if zz_y[k] > min_p:
                            min_p = zz_y[k]
                        if zz_y[k] < max_p:
                            max_p = zz_y[k]

            if count >= min_cluster_count:
                avg_p = (min_p + max_p) / 2.0
                level_top = avg_p + band
                level_bot = avg_p - band

                # Check if this updates an existing level
                # (same start bar) or creates a new one
                updated = False
                for lv in active_buyside:
                    if lv["start_bar"] == st_bar and not lv["broken"]:
                        lv["level"] = st_price
                        lv["top"] = level_top
                        lv["bottom"] = level_bot
                        updated = True
                        break

                if not updated:
                    active_buyside.append(
                        {
                            "level": st_price,
                            "top": level_top,
                            "bottom": level_bot,
                            "start_bar": st_bar,
                            "broken": False,
                        }
                    )

                buyside_level[i] = st_price
                buyside_top[i] = level_top
                buyside_bottom[i] = level_bot

        # ── Handle Pivot LOW at bar i ────────────────────────────
        if not np.isnan(pivot_lows[i]):
            pl_val = pivot_lows[i]
            x2 = i - 1
            y2 = low[i - 1] if i >= 1 else pl_val

            if zz_dir[0] > -1:
                zz_unshift(-1, x2, y2)
            elif zz_dir[0] == -1 and pl_val < zz_y[0]:
                zz_x[0] = x2
                zz_y[0] = y2

            count = 0
            st_bar = 0
            st_price = 0.0
            min_p = 0.0
            max_p = 1e18

            for k in range(max_swings):
                if zz_dir[k] == -1:
                    if not np.isnan(zz_y[k]) and zz_y[k] < pl_val - band:
                        break
                    if (
                        not np.isnan(zz_y[k])
                        and zz_y[k] > pl_val - band
                        and zz_y[k] < pl_val + band
                    ):
                        count += 1
                        st_bar = zz_x[k]
                        st_price = zz_y[k]
                        if zz_y[k] > min_p:
                            min_p = zz_y[k]
                        if zz_y[k] < max_p:
                            max_p = zz_y[k]

            if count >= min_cluster_count:
                avg_p = (min_p + max_p) / 2.0
                level_top = avg_p + band
                level_bot = avg_p - band

                updated = False
                for lv in active_sellside:
                    if lv["start_bar"] == st_bar and not lv["broken"]:
                        lv["level"] = st_price
                        lv["top"] = level_top
                        lv["bottom"] = level_bot
                        updated = True
                        break

                if not updated:
                    active_sellside.append(
                        {
                            "level": st_price,
                            "top": level_top,
                            "bottom": level_bot,
                            "start_bar": st_bar,
                            "broken": False,
                        }
                    )

                sellside_level[i] = st_price
                sellside_top[i] = level_top
                sellside_bottom[i] = level_bot

        # ── Check breaches of active levels ──────────────────────
        for lv in active_buyside:
            if not lv["broken"] and high[i] > lv["top"]:
                lv["broken"] = True
                buyside_broken[i] = 1

        for lv in active_sellside:
            if not lv["broken"] and low[i] < lv["bottom"]:
                lv["broken"] = True
                sellside_broken[i] = 1

        # ── Liquidity voids ──────────────────────────────────────
        if detect_voids and atr_void is not None and i >= 2:
            cur_atr_void = (
                atr_void[i] if not np.isnan(atr_void[i]) else 0.0
            )
            # Bullish void: large gap up
            if (
                low[i] - high[i - 2] > cur_atr_void
                and low[i] > high[i - 2]
                and close[i - 1] > high[i - 2]
            ):
                void_bull[i] = 1
                void_top_arr[i] = low[i]
                void_bot_arr[i] = high[i - 2]

            # Bearish void: large gap down
            if (
                low[i - 2] - high[i] > cur_atr_void
                and high[i] < low[i - 2]
                and close[i - 1] < low[i - 2]
            ):
                void_bear[i] = 1
                void_top_arr[i] = low[i - 2]
                void_bot_arr[i] = high[i]

    # Write outputs
    data[buyside_level_column] = buyside_level
    data[sellside_level_column] = sellside_level
    data[buyside_top_column] = buyside_top
    data[buyside_bottom_column] = buyside_bottom
    data[sellside_top_column] = sellside_top
    data[sellside_bottom_column] = sellside_bottom
    data[buyside_broken_column] = buyside_broken
    data[sellside_broken_column] = sellside_broken

    if detect_voids:
        data[void_bullish_column] = void_bull
        data[void_bearish_column] = void_bear
        data[void_top_column] = void_top_arr
        data[void_bottom_column] = void_bot_arr

    return data
