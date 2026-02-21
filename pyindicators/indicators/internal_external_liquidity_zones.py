"""
Internal vs External Liquidity Zones Indicator

Identifies Internal and External Liquidity Zones based on multi-
timeframe pivot analysis, sweep detection, and market structure
(BOS / CHoCH).

**External Zones** are derived from longer-period pivots
(``external_pivot_length``) and represent major liquidity pools.
**Internal Zones** come from shorter-period pivots
(``internal_pivot_length``) that reside within the external range.

Core concepts:

- **External Pivot** - a swing high/low confirmed over a longer
  lookback window.  These define the outer liquidity range.
- **Internal Pivot** - a swing high/low confirmed over a shorter
  lookback window.  Two modes are supported:
      * ``"every_pivot"`` - every internal pivot creates a zone.
      * ``"equal_hl"`` - only consecutive pivots within an
        ATR-based tolerance create a zone (equal-high/low logic).
- **Zone States** - each zone transitions through:
      * 0 = active
      * 1 = swept (price touched but did not close through)
      * 2 = broken (price closed through the zone)
- **Sweep Mode** - determines how a zone is swept / broken:
      * ``"wick"``  - any wick touch marks a sweep.
      * ``"close"`` - only a close through the zone counts as a
        sweep.
      * ``"wick_close"`` - wick touches sweep; closes break.
- **Structure (BOS / CHoCH)** - for both external and internal
  pivots, Break of Structure and Change of Character events are
  detected based on trend momentum analysis.
"""
from typing import Union, Dict, List
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


# ------------------------------------------------------------------ #
#  Public API                                                        #
# ------------------------------------------------------------------ #

def internal_external_liquidity_zones(
    data: Union[PdDataFrame, PlDataFrame],
    internal_pivot_length: int = 3,
    external_pivot_length: int = 10,
    internal_mode: str = "equal_hl",
    eq_tolerance_atr: float = 0.25,
    require_internal_inside: bool = True,
    reset_internal_on_external: bool = True,
    atr_length: int = 14,
    zone_size_atr: float = 0.40,
    sweep_mode: str = "wick",
    structure_lookback_external: int = 36,
    structure_lookback_internal: int = 2,
    use_closes_for_structure: bool = True,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    ext_high_column: str = "ielz_ext_high",
    ext_low_column: str = "ielz_ext_low",
    ext_high_price_column: str = "ielz_ext_high_price",
    ext_low_price_column: str = "ielz_ext_low_price",
    int_high_column: str = "ielz_int_high",
    int_low_column: str = "ielz_int_low",
    int_high_price_column: str = "ielz_int_high_price",
    int_low_price_column: str = "ielz_int_low_price",
    range_high_column: str = "ielz_range_high",
    range_low_column: str = "ielz_range_low",
    ext_sweep_bull_column: str = "ielz_ext_sweep_bull",
    ext_sweep_bear_column: str = "ielz_ext_sweep_bear",
    int_sweep_bull_column: str = "ielz_int_sweep_bull",
    int_sweep_bear_column: str = "ielz_int_sweep_bear",
    ext_structure_column: str = "ielz_ext_structure",
    int_structure_column: str = "ielz_int_structure",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect Internal and External Liquidity Zones on OHLC data.

    Args:
        data: pandas or polars DataFrame with OHLC columns.
        internal_pivot_length: Lookback/look-ahead for internal
            pivots (default: 3).
        external_pivot_length: Lookback/look-ahead for external
            pivots (default: 10).
        internal_mode: ``"every_pivot"`` or ``"equal_hl"``
            (default: ``"equal_hl"``).  In ``"equal_hl"`` mode
            an internal zone is only created when two consecutive
            internal pivots are within ``eq_tolerance_atr × ATR``.
        eq_tolerance_atr: Tolerance for the equal-high/low test,
            expressed as a fraction of ATR (default: 0.25).
        require_internal_inside: When ``True``, internal pivots
            must fall inside the current external range
            (default: ``True``).
        reset_internal_on_external: When ``True``, the running
            internal pivot tracker resets whenever a new external
            pivot is detected (default: ``True``).
        atr_length: Period for ATR calculation (default: 14).
        zone_size_atr: Half-height of each zone expressed as a
            fraction of ATR (default: 0.40).
        sweep_mode: One of ``"wick"``, ``"close"`` or
            ``"wick_close"`` (default: ``"wick"``).
        structure_lookback_external: Pivot length for external
            BOS/CHoCH detection (default: 36).
        structure_lookback_internal: Pivot length for internal
            BOS/CHoCH detection (default: 2).
        use_closes_for_structure: When ``True``, structure
            analysis uses closes instead of wicks
            (default: ``True``).
        high_column: Input column for highs.
        low_column: Input column for lows.
        close_column: Input column for closes.
        ext_high_column: Output - 1 on bars where an external
            high zone is created.
        ext_low_column: Output - 1 on bars where an external
            low zone is created.
        ext_high_price_column: Output - price level of the
            external high pivot (NaN otherwise).
        ext_low_price_column: Output - price level of the
            external low pivot (NaN otherwise).
        int_high_column: Output - 1 on bars where an internal
            high zone is created.
        int_low_column: Output - 1 on bars where an internal
            low zone is created.
        int_high_price_column: Output - price level of the
            internal high pivot (NaN otherwise).
        int_low_price_column: Output - price level of the
            internal low pivot (NaN otherwise).
        range_high_column: Output - running external range high.
        range_low_column: Output - running external range low.
        ext_sweep_bull_column: Output - 1 on bars with a bullish
            external sweep.
        ext_sweep_bear_column: Output - 1 on bars with a bearish
            external sweep.
        int_sweep_bull_column: Output - 1 on bars with a bullish
            internal sweep.
        int_sweep_bear_column: Output - 1 on bars with a bearish
            internal sweep.
        ext_structure_column: Output - structure label at
            external level (``"eBOS"``, ``"eCHoCH"``, or ``""``).
        int_structure_column: Output - structure label at
            internal level (``"iBOS"``, ``"iCHoCH"``, or ``""``).

    Returns:
        DataFrame with all output columns added.

    Example:
        >>> import pandas as pd
        >>> from pyindicators import internal_external_liquidity_zones
        >>> df = pd.DataFrame({
        ...     'High': [...], 'Low': [...], 'Close': [...]
        ... })
        >>> result = internal_external_liquidity_zones(df)
    """
    valid_modes = ("every_pivot", "equal_hl")
    if internal_mode not in valid_modes:
        raise PyIndicatorException(
            f"internal_mode must be one of {valid_modes}, "
            f"got '{internal_mode}'"
        )

    valid_sweep = ("wick", "close", "wick_close")
    if sweep_mode not in valid_sweep:
        raise PyIndicatorException(
            f"sweep_mode must be one of {valid_sweep}, "
            f"got '{sweep_mode}'"
        )

    if isinstance(data, PdDataFrame):
        return _ielz_pandas(
            data,
            internal_pivot_length=internal_pivot_length,
            external_pivot_length=external_pivot_length,
            internal_mode=internal_mode,
            eq_tolerance_atr=eq_tolerance_atr,
            require_internal_inside=require_internal_inside,
            reset_internal_on_external=reset_internal_on_external,
            atr_length=atr_length,
            zone_size_atr=zone_size_atr,
            sweep_mode=sweep_mode,
            structure_lookback_external=structure_lookback_external,
            structure_lookback_internal=structure_lookback_internal,
            use_closes_for_structure=use_closes_for_structure,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            ext_high_column=ext_high_column,
            ext_low_column=ext_low_column,
            ext_high_price_column=ext_high_price_column,
            ext_low_price_column=ext_low_price_column,
            int_high_column=int_high_column,
            int_low_column=int_low_column,
            int_high_price_column=int_high_price_column,
            int_low_price_column=int_low_price_column,
            range_high_column=range_high_column,
            range_low_column=range_low_column,
            ext_sweep_bull_column=ext_sweep_bull_column,
            ext_sweep_bear_column=ext_sweep_bear_column,
            int_sweep_bull_column=int_sweep_bull_column,
            int_sweep_bear_column=int_sweep_bear_column,
            ext_structure_column=ext_structure_column,
            int_structure_column=int_structure_column,
        )
    elif isinstance(data, PlDataFrame):
        import polars as pl

        pd_data = data.to_pandas()
        result = _ielz_pandas(
            pd_data,
            internal_pivot_length=internal_pivot_length,
            external_pivot_length=external_pivot_length,
            internal_mode=internal_mode,
            eq_tolerance_atr=eq_tolerance_atr,
            require_internal_inside=require_internal_inside,
            reset_internal_on_external=reset_internal_on_external,
            atr_length=atr_length,
            zone_size_atr=zone_size_atr,
            sweep_mode=sweep_mode,
            structure_lookback_external=structure_lookback_external,
            structure_lookback_internal=structure_lookback_internal,
            use_closes_for_structure=use_closes_for_structure,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            ext_high_column=ext_high_column,
            ext_low_column=ext_low_column,
            ext_high_price_column=ext_high_price_column,
            ext_low_price_column=ext_low_price_column,
            int_high_column=int_high_column,
            int_low_column=int_low_column,
            int_high_price_column=int_high_price_column,
            int_low_price_column=int_low_price_column,
            range_high_column=range_high_column,
            range_low_column=range_low_column,
            ext_sweep_bull_column=ext_sweep_bull_column,
            ext_sweep_bear_column=ext_sweep_bear_column,
            int_sweep_bull_column=int_sweep_bull_column,
            int_sweep_bear_column=int_sweep_bear_column,
            ext_structure_column=ext_structure_column,
            int_structure_column=int_structure_column,
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def internal_external_liquidity_zones_signal(
    data: Union[PdDataFrame, PlDataFrame],
    int_sweep_bull_column: str = "ielz_int_sweep_bull",
    int_sweep_bear_column: str = "ielz_int_sweep_bear",
    ext_sweep_bull_column: str = "ielz_ext_sweep_bull",
    ext_sweep_bear_column: str = "ielz_ext_sweep_bear",
    signal_column: str = "ielz_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a combined signal from Internal/External Liquidity
    Zone sweep results.

    Priority: external sweeps > internal sweeps.

    Args:
        data: DataFrame with liquidity zone sweep columns
            (output of
            :func:`internal_external_liquidity_zones`).
        int_sweep_bull_column: Column with internal bullish sweep
            flags.
        int_sweep_bear_column: Column with internal bearish sweep
            flags.
        ext_sweep_bull_column: Column with external bullish sweep
            flags.
        ext_sweep_bear_column: Column with external bearish sweep
            flags.
        signal_column: Output column name.

    Returns:
        DataFrame with ``{signal_column}`` added:

        - ``1`` - bullish sweep detected
        - ``-1`` - bearish sweep detected
        - ``0`` - no sweep
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        signal = np.where(
            data[ext_sweep_bull_column] == 1,
            1,
            np.where(
                data[ext_sweep_bear_column] == 1,
                -1,
                np.where(
                    data[int_sweep_bull_column] == 1,
                    1,
                    np.where(
                        data[int_sweep_bear_column] == 1,
                        -1,
                        0,
                    ),
                ),
            ),
        )
        data[signal_column] = signal
        return data
    elif isinstance(data, PlDataFrame):
        import polars as pl

        return data.with_columns(
            pl.when(pl.col(ext_sweep_bull_column) == 1)
            .then(1)
            .when(pl.col(ext_sweep_bear_column) == 1)
            .then(-1)
            .when(pl.col(int_sweep_bull_column) == 1)
            .then(1)
            .when(pl.col(int_sweep_bear_column) == 1)
            .then(-1)
            .otherwise(0)
            .alias(signal_column)
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def get_internal_external_liquidity_zones_stats(
    data: Union[PdDataFrame, PlDataFrame],
    ext_high_column: str = "ielz_ext_high",
    ext_low_column: str = "ielz_ext_low",
    int_high_column: str = "ielz_int_high",
    int_low_column: str = "ielz_int_low",
    ext_sweep_bull_column: str = "ielz_ext_sweep_bull",
    ext_sweep_bear_column: str = "ielz_ext_sweep_bear",
    int_sweep_bull_column: str = "ielz_int_sweep_bull",
    int_sweep_bear_column: str = "ielz_int_sweep_bear",
    ext_structure_column: str = "ielz_ext_structure",
    int_structure_column: str = "ielz_int_structure",
) -> Dict:
    """
    Return summary statistics for Internal/External Liquidity Zones.

    Args:
        data: DataFrame with IELZ columns (output of
            :func:`internal_external_liquidity_zones`).

    Returns:
        Dictionary with keys:

        - ``total_ext_highs`` - number of external high zones
        - ``total_ext_lows`` - number of external low zones
        - ``total_int_highs`` - number of internal high zones
        - ``total_int_lows`` - number of internal low zones
        - ``total_ext_sweeps`` - total external sweep events
        - ``total_int_sweeps`` - total internal sweep events
        - ``ext_bos_count`` - external BOS events
        - ``ext_choch_count`` - external CHoCH events
        - ``int_bos_count`` - internal BOS events
        - ``int_choch_count`` - internal CHoCH events
        - ``bullish_sweep_ratio`` - ratio of bullish sweeps to
          total sweeps (0-1)
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    ext_h = int(data[ext_high_column].sum())
    ext_l = int(data[ext_low_column].sum())
    int_h = int(data[int_high_column].sum())
    int_l = int(data[int_low_column].sum())

    ext_sb = int(data[ext_sweep_bull_column].sum())
    ext_se = int(data[ext_sweep_bear_column].sum())
    int_sb = int(data[int_sweep_bull_column].sum())
    int_se = int(data[int_sweep_bear_column].sum())

    total_ext_sweeps = ext_sb + ext_se
    total_int_sweeps = int_sb + int_se
    total_bull = ext_sb + int_sb
    total_sweeps = total_ext_sweeps + total_int_sweeps

    ext_bos = int((data[ext_structure_column] == "eBOS").sum())
    ext_choch = int((data[ext_structure_column] == "eCHoCH").sum())
    int_bos = int((data[int_structure_column] == "iBOS").sum())
    int_choch = int((data[int_structure_column] == "iCHoCH").sum())

    return {
        "total_ext_highs": ext_h,
        "total_ext_lows": ext_l,
        "total_int_highs": int_h,
        "total_int_lows": int_l,
        "total_ext_sweeps": total_ext_sweeps,
        "total_int_sweeps": total_int_sweeps,
        "ext_bos_count": ext_bos,
        "ext_choch_count": ext_choch,
        "int_bos_count": int_bos,
        "int_choch_count": int_choch,
        "bullish_sweep_ratio": (
            round(total_bull / total_sweeps, 4)
            if total_sweeps > 0
            else 0.0
        ),
    }


# ------------------------------------------------------------------ #
#  Internal helpers                                                  #
# ------------------------------------------------------------------ #

def _detect_pivot_highs(arr: np.ndarray, length: int) -> np.ndarray:
    """Return array with pivot-high value at the pivot bar, NaN elsewhere."""
    n = len(arr)
    pivots = np.full(n, np.nan)
    for i in range(length, n - length):
        window = arr[i - length: i + length + 1]
        if arr[i] == np.max(window):
            pivots[i] = arr[i]
    return pivots


def _detect_pivot_lows(arr: np.ndarray, length: int) -> np.ndarray:
    """Return array with pivot-low value at the pivot bar, NaN elsewhere."""
    n = len(arr)
    pivots = np.full(n, np.nan)
    for i in range(length, n - length):
        window = arr[i - length: i + length + 1]
        if arr[i] == np.min(window):
            pivots[i] = arr[i]
    return pivots


def _compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Simple ATR: rolling mean of true range."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr_arr = np.full(n, np.nan)
    for i in range(period - 1, n):
        atr_arr[i] = np.mean(tr[i - period + 1: i + 1])
    return atr_arr


def _calc_trend_momentum(close: np.ndarray, i: int, lookback: int) -> float:
    """Momentum score: price change / avg volatility."""
    if i < lookback:
        return 0.0
    price_change = (close[i] - close[i - lookback]) / close[i - lookback] * 100
    # Simple approximation for avg volatility
    diffs = np.abs(np.diff(close[max(0, i - lookback): i + 1]))
    avg_vol = np.mean(diffs) / close[i] * 100 if len(diffs) > 0 else 0.0001
    return price_change / (avg_vol + 0.0001)


def _calc_trend_direction(close: np.ndarray, i: int, lookback: int) -> int:
    """Return 1 (bullish), -1 (bearish), or 0 (neutral)."""
    momentum = _calc_trend_momentum(close, i, lookback)
    if momentum > 0.5:
        return 1
    elif momentum < -0.5:
        return -1
    return 0


def _zone_state_update(
    is_high: bool,
    zone_top: float,
    zone_bottom: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    prev_state: int,
    sweep_mode: str,
) -> int:
    """
    Transition a zone's state.

    States: 0 = active, 1 = swept, 2 = broken.
    """
    if prev_state == 2:
        return 2

    new_state = prev_state

    if sweep_mode == "wick":
        hit = (bar_high >= zone_bottom) if is_high else (bar_low <= zone_top)
        if hit and prev_state == 0:
            new_state = 1

    elif sweep_mode == "close":
        hit = (
            (bar_close >= zone_bottom) if is_high
            else (bar_close <= zone_top)
        )
        if hit and prev_state == 0:
            new_state = 1

    else:  # wick_close
        break_hit = (
            (bar_close > zone_top) if is_high
            else (bar_close < zone_bottom)
        )
        sweep_hit = (
            (bar_high >= zone_bottom and bar_close <= zone_top)
            if is_high
            else (bar_low <= zone_top and bar_close >= zone_bottom)
        )
        if break_hit:
            new_state = 2
        elif sweep_hit and prev_state == 0:
            new_state = 1

    return new_state


# ------------------------------------------------------------------ #
#  Core pandas implementation                                        #
# ------------------------------------------------------------------ #

def _ielz_pandas(
    data: PdDataFrame,
    *,
    internal_pivot_length: int,
    external_pivot_length: int,
    internal_mode: str,
    eq_tolerance_atr: float,
    require_internal_inside: bool,
    reset_internal_on_external: bool,
    atr_length: int,
    zone_size_atr: float,
    sweep_mode: str,
    structure_lookback_external: int,
    structure_lookback_internal: int,
    use_closes_for_structure: bool,
    high_column: str,
    low_column: str,
    close_column: str,
    ext_high_column: str,
    ext_low_column: str,
    ext_high_price_column: str,
    ext_low_price_column: str,
    int_high_column: str,
    int_low_column: str,
    int_high_price_column: str,
    int_low_price_column: str,
    range_high_column: str,
    range_low_column: str,
    ext_sweep_bull_column: str,
    ext_sweep_bear_column: str,
    int_sweep_bull_column: str,
    int_sweep_bear_column: str,
    ext_structure_column: str,
    int_structure_column: str,
) -> PdDataFrame:
    """Core pandas implementation."""
    data = data.copy()

    high = data[high_column].values.astype(float)
    low = data[low_column].values.astype(float)
    close = data[close_column].values.astype(float)
    n = len(data)

    atr_arr = _compute_atr(high, low, close, atr_length)

    # ------ Pivot detection ------ #
    ext_pivot_h = _detect_pivot_highs(high, external_pivot_length)
    ext_pivot_l = _detect_pivot_lows(low, external_pivot_length)

    int_pivot_h = _detect_pivot_highs(high, internal_pivot_length)
    int_pivot_l = _detect_pivot_lows(low, internal_pivot_length)

    # Structure pivots (using close or high/low)
    struct_src_h = (
        close if use_closes_for_structure else high
    )
    struct_src_l = (
        close if use_closes_for_structure else low
    )
    struct_ext_h = _detect_pivot_highs(
        struct_src_h, structure_lookback_external
    )
    struct_ext_l = _detect_pivot_lows(
        struct_src_l, structure_lookback_external
    )
    struct_int_h = _detect_pivot_highs(
        struct_src_h, structure_lookback_internal
    )
    struct_int_l = _detect_pivot_lows(
        struct_src_l, structure_lookback_internal
    )

    # ------ Output arrays ------ #
    out_ext_high = np.zeros(n, dtype=int)
    out_ext_low = np.zeros(n, dtype=int)
    out_ext_high_price = np.full(n, np.nan)
    out_ext_low_price = np.full(n, np.nan)

    out_int_high = np.zeros(n, dtype=int)
    out_int_low = np.zeros(n, dtype=int)
    out_int_high_price = np.full(n, np.nan)
    out_int_low_price = np.full(n, np.nan)

    out_range_high = np.full(n, np.nan)
    out_range_low = np.full(n, np.nan)

    out_ext_sweep_bull = np.zeros(n, dtype=int)
    out_ext_sweep_bear = np.zeros(n, dtype=int)
    out_int_sweep_bull = np.zeros(n, dtype=int)
    out_int_sweep_bear = np.zeros(n, dtype=int)

    out_ext_structure = np.empty(n, dtype=object)
    out_ext_structure[:] = ""
    out_int_structure = np.empty(n, dtype=object)
    out_int_structure[:] = ""

    # ------ Running state ------ #
    cur_ext_high = np.nan
    cur_ext_low = np.nan
    last_int_high = np.nan
    last_int_low = np.nan

    # Zone tracking: list of dicts
    # {price, top, bottom, is_high, state, is_external}
    zones: List[dict] = []
    MAX_ZONES = 500

    # Structure tracking: list of {price, bar, broken}
    ext_upper_pts: List[dict] = []
    ext_lower_pts: List[dict] = []
    int_upper_pts: List[dict] = []
    int_lower_pts: List[dict] = []
    MAX_STRUCT_PTS = 8

    ext_trend_dir = 0
    int_trend_dir = 0

    for i in range(n):
        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 0.0
        half_zone = cur_atr * zone_size_atr * 0.5

        # ============================================================ #
        #  External pivots                                              #
        # ============================================================ #
        if not np.isnan(ext_pivot_h[i]):
            cur_ext_high = ext_pivot_h[i]
            out_ext_high[i] = 1
            out_ext_high_price[i] = cur_ext_high
            top = cur_ext_high + half_zone
            bottom = cur_ext_high - half_zone
            zones.append({
                "price": cur_ext_high, "top": top, "bottom": bottom,
                "is_high": True, "state": 0, "is_external": True,
            })
            if reset_internal_on_external:
                last_int_high = np.nan
                last_int_low = np.nan

        if not np.isnan(ext_pivot_l[i]):
            cur_ext_low = ext_pivot_l[i]
            out_ext_low[i] = 1
            out_ext_low_price[i] = cur_ext_low
            top = cur_ext_low + half_zone
            bottom = cur_ext_low - half_zone
            zones.append({
                "price": cur_ext_low, "top": top, "bottom": bottom,
                "is_high": False, "state": 0, "is_external": True,
            })
            if reset_internal_on_external:
                last_int_high = np.nan
                last_int_low = np.nan

        # Update running range
        if not np.isnan(cur_ext_high) and not np.isnan(cur_ext_low):
            range_top = max(cur_ext_high, cur_ext_low)
            range_bottom = min(cur_ext_high, cur_ext_low)
        else:
            range_top = np.nan
            range_bottom = np.nan

        out_range_high[i] = (
            range_top if not np.isnan(cur_ext_high)
            else np.nan
        )
        out_range_low[i] = (
            range_bottom if not np.isnan(cur_ext_low)
            else np.nan
        )

        # ============================================================ #
        #  Internal pivots                                              #
        # ============================================================ #
        def _in_range(v: float) -> bool:
            if not require_internal_inside:
                return True
            if np.isnan(range_top) or np.isnan(range_bottom):
                return False
            return range_bottom <= v <= range_top

        # Internal high pivot
        if not np.isnan(int_pivot_h[i]):
            piv_val = int_pivot_h[i]
            if _in_range(piv_val):
                if internal_mode == "every_pivot":
                    out_int_high[i] = 1
                    out_int_high_price[i] = piv_val
                    top = piv_val + half_zone
                    bottom = piv_val - half_zone
                    zones.append({
                        "price": piv_val, "top": top, "bottom": bottom,
                        "is_high": True, "state": 0, "is_external": False,
                    })
                else:  # equal_hl
                    tol = cur_atr * eq_tolerance_atr
                    if (
                        not np.isnan(last_int_high)
                        and abs(piv_val - last_int_high) <= tol
                    ):
                        level = (piv_val + last_int_high) * 0.5
                        out_int_high[i] = 1
                        out_int_high_price[i] = level
                        top = level + half_zone
                        bottom = level - half_zone
                        zones.append({
                            "price": level, "top": top, "bottom": bottom,
                            "is_high": True, "state": 0,
                            "is_external": False,
                        })
                    last_int_high = piv_val

        # Internal low pivot
        if not np.isnan(int_pivot_l[i]):
            piv_val = int_pivot_l[i]
            if _in_range(piv_val):
                if internal_mode == "every_pivot":
                    out_int_low[i] = 1
                    out_int_low_price[i] = piv_val
                    top = piv_val + half_zone
                    bottom = piv_val - half_zone
                    zones.append({
                        "price": piv_val, "top": top, "bottom": bottom,
                        "is_high": False, "state": 0, "is_external": False,
                    })
                else:  # equal_hl
                    tol = cur_atr * eq_tolerance_atr
                    if (
                        not np.isnan(last_int_low)
                        and abs(piv_val - last_int_low) <= tol
                    ):
                        level = (piv_val + last_int_low) * 0.5
                        out_int_low[i] = 1
                        out_int_low_price[i] = level
                        top = level + half_zone
                        bottom = level - half_zone
                        zones.append({
                            "price": level, "top": top, "bottom": bottom,
                            "is_high": False, "state": 0,
                            "is_external": False,
                        })
                    last_int_low = piv_val

        # ============================================================ #
        #  Zone state updates & sweep detection                         #
        # ============================================================ #
        remove_ids: List[int] = []
        for j, zone in enumerate(zones):
            if zone["state"] == 2:
                remove_ids.append(j)
                continue

            prev_state = zone["state"]
            new_state = _zone_state_update(
                is_high=zone["is_high"],
                zone_top=zone["top"],
                zone_bottom=zone["bottom"],
                bar_high=high[i],
                bar_low=low[i],
                bar_close=close[i],
                prev_state=prev_state,
                sweep_mode=sweep_mode,
            )
            zone["state"] = new_state

            if new_state != prev_state and prev_state == 0:
                # A sweep or break just happened
                is_ext = zone["is_external"]
                is_high = zone["is_high"]
                if is_ext:
                    if is_high:
                        # Bearish sweep of external high zone
                        out_ext_sweep_bear[i] = 1
                    else:
                        # Bullish sweep of external low zone
                        out_ext_sweep_bull[i] = 1
                else:
                    if is_high:
                        # Bearish sweep of internal high zone
                        out_int_sweep_bear[i] = 1
                    else:
                        # Bullish sweep of internal low zone
                        out_int_sweep_bull[i] = 1

        # Remove broken zones to keep list manageable
        for idx in sorted(remove_ids, reverse=True):
            zones.pop(idx)
        # Cap zone count
        while len(zones) > MAX_ZONES:
            zones.pop(0)

        # ============================================================ #
        #  Structure analysis (BOS / CHoCH)                             #
        # ============================================================ #

        # --- External structure ---
        if not np.isnan(struct_ext_h[i]):
            ext_upper_pts.append({
                "price": struct_ext_h[i],
                "bar": i,
                "broken": False,
            })
            if len(ext_upper_pts) > MAX_STRUCT_PTS:
                ext_upper_pts.pop(0)

        if not np.isnan(struct_ext_l[i]):
            ext_lower_pts.append({
                "price": struct_ext_l[i],
                "bar": i,
                "broken": False,
            })
            if len(ext_lower_pts) > MAX_STRUCT_PTS:
                ext_lower_pts.pop(0)

        calc_ext_dir = _calc_trend_direction(
            close, i, structure_lookback_external
        )

        if ext_upper_pts and ext_lower_pts:
            latest_h = ext_upper_pts[-1]
            latest_l = ext_lower_pts[-1]

            test_val_h = close[i] if use_closes_for_structure else high[i]
            test_val_l = close[i] if use_closes_for_structure else low[i]

            bullish_break = (
                test_val_h > latest_h["price"] and not latest_h["broken"]
            )
            bearish_break = (
                test_val_l < latest_l["price"] and not latest_l["broken"]
            )

            if bullish_break:
                if ext_trend_dir == 1 and calc_ext_dir == 1:
                    out_ext_structure[i] = "eBOS"
                elif ext_trend_dir == -1 and calc_ext_dir == 1:
                    out_ext_structure[i] = "eCHoCH"
                latest_h["broken"] = True

            if bearish_break:
                if ext_trend_dir == -1 and calc_ext_dir == -1:
                    out_ext_structure[i] = "eBOS"
                elif ext_trend_dir == 1 and calc_ext_dir == -1:
                    out_ext_structure[i] = "eCHoCH"
                latest_l["broken"] = True

        ext_trend_dir = calc_ext_dir

        # --- Internal structure ---
        if not np.isnan(struct_int_h[i]):
            int_upper_pts.append({
                "price": struct_int_h[i],
                "bar": i,
                "broken": False,
            })
            if len(int_upper_pts) > MAX_STRUCT_PTS:
                int_upper_pts.pop(0)

        if not np.isnan(struct_int_l[i]):
            int_lower_pts.append({
                "price": struct_int_l[i],
                "bar": i,
                "broken": False,
            })
            if len(int_lower_pts) > MAX_STRUCT_PTS:
                int_lower_pts.pop(0)

        calc_int_dir = _calc_trend_direction(
            close, i, structure_lookback_internal
        )

        if int_upper_pts and int_lower_pts:
            latest_h = int_upper_pts[-1]
            latest_l = int_lower_pts[-1]

            test_val_h = close[i] if use_closes_for_structure else high[i]
            test_val_l = close[i] if use_closes_for_structure else low[i]

            bullish_break = (
                test_val_h > latest_h["price"] and not latest_h["broken"]
            )
            bearish_break = (
                test_val_l < latest_l["price"] and not latest_l["broken"]
            )

            if bullish_break:
                if int_trend_dir == 1 and calc_int_dir == 1:
                    out_int_structure[i] = "iBOS"
                elif int_trend_dir == -1 and calc_int_dir == 1:
                    out_int_structure[i] = "iCHoCH"
                latest_h["broken"] = True

            if bearish_break:
                if int_trend_dir == -1 and calc_int_dir == -1:
                    out_int_structure[i] = "iBOS"
                elif int_trend_dir == 1 and calc_int_dir == -1:
                    out_int_structure[i] = "iCHoCH"
                latest_l["broken"] = True

        int_trend_dir = calc_int_dir

    # ------ Write output columns ------ #
    data[ext_high_column] = out_ext_high
    data[ext_low_column] = out_ext_low
    data[ext_high_price_column] = out_ext_high_price
    data[ext_low_price_column] = out_ext_low_price

    data[int_high_column] = out_int_high
    data[int_low_column] = out_int_low
    data[int_high_price_column] = out_int_high_price
    data[int_low_price_column] = out_int_low_price

    data[range_high_column] = out_range_high
    data[range_low_column] = out_range_low

    data[ext_sweep_bull_column] = out_ext_sweep_bull
    data[ext_sweep_bear_column] = out_ext_sweep_bear
    data[int_sweep_bull_column] = out_int_sweep_bull
    data[int_sweep_bear_column] = out_int_sweep_bear

    data[ext_structure_column] = out_ext_structure
    data[int_structure_column] = out_int_structure

    return data
