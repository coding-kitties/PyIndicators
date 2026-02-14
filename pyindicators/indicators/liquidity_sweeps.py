"""
Liquidity Sweeps Indicator

Identifies liquidity sweep events where price wicks through key swing
highs/lows before reversing. These sweeps target stop-loss clusters
and liquidity pools sitting above/below swing points.

- Detects swing highs and lows using a configurable lookback period
- Identifies three types of sweeps:
    1. **Wick sweeps** – price wicks beyond a level but closes back
    2. **Outbreak & retest** – price breaks a level, then the
       opposite side sweeps it back
    3. **Combined** – both wick and outbreak/retest sweeps

Sweep types:
- **Bullish sweep** – price wicks below a swing low (grabs sell-side
  liquidity) then closes back above → bullish signal
- **Bearish sweep** – price wicks above a swing high (grabs buy-side
  liquidity) then closes back below → bearish signal
- **Bullish retest** – after a bearish break of a swing high, price
  retests and wicks below → bullish signal
- **Bearish retest** – after a bullish break of a swing low, price
  retests and wicks above → bearish signal
"""
from typing import Union, Dict, List
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


def liquidity_sweeps(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 5,
    mode: str = "wicks",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    bullish_sweep_column: str = "liq_sweep_bullish",
    bearish_sweep_column: str = "liq_sweep_bearish",
    sweep_high_column: str = "liq_sweep_high",
    sweep_low_column: str = "liq_sweep_low",
    sweep_type_column: str = "liq_sweep_type",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect Liquidity Sweeps on OHLC data.

    A liquidity sweep occurs when price momentarily pierces a swing
    high or swing low—grabbing resting liquidity—before reversing.

    Three detection modes mirror the Pine Script original:

    * ``"wicks"`` – only wick-through sweeps (high > swing high but
      close < swing high, or low < swing low but close > swing low).
    * ``"outbreak_retest"`` – only outbreak-and-retest sweeps (price
      closes beyond a level, then later the candle on the other side
      wicks through it while closing back).
    * ``"all"`` – both wick and outbreak/retest sweeps.

    Args:
        data: pandas or polars DataFrame with OHLC data.
        swing_length: Lookback/look-ahead period for pivot detection
            (default: 5). A swing high at bar *i* means
            ``high[i]`` is the highest high in ``[i-swing_length,
            i+swing_length]``.
        mode: Detection mode, one of ``"wicks"``,
            ``"outbreak_retest"``, or ``"all"`` (default: ``"wicks"``).
        high_column: Column name for highs (default: ``"High"``).
        low_column: Column name for lows (default: ``"Low"``).
        close_column: Column name for closes (default: ``"Close"``).
        bullish_sweep_column: Output column for bullish sweep signals
            (default: ``"liq_sweep_bullish"``).
        bearish_sweep_column: Output column for bearish sweep signals
            (default: ``"liq_sweep_bearish"``).
        sweep_high_column: Output column – price level of the swept
            swing high (default: ``"liq_sweep_high"``).
        sweep_low_column: Output column – price level of the swept
            swing low (default: ``"liq_sweep_low"``).
        sweep_type_column: Output column – type of sweep detected,
            ``"wick"`` or ``"retest"``
            (default: ``"liq_sweep_type"``).

    Returns:
        DataFrame with added columns:

        - ``{bullish_sweep_column}``: 1 on bars with a bullish
          liquidity sweep, 0 otherwise.
        - ``{bearish_sweep_column}``: 1 on bars with a bearish
          liquidity sweep, 0 otherwise.
        - ``{sweep_high_column}``: The swing-high price that was
          swept (NaN when no bearish sweep).
        - ``{sweep_low_column}``: The swing-low price that was
          swept (NaN when no bullish sweep).
        - ``{sweep_type_column}``: ``"wick"`` or ``"retest"``
          describing how the sweep occurred (empty string when no
          sweep).

    Example:
        >>> import pandas as pd
        >>> from pyindicators import liquidity_sweeps
        >>> df = pd.DataFrame({
        ...     'High': [...],
        ...     'Low': [...],
        ...     'Close': [...]
        ... })
        >>> result = liquidity_sweeps(df, swing_length=5, mode='wicks')
    """
    valid_modes = ("wicks", "outbreak_retest", "all")
    if mode not in valid_modes:
        raise PyIndicatorException(
            f"mode must be one of {valid_modes}, got '{mode}'"
        )

    if isinstance(data, PdDataFrame):
        return _liquidity_sweeps_pandas(
            data,
            swing_length=swing_length,
            mode=mode,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            bullish_sweep_column=bullish_sweep_column,
            bearish_sweep_column=bearish_sweep_column,
            sweep_high_column=sweep_high_column,
            sweep_low_column=sweep_low_column,
            sweep_type_column=sweep_type_column,
        )
    elif isinstance(data, PlDataFrame):
        # Convert to pandas, compute, convert back
        pd_data = data.to_pandas()
        result = _liquidity_sweeps_pandas(
            pd_data,
            swing_length=swing_length,
            mode=mode,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            bullish_sweep_column=bullish_sweep_column,
            bearish_sweep_column=bearish_sweep_column,
            sweep_high_column=sweep_high_column,
            sweep_low_column=sweep_low_column,
            sweep_type_column=sweep_type_column,
        )
        import polars as pl

        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def liquidity_sweep_signal(
    data: Union[PdDataFrame, PlDataFrame],
    bullish_sweep_column: str = "liq_sweep_bullish",
    bearish_sweep_column: str = "liq_sweep_bearish",
    signal_column: str = "liq_sweep_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a combined signal column from liquidity sweep results.

    Args:
        data: DataFrame that already contains liquidity sweep columns
            (output of :func:`liquidity_sweeps`).
        bullish_sweep_column: Column with bullish sweep flags.
        bearish_sweep_column: Column with bearish sweep flags.
        signal_column: Output column name (default:
            ``"liq_sweep_signal"``).

    Returns:
        DataFrame with ``{signal_column}`` added:

        - ``1`` – bullish sweep detected
        - ``-1`` – bearish sweep detected
        - ``0`` – no sweep
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        signal = np.where(
            data[bullish_sweep_column] == 1,
            1,
            np.where(data[bearish_sweep_column] == 1, -1, 0),
        )
        data[signal_column] = signal
        return data
    elif isinstance(data, PlDataFrame):
        import polars as pl

        return data.with_columns(
            pl.when(pl.col(bullish_sweep_column) == 1)
            .then(1)
            .when(pl.col(bearish_sweep_column) == 1)
            .then(-1)
            .otherwise(0)
            .alias(signal_column)
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def get_liquidity_sweep_stats(
    data: Union[PdDataFrame, PlDataFrame],
    bullish_sweep_column: str = "liq_sweep_bullish",
    bearish_sweep_column: str = "liq_sweep_bearish",
) -> Dict:
    """
    Return summary statistics for detected liquidity sweeps.

    Args:
        data: DataFrame that already contains liquidity sweep columns
            (output of :func:`liquidity_sweeps`).
        bullish_sweep_column: Column with bullish sweep flags.
        bearish_sweep_column: Column with bearish sweep flags.

    Returns:
        Dictionary with keys:

        - ``total_bullish_sweeps``
        - ``total_bearish_sweeps``
        - ``total_sweeps``
        - ``bullish_ratio`` (0–1)
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    bull = int(data[bullish_sweep_column].sum())
    bear = int(data[bearish_sweep_column].sum())
    total = bull + bear
    return {
        "total_bullish_sweeps": bull,
        "total_bearish_sweeps": bear,
        "total_sweeps": total,
        "bullish_ratio": round(bull / total, 4) if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _detect_pivot_highs(high: np.ndarray, length: int) -> np.ndarray:
    """
    Detect pivot highs.  A pivot high at index *i* means
    ``high[i]`` >= all highs in ``[i-length … i+length]``.

    The result is delayed by *length* bars (like Pine's
    ``ta.pivothigh(len, len)``).
    """
    n = len(high)
    pivots = np.full(n, np.nan)

    for i in range(length, n - length):
        window = high[i - length: i + length + 1]
        if high[i] == np.max(window):
            pivots[i] = high[i]

    return pivots


def _detect_pivot_lows(low: np.ndarray, length: int) -> np.ndarray:
    """Detect pivot lows (mirror of ``_detect_pivot_highs``)."""
    n = len(low)
    pivots = np.full(n, np.nan)

    for i in range(length, n - length):
        window = low[i - length: i + length + 1]
        if low[i] == np.min(window):
            pivots[i] = low[i]

    return pivots


def _liquidity_sweeps_pandas(
    data: PdDataFrame,
    swing_length: int,
    mode: str,
    high_column: str,
    low_column: str,
    close_column: str,
    bullish_sweep_column: str,
    bearish_sweep_column: str,
    sweep_high_column: str,
    sweep_low_column: str,
    sweep_type_column: str,
) -> PdDataFrame:
    """Core pandas implementation of the Liquidity Sweeps indicator."""
    data = data.copy()
    high = data[high_column].values.astype(float)
    low = data[low_column].values.astype(float)
    close = data[close_column].values.astype(float)
    n = len(data)

    # Flags per mode
    do_wicks = mode in ("wicks", "all")
    do_outbreak = mode in ("outbreak_retest", "all")

    # Detect pivots (confirmed *swing_length* bars after the actual high/low)
    pivot_highs = _detect_pivot_highs(high, swing_length)
    pivot_lows = _detect_pivot_lows(low, swing_length)

    # ------------------------------------------------------------------ #
    #  Pivot tracking structures (mirrors Pine Script UDT arrays)         #
    # ------------------------------------------------------------------ #
    # Each tracked pivot: {price, bar_idx, broken, mitigated, wick_used}
    tracked_highs: List[Dict] = []
    tracked_lows: List[Dict] = []

    # Output arrays
    bullish_sweep = np.zeros(n, dtype=int)
    bearish_sweep = np.zeros(n, dtype=int)
    sweep_high = np.full(n, np.nan)
    sweep_low = np.full(n, np.nan)
    sweep_type = np.empty(n, dtype=object)
    sweep_type[:] = ""

    MAX_AGE = 2000  # bars before a pivot is discarded

    for i in range(swing_length, n):
        # Register new pivots (discovered *swing_length* bars ago)
        ph_idx = i - swing_length
        if not np.isnan(pivot_highs[ph_idx]):
            tracked_highs.append(
                {
                    "price": pivot_highs[ph_idx],
                    "bar": ph_idx,
                    "broken": False,
                    "mitigated": False,
                    "taken": False,
                    "wick_used": False,
                }
            )

        pl_idx = i - swing_length
        if not np.isnan(pivot_lows[pl_idx]):
            tracked_lows.append(
                {
                    "price": pivot_lows[pl_idx],
                    "bar": pl_idx,
                    "broken": False,
                    "mitigated": False,
                    "taken": False,
                    "wick_used": False,
                }
            )

        # ── Process tracked swing HIGHS ──────────────────────────────
        remove_h: List[int] = []

        for j in range(len(tracked_highs) - 1, -1, -1):
            piv = tracked_highs[j]
            if piv["mitigated"] or piv["taken"]:
                remove_h.append(j)
                continue
            if i - piv["bar"] > MAX_AGE:
                remove_h.append(j)
                continue

            if not piv["broken"]:
                # Check if close breaks above the swing high
                if close[i] > piv["price"]:
                    if not do_wicks:
                        # In outbreak mode: mark as broken
                        piv["broken"] = True
                    else:
                        # In wicks-only mode: level is mitigated
                        piv["mitigated"] = True

                # Wick sweep: high pokes above, close stays below
                if do_wicks and not piv["wick_used"]:
                    if high[i] > piv["price"] and close[i] < piv["price"]:
                        # Bearish wick sweep (grabbed buy-side
                        # liquidity above swing high)
                        bearish_sweep[i] = 1
                        sweep_high[i] = piv["price"]
                        sweep_type[i] = "wick"
                        piv["wick_used"] = True
            else:
                # Pivot was broken (close went above); now look for
                # retest sweep from below
                if close[i] < piv["price"]:
                    piv["mitigated"] = True

                if do_outbreak and not piv["taken"]:
                    # Bullish retest sweep: low pokes below the level,
                    # close stays above
                    if low[i] < piv["price"] and close[i] > piv["price"]:
                        bullish_sweep[i] = 1
                        sweep_high[i] = piv["price"]
                        sweep_type[i] = "retest"
                        piv["taken"] = True

        for idx in sorted(remove_h, reverse=True):
            tracked_highs.pop(idx)

        # ── Process tracked swing LOWS ───────────────────────────────
        remove_l: List[int] = []

        for j in range(len(tracked_lows) - 1, -1, -1):
            piv = tracked_lows[j]
            if piv["mitigated"] or piv["taken"]:
                remove_l.append(j)
                continue
            if i - piv["bar"] > MAX_AGE:
                remove_l.append(j)
                continue

            if not piv["broken"]:
                # Check if close breaks below the swing low
                if close[i] < piv["price"]:
                    if not do_wicks:
                        piv["broken"] = True
                    else:
                        piv["mitigated"] = True

                # Wick sweep: low pokes below, close stays above
                if do_wicks and not piv["wick_used"]:
                    if low[i] < piv["price"] and close[i] > piv["price"]:
                        # Bullish wick sweep (grabbed sell-side
                        # liquidity below swing low)
                        bullish_sweep[i] = 1
                        sweep_low[i] = piv["price"]
                        sweep_type[i] = "wick"
                        piv["wick_used"] = True
            else:
                # Pivot was broken (close went below); now look for
                # retest sweep from above
                if close[i] > piv["price"]:
                    piv["mitigated"] = True

                if do_outbreak and not piv["taken"]:
                    # Bearish retest sweep: high pokes above the
                    # level, close stays below
                    if high[i] > piv["price"] and close[i] < piv["price"]:
                        bearish_sweep[i] = 1
                        sweep_low[i] = piv["price"]
                        sweep_type[i] = "retest"
                        piv["taken"] = True

        for idx in sorted(remove_l, reverse=True):
            tracked_lows.pop(idx)

    # Write results
    data[bullish_sweep_column] = bullish_sweep
    data[bearish_sweep_column] = bearish_sweep
    data[sweep_high_column] = sweep_high
    data[sweep_low_column] = sweep_low
    data[sweep_type_column] = sweep_type

    return data
