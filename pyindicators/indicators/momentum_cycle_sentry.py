"""
Momentum Cycle Sentry — multi-layer momentum oscillator with OB/OS detection.

Ported from "Momentum Cycle Sentry [LuxAlgo]" (PineScript v6).

The indicator computes a raw momentum (close − close[length]), applies
EMA smoothing at five different periods to create a layered "glow"
visualisation, and derives dynamic overbought / oversold corridors from
the standard deviation of the fast line.

Components
----------
1. **Layers** — five EMA-smoothed momentum curves (p1 … p5) and their
   mirrors (m1 … m5 = −p1 … −p5).
2. **OB / OS corridors** — inner and outer standard-deviation bands
   around the fast line.
3. **Trend** — bullish when fast line > 0, bearish when < 0.
4. **Retracement** — detected when the fast line is pulling back toward
   zero while the trend is still intact.
"""
from typing import Union, Dict
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import numpy as np

from pyindicators.exceptions import PyIndicatorException


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #
def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Vectorised EMA using the standard multiplier 2/(period+1)."""
    out = np.empty_like(values)
    out[:] = np.nan
    if period < 1 or len(values) < period:
        return out

    k = 2.0 / (period + 1)
    # Seed with the first non-NaN value
    first_valid = 0
    while first_valid < len(values) and np.isnan(values[first_valid]):
        first_valid += 1
    if first_valid >= len(values):
        return out

    out[first_valid] = values[first_valid]
    for i in range(first_valid + 1, len(values)):
        if np.isnan(values[i]):
            out[i] = out[i - 1]
        else:
            out[i] = values[i] * k + out[i - 1] * (1.0 - k)
    return out


def _rolling_stdev(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation (population) over *period* bars."""
    n = len(values)
    out = np.full(n, np.nan)
    if period < 1:
        return out
    for i in range(period - 1, n):
        window = values[i - period + 1: i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 2:
            out[i] = np.std(valid, ddof=0)
    return out


def _falling(values: np.ndarray, length: int) -> np.ndarray:
    """True when *values* has been strictly falling for *length* bars."""
    n = len(values)
    out = np.zeros(n, dtype=bool)
    for i in range(length, n):
        fell = True
        for j in range(1, length + 1):
            if np.isnan(values[i - j + 1]) or np.isnan(values[i - j]):
                fell = False
                break
            if values[i - j + 1] >= values[i - j]:
                fell = False
                break
        out[i] = fell
    return out


def _rising(values: np.ndarray, length: int) -> np.ndarray:
    """True when *values* has been strictly rising for *length* bars."""
    n = len(values)
    out = np.zeros(n, dtype=bool)
    for i in range(length, n):
        rose = True
        for j in range(1, length + 1):
            if np.isnan(values[i - j + 1]) or np.isnan(values[i - j]):
                rose = False
                break
            if values[i - j + 1] <= values[i - j]:
                rose = False
                break
        out[i] = rose
    return out


# ------------------------------------------------------------------ #
#  Core pandas computation                                             #
# ------------------------------------------------------------------ #
def _momentum_cycle_sentry_pandas(
    df: PdDataFrame,
    length: int,
    smoothing: int,
    magnitude: float,
    retrace_len: int,
    ob_lookback: int,
    ob_mult_inner: float,
    ob_mult_outer: float,
    close_col: str,
    # output column names
    mcs_p1_col: str,
    mcs_p2_col: str,
    mcs_p3_col: str,
    mcs_p4_col: str,
    mcs_p5_col: str,
    mcs_ob_inner_col: str,
    mcs_ob_outer_col: str,
    mcs_os_inner_col: str,
    mcs_os_outer_col: str,
    mcs_trend_col: str,
    mcs_retracing_col: str,
) -> PdDataFrame:
    """Core numpy/pandas computation."""
    df = df.copy()
    close = df[close_col].values.astype(float)
    n = len(close)

    # 1. Raw momentum: close - close[length]
    mom = np.full(n, np.nan)
    for i in range(length, n):
        mom[i] = close[i] - close[i - length]

    # 2. Five EMA layers × magnitude
    p1 = _ema(mom, smoothing) * magnitude
    p2 = _ema(mom, smoothing * 2) * magnitude
    p3 = _ema(mom, smoothing * 3) * magnitude
    p4 = _ema(mom, smoothing * 4) * magnitude
    p5 = _ema(mom, smoothing * 5) * magnitude

    # 3. OB / OS corridors
    stdev_vals = _rolling_stdev(p1, ob_lookback)
    ob_inner = stdev_vals * ob_mult_inner
    ob_outer = stdev_vals * ob_mult_outer
    os_inner = -ob_inner
    os_outer = -ob_outer

    # 4. Trend & retracement
    is_up = p1 > 0
    is_down = p1 < 0
    fall = _falling(p1, retrace_len)
    rise = _rising(p1, retrace_len)
    retracing = (is_up & fall) | (is_down & rise)

    trend = np.where(is_up, 1, np.where(is_down, -1, 0)).astype(int)

    # Write columns
    df[mcs_p1_col] = p1
    df[mcs_p2_col] = p2
    df[mcs_p3_col] = p3
    df[mcs_p4_col] = p4
    df[mcs_p5_col] = p5
    df[mcs_ob_inner_col] = ob_inner
    df[mcs_ob_outer_col] = ob_outer
    df[mcs_os_inner_col] = os_inner
    df[mcs_os_outer_col] = os_outer
    df[mcs_trend_col] = trend
    df[mcs_retracing_col] = retracing.astype(int)

    return df


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #
def momentum_cycle_sentry(
    data: Union[PdDataFrame, PlDataFrame],
    length: int = 20,
    smoothing: int = 5,
    magnitude: float = 1.0,
    retrace_len: int = 2,
    ob_lookback: int = 50,
    ob_mult_inner: float = 2.0,
    ob_mult_outer: float = 3.0,
    close_column: str = "Close",
    mcs_p1_column: str = "mcs_p1",
    mcs_p2_column: str = "mcs_p2",
    mcs_p3_column: str = "mcs_p3",
    mcs_p4_column: str = "mcs_p4",
    mcs_p5_column: str = "mcs_p5",
    mcs_ob_inner_column: str = "mcs_ob_inner",
    mcs_ob_outer_column: str = "mcs_ob_outer",
    mcs_os_inner_column: str = "mcs_os_inner",
    mcs_os_outer_column: str = "mcs_os_outer",
    mcs_trend_column: str = "mcs_trend",
    mcs_retracing_column: str = "mcs_retracing",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Momentum Cycle Sentry — multi-layer momentum oscillator.

    Ported from the LuxAlgo PineScript indicator.  Computes five
    EMA-smoothed momentum layers, dynamic overbought/oversold corridors
    based on standard deviation, and trend/retracement detection.

    Args:
        data: OHLCV DataFrame (pandas or polars).
        length: Lookback period for raw momentum (default 20).
        smoothing: Base EMA period — layers use smoothing×1 … ×5
            (default 5).
        magnitude: Scalar multiplier for all layers (default 1.0).
        retrace_len: Number of consecutive falling/rising bars to
            qualify as a retracement (default 2).
        ob_lookback: Rolling window for standard deviation used in
            OB/OS bands (default 50).
        ob_mult_inner: Inner-band multiplier (default 2.0).
        ob_mult_outer: Outer-band multiplier (default 3.0).
        close_column: Name of the Close column.

    Returns:
        DataFrame with added columns:

        - ``mcs_p1`` … ``mcs_p5`` — five smoothed momentum layers
          (mirrors are simply −p1 … −p5)
        - ``mcs_ob_inner`` / ``mcs_ob_outer`` — overbought corridor
        - ``mcs_os_inner`` / ``mcs_os_outer`` — oversold corridor
        - ``mcs_trend`` — 1 bullish, −1 bearish, 0 neutral
        - ``mcs_retracing`` — 1 when the oscillator is pulling back
          within the current trend, 0 otherwise

    Example::

        >>> from pyindicators import momentum_cycle_sentry
        >>> df = momentum_cycle_sentry(df, length=20, smoothing=5)
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        pdf = _momentum_cycle_sentry_pandas(
            pdf, length, smoothing, magnitude, retrace_len,
            ob_lookback, ob_mult_inner, ob_mult_outer,
            close_column,
            mcs_p1_column, mcs_p2_column, mcs_p3_column,
            mcs_p4_column, mcs_p5_column,
            mcs_ob_inner_column, mcs_ob_outer_column,
            mcs_os_inner_column, mcs_os_outer_column,
            mcs_trend_column, mcs_retracing_column,
        )
        import polars as pl
        return pl.from_pandas(pdf)

    if isinstance(data, PdDataFrame):
        return _momentum_cycle_sentry_pandas(
            data, length, smoothing, magnitude, retrace_len,
            ob_lookback, ob_mult_inner, ob_mult_outer,
            close_column,
            mcs_p1_column, mcs_p2_column, mcs_p3_column,
            mcs_p4_column, mcs_p5_column,
            mcs_ob_inner_column, mcs_ob_outer_column,
            mcs_os_inner_column, mcs_os_outer_column,
            mcs_trend_column, mcs_retracing_column,
        )

    raise PyIndicatorException(
        "Input data must be a pandas or polars DataFrame."
    )


def momentum_cycle_sentry_signal(
    data: Union[PdDataFrame, PlDataFrame],
    mcs_trend_column: str = "mcs_trend",
    mcs_retracing_column: str = "mcs_retracing",
    signal_column: str = "mcs_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a directional signal from Momentum Cycle Sentry.

    Signal values:

    -  ``1``  — bullish momentum (fast line > 0, not retracing)
    - ``-1``  — bearish momentum (fast line < 0, not retracing)
    -  ``0``  — neutral or retracing (momentum pulling back)

    Args:
        data: DataFrame with MCS columns already computed.
        mcs_trend_column: Name of the trend column.
        mcs_retracing_column: Name of the retracing column.
        signal_column: Output signal column name.

    Returns:
        DataFrame with added signal column.
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        trend = pdf[mcs_trend_column].fillna(0).astype(int)
        retrace = pdf[mcs_retracing_column].fillna(0).astype(int)
        signal = np.where(retrace == 1, 0, trend)
        pdf[signal_column] = signal.astype(int)
        import polars as pl
        return pl.from_pandas(pdf)

    if isinstance(data, PdDataFrame):
        df = data.copy()
        trend = df[mcs_trend_column].fillna(0).astype(int)
        retrace = df[mcs_retracing_column].fillna(0).astype(int)
        signal = np.where(retrace == 1, 0, trend)
        df[signal_column] = signal.astype(int)
        return df

    raise PyIndicatorException(
        "Input data must be a pandas or polars DataFrame."
    )


def get_momentum_cycle_sentry_stats(
    data: Union[PdDataFrame, PlDataFrame],
    mcs_p1_column: str = "mcs_p1",
    mcs_trend_column: str = "mcs_trend",
    mcs_retracing_column: str = "mcs_retracing",
    mcs_ob_inner_column: str = "mcs_ob_inner",
    mcs_os_inner_column: str = "mcs_os_inner",
) -> Dict[str, object]:
    """
    Compute summary statistics for Momentum Cycle Sentry.

    Args:
        data: DataFrame with MCS columns.

    Returns:
        Dictionary with keys:

        - ``bullish_bars``       — bars where trend == 1
        - ``bearish_bars``       — bars where trend == −1
        - ``bullish_pct``        — percentage of bullish bars
        - ``bearish_pct``        — percentage of bearish bars
        - ``retracing_bars``     — bars flagged as retracing
        - ``retracing_pct``      — percentage of retracing bars
        - ``overbought_bars``    — bars where p1 > ob_inner
        - ``oversold_bars``      — bars where p1 < os_inner
        - ``max_momentum``       — maximum p1 value
        - ``min_momentum``       — minimum p1 value
        - ``avg_momentum``       — mean of absolute p1 values
        - ``zero_crossings``     — number of zero-line crossings
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
    elif isinstance(data, PdDataFrame):
        pdf = data
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )

    p1 = pdf[mcs_p1_column].dropna()
    trend = pdf[mcs_trend_column].fillna(0).astype(int)
    retrace = pdf[mcs_retracing_column].fillna(0).astype(int)

    total = len(trend)
    bull = int((trend == 1).sum())
    bear = int((trend == -1).sum())
    retrace_count = int(retrace.sum())

    # OB / OS counts
    valid_mask = p1.index
    ob_bars = int((pdf.loc[valid_mask, mcs_p1_column]
                   > pdf.loc[valid_mask, mcs_ob_inner_column]).sum())
    os_bars = int((pdf.loc[valid_mask, mcs_p1_column]
                   < pdf.loc[valid_mask, mcs_os_inner_column]).sum())

    # Zero crossings
    p1_vals = pdf[mcs_p1_column].values
    crossings = 0
    for i in range(1, len(p1_vals)):
        if np.isnan(p1_vals[i]) or np.isnan(p1_vals[i - 1]):
            continue
        if (p1_vals[i] > 0 and p1_vals[i - 1] <= 0) or \
           (p1_vals[i] < 0 and p1_vals[i - 1] >= 0):
            crossings += 1

    return {
        "bullish_bars": bull,
        "bearish_bars": bear,
        "bullish_pct": round(bull / total * 100, 1) if total > 0 else 0.0,
        "bearish_pct": round(bear / total * 100, 1) if total > 0 else 0.0,
        "retracing_bars": retrace_count,
        "retracing_pct": (
            round(retrace_count / total * 100, 1)
            if total > 0 else 0.0
        ),
        "overbought_bars": ob_bars,
        "oversold_bars": os_bars,
        "max_momentum": round(float(p1.max()), 4) if len(p1) > 0 else 0.0,
        "min_momentum": round(float(p1.min()), 4) if len(p1) > 0 else 0.0,
        "avg_momentum": (
            round(float(p1.abs().mean()), 4)
            if len(p1) > 0 else 0.0
        ),
        "zero_crossings": crossings,
    }
