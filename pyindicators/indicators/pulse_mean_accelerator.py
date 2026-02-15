"""
Pulse Mean Accelerator (PMA)

The Pulse Mean Accelerator applies a volatility-scaled acceleration
adjustment to a base moving average.  The acceleration is derived from
a lookback comparison of source-price momentum vs. moving-average
momentum.  The result is a trend-following overlay that adapts its
distance from price according to volatility and directional strength.

Output columns
--------------
pma              : The PMA line.
pma_ma           : The base moving average.
pma_trend        : Trend direction (1 = long, -1 = short, 0 = neutral).
pma_long         : 1 on the bar where trend flips to long.
pma_short        : 1 on the bar where trend flips to short.
pma_acceleration : The raw acceleration factor for each bar.
"""
from typing import Union, Dict
import math
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl

from pyindicators.exceptions import PyIndicatorException

# ── Valid option sets ────────────────────────────────────────────────
VALID_MA_TYPES = {"RMA", "SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"}
VALID_VOL_TYPES = {"ATR", "Standard Deviation", "MAD"}
VALID_SMOOTH_TYPES = {
    "NONE", "Exponential", "Extra Moving Average", "Double Moving Average"
}


# =====================================================================
# NumPy-level moving-average helpers
# =====================================================================
def _np_sma(values: np.ndarray, length: int) -> np.ndarray:
    """Simple Moving Average (numpy)."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    cumsum = np.cumsum(values)
    out[length - 1:] = (cumsum[length - 1:]
                        - np.concatenate([[0], cumsum[:-length]])) / length
    # Fill initial bars with expanding mean
    for i in range(length - 1):
        out[i] = np.mean(values[:i + 1])
    return out


def _np_ema(values: np.ndarray, length: int) -> np.ndarray:
    """Exponential Moving Average (numpy)."""
    alpha = 2.0 / (length + 1)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _np_rma(values: np.ndarray, length: int) -> np.ndarray:
    """Wilder's Moving Average / RMA (numpy)."""
    alpha = 1.0 / length
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _np_wma(values: np.ndarray, length: int) -> np.ndarray:
    """Weighted Moving Average (numpy)."""
    weights = np.arange(1, length + 1, dtype=np.float64)
    wsum = weights.sum()
    out = np.full_like(values, np.nan, dtype=np.float64)
    for i in range(length - 1, len(values)):
        out[i] = np.dot(values[i - length + 1:i + 1], weights) / wsum
    # Fill initial bars with expanding WMA
    for i in range(length - 1):
        w = np.arange(1, i + 2, dtype=np.float64)
        out[i] = np.dot(values[:i + 1], w) / w.sum()
    return out


def _np_dema(values: np.ndarray, length: int) -> np.ndarray:
    """Double EMA: 2*EMA - EMA(EMA)."""
    e1 = _np_ema(values, length)
    e2 = _np_ema(e1, length)
    return 2 * e1 - e2


def _np_tema(values: np.ndarray, length: int) -> np.ndarray:
    """Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    e1 = _np_ema(values, length)
    e2 = _np_ema(e1, length)
    e3 = _np_ema(e2, length)
    return 3 * e1 - 3 * e2 + e3


def _np_hma(values: np.ndarray, length: int) -> np.ndarray:
    """Hull Moving Average: WMA(2*WMA(src,len/2) - WMA(src,len), sqrt(len))."""
    half = max(int(round(length / 2)), 1)
    sqr = max(int(round(math.sqrt(length))), 1)
    w1 = _np_wma(values, half)
    w2 = _np_wma(values, length)
    return _np_wma(2 * w1 - w2, sqr)


def _np_ma(values: np.ndarray, length: int, ma_type: str) -> np.ndarray:
    """Dispatch to the correct MA implementation."""
    funcs = {
        "RMA": _np_rma,
        "SMA": _np_sma,
        "EMA": _np_ema,
        "WMA": _np_wma,
        "DEMA": _np_dema,
        "TEMA": _np_tema,
        "HMA": _np_hma,
    }
    return funcs[ma_type](values, length)


def _np_stdev(values: np.ndarray, length: int) -> np.ndarray:
    """Rolling standard deviation (population)."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    for i in range(len(values)):
        window = values[max(0, i - length + 1):i + 1]
        out[i] = np.std(window)
    return out


# =====================================================================
# Main indicator
# =====================================================================
def pulse_mean_accelerator(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str = "Close",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    ma_type: str = "RMA",
    ma_length: int = 20,
    accel_lookback: int = 32,
    max_accel: float = 0.2,
    volatility_type: str = "Standard Deviation",
    smooth_type: str = "Double Moving Average",
    use_confirmation: bool = True,
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the Pulse Mean Accelerator (PMA) indicator.

    The PMA adds a volatility- and momentum-scaled acceleration offset
    to a user-chosen moving average, producing a trend-following overlay
    line.  Trend flips occur when price crosses the PMA line
    (optionally confirmed by combined PMA + MA momentum).

    Parameters
    ----------
    data : DataFrame
        pandas or polars DataFrame with OHLC data.
    source_column : str
        Column to use as the source series (default ``"Close"``).
    high_column, low_column, close_column : str
        Column names for high, low, close prices.
    ma_type : str
        Moving-average type.  One of ``RMA``, ``SMA``, ``EMA``,
        ``WMA``, ``DEMA``, ``TEMA``, ``HMA``.
    ma_length : int
        Lookback period for the base moving average (default 20).
    accel_lookback : int
        Number of past bars over which acceleration is accumulated
        (default 32).
    max_accel : float
        Maximum (absolute) acceleration factor (default 0.2).
    volatility_type : str
        Volatility measure: ``"ATR"``, ``"Standard Deviation"``,
        or ``"MAD"`` (Mean Absolute Deviation from MA).
    smooth_type : str
        Smoothing applied to the raw PMA:
        ``"NONE"``, ``"Exponential"``, ``"Extra Moving Average"``,
        ``"Double Moving Average"`` (default).
    use_confirmation : bool
        When *True* (default), trend flips require the combined
        momentum of PMA and MA to agree with the direction.

    Returns
    -------
    DataFrame
        The input DataFrame with added columns:

        - ``pma`` – The PMA line value.
        - ``pma_ma`` – The base moving average.
        - ``pma_trend`` – Trend direction: 1 (long), -1 (short),
          0 (initial/neutral).
        - ``pma_long`` – 1 on the bar where trend flips to long.
        - ``pma_short`` – 1 on the bar where trend flips to short.
        - ``pma_acceleration`` – Raw acceleration factor per bar.
    """
    # ── Validate inputs ──────────────────────────────────────────
    if ma_type not in VALID_MA_TYPES:
        raise PyIndicatorException(
            f"Invalid ma_type '{ma_type}'. "
            f"Must be one of {sorted(VALID_MA_TYPES)}."
        )
    if volatility_type not in VALID_VOL_TYPES:
        raise PyIndicatorException(
            f"Invalid volatility_type '{volatility_type}'. "
            f"Must be one of {sorted(VALID_VOL_TYPES)}."
        )
    if smooth_type not in VALID_SMOOTH_TYPES:
        raise PyIndicatorException(
            f"Invalid smooth_type '{smooth_type}'. "
            f"Must be one of {sorted(VALID_SMOOTH_TYPES)}."
        )
    if ma_length < 2:
        raise PyIndicatorException("ma_length must be >= 2.")
    if accel_lookback < 2:
        raise PyIndicatorException("accel_lookback must be >= 2.")
    if max_accel <= 0:
        raise PyIndicatorException("max_accel must be > 0.")

    # ── Extract numpy arrays ─────────────────────────────────────
    is_polars = isinstance(data, PlDataFrame)

    if is_polars:
        src = data[source_column].to_numpy().astype(np.float64)
        highs = data[high_column].to_numpy().astype(np.float64)
        lows = data[low_column].to_numpy().astype(np.float64)
        closes = data[close_column].to_numpy().astype(np.float64)
    else:
        src = data[source_column].values.astype(np.float64)
        highs = data[high_column].values.astype(np.float64)
        lows = data[low_column].values.astype(np.float64)
        closes = data[close_column].values.astype(np.float64)

    n = len(src)

    if n < ma_length:
        raise PyIndicatorException(
            f"Data length ({n}) must be >= ma_length ({ma_length})."
        )

    # ── 1. Base moving average ───────────────────────────────────
    ma = _np_ma(src, ma_length, ma_type)

    # ── 2. Rate of change (1-bar differences) ────────────────────
    roc_src = np.zeros(n)
    roc_ma = np.zeros(n)
    roc_src[1:] = src[1:] - src[:-1]
    roc_ma[1:] = ma[1:] - ma[:-1]

    # ── 3. Acceleration per bar ──────────────────────────────────
    # Look back *accel_lookback* bars, accumulating +step when
    # |roc_src| > |roc_ma| and -step otherwise.
    step = max_accel / accel_lookback
    acc = np.zeros(n)

    for bar in range(n):
        a = 0.0
        for k in range(accel_lookback):
            idx = bar - k
            if idx < 1:
                break
            ar_src = abs(roc_src[idx])
            ar_ma = abs(roc_ma[idx])
            if ar_src > ar_ma:
                a += step
            elif ar_ma > ar_src:
                a -= step
        acc[bar] = a

    # ── 4. Volatility ────────────────────────────────────────────
    # True Range
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    if volatility_type == "ATR":
        vol = _np_ma(tr, accel_lookback, ma_type)
    elif volatility_type == "Standard Deviation":
        vol = _np_stdev(ma, accel_lookback)
    else:  # MAD
        mad_raw = np.abs(ma - src)
        vol = _np_ma(mad_raw, accel_lookback, ma_type)

    # ── 5. Direction (sign of average ROC of MA) ─────────────────
    avgr = _np_ma(roc_ma, accel_lookback, ma_type)
    absr = np.abs(avgr)
    direction = np.zeros(n, dtype=np.float64)
    mask = absr != 0.0
    np.divide(avgr, absr, out=direction, where=mask)

    # ── 6. Raw PMA ───────────────────────────────────────────────
    pma_raw = ma + acc * vol * direction

    # ── 7. Smoothing ─────────────────────────────────────────────
    sqrt_len = max(int(round(math.sqrt(ma_length))), 1)

    if smooth_type == "NONE":
        pma = pma_raw.copy()
    elif smooth_type == "Exponential":
        alpha = 2.0 / (1 + ma_length)
        pma = pma_raw * (1 - alpha) + ma * alpha
    elif smooth_type == "Extra Moving Average":
        pma = _np_ma(pma_raw, sqrt_len, ma_type)
    else:  # Double Moving Average
        pma_ma_smooth = _np_ma(pma_raw, sqrt_len, ma_type)
        pma = _np_ma(pma_ma_smooth, sqrt_len, ma_type)

    # ── 8. Confirmation & trend logic ────────────────────────────
    trend = np.zeros(n, dtype=int)

    for bar in range(1, n):
        prev_trend = trend[bar - 1]
        pma_roc = pma[bar] - pma[bar - 1]
        ma_roc = ma[bar] - ma[bar - 1]
        combined = pma_roc + ma_roc

        cl = (combined > 0) if use_confirmation else True
        cs = (combined < 0) if use_confirmation else True

        is_long = (src[bar] > pma[bar]) and cl
        is_short = (src[bar] < pma[bar]) and cs

        if is_long:
            trend[bar] = 1
        elif is_short:
            trend[bar] = -1
        else:
            trend[bar] = prev_trend

    # ── 9. Trend-change signals ──────────────────────────────────
    long_signal = np.zeros(n, dtype=int)
    short_signal = np.zeros(n, dtype=int)

    for bar in range(1, n):
        if trend[bar] == 1 and trend[bar - 1] != 1:
            long_signal[bar] = 1
        if trend[bar] == -1 and trend[bar - 1] != -1:
            short_signal[bar] = 1

    # ── 10. Write output columns ─────────────────────────────────
    if is_polars:
        data = data.with_columns([
            pl.Series("pma", pma),
            pl.Series("pma_ma", ma),
            pl.Series("pma_trend", trend),
            pl.Series("pma_long", long_signal),
            pl.Series("pma_short", short_signal),
            pl.Series("pma_acceleration", acc),
        ])
    else:
        data["pma"] = pma
        data["pma_ma"] = ma
        data["pma_trend"] = trend
        data["pma_long"] = long_signal
        data["pma_short"] = short_signal
        data["pma_acceleration"] = acc

    return data


# =====================================================================
# Signal function
# =====================================================================
def pulse_mean_accelerator_signal(
    data: Union[PdDataFrame, PlDataFrame],
    signal_column: str = "pma_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a simple signal column from PMA trend changes.

    Signal values
    -------------
    - ``1``  – Long (trend just flipped bullish).
    - ``-1`` – Short (trend just flipped bearish).
    - ``0``  – No change.

    The ``pma_long`` and ``pma_short`` columns must already be
    present (call :func:`pulse_mean_accelerator` first).

    Parameters
    ----------
    data : DataFrame
        DataFrame with PMA columns.
    signal_column : str
        Name for the resulting signal column.

    Returns
    -------
    DataFrame
    """
    for col in ("pma_long", "pma_short"):
        if col not in data.columns:
            raise PyIndicatorException(
                f"Column '{col}' not found. "
                "Run pulse_mean_accelerator() first."
            )

    if isinstance(data, PlDataFrame):
        data = data.with_columns(
            pl.when(pl.col("pma_long") == 1)
            .then(1)
            .when(pl.col("pma_short") == 1)
            .then(-1)
            .otherwise(0)
            .alias(signal_column)
        )
    else:
        sig = np.where(
            data["pma_long"] == 1, 1,
            np.where(data["pma_short"] == 1, -1, 0),
        )
        data[signal_column] = sig

    return data


# =====================================================================
# Stats function
# =====================================================================
def get_pulse_mean_accelerator_stats(
    data: Union[PdDataFrame, PlDataFrame],
) -> Dict[str, object]:
    """
    Return summary statistics for the PMA indicator.

    Requires that :func:`pulse_mean_accelerator` has already been
    applied to *data*.

    Returns
    -------
    dict
        Keys:

        - ``total_long_signals`` – Number of bullish-flip bars.
        - ``total_short_signals`` – Number of bearish-flip bars.
        - ``total_signals`` – Sum of long + short.
        - ``current_trend`` – Last observed trend (1 / -1 / 0).
        - ``current_pma`` – Last PMA value.
        - ``current_ma`` – Last base-MA value.
        - ``long_ratio`` – Fraction of bars with trend == 1.
        - ``short_ratio`` – Fraction of bars with trend == -1.
    """
    for col in ("pma", "pma_ma", "pma_trend", "pma_long", "pma_short"):
        if col not in data.columns:
            raise PyIndicatorException(
                f"Column '{col}' not found. "
                "Run pulse_mean_accelerator() first."
            )

    if isinstance(data, PlDataFrame):
        total_long = int(data["pma_long"].sum())
        total_short = int(data["pma_short"].sum())
        current_trend = int(data["pma_trend"][-1])
        current_pma = float(data["pma"][-1])
        current_ma = float(data["pma_ma"][-1])
        bars_long = int((data["pma_trend"] == 1).sum())
        bars_short = int((data["pma_trend"] == -1).sum())
        total_bars = len(data)
    else:
        total_long = int(data["pma_long"].sum())
        total_short = int(data["pma_short"].sum())
        current_trend = int(data["pma_trend"].iloc[-1])
        current_pma = float(data["pma"].iloc[-1])
        current_ma = float(data["pma_ma"].iloc[-1])
        bars_long = int((data["pma_trend"] == 1).sum())
        bars_short = int((data["pma_trend"] == -1).sum())
        total_bars = len(data)

    return {
        "total_long_signals": total_long,
        "total_short_signals": total_short,
        "total_signals": total_long + total_short,
        "current_trend": current_trend,
        "current_pma": round(current_pma, 6),
        "current_ma": round(current_ma, 6),
        "long_ratio": round(bars_long / total_bars, 4) if total_bars else 0,
        "short_ratio": round(bars_short / total_bars, 4) if total_bars else 0,
    }
