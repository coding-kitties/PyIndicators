"""
Volume Weighted Trend Indicator

Based on the Volume Weighted Trend [QuantAlgo] concept from TradingView.
Uses a Volume Weighted Moving Average (VWMA) with ATR-based volatility
bands to determine trend direction.

Core logic:
    1. Compute VWMA of close prices as the trend baseline
    2. Compute ATR over the same period for volatility measurement
    3. Build upper/lower bands: VWMA +/- ATR * multiplier
    4. Trend flips bullish when close > upper band,
       bearish when close < lower band, stays otherwise

Output columns:
    vwt_vwma          - Volume Weighted Moving Average
    vwt_atr           - Average True Range
    vwt_upper         - Upper volatility band
    vwt_lower         - Lower volatility band
    vwt_trend         - Trend direction: +1 bullish, -1 bearish, 0 undefined
    vwt_trend_changed - 1 on bars where trend flipped, 0 otherwise
    vwt_signal        - +1 on bullish flip, -1 on bearish flip, 0 otherwise
"""
from typing import Union, Dict
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame


# ── Helpers ──────────────────────────────────────────────────────

def _vwma(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume Weighted Moving Average."""
    n = len(close)
    out = np.full(n, np.nan)
    pv = close * volume
    for i in range(period - 1, n):
        vol_sum = np.sum(volume[i - period + 1: i + 1])
        if vol_sum > 0:
            out[i] = np.sum(pv[i - period + 1: i + 1]) / vol_sum
        else:
            out[i] = close[i]
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int) -> np.ndarray:
    """Average True Range using Wilder's smoothing (RMA)."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    # RMA (Wilder's smoothing)
    out = np.full(n, np.nan)
    if n >= period:
        out[period - 1] = np.mean(tr[:period])
        alpha = 1.0 / period
        for i in range(period, n):
            out[i] = alpha * tr[i] + (1 - alpha) * out[i - 1]
    return out


# ── Core indicator ───────────────────────────────────────────────

def _volume_weighted_trend_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    vwma_length: int,
    atr_multiplier: float,
):
    """
    Pure-numpy core computation.

    Returns dict of numpy arrays.
    """
    vwma_arr = _vwma(close, volume, vwma_length)
    atr_arr = _atr(high, low, close, vwma_length)

    upper = vwma_arr + atr_arr * atr_multiplier
    lower = vwma_arr - atr_arr * atr_multiplier

    n = len(close)
    trend = np.zeros(n, dtype=int)
    trend_changed = np.zeros(n, dtype=int)
    signal = np.zeros(n, dtype=int)

    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            trend[i] = 0
            continue
        if close[i] > upper[i]:
            trend[i] = 1
        elif close[i] < lower[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1] if i > 0 else 0

        if i > 0 and trend[i] != 0 and trend[i] != trend[i - 1]:
            trend_changed[i] = 1
            signal[i] = trend[i]

    return {
        "vwt_vwma": vwma_arr,
        "vwt_atr": atr_arr,
        "vwt_upper": upper,
        "vwt_lower": lower,
        "vwt_trend": trend,
        "vwt_trend_changed": trend_changed,
        "vwt_signal": signal,
    }


# ── Public API ───────────────────────────────────────────────────

def volume_weighted_trend(
    df: Union[PdDataFrame, PlDataFrame],
    vwma_length: int = 34,
    atr_multiplier: float = 1.5,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    volume_column: str = "Volume",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Compute the Volume Weighted Trend indicator.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Must contain High, Low, Close, Volume columns.
    vwma_length : int
        Period for VWMA and ATR calculation (default 34).
    atr_multiplier : float
        ATR multiplier for band width (default 1.5).
    high_column, low_column, close_column, volume_column : str
        Column name overrides.

    Returns
    -------
    DataFrame with added columns:
        vwt_vwma, vwt_atr, vwt_upper, vwt_lower,
        vwt_trend, vwt_trend_changed, vwt_signal
    """
    is_polars = isinstance(df, PlDataFrame)

    if is_polars:
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    high = pdf[high_column].values.astype(float)
    low = pdf[low_column].values.astype(float)
    close = pdf[close_column].values.astype(float)
    volume = pdf[volume_column].values.astype(float)

    results = _volume_weighted_trend_core(
        high, low, close, volume, vwma_length, atr_multiplier,
    )

    for col, arr in results.items():
        pdf[col] = arr

    if is_polars:
        import polars as pl
        return pl.from_pandas(pdf)

    return pdf


def volume_weighted_trend_signal(
    df: Union[PdDataFrame, PlDataFrame],
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Return the DataFrame unchanged - signals are already computed
    in ``volume_weighted_trend`` as ``vwt_signal``.

    This function exists for API consistency with other indicators.
    """
    return df


def get_volume_weighted_trend_stats(
    df: Union[PdDataFrame, PlDataFrame],
) -> Dict:
    """
    Compute summary statistics for the Volume Weighted Trend indicator.

    Parameters
    ----------
    df : DataFrame with vwt_* columns from ``volume_weighted_trend``.

    Returns
    -------
    dict with keys:
        total_bars, bullish_bars, bearish_bars, neutral_bars,
        bullish_pct, bearish_pct, total_flips,
        bullish_flips, bearish_flips
    """
    is_polars = isinstance(df, PlDataFrame)
    if is_polars:
        pdf = df.to_pandas()
    else:
        pdf = df

    trend = pdf["vwt_trend"].values
    signal = pdf["vwt_signal"].values

    total = len(trend)
    bullish = int((trend == 1).sum())
    bearish = int((trend == -1).sum())
    neutral = total - bullish - bearish

    bull_flips = int((signal == 1).sum())
    bear_flips = int((signal == -1).sum())

    return {
        "total_bars": total,
        "bullish_bars": bullish,
        "bearish_bars": bearish,
        "neutral_bars": neutral,
        "bullish_pct": bullish / total if total else 0,
        "bearish_pct": bearish / total if total else 0,
        "total_flips": bull_flips + bear_flips,
        "bullish_flips": bull_flips,
        "bearish_flips": bear_flips,
    }
