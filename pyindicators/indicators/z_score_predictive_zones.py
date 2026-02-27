"""
Z-Score Predictive Zones Indicator

Based on the Z-Score Predictive Zones [AlgoPoint] concept from TradingView.
Uses a VWMA-smoothed Z-Score with dynamic reversal-based support/resistance
zones to identify overbought/oversold conditions and generate signals when
price enters those predictive zones.

Core logic:
    1. Compute the Z-Score: (close - SMA) / StdDev, smoothed with VWMA
    2. Detect pivot highs/lows on the smoothed Z-Score
    3. Collect reversals above/below a threshold into rolling arrays
    4. Average those reversals to derive dynamic support/resistance levels
    5. Reverse-engineer Z-Score levels back to price to create chart bands
    6. Generate signals when price first enters a support/resistance band

Output columns:
    zspz_z_score          - Smoothed Z-Score oscillator value
    zspz_mean             - SMA of close (Z-Score baseline)
    zspz_std              - Standard deviation of close
    zspz_avg_top_level    - Average Z-Score level of top reversals
    zspz_avg_bot_level    - Average Z-Score level of bottom reversals
    zspz_res_band_high    - Resistance band high (price space, VWMA smoothed)
    zspz_res_band_low     - Resistance band low (price space, VWMA smoothed)
    zspz_sup_band_high    - Support band high (price space, VWMA smoothed)
    zspz_sup_band_low     - Support band low (price space, VWMA smoothed)
    zspz_rsi_ma           - EMA of RSI (for gradient coloring)
    zspz_long_signal      - 1 when price first enters support band, 0 otherwise
    zspz_short_signal     - 1 when price first enters resistance band, 0 otherwise
    zspz_signal           - +1 for long, -1 for short, 0 otherwise
"""
from typing import Union, Dict
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame


# ── Helpers ──────────────────────────────────────────────────────

def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    n = len(data)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.mean(data[i - period + 1: i + 1])
    return out


def _stdev(data: np.ndarray, period: int) -> np.ndarray:
    """Population-style standard deviation (matches Pine Script ta.stdev)."""
    n = len(data)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = data[i - period + 1: i + 1]
        out[i] = np.std(window, ddof=0)
    return out


def _vwma(data: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume Weighted Moving Average."""
    n = len(data)
    out = np.full(n, np.nan)
    pv = data * volume
    for i in range(period - 1, n):
        vol_sum = np.sum(volume[i - period + 1: i + 1])
        if vol_sum > 0:
            out[i] = np.sum(pv[i - period + 1: i + 1]) / vol_sum
        else:
            out[i] = data[i]
    return out


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    n = len(data)
    out = np.full(n, np.nan)
    alpha = 2.0 / (period + 1)
    # Find first non-NaN to seed
    start = 0
    while start < n and np.isnan(data[start]):
        start += 1
    if start >= n:
        return out
    out[start] = data[start]
    for i in range(start + 1, n):
        if np.isnan(data[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index using Wilder's smoothing."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - 100.0 / (1 + rs)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1 + rs)

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
    out = np.full(n, np.nan)
    if n >= period:
        out[period - 1] = np.mean(tr[:period])
        alpha = 1.0 / period
        for i in range(period, n):
            out[i] = alpha * tr[i] + (1 - alpha) * out[i - 1]
    return out


def _pivot_high(data: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    Detect pivot highs.
    A pivot high at index i means data[i] is the highest in
    [i-left .. i+right]. Result is placed at i+right (delayed).
    Returns NaN where no pivot, pivot value where detected.
    """
    n = len(data)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        val = data[i]
        if np.isnan(val):
            continue
        is_pivot = True
        for j in range(i - left, i + right + 1):
            if j == i:
                continue
            if np.isnan(data[j]) or data[j] >= val:
                is_pivot = False
                break
        if is_pivot:
            # Pine Script reports pivot at bar i+right
            out[i + right] = val
    return out


def _pivot_low(data: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    Detect pivot lows.
    Result placed at i+right (delayed like Pine Script).
    """
    n = len(data)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        val = data[i]
        if np.isnan(val):
            continue
        is_pivot = True
        for j in range(i - left, i + right + 1):
            if j == i:
                continue
            if np.isnan(data[j]) or data[j] <= val:
                is_pivot = False
                break
        if is_pivot:
            out[i + right] = val
    return out


# ── Core indicator ───────────────────────────────────────────────

def _z_score_predictive_zones_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    z_length: int,
    smooth: int,
    history_depth: int,
    threshold: float,
    rsi_length: int,
    rsi_ma_length: int,
    band_smooth: int,
    atr_length: int,
):
    """
    Pure-numpy core computation.

    Returns dict of numpy arrays.
    """
    n = len(close)

    # 1. Z-Score calculation
    mean = _sma(close, z_length)
    std_dev = _stdev(close, z_length)

    raw_z = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(mean[i]) and not np.isnan(std_dev[i]) \
                and std_dev[i] > 0:
            raw_z[i] = (close[i] - mean[i]) / std_dev[i]

    z_score = _vwma(raw_z, volume, smooth)

    # 2. Reversal detection (pivots on z_score with left=1, right=1)
    ph = _pivot_high(z_score, 1, 1)
    pl_ = _pivot_low(z_score, 1, 1)

    # 3. Dynamic zone calculation
    z_thresh_top = threshold
    z_thresh_bot = -threshold

    top_reversals = []
    bot_reversals = []
    avg_top_level = np.full(n, np.nan)
    avg_bot_level = np.full(n, np.nan)

    for i in range(n):
        # Check for valid top reversal
        if not np.isnan(ph[i]) and ph[i] > z_thresh_top:
            top_reversals.insert(0, ph[i])
            if len(top_reversals) > history_depth:
                top_reversals.pop()

        # Check for valid bottom reversal
        if not np.isnan(pl_[i]) and pl_[i] < z_thresh_bot:
            bot_reversals.insert(0, pl_[i])
            if len(bot_reversals) > history_depth:
                bot_reversals.pop()

        # Calculate average reversal levels
        avg_top_level[i] = (
            np.mean(top_reversals) if len(top_reversals) > 0 else 2.0
        )
        avg_bot_level[i] = (
            np.mean(bot_reversals) if len(bot_reversals) > 0 else -2.0
        )

    # 4. RSI color logic
    rsi_vals = _rsi(close, rsi_length)
    rsi_ma = _ema(rsi_vals, rsi_ma_length)

    # 5. Price band calculation (reverse-engineer Z-Score → price)
    res_band_low = mean + avg_top_level * std_dev
    res_band_high = mean + (avg_top_level + 0.5) * std_dev
    sup_band_high = mean + avg_bot_level * std_dev
    sup_band_low = mean + (avg_bot_level - 0.5) * std_dev

    # Smooth the bands with VWMA(band_smooth)
    res_band_high_smooth = _vwma(res_band_high, volume, band_smooth)
    res_band_low_smooth = _vwma(res_band_low, volume, band_smooth)
    sup_band_high_smooth = _vwma(sup_band_high, volume, band_smooth)
    sup_band_low_smooth = _vwma(sup_band_low, volume, band_smooth)

    # 6. Signal detection
    # Pine Script uses RAW (unsmoothed) bands for signal detection;
    # the smoothed bands are only used for chart plotting.

    long_signal = np.zeros(n, dtype=int)
    short_signal = np.zeros(n, dtype=int)
    signal = np.zeros(n, dtype=int)

    for i in range(1, n):
        # Short: high crosses into resistance band (raw)
        #   high > res_band_low AND NOT (high[1] > res_band_low[1])
        if (not np.isnan(res_band_low[i])
                and not np.isnan(res_band_low[i - 1])):
            curr_above = high[i] > res_band_low[i]
            prev_above = high[i - 1] > res_band_low[i - 1]
            if curr_above and not prev_above:
                short_signal[i] = 1
                signal[i] = -1

        # Long: low crosses into support band (raw)
        #   low < sup_band_high AND NOT (low[1] < sup_band_high[1])
        if (not np.isnan(sup_band_high[i])
                and not np.isnan(sup_band_high[i - 1])):
            curr_below = low[i] < sup_band_high[i]
            prev_below = low[i - 1] < sup_band_high[i - 1]
            if curr_below and not prev_below:
                long_signal[i] = 1
                # Long signal takes precedence if both fire on same bar
                signal[i] = 1

    return {
        "zspz_z_score": z_score,
        "zspz_mean": mean,
        "zspz_std": std_dev,
        "zspz_avg_top_level": avg_top_level,
        "zspz_avg_bot_level": avg_bot_level,
        "zspz_res_band_high": res_band_high_smooth,
        "zspz_res_band_low": res_band_low_smooth,
        "zspz_sup_band_high": sup_band_high_smooth,
        "zspz_sup_band_low": sup_band_low_smooth,
        "zspz_rsi_ma": rsi_ma,
        "zspz_long_signal": long_signal,
        "zspz_short_signal": short_signal,
        "zspz_signal": signal,
    }


# ── Public API ───────────────────────────────────────────────────

def z_score_predictive_zones(
    df: Union[PdDataFrame, PlDataFrame],
    z_length: int = 144,
    smooth: int = 20,
    history_depth: int = 25,
    threshold: float = 1.5,
    rsi_length: int = 14,
    rsi_ma_length: int = 9,
    band_smooth: int = 4,
    atr_length: int = 30,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    volume_column: str = "Volume",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Compute the Z-Score Predictive Zones indicator.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Must contain High, Low, Close, Volume columns.
    z_length : int
        Period for SMA and StdDev in the Z-Score calculation (default 144).
    smooth : int
        VWMA smoothing period applied to the raw Z-Score (default 20).
    history_depth : int
        Maximum number of recent reversals used to compute the average
        support/resistance Z-Score level (default 25).
    threshold : float
        Z-Score threshold for filtering valid reversals. Only pivot
        highs above +threshold and pivot lows below -threshold are
        included (default 1.5).
    rsi_length : int
        RSI period for the gradient color logic (default 14).
    rsi_ma_length : int
        EMA period applied to RSI for smoothing (default 9).
    band_smooth : int
        VWMA period for smoothing the price bands (default 4).
    atr_length : int
        ATR period used for signal dot placement offset (default 30).
    high_column, low_column, close_column, volume_column : str
        Column name overrides.

    Returns
    -------
    DataFrame with added columns:
        zspz_z_score, zspz_mean, zspz_std,
        zspz_avg_top_level, zspz_avg_bot_level,
        zspz_res_band_high, zspz_res_band_low,
        zspz_sup_band_high, zspz_sup_band_low,
        zspz_rsi_ma,
        zspz_long_signal, zspz_short_signal, zspz_signal
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

    results = _z_score_predictive_zones_core(
        high, low, close, volume,
        z_length, smooth, history_depth, threshold,
        rsi_length, rsi_ma_length, band_smooth, atr_length,
    )

    for col, arr in results.items():
        pdf[col] = arr

    if is_polars:
        import polars as pl
        return pl.from_pandas(pdf)

    return pdf


def z_score_predictive_zones_signal(
    df: Union[PdDataFrame, PlDataFrame],
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Return the DataFrame unchanged - signals are already computed
    in ``z_score_predictive_zones`` as ``zspz_signal``.

    This function exists for API consistency with other indicators.
    """
    return df


def get_z_score_predictive_zones_stats(
    df: Union[PdDataFrame, PlDataFrame],
) -> Dict:
    """
    Compute summary statistics for the Z-Score Predictive Zones indicator.

    Parameters
    ----------
    df : DataFrame with zspz_* columns from ``z_score_predictive_zones``.

    Returns
    -------
    dict with keys:
        total_bars, long_signals, short_signals, total_signals,
        long_pct, short_pct,
        avg_z_score, max_z_score, min_z_score,
        avg_top_level, avg_bot_level,
        avg_res_band_width, avg_sup_band_width
    """
    is_polars = isinstance(df, PlDataFrame)
    if is_polars:
        pdf = df.to_pandas()
    else:
        pdf = df

    total = len(pdf)
    long_signals = int(pdf["zspz_long_signal"].sum())
    short_signals = int(pdf["zspz_short_signal"].sum())
    total_signals = long_signals + short_signals

    z = pdf["zspz_z_score"].dropna()
    avg_z = float(z.mean()) if len(z) > 0 else 0.0
    max_z = float(z.max()) if len(z) > 0 else 0.0
    min_z = float(z.min()) if len(z) > 0 else 0.0

    avg_top = float(pdf["zspz_avg_top_level"].dropna().iloc[-1]) \
        if pdf["zspz_avg_top_level"].notna().any() else 2.0
    avg_bot = float(pdf["zspz_avg_bot_level"].dropna().iloc[-1]) \
        if pdf["zspz_avg_bot_level"].notna().any() else -2.0

    res_width = (
        pdf["zspz_res_band_high"] - pdf["zspz_res_band_low"]
    ).dropna()
    sup_width = (
        pdf["zspz_sup_band_high"] - pdf["zspz_sup_band_low"]
    ).dropna()

    avg_res_width = float(res_width.mean()) if len(res_width) > 0 else 0.0
    avg_sup_width = float(sup_width.mean()) if len(sup_width) > 0 else 0.0

    return {
        "total_bars": total,
        "long_signals": long_signals,
        "short_signals": short_signals,
        "total_signals": total_signals,
        "long_pct": long_signals / total if total else 0,
        "short_pct": short_signals / total if total else 0,
        "avg_z_score": avg_z,
        "max_z_score": max_z,
        "min_z_score": min_z,
        "avg_top_level": avg_top,
        "avg_bot_level": avg_bot,
        "avg_res_band_width": avg_res_width,
        "avg_sup_band_width": avg_sup_width,
    }
