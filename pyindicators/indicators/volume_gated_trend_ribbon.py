"""
Volume-Gated Trend Ribbon Indicator

Based on the Volume-Gated Trend Ribbon [QuantAlgo] concept from TradingView.
Uses volume analysis to gate price data before applying moving average
calculations for trend detection.

The indicator filters price updates by volume significance, computing
fast and slow moving averages on the filtered source. The ribbon between
the MAs (with mid-point interpolations) provides visual trend context.
"""
from typing import Union, Optional
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame


# ── Moving Average helpers (operate on numpy arrays) ─────────────────

def _sma(src: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    out = np.full(len(src), np.nan)
    for i in range(period - 1, len(src)):
        out[i] = np.mean(src[i - period + 1:i + 1])
    return out


def _ema(src: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2.0 / (period + 1)
    out = np.full(len(src), np.nan)
    out[0] = src[0]
    for i in range(1, len(src)):
        out[i] = alpha * src[i] + (1 - alpha) * out[i - 1]
    return out


def _wma(src: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1, dtype=float)
    w_sum = weights.sum()
    out = np.full(len(src), np.nan)
    for i in range(period - 1, len(src)):
        out[i] = np.dot(src[i - period + 1:i + 1], weights) / w_sum
    return out


def _rma(src: np.ndarray, period: int) -> np.ndarray:
    """Wilder's Smoothing (RMA / SMMA)."""
    alpha = 1.0 / period
    out = np.full(len(src), np.nan)
    # Seed with SMA
    if len(src) >= period:
        out[period - 1] = np.mean(src[:period])
        for i in range(period, len(src)):
            out[i] = alpha * src[i] + (1 - alpha) * out[i - 1]
    return out


def _hma(src: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average."""
    half = max(period // 2, 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    wma_half = _wma(src, half)
    wma_full = _wma(src, period)
    diff = 2 * wma_half - wma_full
    # Replace NaN with 0 for final WMA pass then restore NaN positions
    valid = ~np.isnan(diff)
    if valid.sum() < sqrt_p:
        return np.full(len(src), np.nan)
    return _wma(np.where(valid, diff, 0), sqrt_p)


def _vwma(src: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume Weighted Moving Average."""
    out = np.full(len(src), np.nan)
    pv = src * volume
    for i in range(period - 1, len(src)):
        vol_sum = np.sum(volume[i - period + 1:i + 1])
        if vol_sum > 0:
            out[i] = np.sum(pv[i - period + 1:i + 1]) / vol_sum
        else:
            out[i] = src[i]
    return out


def _dema(src: np.ndarray, period: int) -> np.ndarray:
    """Double Exponential Moving Average."""
    e1 = _ema(src, period)
    e2 = _ema(e1, period)
    return 2 * e1 - e2


def _tema(src: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average."""
    e1 = _ema(src, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    return 3 * (e1 - e2) + e3


def _lsma(src: np.ndarray, period: int) -> np.ndarray:
    """Least Squares Moving Average (linear regression value)."""
    out = np.full(len(src), np.nan)
    x = np.arange(period, dtype=float)
    for i in range(period - 1, len(src)):
        y = src[i - period + 1:i + 1]
        slope = (
            (period * np.dot(x, y) - x.sum() * y.sum())
            / (period * np.dot(x, x) - x.sum() ** 2)
        )
        intercept = (y.sum() - slope * x.sum()) / period
        out[i] = intercept + slope * (period - 1)
    return out


def _kama(src: np.ndarray, period: int) -> np.ndarray:
    """Kaufman Adaptive Moving Average."""
    fast_sc = 2.0 / 3.0
    slow_sc = 2.0 / 31.0
    out = np.full(len(src), np.nan)
    out[0] = src[0]
    for i in range(1, len(src)):
        if i < period:
            out[i] = src[i]
            continue
        change = abs(src[i] - src[i - period])
        volatility = np.sum(np.abs(np.diff(src[i - period:i + 1])))
        er = change / volatility if volatility != 0 else 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (src[i] - out[i - 1])
    return out


def _zlema(src: np.ndarray, period: int) -> np.ndarray:
    """Zero-Lag Exponential Moving Average."""
    lag = (period - 1) // 2
    adjusted = np.copy(src)
    for i in range(lag, len(src)):
        adjusted[i] = src[i] + (src[i] - src[i - lag])
    return _ema(adjusted, period)


def _t3(src: np.ndarray, period: int, factor: float = 0.7) -> np.ndarray:
    """Tillson T3 Moving Average."""
    def _gd(s, p, f):
        e1 = _ema(s, p)
        e2 = _ema(e1, p)
        return e1 * (1 + f) - e2 * f
    return _gd(_gd(_gd(src, period, factor), period, factor), period, factor)


def _vidya(src: np.ndarray, period: int) -> np.ndarray:
    """Variable Index Dynamic Average."""
    alpha = 2.0 / (period + 1)
    out = np.full(len(src), np.nan)
    out[0] = src[0]
    for i in range(1, len(src)):
        if i < period:
            out[i] = src[i]
            continue
        mom = np.diff(src[max(0, i - period):i + 1])
        up_sum = np.sum(np.maximum(mom, 0))
        dn_sum = np.sum(np.maximum(-mom, 0))
        total = up_sum + dn_sum
        cmo = abs((up_sum - dn_sum) / total) if total != 0 else 0
        out[i] = src[i] * alpha * cmo + out[i - 1] * (1 - alpha * cmo)
    return out


def _alma(src: np.ndarray, period: int,
          offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    """Arnaud Legoux Moving Average."""
    m = offset * (period - 1)
    s = period / sigma
    weights = np.exp(-((np.arange(period) - m) ** 2) / (2 * s * s))
    w_sum = weights.sum()
    out = np.full(len(src), np.nan)
    for i in range(period - 1, len(src)):
        out[i] = np.dot(src[i - period + 1:i + 1], weights) / w_sum
    return out


def _calc_ma(
    src: np.ndarray,
    period: int,
    ma_type: str,
    volume: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dispatch to the appropriate MA calculation."""
    ma_type = ma_type.upper()
    dispatch = {
        'SMA': lambda: _sma(src, period),
        'EMA': lambda: _ema(src, period),
        'WMA': lambda: _wma(src, period),
        'RMA': lambda: _rma(src, period),
        'SMMA': lambda: _rma(src, period),
        'HMA': lambda: _hma(src, period),
        'DEMA': lambda: _dema(src, period),
        'TEMA': lambda: _tema(src, period),
        'ALMA': lambda: _alma(src, period),
        'LSMA': lambda: _lsma(src, period),
        'KAMA': lambda: _kama(src, period),
        'ZLEMA': lambda: _zlema(src, period),
        'T3': lambda: _t3(src, period),
        'VIDYA': lambda: _vidya(src, period),
        'VWMA': lambda: (
            _vwma(src, volume, period)
            if volume is not None
            else _ema(src, period)
        ),
    }
    fn = dispatch.get(ma_type)
    if fn is None:
        raise ValueError(
            f"Unsupported MA type '{ma_type}'. "
            f"Supported: {sorted(dispatch.keys())}"
        )
    return fn()


# ── Main indicator ───────────────────────────────────────────────────

VALID_MA_TYPES = [
    'SMA', 'EMA', 'WMA', 'RMA', 'HMA', 'VWMA',
    'DEMA', 'TEMA', 'ALMA', 'LSMA', 'SMMA',
    'KAMA', 'ZLEMA', 'T3', 'VIDYA',
]


def volume_gated_trend_ribbon(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str = 'Close',
    volume_column: str = 'Volume',
    vol_mult: float = 1.0,
    vol_period: int = 50,
    ma_type: str = 'EMA',
    fast_length: int = 15,
    slow_length: int = 30,
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the Volume-Gated Trend Ribbon indicator.

    Filters price updates by volume significance, then computes fast / slow
    moving averages on the gated source to detect trend direction.  The
    ribbon is formed by the fast MA, slow MA, and two interpolated mid-MAs.

    Parameters:
        data: DataFrame with OHLCV data.
        source_column: Column to use as the price source (default: 'Close').
        volume_column: Column containing volume data (default: 'Volume').
        vol_mult: Multiplier for average volume threshold (default: 1.0).
                  Lower = more bars qualify, higher = stricter filter.
        vol_period: Lookback for average volume baseline (default: 50).
        ma_type: Moving average type. One of: SMA, EMA, WMA, RMA, HMA,
                 VWMA, DEMA, TEMA, ALMA, LSMA, SMMA, KAMA, ZLEMA, T3,
                 VIDYA. (default: 'EMA')
        fast_length: Period for the fast (inner) MA (default: 15).
        slow_length: Period for the slow (outer) MA (default: 30).

    Returns:
        DataFrame with added columns:
        - vgtr_fast: Fast moving average
        - vgtr_mid_fast: Interpolated mid-fast MA (0.67*fast + 0.33*slow)
        - vgtr_mid_slow: Interpolated mid-slow MA (0.33*fast + 0.67*slow)
        - vgtr_slow: Slow moving average
        - vgtr_trend: Trend state (1=bullish, -1=bearish)
        - vgtr_signal: 1=buy signal, -1=sell signal, 0=no signal
    """
    is_polars = isinstance(data, PlDataFrame)
    if is_polars:
        df = data.to_pandas()
    else:
        df = data.copy()

    ma_type_upper = ma_type.upper()
    if ma_type_upper not in VALID_MA_TYPES:
        raise ValueError(
            f"Unsupported MA type '{ma_type}'. Supported: {VALID_MA_TYPES}"
        )

    src = df[source_column].values.astype(float)
    volume = df[volume_column].values.astype(float)
    n = len(src)

    # ── Volume gating ────────────────────────────────────────────────
    avg_vol = _sma(volume, vol_period)
    gated_close = np.copy(src)

    for i in range(1, n):
        if np.isnan(avg_vol[i]):
            # Before we have enough volume history, keep original price
            continue
        if volume[i] >= avg_vol[i] * vol_mult:
            gated_close[i] = src[i]
        else:
            gated_close[i] = gated_close[i - 1]

    fast_period = max(fast_length, 4) \
        if ma_type_upper == 'HMA' else fast_length
    slow_period = max(slow_length, fast_period + 1)

    # ── Calculate MAs ────────────────────────────────────────────────
    # For VWMA uses raw (un-gated) source with volume
    if ma_type_upper == 'VWMA':
        fast_ma = _vwma(src, volume, fast_period)
        slow_ma = _vwma(src, volume, slow_period)
    else:
        fast_ma = _calc_ma(gated_close, fast_period, ma_type_upper)
        slow_ma = _calc_ma(gated_close, slow_period, ma_type_upper)

    # Interpolated mid-MAs for the ribbon
    mid_fast = fast_ma * 0.67 + slow_ma * 0.33
    mid_slow = fast_ma * 0.33 + slow_ma * 0.67

    # ── Trend detection ──────────────────────────────────────────────
    trend_state = np.zeros(n, dtype=int)
    signal = np.zeros(n, dtype=int)

    for i in range(n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
            continue
        trend_state[i] = 1 if fast_ma[i] > slow_ma[i] else -1

    # Signals on trend change
    for i in range(1, n):
        if trend_state[i] == 1 and trend_state[i - 1] != 1:
            signal[i] = 1   # Bullish flip
        elif trend_state[i] == -1 and trend_state[i - 1] != -1:
            signal[i] = -1  # Bearish flip

    # ── Write results ────────────────────────────────────────────────
    df['vgtr_fast'] = fast_ma
    df['vgtr_mid_fast'] = mid_fast
    df['vgtr_mid_slow'] = mid_slow
    df['vgtr_slow'] = slow_ma
    df['vgtr_trend'] = trend_state
    df['vgtr_signal'] = signal

    if is_polars:
        import polars as pl
        return pl.from_pandas(df)

    return df
