"""
Zero-Lag EMA Envelope Indicator

Combines a Zero-Lag Exponential Moving Average (ZLEMA) with ATR-based
bands and multi-bar swing confirmation to produce trend signals.
"""
from typing import Union

import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl

from pyindicators.exceptions import PyIndicatorException


def zero_lag_ema_envelope(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str = 'Close',
    length: int = 200,
    mult: float = 2.0,
    atr_length: int = 21,
    confirm_bars: int = 2,
    upper_column: str = 'zlema_upper',
    lower_column: str = 'zlema_lower',
    middle_column: str = 'zlema_middle',
    trend_column: str = 'zlema_trend',
    signal_column: str = 'zlema_signal',
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the Zero-Lag EMA Envelope indicator.

    Uses a Zero-Lag EMA (ZLEMA) as the centre line with ATR-based
    upper and lower bands. A trend state is determined by requiring
    multiple consecutive closes beyond a band while the ZLEMA slope
    confirms the direction.

    Calculation:
        - lag = floor((length - 1) / 2)
        - compensated = close + (close - close[lag])
        - ZLEMA = EMA(compensated, length)
        - ATR = Average True Range over atr_length bars
        - Upper = ZLEMA + ATR * mult
        - Lower = ZLEMA - ATR * mult
        - Bull confirmation: close > Upper for confirm_bars
          consecutive bars AND ZLEMA rising
        - Bear confirmation: close < Lower for confirm_bars
          consecutive bars AND ZLEMA falling
        - Trend: 1 (bullish), -1 (bearish), 0 (neutral)
        - Signal: 1 on bull flip, -1 on bear flip, 0 otherwise

    Args:
        data: pandas or polars DataFrame with OHLC price data.
        source_column: Column name for the source prices
            (default: 'Close').
        length: Period for the Zero-Lag EMA (default: 200).
        mult: Multiplier for ATR band width (default: 2.0).
        atr_length: Period for the ATR calculation (default: 21).
        confirm_bars: Number of consecutive bars that must close
            beyond the band to confirm a trend change (default: 2,
            range 1-3).
        upper_column: Result column name for the upper band
            (default: 'zlema_upper').
        lower_column: Result column name for the lower band
            (default: 'zlema_lower').
        middle_column: Result column name for the ZLEMA line
            (default: 'zlema_middle').
        trend_column: Result column name for the trend state
            (default: 'zlema_trend').  1 = bullish, -1 = bearish,
            0 = neutral.
        signal_column: Result column name for the trend flip signal
            (default: 'zlema_signal'). 1 on bull flip, -1 on bear
            flip, 0 otherwise.

    Returns:
        DataFrame with added columns:
            - {upper_column}: Upper boundary of the envelope
            - {lower_column}: Lower boundary of the envelope
            - {middle_column}: Zero-Lag EMA centre line
            - {trend_column}: Trend state (1 / -1 / 0)
            - {signal_column}: Trend flip signals (1 / -1 / 0)

    Example:
        >>> import pandas as pd
        >>> from pyindicators import zero_lag_ema_envelope
        >>> df = pd.DataFrame({
        ...     'High':  [105, 107, 106, 108, 110, 109, 111, 113, 112, 114],
        ...     'Low':   [95, 97, 96, 98, 100, 99, 101, 103, 102, 104],
        ...     'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
        ... })
        >>> result = zero_lag_ema_envelope(df, length=5, atr_length=3)
        >>> print(result[['zlema_upper', 'zlema_middle', 'zlema_lower',
        ...               'zlema_trend']].tail())
    """
    if length < 1:
        raise PyIndicatorException("Length must be at least 1")

    if mult < 0:
        raise PyIndicatorException("Multiplier must be non-negative")

    if atr_length < 1:
        raise PyIndicatorException("ATR length must be at least 1")

    if confirm_bars < 1 or confirm_bars > 3:
        raise PyIndicatorException(
            "Confirm bars must be between 1 and 3"
        )

    for col in [source_column, 'High', 'Low']:
        if col not in data.columns:
            raise PyIndicatorException(
                f"The column '{col}' does not exist in the DataFrame."
            )

    is_polars = isinstance(data, PlDataFrame)

    # --- Extract numpy arrays ---------------------------------------------
    if is_polars:
        close = data[source_column].to_numpy().astype(float)
        high = data['High'].to_numpy().astype(float)
        low = data['Low'].to_numpy().astype(float)
    else:
        close = data[source_column].values.astype(float)
        high = data['High'].values.astype(float)
        low = data['Low'].values.astype(float)

    n = len(close)

    # --- Zero-Lag EMA (ZLEMA) ---------------------------------------------
    lag = max(int(np.floor((length - 1) / 2)), 0)

    # Build the lag-compensated series: close + (close - close[lag])
    compensated = np.empty(n)

    for i in range(n):
        if lag > 0 and i >= lag:
            compensated[i] = close[i] + (close[i] - close[i - lag])
        else:
            compensated[i] = close[i]

    # Calculate EMA of the compensated series
    alpha = 2.0 / (length + 1)
    zlema = np.empty(n)
    zlema[0] = compensated[0]

    for i in range(1, n):
        zlema[i] = compensated[i] * alpha + zlema[i - 1] * (1.0 - alpha)

    # --- ATR ---------------------------------------------------------------
    # True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Rolling mean ATR
    atr_vals = np.full(n, np.nan)

    for i in range(n):
        window = min(i + 1, atr_length)
        atr_vals[i] = np.mean(tr[i - window + 1: i + 1])

    # --- Bands -------------------------------------------------------------
    upper = zlema + atr_vals * mult
    lower = zlema - atr_vals * mult

    # --- Trend confirmation ------------------------------------------------
    trend = np.zeros(n, dtype=int)
    signal = np.zeros(n, dtype=int)

    for t in range(n):
        # Check bull confirmation: close > upper for confirm_bars
        # consecutive bars AND zlema rising
        bull = True

        for k in range(confirm_bars):
            idx = t - k

            if idx < 0 or np.isnan(upper[idx]):
                bull = False
                break

            if close[idx] <= upper[idx]:
                bull = False
                break

        if bull and t >= 1:
            bull = zlema[t] > zlema[t - 1]

        # Check bear confirmation: close < lower for confirm_bars
        # consecutive bars AND zlema falling
        bear = True

        for k in range(confirm_bars):
            idx = t - k

            if idx < 0 or np.isnan(lower[idx]):
                bear = False
                break

            if close[idx] >= lower[idx]:
                bear = False
                break

        if bear and t >= 1:
            bear = zlema[t] < zlema[t - 1]

        # Update trend state
        if bull:
            trend[t] = 1
        elif bear:
            trend[t] = -1
        else:
            trend[t] = trend[t - 1] if t > 0 else 0

        # Signal on trend flip
        prev_trend = trend[t - 1] if t > 0 else 0

        if trend[t] != prev_trend and trend[t] != 0:
            signal[t] = trend[t]

    # --- Write results back ------------------------------------------------
    if is_polars:
        data = data.with_columns([
            pl.Series(name=middle_column, values=zlema),
            pl.Series(name=upper_column, values=upper),
            pl.Series(name=lower_column, values=lower),
            pl.Series(name=trend_column, values=trend.tolist()),
            pl.Series(name=signal_column, values=signal.tolist()),
        ])
    else:
        data[middle_column] = zlema
        data[upper_column] = upper
        data[lower_column] = lower
        data[trend_column] = trend
        data[signal_column] = signal

    return data
