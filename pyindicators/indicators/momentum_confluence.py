"""
Momentum Confluence - Multi-component oscillator for trend/reversal detection.

A comprehensive indicator combining multiple components for confluence-based
analysis including money flow, trend oscillators, divergences, and reversals.

Components:
    1. Money Flow: Measures buying/selling liquidity
    2. Thresholds: Dynamic levels showing significant activity
    3. Overflow: Detects excess buying/selling predicting reversals
    4. Trend Wave: Reactive trend-following oscillator (0-100)
    5. Real-Time Divergences: Price vs oscillator divergence
    6. Reversal Signals: High-frequency and strong reversals
    7. Confluence: Combined signal from all components
"""
from typing import Union, Dict, Tuple
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
import numpy as np

from pyindicators.exceptions import PyIndicatorException
from pyindicators.indicators.rsi import rsi
from pyindicators.indicators.stochastic_oscillator import stochastic_oscillator
from pyindicators.indicators.exponential_moving_average import ema


def momentum_confluence(
    data: Union[PdDataFrame, PlDataFrame],
    money_flow_length: int = 14,
    trend_wave_length: int = 10,
    threshold_mult: float = 1.5,
    overflow_threshold: float = 0.8,
    divergence_lookback: int = 5,
    high_column: str = 'High',
    low_column: str = 'Low',
    close_column: str = 'Close',
    volume_column: str = 'Volume',
    money_flow_column: str = 'money_flow',
    upper_threshold_column: str = 'mf_upper_threshold',
    lower_threshold_column: str = 'mf_lower_threshold',
    overflow_bullish_column: str = 'overflow_bullish',
    overflow_bearish_column: str = 'overflow_bearish',
    trend_wave_column: str = 'trend_wave',
    trend_wave_signal_column: str = 'trend_wave_signal',
    divergence_bullish_column: str = 'divergence_bullish',
    divergence_bearish_column: str = 'divergence_bearish',
    reversal_bullish_column: str = 'reversal_bullish',
    reversal_bearish_column: str = 'reversal_bearish',
    reversal_strong_bullish_column: str = 'reversal_strong_bullish',
    reversal_strong_bearish_column: str = 'reversal_strong_bearish',
    confluence_column: str = 'confluence',
    trend_column: str = 'mc_trend'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Momentum Confluence - Multi-component oscillator for trend/reversals.

    This indicator combines multiple components for confluence-based analysis:

    **Components:**
        1. **Money Flow**: Measures buying/selling liquidity (-100 to +100)
        2. **Thresholds**: Dynamic levels showing significant activity
        3. **Overflow**: Detects excess buying/selling predicting reversals
        4. **Trend Wave**: Reactive trend-following oscillator (0-100)
        5. **Real-Time Divergences**: Price vs oscillator divergence
        6. **Reversal Signals**: High-frequency (dots) and strong (arrows)
        7. **Confluence**: Combined signal strength (-100 to +100)

    Args:
        data: DataFrame with OHLCV data
        money_flow_length: Period for money flow calculation (default: 14)
        trend_wave_length: Period for trend wave oscillator (default: 10)
        threshold_mult: Multiplier for dynamic thresholds (default: 1.5)
        overflow_threshold: Threshold for overflow detection (default: 0.8)
        divergence_lookback: Bars to look back for divergences (default: 5)
        high_column: Column name for high prices
        low_column: Column name for low prices
        close_column: Column name for close prices
        volume_column: Column name for volume
        money_flow_column: Result column for money flow oscillator
        upper_threshold_column: Result column for upper threshold
        lower_threshold_column: Result column for lower threshold
        overflow_bullish_column: Result column for bullish overflow
        overflow_bearish_column: Result column for bearish overflow
        trend_wave_column: Result column for trend wave value (0-100)
        trend_wave_signal_column: Result column for trend wave trend
        divergence_bullish_column: Result column for bullish divergence
        divergence_bearish_column: Result column for bearish divergence
        reversal_bullish_column: Result column for bullish reversal
        reversal_bearish_column: Result column for bearish reversal
        reversal_strong_bullish_column: Result for strong bull reversal
        reversal_strong_bearish_column: Result for strong bear reversal
        confluence_column: Result column for confluence score (-100 to 100)
        trend_column: Result column for overall trend (1=bull, -1=bear)

    Returns:
        DataFrame with all momentum confluence components added as columns.

    Example:
        >>> import pandas as pd
        >>> from pyindicators import momentum_confluence
        >>> df = pd.DataFrame({
        ...     'High': [...], 'Low': [...], 'Close': [...], 'Volume': [...]
        ... })
        >>> result = momentum_confluence(df)
        >>> # Check for strong reversal signals with weak money flow
        >>> bullish = result[
        ...     (result['reversal_strong_bullish'] == 1) &
        ...     (result['money_flow'] > -50)
        ... ]
    """
    if isinstance(data, PdDataFrame):
        return _momentum_confluence_pandas(
            data, money_flow_length, trend_wave_length, threshold_mult,
            overflow_threshold, divergence_lookback,
            high_column, low_column, close_column, volume_column,
            money_flow_column, upper_threshold_column, lower_threshold_column,
            overflow_bullish_column, overflow_bearish_column,
            trend_wave_column, trend_wave_signal_column,
            divergence_bullish_column, divergence_bearish_column,
            reversal_bullish_column, reversal_bearish_column,
            reversal_strong_bullish_column, reversal_strong_bearish_column,
            confluence_column, trend_column
        )
    elif isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        result = _momentum_confluence_pandas(
            pdf, money_flow_length, trend_wave_length, threshold_mult,
            overflow_threshold, divergence_lookback,
            high_column, low_column, close_column, volume_column,
            money_flow_column, upper_threshold_column, lower_threshold_column,
            overflow_bullish_column, overflow_bearish_column,
            trend_wave_column, trend_wave_signal_column,
            divergence_bullish_column, divergence_bearish_column,
            reversal_bullish_column, reversal_bearish_column,
            reversal_strong_bullish_column, reversal_strong_bearish_column,
            confluence_column, trend_column
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def _momentum_confluence_pandas(
    data: PdDataFrame,
    money_flow_length: int,
    trend_wave_length: int,
    threshold_mult: float,
    overflow_threshold: float,
    divergence_lookback: int,
    high_column: str,
    low_column: str,
    close_column: str,
    volume_column: str,
    money_flow_column: str,
    upper_threshold_column: str,
    lower_threshold_column: str,
    overflow_bullish_column: str,
    overflow_bearish_column: str,
    trend_wave_column: str,
    trend_wave_signal_column: str,
    divergence_bullish_column: str,
    divergence_bearish_column: str,
    reversal_bullish_column: str,
    reversal_bearish_column: str,
    reversal_strong_bullish_column: str,
    reversal_strong_bearish_column: str,
    confluence_column: str,
    trend_column: str
) -> PdDataFrame:
    """Pandas implementation of Momentum Confluence."""
    data = data.copy()

    n = len(data)

    # Check for volume
    has_volume = volume_column in data.columns
    if has_volume:
        volume = data[volume_column].values
    else:
        volume = np.ones(n)

    high = data[high_column].values
    low = data[low_column].values
    close = data[close_column].values

    # 1. Calculate Money Flow (similar to MFI but normalized to -100 to 100)
    money_flow = _calculate_money_flow(
        high, low, close, volume, money_flow_length
    )

    # 2. Calculate Dynamic Thresholds
    upper_threshold, lower_threshold = _calculate_thresholds(
        money_flow, money_flow_length, threshold_mult
    )

    # 3. Detect Overflow (excess buying/selling)
    overflow_bullish, overflow_bearish = _detect_overflow(
        money_flow, upper_threshold, lower_threshold, overflow_threshold
    )

    # 4. Calculate Trend Wave Oscillator using existing indicators
    trend_wave, trend_wave_signal = _calculate_trend_wave(
        data, high_column, low_column, close_column, trend_wave_length
    )

    # 5. Detect Real-Time Divergences (specific implementation)
    divergence_bullish, divergence_bearish = _detect_divergences(
        close, trend_wave, divergence_lookback
    )

    # 6. Generate Reversal Signals
    (reversal_bullish, reversal_bearish,
     reversal_strong_bullish, reversal_strong_bearish) = _generate_reversals(
        money_flow, trend_wave, overflow_bullish, overflow_bearish,
        upper_threshold, lower_threshold
    )

    # 7. Calculate Confluence Score
    confluence, trend = _calculate_confluence(
        money_flow, trend_wave, overflow_bullish, overflow_bearish,
        divergence_bullish, divergence_bearish,
        reversal_bullish, reversal_bearish
    )

    # Add results to dataframe
    data[money_flow_column] = money_flow
    data[upper_threshold_column] = upper_threshold
    data[lower_threshold_column] = lower_threshold
    data[overflow_bullish_column] = overflow_bullish
    data[overflow_bearish_column] = overflow_bearish
    data[trend_wave_column] = trend_wave
    data[trend_wave_signal_column] = trend_wave_signal
    data[divergence_bullish_column] = divergence_bullish
    data[divergence_bearish_column] = divergence_bearish
    data[reversal_bullish_column] = reversal_bullish
    data[reversal_bearish_column] = reversal_bearish
    data[reversal_strong_bullish_column] = reversal_strong_bullish
    data[reversal_strong_bearish_column] = reversal_strong_bearish
    data[confluence_column] = confluence
    data[trend_column] = trend

    return data


def _calculate_money_flow(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    length: int
) -> np.ndarray:
    """
    Calculate Money Flow oscillator.

    Returns values from -100 (strong selling) to +100 (strong buying).
    Based on Money Flow Index (MFI) normalized to symmetric range.
    """
    n = len(close)

    # Typical price
    tp = (high + low + close) / 3

    # Raw money flow
    raw_mf = tp * volume

    # Positive and negative money flow
    mf_positive = np.zeros(n)
    mf_negative = np.zeros(n)

    for i in range(1, n):
        if tp[i] > tp[i - 1]:
            mf_positive[i] = raw_mf[i]
        elif tp[i] < tp[i - 1]:
            mf_negative[i] = raw_mf[i]

    # Rolling sums
    pos_sum = _rolling_sum(mf_positive, length)
    neg_sum = _rolling_sum(mf_negative, length)

    # Money Flow Ratio and Index (normalized to -100 to 100)
    money_flow = np.zeros(n)
    for i in range(length - 1, n):
        total = pos_sum[i] + neg_sum[i]
        if total > 0:
            if neg_sum[i] > 0:
                ratio = pos_sum[i] / neg_sum[i]
                mfi = 100 - (100 / (1 + ratio))
            else:
                mfi = 100
            # Normalize to -100 to 100 range
            money_flow[i] = (mfi - 50) * 2

    return money_flow


def _calculate_thresholds(
    money_flow: np.ndarray,
    length: int,
    mult: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate dynamic thresholds based on money flow volatility."""
    n = len(money_flow)
    window = length * 2

    # Calculate rolling standard deviation and mean
    mf_std = np.full(n, np.nan)
    mf_mean = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = money_flow[i - window + 1:i + 1]
        mf_mean[i] = np.mean(window_data)
        mf_std[i] = np.std(window_data, ddof=1)

    upper_threshold = mf_mean + (mf_std * mult)
    lower_threshold = mf_mean - (mf_std * mult)

    # Clip thresholds to reasonable bounds
    upper_threshold = np.clip(upper_threshold, 20, 80)
    lower_threshold = np.clip(lower_threshold, -80, -20)

    return upper_threshold, lower_threshold


def _detect_overflow(
    money_flow: np.ndarray,
    upper_threshold: np.ndarray,
    lower_threshold: np.ndarray,
    overflow_pct: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect overflow conditions (excess buying/selling).

    Overflow occurs when money flow exceeds thresholds significantly,
    indicating potential reversal due to exhaustion.
    """
    n = len(money_flow)
    overflow_bullish = np.zeros(n, dtype=int)
    overflow_bearish = np.zeros(n, dtype=int)

    for i in range(1, n):
        if not np.isnan(upper_threshold[i]):
            # Bullish overflow: money flow significantly above threshold
            if money_flow[i] > upper_threshold[i] * (1 + overflow_pct):
                overflow_bullish[i] = 1

            # Bearish overflow: money flow significantly below threshold
            if money_flow[i] < lower_threshold[i] * (1 + overflow_pct):
                overflow_bearish[i] = 1

    return overflow_bullish, overflow_bearish


def _calculate_trend_wave(
    data: PdDataFrame,
    high_column: str,
    low_column: str,
    close_column: str,
    length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Trend Wave oscillator using existing RSI and Stochastic.

    Combines RSI and Stochastic for a reactive trend-following oscillator.
    Returns values 0-100 and trend signal (1/-1/0).
    """
    n = len(data)
    work_df = data.copy()

    # Use existing RSI indicator
    work_df = rsi(
        work_df,
        source_column=close_column,
        period=length,
        result_column='_tw_rsi'
    )

    # Use existing Stochastic indicator
    work_df = stochastic_oscillator(
        work_df,
        high_column=high_column,
        low_column=low_column,
        close_column=close_column,
        k_period=length,
        k_slowing=3,
        d_period=3,
        result_column='_tw_stoch'
    )

    rsi_vals = work_df['_tw_rsi'].fillna(50).values
    stoch_vals = work_df['_tw_stoch_%K'].fillna(50).values

    # Combine RSI and Stochastic (weighted average)
    trend_wave_raw = (rsi_vals * 0.5 + stoch_vals * 0.5)

    # Smooth with EMA using existing indicator
    work_df['_tw_raw'] = trend_wave_raw
    work_df = ema(
        work_df, source_column='_tw_raw', period=3, result_column='_tw_smooth'
    )
    trend_wave = work_df['_tw_smooth'].values

    # Generate signal
    trend_wave_signal = np.zeros(n, dtype=int)
    for i in range(1, n):
        if trend_wave[i] > 50 and trend_wave[i] > trend_wave[i - 1]:
            trend_wave_signal[i] = 1
        elif trend_wave[i] < 50 and trend_wave[i] < trend_wave[i - 1]:
            trend_wave_signal[i] = -1

    return trend_wave, trend_wave_signal


def _detect_divergences(
    close: np.ndarray,
    oscillator: np.ndarray,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect real-time divergences between price and oscillator.

    This is a specific implementation for momentum confluence that detects:
    - Bullish divergence: Price makes lower low, oscillator makes higher low
    - Bearish divergence: Price makes higher high, oscillator makes lower high
    """
    n = len(close)
    divergence_bullish = np.zeros(n, dtype=int)
    divergence_bearish = np.zeros(n, dtype=int)

    for i in range(lookback * 2, n):
        # Check for bullish divergence (at potential lows)
        if oscillator[i] < 40:
            price_prev_low = np.min(close[i - lookback * 2:i - lookback])
            price_curr_low = np.min(close[i - lookback:i + 1])

            osc_prev_low = np.min(oscillator[i - lookback * 2:i - lookback])
            osc_curr_low = np.min(oscillator[i - lookback:i + 1])

            if price_curr_low < price_prev_low and osc_curr_low > osc_prev_low:
                divergence_bullish[i] = 1

        # Check for bearish divergence (at potential highs)
        if oscillator[i] > 60:
            price_prev_high = np.max(close[i - lookback * 2:i - lookback])
            price_curr_high = np.max(close[i - lookback:i + 1])

            osc_prev_high = np.max(oscillator[i - lookback * 2:i - lookback])
            osc_curr_high = np.max(oscillator[i - lookback:i + 1])

            if (price_curr_high > price_prev_high and
                    osc_curr_high < osc_prev_high):
                divergence_bearish[i] = 1

    return divergence_bullish, divergence_bearish


def _generate_reversals(
    money_flow: np.ndarray,
    trend_wave: np.ndarray,
    overflow_bullish: np.ndarray,
    overflow_bearish: np.ndarray,
    upper_threshold: np.ndarray,
    lower_threshold: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate reversal signals.

    High-frequency reversals: Small dots indicating potential local reversals
    Strong reversals: Large arrows indicating significant reversal potential
    """
    n = len(money_flow)
    reversal_bullish = np.zeros(n, dtype=int)
    reversal_bearish = np.zeros(n, dtype=int)
    reversal_strong_bullish = np.zeros(n, dtype=int)
    reversal_strong_bearish = np.zeros(n, dtype=int)

    for i in range(2, n):
        # High-frequency bullish reversal
        # Trend wave turning up from low levels
        if (trend_wave[i] < 30 and
                trend_wave[i] > trend_wave[i - 1] and
                trend_wave[i - 1] <= trend_wave[i - 2]):
            reversal_bullish[i] = 1

        # High-frequency bearish reversal
        # Trend wave turning down from high levels
        if (trend_wave[i] > 70 and
                trend_wave[i] < trend_wave[i - 1] and
                trend_wave[i - 1] >= trend_wave[i - 2]):
            reversal_bearish[i] = 1

        # Strong bullish reversal
        # Overflow + money flow not near threshold (weak selling)
        if overflow_bearish[i] == 1 or overflow_bearish[i - 1] == 1:
            if not np.isnan(lower_threshold[i]):
                if money_flow[i] > lower_threshold[i] * 0.5:
                    reversal_strong_bullish[i] = 1

        # Strong bearish reversal
        # Overflow + money flow not near threshold (weak buying)
        if overflow_bullish[i] == 1 or overflow_bullish[i - 1] == 1:
            if not np.isnan(upper_threshold[i]):
                if money_flow[i] < upper_threshold[i] * 0.5:
                    reversal_strong_bearish[i] = 1

    return (reversal_bullish, reversal_bearish,
            reversal_strong_bullish, reversal_strong_bearish)


def _calculate_confluence(
    money_flow: np.ndarray,
    trend_wave: np.ndarray,
    overflow_bullish: np.ndarray,
    overflow_bearish: np.ndarray,
    divergence_bullish: np.ndarray,
    divergence_bearish: np.ndarray,
    reversal_bullish: np.ndarray,
    reversal_bearish: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confluence score combining all components.

    Returns:
        confluence: Score from -100 (strong bearish) to +100 (strong bullish)
        trend: Overall trend direction (1=bull, -1=bear, 0=neutral)
    """
    n = len(money_flow)
    confluence = np.zeros(n)
    trend = np.zeros(n, dtype=int)

    for i in range(n):
        score = 0.0

        # Money flow contribution (weight: 30%)
        score += money_flow[i] * 0.3

        # Trend wave contribution (weight: 30%)
        # Convert 0-100 to -50 to +50
        tw_contrib = (trend_wave[i] - 50)
        score += tw_contrib * 0.6

        # Overflow contribution (weight: 15%)
        # Overflow suggests reversal
        if overflow_bullish[i]:
            score -= 15
        if overflow_bearish[i]:
            score += 15

        # Divergence contribution (weight: 15%)
        if divergence_bullish[i]:
            score += 15
        if divergence_bearish[i]:
            score -= 15

        # Reversal signals contribution (weight: 10%)
        if reversal_bullish[i]:
            score += 10
        if reversal_bearish[i]:
            score -= 10

        # Clip to -100 to 100
        confluence[i] = np.clip(score, -100, 100)

        # Determine trend
        if confluence[i] > 20:
            trend[i] = 1
        elif confluence[i] < -20:
            trend[i] = -1
        else:
            trend[i] = 0

    return confluence, trend


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling sum."""
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.sum(arr[i - window + 1:i + 1])
    return result


def momentum_confluence_signal(
    data: Union[PdDataFrame, PlDataFrame],
    confluence_column: str = 'confluence',
    reversal_strong_bullish_column: str = 'reversal_strong_bullish',
    reversal_strong_bearish_column: str = 'reversal_strong_bearish',
    signal_column: str = 'mc_signal'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate trading signals from Momentum Confluence.

    Signal values:
        - 2: Strong bullish reversal signal
        - 1: Bullish confluence
        - -1: Bearish confluence
        - -2: Strong bearish reversal signal
        - 0: Neutral

    Args:
        data: DataFrame with momentum confluence columns
        confluence_column: Column for confluence score
        reversal_strong_bullish_column: Column for strong bullish reversals
        reversal_strong_bearish_column: Column for strong bearish reversals
        signal_column: Result column name

    Returns:
        DataFrame with added signal column

    Example:
        >>> df = momentum_confluence(df)
        >>> df = momentum_confluence_signal(df)
        >>> buy_signals = df[df['mc_signal'] == 2]
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        confluence = data[confluence_column].values
        strong_bull = data[reversal_strong_bullish_column].values
        strong_bear = data[reversal_strong_bearish_column].values

        signal = np.zeros(len(data), dtype=int)
        signal = np.where(strong_bull == 1, 2, signal)
        signal = np.where(strong_bear == 1, -2, signal)
        signal = np.where((signal == 0) & (confluence > 30), 1, signal)
        signal = np.where((signal == 0) & (confluence < -30), -1, signal)

        data[signal_column] = signal
        return data
    elif isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        result = momentum_confluence_signal(
            pdf, confluence_column, reversal_strong_bullish_column,
            reversal_strong_bearish_column, signal_column
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def get_momentum_confluence_stats(
    data: Union[PdDataFrame, PlDataFrame],
    reversal_bullish_column: str = 'reversal_bullish',
    reversal_bearish_column: str = 'reversal_bearish',
    reversal_strong_bullish_column: str = 'reversal_strong_bullish',
    reversal_strong_bearish_column: str = 'reversal_strong_bearish',
    divergence_bullish_column: str = 'divergence_bullish',
    divergence_bearish_column: str = 'divergence_bearish',
    overflow_bullish_column: str = 'overflow_bullish',
    overflow_bearish_column: str = 'overflow_bearish'
) -> Dict[str, int]:
    """
    Calculate statistics for Momentum Confluence signals.

    Args:
        data: DataFrame with momentum confluence columns
        reversal_bullish_column: Column for bullish reversals
        reversal_bearish_column: Column for bearish reversals
        reversal_strong_bullish_column: Column for strong bullish reversals
        reversal_strong_bearish_column: Column for strong bearish reversals
        divergence_bullish_column: Column for bullish divergences
        divergence_bearish_column: Column for bearish divergences
        overflow_bullish_column: Column for bullish overflow
        overflow_bearish_column: Column for bearish overflow

    Returns:
        Dictionary with signal counts for all components.

    Example:
        >>> df = momentum_confluence(df)
        >>> stats = get_momentum_confluence_stats(df)
        >>> print(f"Bull reversals: {stats['strong_reversal_bullish_count']}")
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    bull_rev = reversal_strong_bullish_column
    bear_rev = reversal_strong_bearish_column
    div_bull = divergence_bullish_column
    div_bear = divergence_bearish_column

    return {
        'reversal_bullish_count': int(data[reversal_bullish_column].sum()),
        'reversal_bearish_count': int(data[reversal_bearish_column].sum()),
        'strong_reversal_bullish_count': int(data[bull_rev].sum()),
        'strong_reversal_bearish_count': int(data[bear_rev].sum()),
        'divergence_bullish_count': int(data[div_bull].sum()),
        'divergence_bearish_count': int(data[div_bear].sum()),
        'overflow_bullish_count': int(data[overflow_bullish_column].sum()),
        'overflow_bearish_count': int(data[overflow_bearish_column].sum())
    }
