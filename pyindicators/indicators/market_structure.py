from typing import Union, List, Dict
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
import numpy as np
from pyindicators.exceptions import PyIndicatorException


def market_structure_break(
    data: Union[PdDataFrame, PlDataFrame],
    pivot_length: int = 7,
    momentum_zscore_threshold: float = 0.5,
    high_column: str = 'High',
    low_column: str = 'Low',
    close_column: str = 'Close',
    volume_column: str = 'Volume',
    msb_bullish_column: str = 'msb_bullish',
    msb_bearish_column: str = 'msb_bearish',
    last_pivot_high_column: str = 'last_pivot_high',
    last_pivot_low_column: str = 'last_pivot_low',
    momentum_z_column: str = 'momentum_z'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect Market Structure Breaks (MSB) in price data.

    A Market Structure Break occurs when price breaks through a significant
    pivot point with sufficient momentum. This is often used to identify
    potential trend changes or continuation patterns.

    **Bullish MSB:**
        Detected when price closes above the last pivot high AND
        momentum z-score is above the threshold.

    **Bearish MSB:**
        Detected when price closes below the last pivot low AND
        momentum z-score is below the negative threshold.

    Args:
        data: pandas or polars DataFrame with OHLCV price data
        pivot_length: Lookback period for pivot detection (default: 7)
        momentum_zscore_threshold: Z-score threshold for momentum
            confirmation (default: 0.5)
        high_column: Column name for high prices (default: 'High')
        low_column: Column name for low prices (default: 'Low')
        close_column: Column name for close prices (default: 'Close')
        volume_column: Column name for volume (default: 'Volume')
        msb_bullish_column: Result column for bullish MSB signal
            (default: 'msb_bullish')
        msb_bearish_column: Result column for bearish MSB signal
            (default: 'msb_bearish')
        last_pivot_high_column: Result column for last pivot high
            (default: 'last_pivot_high')
        last_pivot_low_column: Result column for last pivot low
            (default: 'last_pivot_low')
        momentum_z_column: Result column for momentum z-score
            (default: 'momentum_z')

    Returns:
        DataFrame with added columns:
            - {msb_bullish_column}: 1 when bullish MSB, 0 otherwise
            - {msb_bearish_column}: 1 when bearish MSB, 0 otherwise
            - {last_pivot_high_column}: Most recent pivot high price
            - {last_pivot_low_column}: Most recent pivot low price
            - {momentum_z_column}: Momentum z-score value

    Example:
        >>> import pandas as pd
        >>> from pyindicators import market_structure_break
        >>> df = pd.DataFrame({
        ...     'Open': [...],
        ...     'High': [...],
        ...     'Low': [...],
        ...     'Close': [...],
        ...     'Volume': [...]
        ... })
        >>> result = market_structure_break(df, pivot_length=7)
    """
    if isinstance(data, PdDataFrame):
        return _msb_pandas(
            data, pivot_length, momentum_zscore_threshold,
            high_column, low_column, close_column, volume_column,
            msb_bullish_column, msb_bearish_column,
            last_pivot_high_column, last_pivot_low_column,
            momentum_z_column
        )
    elif isinstance(data, PlDataFrame):
        return _msb_polars(
            data, pivot_length, momentum_zscore_threshold,
            high_column, low_column, close_column, volume_column,
            msb_bullish_column, msb_bearish_column,
            last_pivot_high_column, last_pivot_low_column,
            momentum_z_column
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def _msb_pandas(
    data: PdDataFrame,
    pivot_length: int,
    momentum_zscore_threshold: float,
    high_column: str,
    low_column: str,
    close_column: str,
    volume_column: str,
    msb_bullish_column: str,
    msb_bearish_column: str,
    last_pivot_high_column: str,
    last_pivot_low_column: str,
    momentum_z_column: str
) -> PdDataFrame:
    """Pandas implementation of market structure break."""
    high = data[high_column].values
    low = data[low_column].values
    close = data[close_column].values
    n = len(data)

    # Calculate momentum z-score
    price_change = np.diff(close, prepend=close[0])
    avg_change = _rolling_mean(price_change, 50)
    std_change = _rolling_std(price_change, 50)
    momentum_z = np.where(std_change > 0,
                          (price_change - avg_change) / std_change, 0)

    # Detect pivot highs and lows
    pivot_highs, pivot_lows = _detect_pivots(high, low, pivot_length)

    # Initialize result arrays
    msb_bullish = np.zeros(n, dtype=int)
    msb_bearish = np.zeros(n, dtype=int)
    last_pivot_high = np.full(n, np.nan)
    last_pivot_low = np.full(n, np.nan)
    last_pivot_high_idx = np.full(n, -1, dtype=int)
    last_pivot_low_idx = np.full(n, -1, dtype=int)

    # Track last pivot values
    current_pivot_high = np.nan
    current_pivot_low = np.nan
    current_pivot_high_idx = -1
    current_pivot_low_idx = -1
    pivot_high_crossed = True
    pivot_low_crossed = True

    for i in range(pivot_length, n):
        # Check for new pivot high
        if pivot_highs[i]:
            current_pivot_high = high[i]
            current_pivot_high_idx = i
            pivot_high_crossed = False

        # Check for new pivot low
        if pivot_lows[i]:
            current_pivot_low = low[i]
            current_pivot_low_idx = i
            pivot_low_crossed = False

        # Record current pivot values
        last_pivot_high[i] = current_pivot_high
        last_pivot_low[i] = current_pivot_low
        last_pivot_high_idx[i] = current_pivot_high_idx
        last_pivot_low_idx[i] = current_pivot_low_idx

        # Check for bullish MSB
        if (not pivot_high_crossed and
                not np.isnan(current_pivot_high) and
                close[i] > current_pivot_high and
                close[i - 1] <= current_pivot_high and
                momentum_z[i] > momentum_zscore_threshold):
            msb_bullish[i] = 1
            pivot_high_crossed = True

        # Check for bearish MSB
        if (not pivot_low_crossed and
                not np.isnan(current_pivot_low) and
                close[i] < current_pivot_low and
                close[i - 1] >= current_pivot_low and
                momentum_z[i] < -momentum_zscore_threshold):
            msb_bearish[i] = 1
            pivot_low_crossed = True

    # Add results to dataframe
    data[msb_bullish_column] = msb_bullish
    data[msb_bearish_column] = msb_bearish
    data[last_pivot_high_column] = last_pivot_high
    data[last_pivot_low_column] = last_pivot_low
    data[momentum_z_column] = momentum_z

    return data


def _msb_polars(
    data: PlDataFrame,
    pivot_length: int,
    momentum_zscore_threshold: float,
    high_column: str,
    low_column: str,
    close_column: str,
    volume_column: str,
    msb_bullish_column: str,
    msb_bearish_column: str,
    last_pivot_high_column: str,
    last_pivot_low_column: str,
    momentum_z_column: str
) -> PlDataFrame:
    """Polars implementation of market structure break."""
    pdf = data.to_pandas()
    result = _msb_pandas(
        pdf, pivot_length, momentum_zscore_threshold,
        high_column, low_column, close_column, volume_column,
        msb_bullish_column, msb_bearish_column,
        last_pivot_high_column, last_pivot_low_column,
        momentum_z_column
    )
    return pl.from_pandas(result)


def _detect_pivots(
    high: np.ndarray,
    low: np.ndarray,
    length: int
) -> tuple:
    """Detect pivot highs and lows."""
    n = len(high)
    pivot_highs = np.zeros(n, dtype=bool)
    pivot_lows = np.zeros(n, dtype=bool)

    for i in range(length, n - length):
        # Check for pivot high
        is_pivot_high = True
        for j in range(i - length, i + length + 1):
            if j != i and high[j] > high[i]:
                is_pivot_high = False
                break
        if is_pivot_high:
            pivot_highs[i] = True

        # Check for pivot low
        is_pivot_low = True
        for j in range(i - length, i + length + 1):
            if j != i and low[j] < low[i]:
                is_pivot_low = False
                break
        if is_pivot_low:
            pivot_lows[i] = True

    return pivot_highs, pivot_lows


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling mean."""
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling standard deviation."""
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.std(arr[i - window + 1:i + 1], ddof=1)
    return result


def market_structure_ob(
    data: Union[PdDataFrame, PlDataFrame],
    pivot_length: int = 7,
    momentum_zscore_threshold: float = 0.5,
    max_active_obs: int = 10,
    high_column: str = 'High',
    low_column: str = 'Low',
    open_column: str = 'Open',
    close_column: str = 'Close',
    volume_column: str = 'Volume',
    msb_bullish_column: str = 'msb_bullish',
    msb_bearish_column: str = 'msb_bearish',
    ob_bullish_column: str = 'ob_bullish',
    ob_bearish_column: str = 'ob_bearish',
    ob_top_column: str = 'ob_top',
    ob_bottom_column: str = 'ob_bottom',
    ob_quality_column: str = 'ob_quality',
    ob_is_hpz_column: str = 'ob_is_hpz',
    ob_mitigated_column: str = 'ob_mitigated'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect Market Structure Breaks with Order Block Probability scoring.

    This indicator combines Market Structure Breaks (MSB) with Order Block
    detection and quality scoring. Order blocks are scored based on
    momentum and volume to identify High Probability Zones (HPZ).

    **Order Block Quality Score (0-100):**
        - Based on momentum z-score (higher momentum = higher score)
        - Based on volume percentile rank (higher volume = higher score)
        - Score > 80 indicates a High Probability Zone (HPZ)

    **Order Block Detection:**
        - Bullish OB: Bearish candle before MSB breakout
        - Bearish OB: Bullish candle before MSB breakdown
        - Uses the candle closest to the breakout point

    Args:
        data: pandas or polars DataFrame with OHLCV price data
        pivot_length: Lookback period for pivot detection (default: 7)
        momentum_zscore_threshold: Z-score threshold for MSB (default: 0.5)
        max_active_obs: Maximum active order blocks to track (default: 10)
        high_column: Column name for high prices (default: 'High')
        low_column: Column name for low prices (default: 'Low')
        open_column: Column name for open prices (default: 'Open')
        close_column: Column name for close prices (default: 'Close')
        volume_column: Column name for volume (default: 'Volume')
        msb_bullish_column: Result column for bullish MSB
        msb_bearish_column: Result column for bearish MSB
        ob_bullish_column: Result column for bullish OB signal
        ob_bearish_column: Result column for bearish OB signal
        ob_top_column: Result column for OB top price
        ob_bottom_column: Result column for OB bottom price
        ob_quality_column: Result column for OB quality score (0-100)
        ob_is_hpz_column: Result column for HPZ flag (score > 80)
        ob_mitigated_column: Result column for OB mitigation flag

    Returns:
        DataFrame with added columns for MSB signals, OB zones,
        quality scores, HPZ flags, and mitigation status.

    Example:
        >>> import pandas as pd
        >>> from pyindicators import market_structure_ob
        >>> df = pd.DataFrame({
        ...     'Open': [...],
        ...     'High': [...],
        ...     'Low': [...],
        ...     'Close': [...],
        ...     'Volume': [...]
        ... })
        >>> result = market_structure_ob(df)
        >>> hpz_zones = result[result['ob_is_hpz'] == True]
    """
    if isinstance(data, PdDataFrame):
        return _market_structure_ob_pandas(
            data, pivot_length, momentum_zscore_threshold, max_active_obs,
            high_column, low_column, open_column, close_column, volume_column,
            msb_bullish_column, msb_bearish_column,
            ob_bullish_column, ob_bearish_column,
            ob_top_column, ob_bottom_column,
            ob_quality_column, ob_is_hpz_column, ob_mitigated_column
        )
    elif isinstance(data, PlDataFrame):
        return _market_structure_ob_polars(
            data, pivot_length, momentum_zscore_threshold, max_active_obs,
            high_column, low_column, open_column, close_column, volume_column,
            msb_bullish_column, msb_bearish_column,
            ob_bullish_column, ob_bearish_column,
            ob_top_column, ob_bottom_column,
            ob_quality_column, ob_is_hpz_column, ob_mitigated_column
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def _market_structure_ob_pandas(
    data: PdDataFrame,
    pivot_length: int,
    momentum_zscore_threshold: float,
    max_active_obs: int,
    high_column: str,
    low_column: str,
    open_column: str,
    close_column: str,
    volume_column: str,
    msb_bullish_column: str,
    msb_bearish_column: str,
    ob_bullish_column: str,
    ob_bearish_column: str,
    ob_top_column: str,
    ob_bottom_column: str,
    ob_quality_column: str,
    ob_is_hpz_column: str,
    ob_mitigated_column: str
) -> PdDataFrame:
    """Pandas implementation of market structure with OB probability."""
    high = data[high_column].values
    low = data[low_column].values
    open_price = data[open_column].values
    close = data[close_column].values
    n = len(data)

    # Check if volume column exists
    has_volume = volume_column in data.columns
    if has_volume:
        volume = data[volume_column].values
        vol_percentile = _rolling_percentile_rank(volume, 100)
    else:
        vol_percentile = np.full(n, 50.0)

    # Calculate momentum z-score
    price_change = np.diff(close, prepend=close[0])
    avg_change = _rolling_mean(price_change, 50)
    std_change = _rolling_std(price_change, 50)
    momentum_z = np.where(std_change > 0,
                          (price_change - avg_change) / std_change, 0)

    # Detect pivot highs and lows
    pivot_highs, pivot_lows = _detect_pivots(high, low, pivot_length)

    # Initialize result arrays
    msb_bullish = np.zeros(n, dtype=int)
    msb_bearish = np.zeros(n, dtype=int)
    ob_bullish = np.zeros(n, dtype=int)
    ob_bearish = np.zeros(n, dtype=int)
    ob_top = np.full(n, np.nan)
    ob_bottom = np.full(n, np.nan)
    ob_quality = np.full(n, np.nan)
    ob_is_hpz = np.zeros(n, dtype=bool)
    ob_mitigated = np.zeros(n, dtype=int)

    # Track last pivot values
    last_pivot_high = np.nan
    last_pivot_low = np.nan
    pivot_high_crossed = True
    pivot_low_crossed = True

    # Track active order blocks
    active_obs: List[Dict] = []

    for i in range(pivot_length, n):
        # Check for new pivot high
        if pivot_highs[i]:
            last_pivot_high = high[i]
            pivot_high_crossed = False

        # Check for new pivot low
        if pivot_lows[i]:
            last_pivot_low = low[i]
            pivot_low_crossed = False

        # Check for bullish MSB
        is_msb_bull = False
        if (not pivot_high_crossed and
                not np.isnan(last_pivot_high) and
                close[i] > last_pivot_high and
                close[i - 1] <= last_pivot_high and
                momentum_z[i] > momentum_zscore_threshold):
            msb_bullish[i] = 1
            is_msb_bull = True
            pivot_high_crossed = True

        # Check for bearish MSB
        is_msb_bear = False
        if (not pivot_low_crossed and
                not np.isnan(last_pivot_low) and
                close[i] < last_pivot_low and
                close[i - 1] >= last_pivot_low and
                momentum_z[i] < -momentum_zscore_threshold):
            msb_bearish[i] = 1
            is_msb_bear = True
            pivot_low_crossed = True

        # Create order block on MSB
        if is_msb_bull or is_msb_bear:
            # Find the opposite candle before MSB
            ob_idx = 0
            for j in range(1, min(11, i)):
                if is_msb_bull and close[i - j] < open_price[i - j]:
                    ob_idx = j
                    break
                elif is_msb_bear and close[i - j] > open_price[i - j]:
                    ob_idx = j
                    break

            if ob_idx > 0:
                ob_top_val = high[i - ob_idx]
                ob_bottom_val = low[i - ob_idx]

                # Calculate quality score
                score = min(100.0,
                            (abs(momentum_z[i]) * 20) +
                            (vol_percentile[i] * 0.5))
                is_hpz = score > 80

                if is_msb_bull:
                    ob_bullish[i] = 1
                else:
                    ob_bearish[i] = 1

                ob_top[i] = ob_top_val
                ob_bottom[i] = ob_bottom_val
                ob_quality[i] = score
                ob_is_hpz[i] = is_hpz

                # Add to active OBs
                active_obs.append({
                    'top': ob_top_val,
                    'bottom': ob_bottom_val,
                    'idx': i - ob_idx,
                    'is_bull': is_msb_bull,
                    'is_hpz': is_hpz,
                    'score': score,
                    'mitigated': False,
                    'created_at': i
                })

                # Limit active OBs
                if len(active_obs) > max_active_obs:
                    active_obs.pop(0)

        # Check for OB mitigation
        for ob in active_obs[:]:
            if not ob['mitigated']:
                if ob['is_bull'] and low[i] < ob['bottom']:
                    ob['mitigated'] = True
                    ob_mitigated[i] = 1
                elif not ob['is_bull'] and high[i] > ob['top']:
                    ob['mitigated'] = True
                    ob_mitigated[i] = 1

    # Add results to dataframe
    data = data.copy()
    data[msb_bullish_column] = msb_bullish
    data[msb_bearish_column] = msb_bearish
    data[ob_bullish_column] = ob_bullish
    data[ob_bearish_column] = ob_bearish
    data[ob_top_column] = ob_top
    data[ob_bottom_column] = ob_bottom
    data[ob_quality_column] = ob_quality
    data[ob_is_hpz_column] = ob_is_hpz
    data[ob_mitigated_column] = ob_mitigated

    return data


def _market_structure_ob_polars(
    data: PlDataFrame,
    pivot_length: int,
    momentum_zscore_threshold: float,
    max_active_obs: int,
    high_column: str,
    low_column: str,
    open_column: str,
    close_column: str,
    volume_column: str,
    msb_bullish_column: str,
    msb_bearish_column: str,
    ob_bullish_column: str,
    ob_bearish_column: str,
    ob_top_column: str,
    ob_bottom_column: str,
    ob_quality_column: str,
    ob_is_hpz_column: str,
    ob_mitigated_column: str
) -> PlDataFrame:
    """Polars implementation of market structure with OB probability."""
    pdf = data.to_pandas()
    result = _market_structure_ob_pandas(
        pdf, pivot_length, momentum_zscore_threshold, max_active_obs,
        high_column, low_column, open_column, close_column, volume_column,
        msb_bullish_column, msb_bearish_column,
        ob_bullish_column, ob_bearish_column,
        ob_top_column, ob_bottom_column,
        ob_quality_column, ob_is_hpz_column, ob_mitigated_column
    )
    return pl.from_pandas(result)


def _rolling_percentile_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling percentile rank (0-100)."""
    result = np.full_like(arr, 50.0, dtype=float)
    for i in range(window - 1, len(arr)):
        window_data = arr[i - window + 1:i + 1]
        current_val = arr[i]
        rank = np.sum(window_data <= current_val) / len(window_data) * 100
        result[i] = rank
    return result


def msb_signal(
    data: Union[PdDataFrame, PlDataFrame],
    close_column: str = 'Close',
    msb_bullish_column: str = 'msb_bullish',
    msb_bearish_column: str = 'msb_bearish',
    signal_column: str = 'msb_signal'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate trading signals based on Market Structure Breaks.

    Signal values:
        - 1: Bullish MSB detected (potential long opportunity)
        - -1: Bearish MSB detected (potential short opportunity)
        - 0: No MSB detected

    Args:
        data: DataFrame with MSB columns calculated
        close_column: Column name for close prices (default: 'Close')
        msb_bullish_column: Column for bullish MSB (default: 'msb_bullish')
        msb_bearish_column: Column for bearish MSB (default: 'msb_bearish')
        signal_column: Result column name (default: 'msb_signal')

    Returns:
        DataFrame with added signal column

    Example:
        >>> df = market_structure_break(df)
        >>> df = msb_signal(df)
        >>> bullish_signals = df[df['msb_signal'] == 1]
    """
    if isinstance(data, PdDataFrame):
        msb_bull = data[msb_bullish_column].values
        msb_bear = data[msb_bearish_column].values
        signal = np.where(msb_bull == 1, 1, np.where(msb_bear == 1, -1, 0))
        data = data.copy()
        data[signal_column] = signal
        return data
    elif isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        result = msb_signal(
            pdf, close_column, msb_bullish_column,
            msb_bearish_column, signal_column
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def ob_quality_signal(
    data: Union[PdDataFrame, PlDataFrame],
    min_quality: float = 50.0,
    hpz_only: bool = False,
    close_column: str = 'Close',
    ob_bullish_column: str = 'ob_bullish',
    ob_bearish_column: str = 'ob_bearish',
    ob_top_column: str = 'ob_top',
    ob_bottom_column: str = 'ob_bottom',
    ob_quality_column: str = 'ob_quality',
    ob_is_hpz_column: str = 'ob_is_hpz',
    signal_column: str = 'ob_quality_signal'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate signals when price enters a quality Order Block zone.

    Signal values:
        - 1: Price is within a quality bullish OB zone
        - -1: Price is within a quality bearish OB zone
        - 0: Price is not within any quality OB zone

    Args:
        data: DataFrame with OB columns from market_structure_ob()
        min_quality: Minimum quality score to generate signal (default: 50)
        hpz_only: Only signal on High Probability Zones (default: False)
        close_column: Column name for close prices (default: 'Close')
        ob_bullish_column: Column for bullish OB (default: 'ob_bullish')
        ob_bearish_column: Column for bearish OB (default: 'ob_bearish')
        ob_top_column: Column for OB top (default: 'ob_top')
        ob_bottom_column: Column for OB bottom (default: 'ob_bottom')
        ob_quality_column: Column for OB quality (default: 'ob_quality')
        ob_is_hpz_column: Column for HPZ flag (default: 'ob_is_hpz')
        signal_column: Result column name (default: 'ob_quality_signal')

    Returns:
        DataFrame with added signal column

    Example:
        >>> df = market_structure_ob(df)
        >>> df = ob_quality_signal(df, min_quality=70, hpz_only=True)
        >>> high_quality_signals = df[df['ob_quality_signal'] != 0]
    """
    if isinstance(data, PdDataFrame):
        return _ob_quality_signal_pandas(
            data, min_quality, hpz_only, close_column,
            ob_bullish_column, ob_bearish_column,
            ob_top_column, ob_bottom_column,
            ob_quality_column, ob_is_hpz_column, signal_column
        )
    elif isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        result = _ob_quality_signal_pandas(
            pdf, min_quality, hpz_only, close_column,
            ob_bullish_column, ob_bearish_column,
            ob_top_column, ob_bottom_column,
            ob_quality_column, ob_is_hpz_column, signal_column
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def _ob_quality_signal_pandas(
    data: PdDataFrame,
    min_quality: float,
    hpz_only: bool,
    close_column: str,
    ob_bullish_column: str,
    ob_bearish_column: str,
    ob_top_column: str,
    ob_bottom_column: str,
    ob_quality_column: str,
    ob_is_hpz_column: str,
    signal_column: str
) -> PdDataFrame:
    """Pandas implementation of OB quality signal."""
    close = data[close_column].values
    n = len(data)

    # Forward fill OB zones to track active zones
    bull_ob = data[ob_bullish_column].values
    bear_ob = data[ob_bearish_column].values
    ob_top_vals = data[ob_top_column].values
    ob_bottom_vals = data[ob_bottom_column].values
    quality_vals = data[ob_quality_column].values
    hpz_vals = data[ob_is_hpz_column].values

    # Track active zones with forward fill
    active_bull_top = np.nan
    active_bull_bottom = np.nan
    active_bull_quality = np.nan
    active_bull_hpz = False

    active_bear_top = np.nan
    active_bear_bottom = np.nan
    active_bear_quality = np.nan
    active_bear_hpz = False

    signal = np.zeros(n, dtype=int)

    for i in range(n):
        # Update active bullish zone
        if bull_ob[i] == 1 and not np.isnan(ob_top_vals[i]):
            active_bull_top = ob_top_vals[i]
            active_bull_bottom = ob_bottom_vals[i]
            active_bull_quality = quality_vals[i]
            active_bull_hpz = bool(hpz_vals[i])

        # Update active bearish zone
        if bear_ob[i] == 1 and not np.isnan(ob_top_vals[i]):
            active_bear_top = ob_top_vals[i]
            active_bear_bottom = ob_bottom_vals[i]
            active_bear_quality = quality_vals[i]
            active_bear_hpz = bool(hpz_vals[i])

        # Check if price is in bullish zone with quality criteria
        if (not np.isnan(active_bull_top) and
                active_bull_bottom <= close[i] <= active_bull_top):
            if hpz_only and active_bull_hpz:
                signal[i] = 1
            elif not hpz_only and active_bull_quality >= min_quality:
                signal[i] = 1

        # Check if price is in bearish zone with quality criteria
        if (not np.isnan(active_bear_top) and
                active_bear_bottom <= close[i] <= active_bear_top):
            if hpz_only and active_bear_hpz:
                signal[i] = -1
            elif not hpz_only and active_bear_quality >= min_quality:
                signal[i] = -1

        # Reset zones if mitigated
        if not np.isnan(active_bull_bottom) and close[i] < active_bull_bottom:
            active_bull_top = np.nan
            active_bull_bottom = np.nan

        if not np.isnan(active_bear_top) and close[i] > active_bear_top:
            active_bear_top = np.nan
            active_bear_bottom = np.nan

    data = data.copy()
    data[signal_column] = signal
    return data


def get_market_structure_stats(
    data: Union[PdDataFrame, PlDataFrame],
    msb_bullish_column: str = 'msb_bullish',
    msb_bearish_column: str = 'msb_bearish',
    ob_mitigated_column: str = 'ob_mitigated',
    ob_is_hpz_column: str = 'ob_is_hpz',
    ob_bullish_column: str = 'ob_bullish',
    ob_bearish_column: str = 'ob_bearish'
) -> Dict[str, float]:
    """
    Calculate statistics for Market Structure analysis.

    Returns reliability metrics and counts for order blocks and MSBs.

    Args:
        data: DataFrame with market structure columns calculated
        msb_bullish_column: Column for bullish MSB
        msb_bearish_column: Column for bearish MSB
        ob_mitigated_column: Column for OB mitigation
        ob_is_hpz_column: Column for HPZ flag
        ob_bullish_column: Column for bullish OB
        ob_bearish_column: Column for bearish OB

    Returns:
        Dictionary with:
            - 'total_obs': Total order blocks detected
            - 'total_mitigated': Total order blocks mitigated
            - 'reliability': Percentage of OBs that were mitigated
            - 'hpz_count': Number of High Probability Zones
            - 'bullish_msb_count': Number of bullish MSBs
            - 'bearish_msb_count': Number of bearish MSBs

    Example:
        >>> df = market_structure_ob(df)
        >>> stats = get_market_structure_stats(df)
        >>> print(f"Reliability: {stats['reliability']:.1f}%")
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    total_obs = (data[ob_bullish_column].sum() +
                 data[ob_bearish_column].sum())
    total_mitigated = data[ob_mitigated_column].sum()
    reliability = (total_mitigated / total_obs * 100) if total_obs > 0 else 0

    hpz_mask = data[ob_is_hpz_column] == True  # noqa: E712
    ob_mask = (data[ob_bullish_column] == 1) | (data[ob_bearish_column] == 1)
    hpz_count = (hpz_mask & ob_mask).sum()

    return {
        'total_obs': int(total_obs),
        'total_mitigated': int(total_mitigated),
        'reliability': float(reliability),
        'hpz_count': int(hpz_count),
        'bullish_msb_count': int(data[msb_bullish_column].sum()),
        'bearish_msb_count': int(data[msb_bearish_column].sum())
    }


def market_structure_choch_bos(
    data: Union[PdDataFrame, PlDataFrame],
    length: int = 5,
    high_column: str = 'High',
    low_column: str = 'Low',
    close_column: str = 'Close',
    choch_bullish_column: str = 'choch_bullish',
    choch_bearish_column: str = 'choch_bearish',
    choch_plus_bullish_column: str = 'choch_plus_bullish',
    choch_plus_bearish_column: str = 'choch_plus_bearish',
    bos_bullish_column: str = 'bos_bullish',
    bos_bearish_column: str = 'bos_bearish',
    support_column: str = 'support_level',
    resistance_column: str = 'resistance_level',
    support_broken_column: str = 'support_broken',
    resistance_broken_column: str = 'resistance_broken',
    trend_column: str = 'market_trend'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect Market Structure using fractal-based CHoCH, CHoCH+ and BOS signals.

    This indicator uses fractal detection to identify significant swing
    highs and lows, then determines whether price breaks are:
    - **CHoCH (Change of Character)**: Trend reversal signal
    - **CHoCH+ (Confirmed Change of Character)**: Stronger reversal signal
      preceded by a warning swing (LH before bearish reversal, HL before
      bullish reversal)
    - **BOS (Break of Structure)**: Trend continuation signal

    The distinction is based on the current market trend (order flow state):
    - If trend is bearish (-1) and price breaks above a swing high → CHoCH
    - If trend is bullish (1) and price breaks above a swing high → BOS
    - If trend is bullish (1) and price breaks below a swing low → CHoCH
    - If trend is bearish (-1) and price breaks below a swing low → BOS

    CHoCH+ upgrade logic (uses internally tracked swing classifications):
    - A bearish CHoCH is upgraded to CHoCH+ when the most recent swing
      high was a **Lower High (LH)** — i.e., bulls already showed weakness.
    - A bullish CHoCH is upgraded to CHoCH+ when the most recent swing
      low was a **Higher Low (HL)** — i.e., bears already showed weakness.
    - When a CHoCH is upgraded to CHoCH+, the regular CHoCH column stays 0
      and only the CHoCH+ column fires.

    Args:
        data: pandas or polars DataFrame with OHLC price data
        length: Lookback period for fractal detection (default: 5, min: 3)
        high_column: Column name for high prices (default: 'High')
        low_column: Column name for low prices (default: 'Low')
        close_column: Column name for close prices (default: 'Close')
        choch_bullish_column: Result column for bullish CHoCH
        choch_bearish_column: Result column for bearish CHoCH
        choch_plus_bullish_column: Result column for bullish CHoCH+
        choch_plus_bearish_column: Result column for bearish CHoCH+
        bos_bullish_column: Result column for bullish BOS
        bos_bearish_column: Result column for bearish BOS
        support_column: Result column for support level
        resistance_column: Result column for resistance level
        support_broken_column: Result column for support breakout
        resistance_broken_column: Result column for resistance breakout
        trend_column: Result column for current market trend (1=bull, -1=bear)

    Returns:
        DataFrame with added columns:
            - {choch_bullish_column}: 1 when bullish CHoCH, 0 otherwise
            - {choch_bearish_column}: 1 when bearish CHoCH, 0 otherwise
            - {choch_plus_bullish_column}: 1 when bullish CHoCH+, 0 otherwise
            - {choch_plus_bearish_column}: 1 when bearish CHoCH+, 0 otherwise
            - {bos_bullish_column}: 1 when bullish BOS, 0 otherwise
            - {bos_bearish_column}: 1 when bearish BOS, 0 otherwise
            - {support_column}: Current support level price
            - {resistance_column}: Current resistance level price
            - {support_broken_column}: 1 when support is broken
            - {resistance_broken_column}: 1 when resistance is broken
            - {trend_column}: Current market trend
              (1=bullish, -1=bearish, 0=neutral)

    Example:
        >>> import pandas as pd
        >>> from pyindicators import market_structure_choch_bos
        >>> df = pd.DataFrame({
        ...     'High': [...],
        ...     'Low': [...],
        ...     'Close': [...]
        ... })
        >>> result = market_structure_choch_bos(df, length=5)
        >>> choch_signals = result[result['choch_bullish'] == 1]
        >>> choch_plus = result[result['choch_plus_bullish'] == 1]
        >>> bos_signals = result[result['bos_bullish'] == 1]
    """
    if length < 3:
        raise PyIndicatorException("Length must be at least 3")

    if isinstance(data, PdDataFrame):
        return _choch_bos_pandas(
            data, length,
            high_column, low_column, close_column,
            choch_bullish_column, choch_bearish_column,
            choch_plus_bullish_column, choch_plus_bearish_column,
            bos_bullish_column, bos_bearish_column,
            support_column, resistance_column,
            support_broken_column, resistance_broken_column,
            trend_column
        )
    elif isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        result = _choch_bos_pandas(
            pdf, length,
            high_column, low_column, close_column,
            choch_bullish_column, choch_bearish_column,
            choch_plus_bullish_column, choch_plus_bearish_column,
            bos_bullish_column, bos_bearish_column,
            support_column, resistance_column,
            support_broken_column, resistance_broken_column,
            trend_column
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def _choch_bos_pandas(
    data: PdDataFrame,
    length: int,
    high_column: str,
    low_column: str,
    close_column: str,
    choch_bullish_column: str,
    choch_bearish_column: str,
    choch_plus_bullish_column: str,
    choch_plus_bearish_column: str,
    bos_bullish_column: str,
    bos_bearish_column: str,
    support_column: str,
    resistance_column: str,
    support_broken_column: str,
    resistance_broken_column: str,
    trend_column: str
) -> PdDataFrame:
    """Pandas implementation of CHoCH/CHoCH+/BOS detection."""
    high = data[high_column].values
    low = data[low_column].values
    close = data[close_column].values
    n = len(data)
    p = length // 2

    # Initialize result arrays
    choch_bullish = np.zeros(n, dtype=int)
    choch_bearish = np.zeros(n, dtype=int)
    choch_plus_bullish = np.zeros(n, dtype=int)
    choch_plus_bearish = np.zeros(n, dtype=int)
    bos_bullish = np.zeros(n, dtype=int)
    bos_bearish = np.zeros(n, dtype=int)
    support_level = np.full(n, np.nan)
    resistance_level = np.full(n, np.nan)
    support_broken = np.zeros(n, dtype=int)
    resistance_broken = np.zeros(n, dtype=int)
    trend = np.zeros(n, dtype=int)

    # Detect fractals using directional momentum
    bullish_fractals, bearish_fractals = _detect_fractals(high, low, length)

    # Track swing high/low for structure breaks
    upper_value = np.nan
    upper_loc = -1
    upper_crossed = True

    lower_value = np.nan
    lower_loc = -1
    lower_crossed = True

    # Order flow state: 1 = bullish, -1 = bearish, 0 = neutral
    os_state = 0

    # CHoCH+ tracking: previous swing values for HH/HL/LH/LL classification
    prev_upper_value = np.nan   # previous swing high price
    prev_lower_value = np.nan   # previous swing low price
    last_sh_is_lower = False    # True when most recent swing high is LH
    last_sl_is_higher = False   # True when most recent swing low is HL

    # Support/resistance tracking
    current_support = np.nan
    current_resistance = np.nan
    sup_broken = True
    res_broken = True

    for i in range(length, n):
        # Update swing high on bullish fractal
        if bullish_fractals[i]:
            new_high = high[i - p]
            # Classify this swing high as HH or LH
            if not np.isnan(upper_value):
                last_sh_is_lower = new_high < upper_value  # LH
                prev_upper_value = upper_value
            upper_value = new_high
            upper_loc = i - p
            upper_crossed = False

        # Update swing low on bearish fractal
        if bearish_fractals[i]:
            new_low = low[i - p]
            # Classify this swing low as HL or LL
            if not np.isnan(lower_value):
                last_sl_is_higher = new_low > lower_value  # HL
                prev_lower_value = lower_value
            lower_value = new_low
            lower_loc = i - p
            lower_crossed = False

        # Check for bullish structure break (close crosses above swing high)
        if (not upper_crossed and
                not np.isnan(upper_value) and
                close[i] > upper_value and
                close[i - 1] <= upper_value):

            # Determine if CHoCH / CHoCH+ / BOS
            if os_state == -1:
                # Trend was bearish, now breaking up = Change of Character
                # Upgrade to CHoCH+ if last swing low was HL (warning swing)
                if last_sl_is_higher and not np.isnan(prev_lower_value):
                    choch_plus_bullish[i] = 1
                else:
                    choch_bullish[i] = 1
            else:
                # Trend was bullish or neutral = Break of Structure
                bos_bullish[i] = 1

            # Find support level (lowest low since swing high)
            if upper_loc >= 0 and i > upper_loc:
                min_low = low[upper_loc + 1]
                for j in range(upper_loc + 1, i):
                    if low[j] < min_low:
                        min_low = low[j]
                current_support = min_low
                sup_broken = False

            upper_crossed = True
            os_state = 1

        # Check for bearish structure break (close crosses below swing low)
        if (not lower_crossed and
                not np.isnan(lower_value) and
                close[i] < lower_value and
                close[i - 1] >= lower_value):

            # Determine if CHoCH / CHoCH+ / BOS
            if os_state == 1:
                # Trend was bullish, now breaking down = Change of Character
                # Upgrade to CHoCH+ if last swing high was LH (warning swing)
                if last_sh_is_lower and not np.isnan(prev_upper_value):
                    choch_plus_bearish[i] = 1
                else:
                    choch_bearish[i] = 1
            else:
                # Trend was bearish or neutral = Break of Structure
                bos_bearish[i] = 1

            # Find resistance level (highest high since swing low)
            if lower_loc >= 0 and i > lower_loc:
                max_high = high[lower_loc + 1]
                for j in range(lower_loc + 1, i):
                    if high[j] > max_high:
                        max_high = high[j]
                current_resistance = max_high
                res_broken = False

            lower_crossed = True
            os_state = -1

        # Check for support/resistance breakouts
        if not sup_broken and not np.isnan(current_support):
            if close[i] < current_support:
                support_broken[i] = 1
                sup_broken = True

        if not res_broken and not np.isnan(current_resistance):
            if close[i] > current_resistance:
                resistance_broken[i] = 1
                res_broken = True

        # Record current levels and trend
        support_level[i] = current_support if not sup_broken else np.nan
        resistance_level[i] = current_resistance if not res_broken else np.nan
        trend[i] = os_state

    # Add results to dataframe
    data = data.copy()
    data[choch_bullish_column] = choch_bullish
    data[choch_bearish_column] = choch_bearish
    data[choch_plus_bullish_column] = choch_plus_bullish
    data[choch_plus_bearish_column] = choch_plus_bearish
    data[bos_bullish_column] = bos_bullish
    data[bos_bearish_column] = bos_bearish
    data[support_column] = support_level
    data[resistance_column] = resistance_level
    data[support_broken_column] = support_broken
    data[resistance_broken_column] = resistance_broken
    data[trend_column] = trend

    return data


def _detect_fractals(
    high: np.ndarray,
    low: np.ndarray,
    length: int
) -> tuple:
    """
    Detect fractals using directional momentum method.

    A bullish fractal (swing high) occurs when:
    - The sum of sign changes in highs over p bars equals -p (falling)
    - AND the sum p bars ago equaled +p (rising)
    - AND the high at p bars ago is the highest in the length

    A bearish fractal (swing low) is the inverse.
    """
    n = len(high)
    p = length // 2
    bullish_fractals = np.zeros(n, dtype=bool)
    bearish_fractals = np.zeros(n, dtype=bool)

    # Calculate directional sums
    high_sign = np.zeros(n)
    low_sign = np.zeros(n)

    for i in range(1, n):
        high_sign[i] = np.sign(high[i] - high[i - 1])
        low_sign[i] = np.sign(low[i] - low[i - 1])

    for i in range(length, n):
        # Sum of high direction over last p bars
        dh = np.sum(high_sign[i - p + 1:i + 1])
        # Sum of high direction p bars before that
        dh_prev = np.sum(high_sign[i - 2 * p + 1:i - p + 1])

        # Sum of low direction over last p bars
        dl = np.sum(low_sign[i - p + 1:i + 1])
        # Sum of low direction p bars before that
        dl_prev = np.sum(low_sign[i - 2 * p + 1:i - p + 1])

        # Check for bullish fractal (swing high)
        # Direction was rising (+p), now falling (-p), and high[i-p] is highest
        if dh == -p and dh_prev == p:
            # Verify it's the highest in the window
            window_start = max(0, i - length + 1)
            if high[i - p] == np.max(high[window_start:i + 1]):
                bullish_fractals[i] = True

        # Check for bearish fractal (swing low)
        # Direction was falling (-p), now rising (+p), and low[i-p] is lowest
        if dl == p and dl_prev == -p:
            # Verify it's the lowest in the window
            window_start = max(0, i - length + 1)
            if low[i - p] == np.min(low[window_start:i + 1]):
                bearish_fractals[i] = True

    return bullish_fractals, bearish_fractals


def choch_bos_signal(
    data: Union[PdDataFrame, PlDataFrame],
    choch_bullish_column: str = 'choch_bullish',
    choch_bearish_column: str = 'choch_bearish',
    choch_plus_bullish_column: str = 'choch_plus_bullish',
    choch_plus_bearish_column: str = 'choch_plus_bearish',
    bos_bullish_column: str = 'bos_bullish',
    bos_bearish_column: str = 'bos_bearish',
    signal_column: str = 'structure_signal'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate trading signals from CHoCH/CHoCH+/BOS detection.

    Signal values:
        - 3: Bullish CHoCH+ (confirmed reversal - trend change to bullish
              preceded by a warning Higher Low)
        - 2: Bullish CHoCH (reversal signal - trend change to bullish)
        - 1: Bullish BOS (continuation signal - bullish trend continues)
        - -1: Bearish BOS (continuation signal - bearish trend continues)
        - -2: Bearish CHoCH (reversal signal - trend change to bearish)
        - -3: Bearish CHoCH+ (confirmed reversal - trend change to bearish
               preceded by a warning Lower High)
        - 0: No signal

    Args:
        data: DataFrame with CHoCH/CHoCH+/BOS columns calculated
        choch_bullish_column: Column for bullish CHoCH
        choch_bearish_column: Column for bearish CHoCH
        choch_plus_bullish_column: Column for bullish CHoCH+
        choch_plus_bearish_column: Column for bearish CHoCH+
        bos_bullish_column: Column for bullish BOS
        bos_bearish_column: Column for bearish BOS
        signal_column: Result column name

    Returns:
        DataFrame with added signal column

    Example:
        >>> df = market_structure_choch_bos(df)
        >>> df = choch_bos_signal(df)
        >>> confirmed_reversals = df[abs(df['structure_signal']) == 3]
        >>> reversal_signals = df[abs(df['structure_signal']) >= 2]
    """
    if isinstance(data, PdDataFrame):
        choch_bull = data[choch_bullish_column].values
        choch_bear = data[choch_bearish_column].values
        choch_plus_bull = data[choch_plus_bullish_column].values
        choch_plus_bear = data[choch_plus_bearish_column].values
        bos_bull = data[bos_bullish_column].values
        bos_bear = data[bos_bearish_column].values

        signal = np.zeros(len(data), dtype=int)
        signal = np.where(choch_bull == 1, 2, signal)
        signal = np.where(choch_bear == 1, -2, signal)
        signal = np.where(choch_plus_bull == 1, 3, signal)
        signal = np.where(choch_plus_bear == 1, -3, signal)
        signal = np.where(bos_bull == 1, 1, signal)
        signal = np.where(bos_bear == 1, -1, signal)

        data = data.copy()
        data[signal_column] = signal
        return data
    elif isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        result = choch_bos_signal(
            pdf, choch_bullish_column, choch_bearish_column,
            choch_plus_bullish_column, choch_plus_bearish_column,
            bos_bullish_column, bos_bearish_column, signal_column
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def get_choch_bos_stats(
    data: Union[PdDataFrame, PlDataFrame],
    choch_bullish_column: str = 'choch_bullish',
    choch_bearish_column: str = 'choch_bearish',
    choch_plus_bullish_column: str = 'choch_plus_bullish',
    choch_plus_bearish_column: str = 'choch_plus_bearish',
    bos_bullish_column: str = 'bos_bullish',
    bos_bearish_column: str = 'bos_bearish'
) -> Dict[str, int]:
    """
    Calculate statistics for CHoCH/CHoCH+/BOS signals.

    Args:
        data: DataFrame with CHoCH/CHoCH+/BOS columns calculated
        choch_bullish_column: Column for bullish CHoCH
        choch_bearish_column: Column for bearish CHoCH
        choch_plus_bullish_column: Column for bullish CHoCH+
        choch_plus_bearish_column: Column for bearish CHoCH+
        bos_bullish_column: Column for bullish BOS
        bos_bearish_column: Column for bearish BOS

    Returns:
        Dictionary with:
            - 'choch_bullish_count': Number of bullish CHoCH signals
            - 'choch_bearish_count': Number of bearish CHoCH signals
            - 'choch_plus_bullish_count': Number of bullish CHoCH+ signals
            - 'choch_plus_bearish_count': Number of bearish CHoCH+ signals
            - 'bos_bullish_count': Number of bullish BOS signals
            - 'bos_bearish_count': Number of bearish BOS signals
            - 'total_choch': Total CHoCH signals (excluding CHoCH+)
            - 'total_choch_plus': Total CHoCH+ signals (confirmed reversals)
            - 'total_bos': Total BOS signals (continuations)

    Example:
        >>> df = market_structure_choch_bos(df)
        >>> stats = get_choch_bos_stats(df)
        >>> print(f"Reversals: {stats['total_choch']}")
        >>> print(f"Confirmed reversals: {stats['total_choch_plus']}")
        >>> print(f"Continuations: {stats['total_bos']}")
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    choch_bull = int(data[choch_bullish_column].sum())
    choch_bear = int(data[choch_bearish_column].sum())
    choch_plus_bull = int(data[choch_plus_bullish_column].sum())
    choch_plus_bear = int(data[choch_plus_bearish_column].sum())
    bos_bull = int(data[bos_bullish_column].sum())
    bos_bear = int(data[bos_bearish_column].sum())

    return {
        'choch_bullish_count': choch_bull,
        'choch_bearish_count': choch_bear,
        'choch_plus_bullish_count': choch_plus_bull,
        'choch_plus_bearish_count': choch_plus_bear,
        'bos_bullish_count': bos_bull,
        'bos_bearish_count': bos_bear,
        'total_choch': choch_bull + choch_bear,
        'total_choch_plus': choch_plus_bull + choch_plus_bear,
        'total_bos': bos_bull + bos_bear
    }
