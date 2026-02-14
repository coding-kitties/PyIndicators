"""
SuperTrend Clustering Indicator

This indicator implements the SuperTrend with optimization using
K-means clustering to find the optimal ATR multiplier factor.
"""
from typing import Union, Dict, List, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.indicators.average_true_range import atr


def _calculate_supertrend_single(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_values: np.ndarray,
    factor: float,
    perf_alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate SuperTrend for a single factor.

    Returns:
        Tuple of (output, trend, upper, lower, performance)
    """
    n = len(close)
    hl2 = (high + low) / 2

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    output = np.full(n, np.nan)
    perf = np.zeros(n)

    # Initialize
    upper[0] = hl2[0] + atr_values[0] * factor
    lower[0] = hl2[0] - atr_values[0] * factor
    output[0] = upper[0]

    alpha = 2 / (perf_alpha + 1)

    for i in range(1, n):
        up = hl2[i] + atr_values[i] * factor
        dn = hl2[i] - atr_values[i] * factor

        # Update trend
        if close[i] > upper[i-1]:
            trend[i] = 1
        elif close[i] < lower[i-1]:
            trend[i] = 0
        else:
            trend[i] = trend[i-1]

        # Update upper/lower bands
        if close[i-1] < upper[i-1]:
            upper[i] = min(up, upper[i-1])
        else:
            upper[i] = up

        if close[i-1] > lower[i-1]:
            lower[i] = max(dn, lower[i-1])
        else:
            lower[i] = dn

        # Output based on trend
        output[i] = lower[i] if trend[i] == 1 else upper[i]

        # Performance calculation (EMA of returns * direction)
        if i > 0 and not np.isnan(output[i-1]):
            diff = np.sign(close[i-1] - output[i-1])
            price_change = close[i] - close[i-1]
            perf[i] = perf[i-1] + alpha * (price_change * diff - perf[i-1])

    return output, trend, upper, lower, perf


def _kmeans_clustering(
    data: np.ndarray,
    factors: np.ndarray,
    max_iter: int = 1000
) -> Tuple[List[List[float]], List[List[float]], np.ndarray]:
    """
    Perform K-means clustering with 3 clusters (Best, Average, Worst).

    Uses scikit-learn's KMeans for robust clustering.

    Returns:
        Tuple of (performance_clusters, factor_clusters, centroids)
    """
    try:
        from sklearn.cluster import KMeans

        # Reshape data for sklearn (needs 2D array)
        X = data.reshape(-1, 1)

        # Initialize with quartiles for consistent ordering
        init_centroids = np.array([
            [np.percentile(data, 25)],
            [np.percentile(data, 50)],
            [np.percentile(data, 75)]
        ])

        kmeans = KMeans(
            n_clusters=3,
            init=init_centroids,
            n_init=1,
            max_iter=max_iter,
            random_state=42
        )
        kmeans.fit(X)

        # Get cluster labels and centroids
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_.flatten()

        # Sort clusters by centroid value (worst=0, average=1, best=2)
        sorted_indices = np.argsort(centroids)

        # Build cluster lists
        perf_clusters = [[], [], []]
        factor_clusters = [[], [], []]

        for i, label in enumerate(labels):
            # Map to sorted cluster index
            sorted_label = np.where(sorted_indices == label)[0][0]
            perf_clusters[sorted_label].append(data[i])
            factor_clusters[sorted_label].append(factors[i])

        centroids = centroids[sorted_indices]

        return perf_clusters, factor_clusters, centroids

    except ImportError:
        # Fallback to simple implementation if sklearn not available
        return _kmeans_clustering_simple(data, factors, max_iter)


def _kmeans_clustering_simple(
    data: np.ndarray,
    factors: np.ndarray,
    max_iter: int = 1000
) -> Tuple[List[List[float]], List[List[float]], np.ndarray]:
    """
    Simple K-means fallback without sklearn dependency.
    """
    # Initialize centroids using quartiles
    centroids = np.array([
        np.percentile(data, 25),
        np.percentile(data, 50),
        np.percentile(data, 75)
    ])

    perf_clusters = [[], [], []]
    factor_clusters = [[], [], []]

    for _ in range(max_iter):
        # Assign values to clusters
        perf_clusters = [[], [], []]
        factor_clusters = [[], [], []]

        for i, value in enumerate(data):
            distances = np.abs(value - centroids)
            idx = np.argmin(distances)
            perf_clusters[idx].append(value)
            factor_clusters[idx].append(factors[i])

        # Update centroids
        new_centroids = np.array([
            np.mean(cluster) if len(cluster) > 0 else centroids[i]
            for i, cluster in enumerate(perf_clusters)
        ])

        # Check for convergence
        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return perf_clusters, factor_clusters, centroids


def supertrend_clustering(
    data: Union[PdDataFrame, PlDataFrame],
    atr_length: int = 10,
    min_mult: float = 1.0,
    max_mult: float = 5.0,
    step: float = 0.5,
    perf_alpha: float = 10.0,
    from_cluster: str = 'best',
    max_iter: int = 1000,
    max_data: int = 10000
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the SuperTrend Clustering indicator using K-means clustering
    to optimize the ATR multiplier factor.

    This indicator computes multiple SuperTrend variations with different
    factors, evaluates their performance, and uses K-means clustering to
    identify the best performing factor.

    Parameters:
        data: DataFrame with OHLC data (High, Low, Close columns required)
        atr_length: Period for ATR calculation (default: 10)
        min_mult: Minimum ATR multiplier factor (default: 1.0)
        max_mult: Maximum ATR multiplier factor (default: 5.0)
        step: Step size for factor range (default: 0.5)
        perf_alpha: Performance memory/smoothing period (default: 10.0)
        from_cluster: Which cluster to use - 'best', 'average', or 'worst'
        max_iter: Maximum K-means iteration steps (default: 1000)
        max_data: Maximum historical bars for calculation (default: 10000)

    Returns:
        DataFrame with added columns:
        - supertrend: The optimized SuperTrend trailing stop
        - supertrend_trend: Current trend (1=bullish, 0=bearish)
        - supertrend_ama: Adaptive moving average of SuperTrend
        - supertrend_perf_idx: Performance index (0-1 scale)
        - supertrend_factor: Currently used ATR factor
        - supertrend_signal: 1=buy signal, -1=sell signal, 0=no signal
    """
    if min_mult > max_mult:
        raise ValueError(
            'Minimum factor is greater than maximum factor in the range'
        )

    is_polars = isinstance(data, PlDataFrame)
    if is_polars:
        df = data.to_pandas()
    else:
        df = data.copy()

    # Calculate ATR
    df = atr(df, period=atr_length, result_column='_atr')

    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    atr_values = df['_atr'].values
    hl2 = (high + low) / 2
    n = len(df)

    # Generate factor range
    factors = np.arange(min_mult, max_mult + step, step)

    # Calculate SuperTrend for each factor
    all_outputs = []
    all_trends = []
    all_perfs = []

    for factor in factors:
        output, trend, _, _, perf = _calculate_supertrend_single(
            high, low, close, atr_values, factor, perf_alpha
        )
        all_outputs.append(output)
        all_trends.append(trend)
        all_perfs.append(perf)

    all_outputs = np.array(all_outputs)
    all_trends = np.array(all_trends)
    all_perfs = np.array(all_perfs)

    # Initialize result arrays
    supertrend = np.full(n, np.nan)
    supertrend_trend = np.zeros(n, dtype=int)
    supertrend_ama = np.full(n, np.nan)
    supertrend_perf_idx = np.zeros(n)
    supertrend_factor = np.full(n, np.nan)
    supertrend_signal = np.zeros(n, dtype=int)

    # Process each bar (use last max_data bars for clustering)
    start_idx = max(0, n - max_data)

    # Get cluster mapping
    cluster_idx_map = {'best': 2, 'average': 1, 'worst': 0}
    target_cluster_idx = cluster_idx_map.get(from_cluster.lower(), 2)

    # For optimization, we'll do clustering at each bar using
    # the performance up to that point
    prev_upper = hl2.copy()
    prev_lower = hl2.copy()
    os = np.zeros(n, dtype=int)

    # Calculate denominator for performance index
    price_changes = np.abs(np.diff(close, prepend=close[0]))
    den = pd.Series(price_changes).ewm(span=int(perf_alpha)).mean().values

    for i in range(start_idx, n):
        # Get performances at this bar
        perfs_at_i = all_perfs[:, i]
        factors_array = factors.copy()

        # Perform clustering
        perf_clusters, factor_clusters, centroids = _kmeans_clustering(
            perfs_at_i, factors_array, max_iter
        )

        # Get target factor from cluster
        target_factors = factor_clusters[target_cluster_idx]
        if len(target_factors) > 0:
            target_factor = np.mean(target_factors)
        else:
            # Fallback to middle factor
            target_factor = (min_mult + max_mult) / 2

        supertrend_factor[i] = target_factor

        # Get performance index
        target_perfs = perf_clusters[target_cluster_idx]
        if len(target_perfs) > 0 and den[i] > 0:
            perf_val = max(np.mean(target_perfs), 0) / den[i]
            supertrend_perf_idx[i] = min(perf_val, 1.0)  # Cap at 1

        # Calculate new SuperTrend with target factor
        up = hl2[i] + atr_values[i] * target_factor
        dn = hl2[i] - atr_values[i] * target_factor

        if i > 0:
            if close[i-1] < prev_upper[i-1]:
                prev_upper[i] = min(up, prev_upper[i-1])
            else:
                prev_upper[i] = up

            if close[i-1] > prev_lower[i-1]:
                prev_lower[i] = max(dn, prev_lower[i-1])
            else:
                prev_lower[i] = dn

            if close[i] > prev_upper[i]:
                os[i] = 1
            elif close[i] < prev_lower[i]:
                os[i] = 0
            else:
                os[i] = os[i-1]
        else:
            prev_upper[i] = up
            prev_lower[i] = dn

        # Set trailing stop
        ts = prev_lower[i] if os[i] == 1 else prev_upper[i]
        supertrend[i] = ts
        supertrend_trend[i] = os[i]

        # Adaptive MA
        if i == start_idx or np.isnan(supertrend_ama[i-1]):
            supertrend_ama[i] = ts
        else:
            perf_idx = supertrend_perf_idx[i]
            supertrend_ama[i] = (
                supertrend_ama[i-1] + perf_idx * (ts - supertrend_ama[i-1])
            )

        # Generate signals
        if i > 0:
            if os[i] > os[i-1]:
                supertrend_signal[i] = 1  # Buy signal
            elif os[i] < os[i-1]:
                supertrend_signal[i] = -1  # Sell signal

    # Add results to DataFrame
    df['supertrend'] = supertrend
    df['supertrend_trend'] = supertrend_trend
    df['supertrend_ama'] = supertrend_ama
    df['supertrend_perf_idx'] = supertrend_perf_idx
    df['supertrend_factor'] = supertrend_factor
    df['supertrend_signal'] = supertrend_signal

    # Clean up temporary column
    df = df.drop(columns=['_atr'])

    if is_polars:
        import polars as pl
        return pl.from_pandas(df)

    return df


def supertrend(
    data: Union[PdDataFrame, PlDataFrame],
    atr_length: int = 10,
    factor: float = 3.0,
    wicks: bool = True
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Calculate the SuperTrend indicator.

    Based on the standard SuperTrend formula by Alex Orekhov (everget).
    Uses ATR-based trailing stops with trend detection.

    Parameters:
        data: DataFrame with OHLC data
              (Open, High, Low, Close columns required)
        atr_length: Period for ATR calculation (default: 10)
        factor: ATR multiplier factor (default: 3.0)
        wicks: If True, use High/Low for trend detection and band
               clamping. If False, use Close only. (default: True)

    Returns:
        DataFrame with added columns:
        - supertrend: The SuperTrend trailing stop value
        - supertrend_trend: Current trend (1=bullish, 0=bearish)
        - supertrend_upper: Upper band (short stop)
        - supertrend_lower: Lower band (long stop)
        - supertrend_signal: 1=buy signal, -1=sell signal, 0=no signal
    """
    is_polars = isinstance(data, PlDataFrame)
    if is_polars:
        df = data.to_pandas()
    else:
        df = data.copy()

    # Calculate ATR
    df = atr(df, period=atr_length, result_column='_atr')

    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    open_price = df['Open'].values
    atr_values = df['_atr'].values
    hl2 = (high + low) / 2
    n = len(df)

    # Price references for trend detection and band clamping
    high_price = high if wicks else close
    low_price = low if wicks else close

    # Initialize arrays
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    supertrend_val = np.full(n, np.nan)
    signal = np.zeros(n, dtype=int)

    # Initial values
    upper[0] = hl2[0] + atr_values[0] * factor
    lower[0] = hl2[0] - atr_values[0] * factor
    trend[0] = 1
    supertrend_val[0] = lower[0]

    for i in range(1, n):
        up = hl2[i] + atr_values[i] * factor
        dn = hl2[i] - atr_values[i] * factor

        # Doji4price: all OHLC values equal — preserve previous stops
        doji4price = (
            open_price[i] == close[i]
            and open_price[i] == low[i]
            and open_price[i] == high[i]
        )

        # Update lower band (long stop)
        if doji4price:
            lower[i] = lower[i-1]
        elif low_price[i-1] > lower[i-1]:
            lower[i] = max(dn, lower[i-1])
        else:
            lower[i] = dn

        # Update upper band (short stop)
        if doji4price:
            upper[i] = upper[i-1]
        elif high_price[i-1] < upper[i-1]:
            upper[i] = min(up, upper[i-1])
        else:
            upper[i] = up

        # Determine trend using previous bar's stops
        if trend[i-1] == 0 and high_price[i] > upper[i-1]:
            trend[i] = 1  # Flip to bullish
        elif trend[i-1] == 1 and low_price[i] < lower[i-1]:
            trend[i] = 0  # Flip to bearish
        else:
            trend[i] = trend[i-1]

        # Set SuperTrend value
        supertrend_val[i] = lower[i] if trend[i] == 1 else upper[i]

        # Generate signals
        if trend[i] > trend[i-1]:
            signal[i] = 1  # Buy signal
        elif trend[i] < trend[i-1]:
            signal[i] = -1  # Sell signal

    # Add results to DataFrame
    df['supertrend'] = supertrend_val
    df['supertrend_trend'] = trend
    df['supertrend_upper'] = upper
    df['supertrend_lower'] = lower
    df['supertrend_signal'] = signal

    # Clean up
    df = df.drop(columns=['_atr'])

    if is_polars:
        import polars as pl
        return pl.from_pandas(df)

    return df


def supertrend_signal(
    data: Union[PdDataFrame, PlDataFrame]
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate trading signals from SuperTrend indicator.

    Expects the DataFrame to already have 'supertrend_signal' column.

    Parameters:
        data: DataFrame with supertrend_signal column

    Returns:
        DataFrame with additional signal columns:
        - supertrend_buy: 1 where buy signal occurs
        - supertrend_sell: 1 where sell signal occurs
    """
    is_polars = isinstance(data, PlDataFrame)
    if is_polars:
        df = data.to_pandas()
    else:
        df = data.copy()

    df['supertrend_buy'] = (df['supertrend_signal'] == 1).astype(int)
    df['supertrend_sell'] = (df['supertrend_signal'] == -1).astype(int)

    if is_polars:
        import polars as pl
        return pl.from_pandas(df)

    return df


def get_supertrend_stats(data: Union[PdDataFrame, PlDataFrame]) -> Dict:
    """
    Get statistics from SuperTrend indicator calculation.

    Parameters:
        data: DataFrame with SuperTrend columns

    Returns:
        Dictionary with statistics:
        - buy_signals: Number of buy signals
        - sell_signals: Number of sell signals
        - current_trend: Current trend ('bullish' or 'bearish')
        - avg_factor: Average factor used (for AI version)
        - avg_perf_idx: Average performance index (for AI version)
    """
    is_polars = isinstance(data, PlDataFrame)
    if is_polars:
        df = data.to_pandas()
    else:
        df = data

    stats = {
        'buy_signals': int((df['supertrend_signal'] == 1).sum()),
        'sell_signals': int((df['supertrend_signal'] == -1).sum()),
        'current_trend': 'bullish' if df['supertrend_trend'].iloc[-1] == 1
                         else 'bearish',
    }

    # AI-specific stats
    if 'supertrend_factor' in df.columns:
        stats['avg_factor'] = float(df['supertrend_factor'].mean())

    if 'supertrend_perf_idx' in df.columns:
        stats['avg_perf_idx'] = float(df['supertrend_perf_idx'].mean())

    return stats
