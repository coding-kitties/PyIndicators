from typing import Union, List, Dict
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
import numpy as np
from pyindicators.exceptions import PyIndicatorException


def order_blocks(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 10,
    use_body: bool = False,
    high_column: str = 'High',
    low_column: str = 'Low',
    open_column: str = 'Open',
    close_column: str = 'Close',
    bullish_ob_column: str = 'bullish_ob',
    bearish_ob_column: str = 'bearish_ob',
    bullish_ob_top_column: str = 'bullish_ob_top',
    bullish_ob_bottom_column: str = 'bullish_ob_bottom',
    bearish_ob_top_column: str = 'bearish_ob_top',
    bearish_ob_bottom_column: str = 'bearish_ob_bottom',
    bullish_breaker_column: str = 'bullish_breaker',
    bearish_breaker_column: str = 'bearish_breaker'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Identify Order Blocks (OB) in price data.

    Order Blocks are zones where institutional traders have placed large
    orders, creating significant price movements. They often act as
    strong support/resistance levels.

    This implementation follows the LuxAlgo methodology:
    - Uses swing highs/lows to identify potential reversal points
    - Bullish OB: Forms at swing low, detected when price breaks above
      the swing high
    - Bearish OB: Forms at swing high, detected when price breaks below
      the swing low
    - Breaker blocks: When an order block is broken (price goes through)

    **Bullish Order Block:**
        Identified when price closes above a swing high. The OB zone is
        the candle with the lowest low before that swing high was broken.

    **Bearish Order Block:**
        Identified when price closes below a swing low. The OB zone is
        the candle with the highest high before that swing low was broken.

    Args:
        data: pandas or polars DataFrame with OHLC price data
        swing_length: Lookback period for swing detection (default: 10)
        use_body: Use candle body instead of high/low for zone boundaries
            (default: False)
        high_column: Column name for high prices (default: 'High')
        low_column: Column name for low prices (default: 'Low')
        open_column: Column name for open prices (default: 'Open')
        close_column: Column name for close prices (default: 'Close')
        bullish_ob_column: Result column for bullish OB signal
            (default: 'bullish_ob')
        bearish_ob_column: Result column for bearish OB signal
            (default: 'bearish_ob')
        bullish_ob_top_column: Result column for bullish OB top
            (default: 'bullish_ob_top')
        bullish_ob_bottom_column: Result column for bullish OB bottom
            (default: 'bullish_ob_bottom')
        bearish_ob_top_column: Result column for bearish OB top
            (default: 'bearish_ob_top')
        bearish_ob_bottom_column: Result column for bearish OB bottom
            (default: 'bearish_ob_bottom')
        bullish_breaker_column: Result column for bullish breaker signal
            (default: 'bullish_breaker')
        bearish_breaker_column: Result column for bearish breaker signal
            (default: 'bearish_breaker')

    Returns:
        DataFrame with added columns:
            - {bullish_ob_column}: 1 when bullish OB is formed, 0 otherwise
            - {bearish_ob_column}: 1 when bearish OB is formed, 0 otherwise
            - {bullish_ob_top_column}: Top of bullish OB zone
            - {bullish_ob_bottom_column}: Bottom of bullish OB zone
            - {bearish_ob_top_column}: Top of bearish OB zone
            - {bearish_ob_bottom_column}: Bottom of bearish OB zone
            - {bullish_breaker_column}: 1 when bullish OB becomes breaker
            - {bearish_breaker_column}: 1 when bearish OB becomes breaker

    Example:
        >>> import pandas as pd
        >>> from pyindicators import order_blocks
        >>> df = pd.DataFrame({
        ...     'Open': [...],
        ...     'High': [...],
        ...     'Low': [...],
        ...     'Close': [...]
        ... })
        >>> result = order_blocks(df, swing_length=10)
    """
    if isinstance(data, PdDataFrame):
        return _order_blocks_pandas(
            data, swing_length, use_body,
            high_column, low_column, open_column, close_column,
            bullish_ob_column, bearish_ob_column,
            bullish_ob_top_column, bullish_ob_bottom_column,
            bearish_ob_top_column, bearish_ob_bottom_column,
            bullish_breaker_column, bearish_breaker_column
        )
    elif isinstance(data, PlDataFrame):
        return _order_blocks_polars(
            data, swing_length, use_body,
            high_column, low_column, open_column, close_column,
            bullish_ob_column, bearish_ob_column,
            bullish_ob_top_column, bullish_ob_bottom_column,
            bearish_ob_top_column, bearish_ob_bottom_column,
            bullish_breaker_column, bearish_breaker_column
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def _order_blocks_pandas(
    data: PdDataFrame,
    swing_length: int,
    use_body: bool,
    high_column: str,
    low_column: str,
    open_column: str,
    close_column: str,
    bullish_ob_column: str,
    bearish_ob_column: str,
    bullish_ob_top_column: str,
    bullish_ob_bottom_column: str,
    bearish_ob_top_column: str,
    bearish_ob_bottom_column: str,
    bullish_breaker_column: str,
    bearish_breaker_column: str
) -> PdDataFrame:
    """Pandas implementation of order blocks."""
    high = data[high_column].values
    low = data[low_column].values
    open_price = data[open_column].values
    close = data[close_column].values
    n = len(data)

    # Calculate swing highs and lows
    swing_highs, swing_lows = _detect_swings_pandas(
        high, low, swing_length
    )

    # Initialize result arrays
    bullish_ob = np.zeros(n, dtype=int)
    bearish_ob = np.zeros(n, dtype=int)
    bullish_ob_top = np.full(n, np.nan)
    bullish_ob_bottom = np.full(n, np.nan)
    bearish_ob_top = np.full(n, np.nan)
    bearish_ob_bottom = np.full(n, np.nan)
    bullish_breaker = np.zeros(n, dtype=int)
    bearish_breaker = np.zeros(n, dtype=int)

    # Track active order blocks
    active_bullish_obs: List[Dict] = []
    active_bearish_obs: List[Dict] = []

    # Track last swing high/low
    last_swing_high_price = np.nan
    last_swing_high_idx = -1
    last_swing_high_crossed = True

    last_swing_low_price = np.nan
    last_swing_low_idx = -1
    last_swing_low_crossed = True

    # Use body or wick for zone boundaries
    if use_body:
        max_price = np.maximum(close, open_price)
        min_price = np.minimum(close, open_price)
    else:
        max_price = high
        min_price = low

    for i in range(swing_length, n):
        # Check for new swing high
        if swing_highs[i]:
            last_swing_high_price = high[i]
            last_swing_high_idx = i
            last_swing_high_crossed = False

        # Check for new swing low
        if swing_lows[i]:
            last_swing_low_price = low[i]
            last_swing_low_idx = i
            last_swing_low_crossed = False

        # Check for bullish OB formation (price breaks above swing high)
        if (not last_swing_high_crossed and
                not np.isnan(last_swing_high_price) and
                close[i] > last_swing_high_price):
            last_swing_high_crossed = True

            # Find the candle with lowest low between swing high and now
            search_start = last_swing_high_idx + 1
            search_end = i

            if search_start < search_end:
                ob_bottom = min_price[search_start]
                ob_top = max_price[search_start]
                ob_idx = search_start

                for j in range(search_start, search_end):
                    if min_price[j] < ob_bottom:
                        ob_bottom = min_price[j]
                        ob_top = max_price[j]
                        ob_idx = j

                # Record bullish OB
                bullish_ob[i] = 1
                bullish_ob_top[i] = ob_top
                bullish_ob_bottom[i] = ob_bottom

                active_bullish_obs.append({
                    'top': ob_top,
                    'bottom': ob_bottom,
                    'idx': ob_idx,
                    'breaker': False
                })

        # Check for bearish OB formation (price breaks below swing low)
        if (not last_swing_low_crossed and
                not np.isnan(last_swing_low_price) and
                close[i] < last_swing_low_price):
            last_swing_low_crossed = True

            # Find the candle with highest high between swing low and now
            search_start = last_swing_low_idx + 1
            search_end = i

            if search_start < search_end:
                ob_top = max_price[search_start]
                ob_bottom = min_price[search_start]
                ob_idx = search_start

                for j in range(search_start, search_end):
                    if max_price[j] > ob_top:
                        ob_top = max_price[j]
                        ob_bottom = min_price[j]
                        ob_idx = j

                # Record bearish OB
                bearish_ob[i] = 1
                bearish_ob_top[i] = ob_top
                bearish_ob_bottom[i] = ob_bottom

                active_bearish_obs.append({
                    'top': ob_top,
                    'bottom': ob_bottom,
                    'idx': ob_idx,
                    'breaker': False
                })

        # Check for breaker blocks (OB invalidation)
        body_low = min(close[i], open_price[i])
        body_high = max(close[i], open_price[i])

        # Check bullish OBs for breaks
        for ob in active_bullish_obs[:]:
            if not ob['breaker']:
                # Bullish OB becomes breaker when price breaks below bottom
                if body_low < ob['bottom']:
                    ob['breaker'] = True
                    bullish_breaker[i] = 1
            else:
                # Remove breaker if price closes above top
                if close[i] > ob['top']:
                    active_bullish_obs.remove(ob)

        # Check bearish OBs for breaks
        for ob in active_bearish_obs[:]:
            if not ob['breaker']:
                # Bearish OB becomes breaker when price breaks above top
                if body_high > ob['top']:
                    ob['breaker'] = True
                    bearish_breaker[i] = 1
            else:
                # Remove breaker if price closes below bottom
                if close[i] < ob['bottom']:
                    active_bearish_obs.remove(ob)

    # Add results to dataframe
    data[bullish_ob_column] = bullish_ob
    data[bearish_ob_column] = bearish_ob
    data[bullish_ob_top_column] = bullish_ob_top
    data[bullish_ob_bottom_column] = bullish_ob_bottom
    data[bearish_ob_top_column] = bearish_ob_top
    data[bearish_ob_bottom_column] = bearish_ob_bottom
    data[bullish_breaker_column] = bullish_breaker
    data[bearish_breaker_column] = bearish_breaker

    return data


def _detect_swings_pandas(
    high: np.ndarray,
    low: np.ndarray,
    length: int
) -> tuple:
    """Detect swing highs and lows using rolling window."""
    n = len(high)
    swing_highs = np.zeros(n, dtype=bool)
    swing_lows = np.zeros(n, dtype=bool)

    # Calculate rolling highest high and lowest low
    for i in range(length, n):
        window_high = high[i - length:i]
        window_low = low[i - length:i]

        highest = np.max(window_high)
        lowest = np.min(window_low)

        # Check if the bar at [i-length] is a swing point
        check_idx = i - length

        # Swing high: high at check_idx equals the highest in window
        if high[check_idx] == highest:
            # Verify it's a local maximum
            is_swing_high = True
            half_len = length // 2
            start = max(0, check_idx - half_len)
            end = min(n, check_idx + half_len + 1)
            for j in range(start, end):
                if j != check_idx and high[j] > high[check_idx]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs[check_idx] = True

        # Swing low: low at check_idx equals the lowest in window
        if low[check_idx] == lowest:
            # Verify it's a local minimum
            is_swing_low = True
            half_len = length // 2
            start = max(0, check_idx - half_len)
            end = min(n, check_idx + half_len + 1)
            for j in range(start, end):
                if j != check_idx and low[j] < low[check_idx]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows[check_idx] = True

    return swing_highs, swing_lows


def _order_blocks_polars(
    data: PlDataFrame,
    swing_length: int,
    use_body: bool,
    high_column: str,
    low_column: str,
    open_column: str,
    close_column: str,
    bullish_ob_column: str,
    bearish_ob_column: str,
    bullish_ob_top_column: str,
    bullish_ob_bottom_column: str,
    bearish_ob_top_column: str,
    bearish_ob_bottom_column: str,
    bullish_breaker_column: str,
    bearish_breaker_column: str
) -> PlDataFrame:
    """Polars implementation of order blocks."""
    # Convert to pandas for processing, then back to polars
    # (Complex stateful logic is easier in pandas)
    pdf = data.to_pandas()

    result = _order_blocks_pandas(
        pdf, swing_length, use_body,
        high_column, low_column, open_column, close_column,
        bullish_ob_column, bearish_ob_column,
        bullish_ob_top_column, bullish_ob_bottom_column,
        bearish_ob_top_column, bearish_ob_bottom_column,
        bullish_breaker_column, bearish_breaker_column
    )

    return pl.from_pandas(result)


def get_active_order_blocks(
    data: Union[PdDataFrame, PlDataFrame],
    max_bullish: int = 3,
    max_bearish: int = 3,
    bullish_ob_column: str = 'bullish_ob',
    bearish_ob_column: str = 'bearish_ob',
    bullish_ob_top_column: str = 'bullish_ob_top',
    bullish_ob_bottom_column: str = 'bullish_ob_bottom',
    bearish_ob_top_column: str = 'bearish_ob_top',
    bearish_ob_bottom_column: str = 'bearish_ob_bottom',
    bullish_breaker_column: str = 'bullish_breaker',
    bearish_breaker_column: str = 'bearish_breaker',
    close_column: str = 'Close'
) -> Dict[str, List[Dict]]:
    """
    Get currently active order blocks from the data.

    Returns the most recent unbroken order blocks that can be used
    for visualization or signal generation.

    Args:
        data: DataFrame with order block columns calculated
        max_bullish: Maximum number of bullish OBs to return (default: 3)
        max_bearish: Maximum number of bearish OBs to return (default: 3)
        [column parameters]: Column names matching order_blocks() output

    Returns:
        Dictionary with 'bullish' and 'bearish' keys, each containing
        a list of order block dictionaries with 'top', 'bottom', 'idx',
        and 'breaker' keys.

    Example:
        >>> df = order_blocks(df)
        >>> active = get_active_order_blocks(df)
        >>> print(active['bullish'])  # List of bullish OBs
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    close = data[close_column].values
    n = len(data)

    # Find all order blocks and track their status
    bullish_obs = []
    bearish_obs = []

    # Get all bullish OB formations
    bull_mask = data[bullish_ob_column] == 1
    for idx in data[bull_mask].index:
        i = data.index.get_loc(idx) if hasattr(data.index, 'get_loc') else idx
        ob = {
            'top': data[bullish_ob_top_column].iloc[i],
            'bottom': data[bullish_ob_bottom_column].iloc[i],
            'idx': i,
            'breaker': False
        }
        bullish_obs.append(ob)

    # Get all bearish OB formations
    bear_mask = data[bearish_ob_column] == 1
    for idx in data[bear_mask].index:
        i = data.index.get_loc(idx) if hasattr(data.index, 'get_loc') else idx
        ob = {
            'top': data[bearish_ob_top_column].iloc[i],
            'bottom': data[bearish_ob_bottom_column].iloc[i],
            'idx': i,
            'breaker': False
        }
        bearish_obs.append(ob)

    # Check which OBs are still active (not broken through)

    active_bullish = []
    for ob in reversed(bullish_obs):
        if len(active_bullish) >= max_bullish:
            break
        # Check if OB has been broken
        ob_broken = False
        for i in range(ob['idx'] + 1, n):
            body_low = min(close[i], data[close_column].iloc[i])
            if body_low < ob['bottom']:
                ob['breaker'] = True
                # Check if breaker is still valid
                if close[i] > ob['top']:
                    ob_broken = True
                    break
        if not ob_broken:
            active_bullish.append(ob)

    active_bearish = []
    for ob in reversed(bearish_obs):
        if len(active_bearish) >= max_bearish:
            break
        # Check if OB has been broken
        ob_broken = False
        for i in range(ob['idx'] + 1, n):
            body_high = max(close[i], data[close_column].iloc[i])
            if body_high > ob['top']:
                ob['breaker'] = True
                # Check if breaker is still valid
                if close[i] < ob['bottom']:
                    ob_broken = True
                    break
        if not ob_broken:
            active_bearish.append(ob)

    return {
        'bullish': active_bullish,
        'bearish': active_bearish
    }


def ob_signal(
    data: Union[PdDataFrame, PlDataFrame],
    close_column: str = 'Close',
    bullish_ob_top_column: str = 'bullish_ob_top',
    bullish_ob_bottom_column: str = 'bullish_ob_bottom',
    bearish_ob_top_column: str = 'bearish_ob_top',
    bearish_ob_bottom_column: str = 'bearish_ob_bottom',
    signal_column: str = 'ob_signal'
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate signals when price enters an Order Block zone.

    Signal values:
        - 1: Price is within a bullish OB zone (potential long entry)
        - -1: Price is within a bearish OB zone (potential short entry)
        - 0: Price is not within any OB zone

    Args:
        data: DataFrame with order block columns calculated
        close_column: Column name for close prices (default: 'Close')
        bullish_ob_top_column: Column for bullish OB top
        bullish_ob_bottom_column: Column for bullish OB bottom
        bearish_ob_top_column: Column for bearish OB top
        bearish_ob_bottom_column: Column for bearish OB bottom
        signal_column: Result column name (default: 'ob_signal')

    Returns:
        DataFrame with added signal column

    Example:
        >>> df = order_blocks(df)
        >>> df = ob_signal(df)
        >>> buy_signals = df[df['ob_signal'] == 1]
    """
    if isinstance(data, PdDataFrame):
        # Forward fill OB zones
        bull_top_ff = data[bullish_ob_top_column].ffill()
        bull_bottom_ff = data[bullish_ob_bottom_column].ffill()
        bear_top_ff = data[bearish_ob_top_column].ffill()
        bear_bottom_ff = data[bearish_ob_bottom_column].ffill()

        close = data[close_column]

        # Check if price is in OB zones
        in_bullish_ob = (close >= bull_bottom_ff) & (close <= bull_top_ff)
        in_bearish_ob = (close >= bear_bottom_ff) & (close <= bear_top_ff)

        signal = np.where(
            in_bullish_ob, 1,
            np.where(in_bearish_ob, -1, 0)
        )

        data[signal_column] = signal
        return data

    elif isinstance(data, PlDataFrame):
        # Convert to pandas, process, convert back
        pdf = data.to_pandas()
        result = ob_signal(
            pdf, close_column,
            bullish_ob_top_column, bullish_ob_bottom_column,
            bearish_ob_top_column, bearish_ob_bottom_column,
            signal_column
        )
        return pl.from_pandas(result)

    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )
