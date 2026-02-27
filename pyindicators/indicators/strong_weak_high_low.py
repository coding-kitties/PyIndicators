"""
Strong / Weak High-Low Indicator

Classifies swing highs and lows as **strong** or **weak** based on
volume analysis.  This concept comes from ICT / Smart Money theory:

- **Strong High:**  A swing high formed during a *bearish* trend
  that has *not* been broken.  It represents genuine institutional
  selling and is expected to hold as resistance.
- **Weak High:**  A swing high formed during a *bullish* trend.
  It is considered vulnerable to being taken out.
- **Strong Low:**  A swing low formed during a *bullish* trend
  that has *not* been broken.  Represents institutional demand.
- **Weak Low:**  A swing low formed during a *bearish* trend.
  Vulnerable to being swept.

Additionally, each swing point reports the volume that was
present at the extremum and a percentage of total volume for the
high/low pair (identical to the RUDYINDICATOR's volume labels).

Based on the "Price Action Concepts [RUDYINDICATOR]" methodology.

Three exported functions:
    - ``strong_weak_high_low()``       — core detector
    - ``strong_weak_high_low_signal()`` — directional signal
    - ``get_strong_weak_high_low_stats()`` — summary statistics
"""
from typing import Union, Dict
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


# -------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------

def strong_weak_high_low(
    data: Union[PdDataFrame, PlDataFrame],
    swing_lookback: int = 50,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    volume_column: str = "Volume",
    sw_high_column: str = "sw_high",
    sw_low_column: str = "sw_low",
    sw_high_price_column: str = "sw_high_price",
    sw_low_price_column: str = "sw_low_price",
    sw_high_type_column: str = "sw_high_type",
    sw_low_type_column: str = "sw_low_type",
    sw_high_volume_column: str = "sw_high_volume",
    sw_low_volume_column: str = "sw_low_volume",
    sw_high_vol_pct_column: str = "sw_high_vol_pct",
    sw_low_vol_pct_column: str = "sw_low_vol_pct",
    equilibrium_column: str = "sw_equilibrium",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect and classify swing highs/lows as Strong or Weak.

    The classification depends on the prevailing trend at the time
    of the swing point:

    - If trend is bearish when a swing high forms → **Strong High**
    - If trend is bullish when a swing high forms → **Weak High**
    - If trend is bullish when a swing low forms → **Strong Low**
    - If trend is bearish when a swing low forms → **Weak Low**

    Trend is determined using a simple highest/lowest breakout
    method over ``swing_lookback`` bars (matching TradingView's
    approach in the RUDYINDICATOR).

    Args:
        data: pandas or polars DataFrame with OHLCV data.
        swing_lookback: Lookback period for swing detection and
            trend determination (default: 50).
        high_column: Column name for highs (default: ``"High"``).
        low_column: Column name for lows (default: ``"Low"``).
        close_column: Column name for closes (default: ``"Close"``).
        volume_column: Column name for volume (default: ``"Volume"``).
        sw_high_column: Output — 1 when a swing high is confirmed
            (default: ``"sw_high"``).
        sw_low_column: Output — 1 when a swing low is confirmed
            (default: ``"sw_low"``).
        sw_high_price_column: Price of the swing high
            (default: ``"sw_high_price"``).
        sw_low_price_column: Price of the swing low
            (default: ``"sw_low_price"``).
        sw_high_type_column: ``"Strong"`` or ``"Weak"``
            (default: ``"sw_high_type"``).
        sw_low_type_column: ``"Strong"`` or ``"Weak"``
            (default: ``"sw_low_type"``).
        sw_high_volume_column: Volume at the swing high bar
            (default: ``"sw_high_volume"``).
        sw_low_volume_column: Volume at the swing low bar
            (default: ``"sw_low_volume"``).
        sw_high_vol_pct_column: Volume % of high vs (high+low) vol
            (default: ``"sw_high_vol_pct"``).
        sw_low_vol_pct_column: Volume % of low vs (high+low) vol
            (default: ``"sw_low_vol_pct"``).
        equilibrium_column: Midpoint between current strong high
            and strong low (default: ``"sw_equilibrium"``).

    Returns:
        DataFrame with added columns.

    Example:
        >>> import pandas as pd
        >>> from pyindicators import strong_weak_high_low
        >>> df = strong_weak_high_low(df, swing_lookback=50)
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        pdf = _compute_sw(
            pdf, swing_lookback,
            high_column, low_column, close_column, volume_column,
            sw_high_column, sw_low_column,
            sw_high_price_column, sw_low_price_column,
            sw_high_type_column, sw_low_type_column,
            sw_high_volume_column, sw_low_volume_column,
            sw_high_vol_pct_column, sw_low_vol_pct_column,
            equilibrium_column,
        )
        import polars as pl
        return pl.from_pandas(pdf)

    if isinstance(data, PdDataFrame):
        return _compute_sw(
            data, swing_lookback,
            high_column, low_column, close_column, volume_column,
            sw_high_column, sw_low_column,
            sw_high_price_column, sw_low_price_column,
            sw_high_type_column, sw_low_type_column,
            sw_high_volume_column, sw_low_volume_column,
            sw_high_vol_pct_column, sw_low_vol_pct_column,
            equilibrium_column,
        )

    raise PyIndicatorException(
        "Input data must be a pandas or polars DataFrame."
    )


def strong_weak_high_low_signal(
    data: Union[PdDataFrame, PlDataFrame],
    sw_high_column: str = "sw_high",
    sw_low_column: str = "sw_low",
    sw_high_type_column: str = "sw_high_type",
    sw_low_type_column: str = "sw_low_type",
    signal_column: str = "sw_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a directional signal from Strong/Weak High-Low.

    Signal values:
        -  1 : Strong Low detected (bullish — institution buying)
        - -1 : Strong High detected (bearish — institution selling)
        -  0 : Weak swing or no swing

    Args:
        data: DataFrame with SW columns already computed.
        sw_high_column: Column with swing high flags.
        sw_low_column: Column with swing low flags.
        sw_high_type_column: Column with high type classification.
        sw_low_type_column: Column with low type classification.
        signal_column: Output signal column name.

    Returns:
        DataFrame with added signal column.
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        pdf = _compute_signal(
            pdf, sw_high_column, sw_low_column,
            sw_high_type_column, sw_low_type_column, signal_column,
        )
        import polars as pl
        return pl.from_pandas(pdf)

    if isinstance(data, PdDataFrame):
        return _compute_signal(
            data, sw_high_column, sw_low_column,
            sw_high_type_column, sw_low_type_column, signal_column,
        )

    raise PyIndicatorException(
        "Input data must be a pandas or polars DataFrame."
    )


def get_strong_weak_high_low_stats(
    data: Union[PdDataFrame, PlDataFrame],
    sw_high_column: str = "sw_high",
    sw_low_column: str = "sw_low",
    sw_high_type_column: str = "sw_high_type",
    sw_low_type_column: str = "sw_low_type",
) -> Dict[str, object]:
    """
    Compute summary statistics for Strong/Weak High-Low detections.

    Returns:
        dict with keys:
            - total_swing_highs  : int
            - total_swing_lows   : int
            - strong_highs       : int
            - weak_highs         : int
            - strong_lows        : int
            - weak_lows          : int
            - total              : int
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    if not isinstance(data, PdDataFrame):
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )

    sh = int(data[sw_high_column].sum())
    sl = int(data[sw_low_column].sum())

    high_mask = data[sw_high_column] == 1
    low_mask = data[sw_low_column] == 1

    strong_h = int((data.loc[high_mask, sw_high_type_column] == "Strong").sum()) if sh else 0
    weak_h = sh - strong_h
    strong_l = int((data.loc[low_mask, sw_low_type_column] == "Strong").sum()) if sl else 0
    weak_l = sl - strong_l

    return {
        "total_swing_highs": sh,
        "total_swing_lows": sl,
        "strong_highs": strong_h,
        "weak_highs": weak_h,
        "strong_lows": strong_l,
        "weak_lows": weak_l,
        "total": sh + sl,
    }


# -------------------------------------------------------------------
#  Internal helpers
# -------------------------------------------------------------------

def _detect_pivot_highs(high: np.ndarray, length: int) -> np.ndarray:
    """Return boolean array — True at confirmed pivot high positions."""
    n = len(high)
    result = np.zeros(n, dtype=bool)
    for i in range(length, n - length):
        is_pivot = True
        for j in range(1, length + 1):
            if high[i] < high[i - j] or high[i] < high[i + j]:
                is_pivot = False
                break
        if is_pivot:
            result[i] = True
    return result


def _detect_pivot_lows(low: np.ndarray, length: int) -> np.ndarray:
    """Return boolean array — True at confirmed pivot low positions."""
    n = len(low)
    result = np.zeros(n, dtype=bool)
    for i in range(length, n - length):
        is_pivot = True
        for j in range(1, length + 1):
            if low[i] > low[i - j] or low[i] > low[i + j]:
                is_pivot = False
                break
        if is_pivot:
            result[i] = True
    return result


def _compute_sw(
    df: PdDataFrame,
    swing_lookback: int,
    high_col: str, low_col: str, close_col: str, vol_col: str,
    sh_col: str, sl_col: str,
    sh_price: str, sl_price: str,
    sh_type: str, sl_type: str,
    sh_vol: str, sl_vol: str,
    sh_vpct: str, sl_vpct: str,
    eq_col: str,
) -> PdDataFrame:
    """Core numpy computation for Strong/Weak High/Low."""
    n = len(df)
    h = df[high_col].values.astype(float)
    low = df[low_col].values.astype(float)
    v = df[vol_col].values.astype(float)

    # Swing detection using pivot logic with the lookback
    piv_len = max(swing_lookback // 5, 2)  # scale pivot length
    piv_highs = _detect_pivot_highs(h, piv_len)
    piv_lows = _detect_pivot_lows(low, piv_len)

    # Trend detection: rolling highest/lowest breakout
    # trend = 0 initially, 1 = bullish, -1 = bearish
    trend = np.zeros(n, dtype=int)
    for i in range(swing_lookback, n):
        upper = np.max(h[i - swing_lookback:i])
        lower = np.min(low[i - swing_lookback:i])
        if h[i] > upper:
            trend[i] = 0  # breakout up → marks top → trend was bullish
        elif low[i] < lower:
            trend[i] = 1  # breakout down → marks bottom → trend was bearish
        else:
            trend[i] = trend[i - 1]

    # Classify and output
    sh_flag = np.zeros(n, dtype=int)
    sl_flag = np.zeros(n, dtype=int)
    sh_p = np.full(n, np.nan)
    sl_p = np.full(n, np.nan)
    sh_t = np.full(n, None, dtype=object)
    sl_t = np.full(n, None, dtype=object)
    sh_v = np.full(n, np.nan)
    sl_v = np.full(n, np.nan)
    sh_vp = np.full(n, np.nan)
    sl_vp = np.full(n, np.nan)
    eq = np.full(n, np.nan)

    last_sh_price = np.nan
    last_sl_price = np.nan
    last_high_vol = np.nan
    last_low_vol = np.nan

    for i in range(n):
        # Swing high confirmed (delayed by piv_len bars for look-ahead)
        if piv_highs[i]:
            sh_flag[i] = 1
            sh_p[i] = h[i]
            sh_v[i] = v[i]
            last_high_vol = v[i]

            # Trend-based classification
            # Bearish trend → high is "strong" (unlikely to break)
            # Bullish trend → high is "weak"
            if trend[i] <= 0:
                sh_t[i] = "Strong"
            else:
                sh_t[i] = "Weak"

            last_sh_price = h[i]

            # Volume percentage
            if not np.isnan(last_low_vol) and (last_high_vol + last_low_vol) > 0:
                sh_vp[i] = round(last_high_vol / (last_high_vol + last_low_vol) * 100, 1)

        if piv_lows[i]:
            sl_flag[i] = 1
            sl_p[i] = low[i]
            sl_v[i] = v[i]
            last_low_vol = v[i]

            # Bullish trend → low is "strong" (unlikely to break)
            # Bearish trend → low is "weak"
            if trend[i] >= 1:
                sl_t[i] = "Strong"
            else:
                sl_t[i] = "Weak"

            last_sl_price = low[i]

            # Volume percentage
            if not np.isnan(last_high_vol) and (last_high_vol + last_low_vol) > 0:
                sl_vp[i] = round(last_low_vol / (last_high_vol + last_low_vol) * 100, 1)

        # Equilibrium
        if not np.isnan(last_sh_price) and not np.isnan(last_sl_price):
            eq[i] = (last_sh_price + last_sl_price) / 2.0

    df[sh_col] = sh_flag
    df[sl_col] = sl_flag
    df[sh_price] = sh_p
    df[sl_price] = sl_p
    df[sh_type] = sh_t
    df[sl_type] = sl_t
    df[sh_vol] = sh_v
    df[sl_vol] = sl_v
    df[sh_vpct] = sh_vp
    df[sl_vpct] = sl_vp
    df[eq_col] = eq
    return df


def _compute_signal(
    df: PdDataFrame,
    sh_col: str, sl_col: str,
    sh_type: str, sl_type: str,
    sig_col: str,
) -> PdDataFrame:
    """Generate signal: +1 for strong low, -1 for strong high, else 0."""
    n = len(df)
    sig = np.zeros(n, dtype=int)
    sh_flags = df[sh_col].values
    sl_flags = df[sl_col].values
    sh_types = df[sh_type].values
    sl_types = df[sl_type].values

    for i in range(n):
        if sl_flags[i] == 1 and sl_types[i] == "Strong":
            sig[i] = 1
        elif sh_flags[i] == 1 and sh_types[i] == "Strong":
            sig[i] = -1

    df[sig_col] = sig
    return df
