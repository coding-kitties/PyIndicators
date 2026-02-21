"""
Premium & Discount Zones

Identifies Premium, Discount, and Equilibrium zones based on the
current market range defined by swing highs and swing lows.

**Concept:**
    In Smart Money Concepts (SMC) trading, the market is divided into
    zones relative to the most recent significant swing range:

    - **Premium Zone** – the upper half of the range (above equilibrium).
      Price in this zone is considered *expensive* / overvalued.  Smart
      money is more likely to *sell* in this zone.

    - **Discount Zone** – the lower half of the range (below
      equilibrium).  Price in this zone is considered *cheap* /
      undervalued.  Smart money is more likely to *buy* here.

    - **Equilibrium** – the exact midpoint (50 %) of the range.  It
      acts as a decision boundary between premium and discount.

**Calculation:**
    1. Detect swing highs and swing lows using a configurable pivot
       length.
    2. Track the most recent swing high (``range_high``) and swing
       low (``range_low``) to define the current range.
    3. Compute:
       - ``equilibrium = (range_high + range_low) / 2``
       - ``premium_zone_lower = equilibrium`` (lower bound of
         premium zone)
       - ``premium_zone_upper = range_high``
       - ``discount_zone_upper = equilibrium`` (upper bound of
         discount zone)
       - ``discount_zone_lower = range_low``
    4. Classify the current close as:
       - ``"premium"``  if ``close > equilibrium``
       - ``"discount"`` if ``close < equilibrium``
       - ``"equilibrium"`` if ``close == equilibrium``
    5. Compute a *premium/discount percentage* indicating how deep
       into the zone the price sits:
       - In premium: ``pct = (close - equilibrium) /
         (range_high - equilibrium) * 100``
       - In discount: ``pct = (equilibrium - close) /
         (equilibrium - range_low) * 100``
       A value of 100 % means price is at the extreme of the zone.
"""
from typing import Union, Dict
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


def premium_discount_zones(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 10,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    range_high_column: str = "pdz_range_high",
    range_low_column: str = "pdz_range_low",
    equilibrium_column: str = "pdz_equilibrium",
    zone_column: str = "pdz_zone",
    zone_pct_column: str = "pdz_zone_pct",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Compute Premium / Discount Zones for a price series.

    Uses pivot-based swing detection to identify the current range,
    then classifies each bar as premium, discount, or equilibrium.

    Args:
        data: pandas or polars DataFrame with OHLC price data.
        swing_length: Number of bars on each side required to
            confirm a swing high or swing low (default: 10).
        high_column: Column name for highs (default: ``"High"``).
        low_column: Column name for lows (default: ``"Low"``).
        close_column: Column name for closes (default: ``"Close"``).
        range_high_column: Output column for the upper boundary of
            the current range (default: ``"pdz_range_high"``).
        range_low_column: Output column for the lower boundary of
            the current range (default: ``"pdz_range_low"``).
        equilibrium_column: Output column for the equilibrium /
            midpoint of the current range
            (default: ``"pdz_equilibrium"``).
        zone_column: Output column indicating the zone the close
            price falls in – ``"premium"``, ``"discount"``, or
            ``"equilibrium"`` (default: ``"pdz_zone"``).
        zone_pct_column: Output column with the depth percentage
            into the current zone (0–100). A value of 100 means
            price is at the extreme boundary of the zone.
            (default: ``"pdz_zone_pct"``).

    Returns:
        DataFrame with added columns:

        - ``{range_high_column}``: Current swing range high.
        - ``{range_low_column}``: Current swing range low.
        - ``{equilibrium_column}``: Midpoint of the range.
        - ``{zone_column}``: ``"premium"`` / ``"discount"`` /
          ``"equilibrium"`` (or ``""`` when no range established).
        - ``{zone_pct_column}``: How deep into the zone (0-100).

    Example:
        >>> import pandas as pd
        >>> from pyindicators import premium_discount_zones
        >>> df = pd.DataFrame({
        ...     'High': [...],
        ...     'Low': [...],
        ...     'Close': [...]
        ... })
        >>> result = premium_discount_zones(df, swing_length=10)
    """
    if isinstance(data, PdDataFrame):
        return _pdz_pandas(
            data,
            swing_length=swing_length,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            range_high_column=range_high_column,
            range_low_column=range_low_column,
            equilibrium_column=equilibrium_column,
            zone_column=zone_column,
            zone_pct_column=zone_pct_column,
        )
    elif isinstance(data, PlDataFrame):
        pd_data = data.to_pandas()
        result = _pdz_pandas(
            pd_data,
            swing_length=swing_length,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            range_high_column=range_high_column,
            range_low_column=range_low_column,
            equilibrium_column=equilibrium_column,
            zone_column=zone_column,
            zone_pct_column=zone_pct_column,
        )
        import polars as pl

        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def premium_discount_zones_signal(
    data: Union[PdDataFrame, PlDataFrame],
    zone_column: str = "pdz_zone",
    signal_column: str = "pdz_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a trading signal from Premium / Discount Zones.

    Args:
        data: DataFrame containing the ``zone_column`` (output of
            :func:`premium_discount_zones`).
        zone_column: Column with zone labels (default:
            ``"pdz_zone"``).
        signal_column: Output column name (default:
            ``"pdz_signal"``).

    Returns:
        DataFrame with ``{signal_column}`` added:

        - ``1``  – price is in the **discount** zone (potential buy).
        - ``-1`` – price is in the **premium** zone (potential sell).
        - ``0``  – equilibrium or no range established.
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        signal = np.where(
            data[zone_column] == "discount",
            1,
            np.where(
                data[zone_column] == "premium",
                -1,
                0,
            ),
        )
        data[signal_column] = signal.astype(int)
        return data
    elif isinstance(data, PlDataFrame):
        import polars as pl

        return data.with_columns(
            pl.when(pl.col(zone_column) == "discount")
            .then(1)
            .when(pl.col(zone_column) == "premium")
            .then(-1)
            .otherwise(0)
            .cast(pl.Int64)
            .alias(signal_column)
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


def get_premium_discount_zones_stats(
    data: Union[PdDataFrame, PlDataFrame],
    zone_column: str = "pdz_zone",
    zone_pct_column: str = "pdz_zone_pct",
) -> Dict:
    """
    Return summary statistics for Premium / Discount Zones.

    Args:
        data: DataFrame containing zone columns (output of
            :func:`premium_discount_zones`).
        zone_column: Column with zone labels.
        zone_pct_column: Column with zone depth percentages.

    Returns:
        Dictionary with keys:

        - ``total_bars`` – total number of bars.
        - ``premium_bars`` – bars in premium zone.
        - ``discount_bars`` – bars in discount zone.
        - ``equilibrium_bars`` – bars exactly at equilibrium.
        - ``no_zone_bars`` – bars with no range established yet.
        - ``premium_ratio`` – fraction of classified bars in
          premium (0-1).
        - ``discount_ratio`` – fraction of classified bars in
          discount (0-1).
        - ``avg_premium_pct`` – average depth percentage when in
          premium zone.
        - ``avg_discount_pct`` – average depth percentage when in
          discount zone.
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    total = len(data)
    premium = int((data[zone_column] == "premium").sum())
    discount = int((data[zone_column] == "discount").sum())
    equilibrium = int((data[zone_column] == "equilibrium").sum())
    no_zone = int((data[zone_column] == "").sum())
    classified = premium + discount + equilibrium

    # Average zone depth percentages
    premium_mask = data[zone_column] == "premium"
    discount_mask = data[zone_column] == "discount"
    avg_premium_pct = (
        round(float(data.loc[premium_mask, zone_pct_column].mean()), 2)
        if premium > 0
        else 0.0
    )
    avg_discount_pct = (
        round(float(data.loc[discount_mask, zone_pct_column].mean()), 2)
        if discount > 0
        else 0.0
    )

    return {
        "total_bars": total,
        "premium_bars": premium,
        "discount_bars": discount,
        "equilibrium_bars": equilibrium,
        "no_zone_bars": no_zone,
        "premium_ratio": (
            round(premium / classified, 4) if classified > 0 else 0.0
        ),
        "discount_ratio": (
            round(discount / classified, 4) if classified > 0 else 0.0
        ),
        "avg_premium_pct": avg_premium_pct,
        "avg_discount_pct": avg_discount_pct,
    }


# -------------------------------------------------------------------
#  Internal implementation
# -------------------------------------------------------------------


def _detect_pivots_high(high: np.ndarray, length: int) -> np.ndarray:
    """Return boolean array where True at position p means high[p] is
    a swing high (higher than all neighbours within ``length`` bars)."""
    n = len(high)
    result = np.zeros(n, dtype=bool)
    for i in range(length, n - length):
        window = high[i - length:i + length + 1]
        if high[i] >= np.max(window):
            result[i] = True
    return result


def _detect_pivots_low(low: np.ndarray, length: int) -> np.ndarray:
    """Return boolean array where True at position p means low[p] is
    a swing low (lower than all neighbours within ``length`` bars)."""
    n = len(low)
    result = np.zeros(n, dtype=bool)
    for i in range(length, n - length):
        window = low[i - length:i + length + 1]
        if low[i] <= np.min(window):
            result[i] = True
    return result


def _pdz_pandas(
    data: PdDataFrame,
    swing_length: int,
    high_column: str,
    low_column: str,
    close_column: str,
    range_high_column: str,
    range_low_column: str,
    equilibrium_column: str,
    zone_column: str,
    zone_pct_column: str,
) -> PdDataFrame:
    """Core pandas implementation of Premium / Discount Zones.

    Steps
    -----
    1. Detect swing highs and swing lows using a standard pivot
       algorithm (bar must be the extreme within ``swing_length``
       bars on each side).  A swing point at bar *p* is confirmed
       at bar ``p + swing_length``.
    2. Maintain a running *range_high* and *range_low* from the
       most recently confirmed swing high / low.  The range updates
       each time a new swing is confirmed.
    3. Compute equilibrium = (range_high + range_low) / 2.
    4. Classify the close as premium / discount / equilibrium.
    5. Compute the depth percentage into the current zone.
    """
    data = data.copy()
    high = data[high_column].values.astype(float)
    low = data[low_column].values.astype(float)
    close = data[close_column].values.astype(float)
    n = len(data)

    # Detect pivots
    pivot_highs = _detect_pivots_high(high, swing_length)
    pivot_lows = _detect_pivots_low(low, swing_length)

    # Output arrays
    range_high_arr = np.full(n, np.nan)
    range_low_arr = np.full(n, np.nan)
    equilibrium_arr = np.full(n, np.nan)
    zone_arr = np.full(n, "", dtype=object)
    zone_pct_arr = np.full(n, np.nan)

    # State
    current_range_high = np.nan
    current_range_low = np.nan

    for i in range(n):
        # Confirmation bar: a pivot at position p is confirmed
        # at bar p + swing_length
        p = i - swing_length
        if p >= swing_length:
            if pivot_highs[p]:
                current_range_high = high[p]
            if pivot_lows[p]:
                current_range_low = low[p]

        # Record current range values
        range_high_arr[i] = current_range_high
        range_low_arr[i] = current_range_low

        # Compute equilibrium and classify if we have a valid range
        if not np.isnan(current_range_high) and not np.isnan(
            current_range_low
        ):
            eq = (current_range_high + current_range_low) / 2.0
            equilibrium_arr[i] = eq

            price = close[i]
            half_range = current_range_high - eq  # same as eq - range_low

            if half_range > 0:
                if price > eq:
                    zone_arr[i] = "premium"
                    # Depth into premium zone (0..100+)
                    zone_pct_arr[i] = min(
                        (price - eq) / half_range * 100.0, 100.0
                    )
                elif price < eq:
                    zone_arr[i] = "discount"
                    # Depth into discount zone (0..100+)
                    zone_pct_arr[i] = min(
                        (eq - price) / half_range * 100.0, 100.0
                    )
                else:
                    zone_arr[i] = "equilibrium"
                    zone_pct_arr[i] = 0.0
            else:
                # Degenerate range (high == low); treat as equilibrium
                zone_arr[i] = "equilibrium"
                zone_pct_arr[i] = 0.0

    # Assign to DataFrame
    data[range_high_column] = range_high_arr
    data[range_low_column] = range_low_arr
    data[equilibrium_column] = equilibrium_arr
    data[zone_column] = zone_arr
    data[zone_pct_column] = zone_pct_arr

    return data
