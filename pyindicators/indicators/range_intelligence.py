"""
Range Intelligence Suite

Detects consolidation ranges where price compresses into a tight band,
enriched with volume profiling, Point of Control (POC), net delta
analysis, and liquidity sweep detection.

Ported from the TradingView PineScript indicator
"Range Intelligence Suite [LuxAlgo]" (CC BY-NC-SA 4.0 © LuxAlgo).

**Core Concept:**
    A consolidation range forms when the rolling ``highest(high, length)
    - lowest(low, length)`` is less than ``sensitivity * ATR(length)``.
    This identifies periods of compression/accumulation before breakouts.

**Range Boundaries:**
    - ``range_high`` = highest high over the detection window
    - ``range_low`` = lowest low over the detection window
    - ``range_mid`` = (range_high + range_low) / 2

**Volume Profile:**
    Within each range, volume is distributed across ``vp_rows``
    horizontal bins.  Each bar's volume is allocated proportionally
    to the bins its wick covers, split into buy (bullish candle)
    and sell (bearish candle) volume.

**Point of Control (POC):**
    The price level (bin midpoint) with the highest total volume.

**Net Delta:**
    Sum of all bullish volume minus bearish volume within the range.
    Positive → accumulation (institutional buying).
    Negative → distribution (institutional selling).

**Liquidity Sweeps (Fakeouts):**
    A bar that wicks beyond the range boundary but closes back
    inside the range.  These indicate stop hunts:
    - Sweep High: ``high > range_high and close < range_high``
    - Sweep Low:  ``low  < range_low  and close > range_low``

**Breakout:**
    The range ends when ``close > range_high`` (bullish breakout)
    or ``close < range_low`` (bearish breakout).

**Ready Score (Intensity):**
    A 0–100 composite score combining range duration and volume
    imbalance, indicating how "ready" the range is for a breakout.

**Signals:**
    -  ``1`` — bullish breakout (close above range high)
    - ``-1`` — bearish breakout (close below range low)
    -  ``0`` — no breakout

Three exported functions follow the library convention:
    - ``range_intelligence()``             — core detector
    - ``range_intelligence_signal()``      — breakout signal
    - ``get_range_intelligence_stats()``   — summary statistics
"""
from typing import Union, Dict, List
from dataclasses import dataclass, field
import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


# ------------------------------------------------------------------ #
#  Data structures                                                    #
# ------------------------------------------------------------------ #
@dataclass
class _VPBin:
    """One horizontal bin of the volume profile."""
    price_low: float
    price_high: float
    buy_volume: float
    sell_volume: float
    total_volume: float
    width_pct: float
    is_poc: bool


@dataclass
class _Range:
    """A detected consolidation range."""
    start_bar: int
    end_bar: int
    range_high: float
    range_low: float
    mid: float
    poc_price: float
    total_volume: float
    net_delta: float
    state: str        # "Accumulation" or "Distribution"
    breakout: str     # "Bullish", "Bearish", or "Active"
    ready_score: float
    sweep_highs: int
    sweep_lows: int
    duration: int
    profile: List[_VPBin] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def _compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int,
) -> np.ndarray:
    """ATR using EMA-style smoothing after initial SMA seed."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr_arr = np.full(n, np.nan)
    if n >= period:
        atr_arr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period

    return atr_arr


def _build_volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    opn: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    range_high: float,
    range_low: float,
    start_idx: int,
    end_idx: int,
    vp_rows: int,
):
    """
    Distribute volume across horizontal bins within the range.

    Returns: (profile_list, poc_price, total_volume, net_delta)
    """
    zone_height = range_high - range_low
    if zone_height <= 0:
        zone_height = 1e-10
    bin_height = zone_height / vp_rows

    buy_bins = np.zeros(vp_rows)
    sell_bins = np.zeros(vp_rows)

    for i in range(start_idx, end_idx + 1):
        if i < 0 or i >= len(high):
            continue

        bar_high = high[i]
        bar_low = low[i]
        bar_vol = volume[i] if not np.isnan(volume[i]) else 1.0
        is_bull = close[i] >= opn[i]

        min_bin = max(0, min(vp_rows - 1, int(
            np.floor((bar_low - range_low)
                     / bin_height))))
        max_bin = max(0, min(vp_rows - 1, int(
            np.floor((bar_high - range_low)
                     / bin_height))))
        bins_covered = max_bin - min_bin + 1
        vol_share = bar_vol / bins_covered if bins_covered > 0 else 0.0

        for b in range(min_bin, max_bin + 1):
            if is_bull:
                buy_bins[b] += vol_share
            else:
                sell_bins[b] += vol_share

    total_bins = buy_bins + sell_bins
    max_vol = np.max(total_bins) if np.max(total_bins) > 0 else 1.0
    poc_idx = int(np.argmax(total_bins))
    poc_price = range_low + poc_idx * bin_height + bin_height / 2.0
    total_volume = float(np.sum(total_bins))
    net_delta = float(np.sum(buy_bins) - np.sum(sell_bins))

    profile = []
    for r in range(vp_rows):
        r_low = range_low + r * bin_height
        r_high = r_low + bin_height
        profile.append(_VPBin(
            price_low=r_low,
            price_high=r_high,
            buy_volume=float(buy_bins[r]),
            sell_volume=float(sell_bins[r]),
            total_volume=float(total_bins[r]),
            width_pct=float(total_bins[r] / max_vol),
            is_poc=(r == poc_idx),
        ))

    return profile, poc_price, total_volume, net_delta


# ------------------------------------------------------------------ #
#  Core pandas implementation                                         #
# ------------------------------------------------------------------ #
def _range_intelligence_pandas(
    data: PdDataFrame,
    length: int,
    sensitivity: float,
    vp_rows: int,
    hide_overlaps: bool,
    high_col: str,
    low_col: str,
    open_col: str,
    close_col: str,
    volume_col: str,
    # Output column names
    ri_active_col: str,
    ri_high_col: str,
    ri_low_col: str,
    ri_mid_col: str,
    ri_poc_col: str,
    ri_delta_col: str,
    ri_state_col: str,
    ri_ready_col: str,
    ri_sweep_high_col: str,
    ri_sweep_low_col: str,
    ri_breakout_col: str,
    ri_duration_col: str,
) -> PdDataFrame:
    """Core numpy/pandas computation."""
    df = data.copy()
    n = len(df)

    high = df[high_col].values.astype(float)
    low = df[low_col].values.astype(float)
    opn = df[open_col].values.astype(float)
    close = df[close_col].values.astype(float)
    vol = df[volume_col].values.astype(float)

    # ATR
    atr = _compute_atr(high, low, close, length)

    # Rolling highest / lowest
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(length - 1, n):
        highest_high[i] = np.max(high[i - length + 1:i + 1])
        lowest_low[i] = np.min(low[i - length + 1:i + 1])

    range_width = highest_high - lowest_low

    # Output arrays
    out_active = np.zeros(n, dtype=int)
    out_high = np.full(n, np.nan)
    out_low = np.full(n, np.nan)
    out_mid = np.full(n, np.nan)
    out_poc = np.full(n, np.nan)
    out_delta = np.full(n, np.nan)
    out_state = np.full(n, "", dtype=object)
    out_ready = np.full(n, np.nan)
    out_sweep_high = np.zeros(n, dtype=int)
    out_sweep_low = np.zeros(n, dtype=int)
    out_breakout = np.zeros(n, dtype=int)   # +1 bull, -1 bear
    out_duration = np.zeros(n, dtype=int)

    # State machine
    range_active = False
    r_high = np.nan
    r_low = np.nan
    r_mid = np.nan
    r_start = 0
    net_delta_acc = 0.0
    poc_price = np.nan
    sweep_h_count = 0
    sweep_l_count = 0

    # Volume bins for the active range (re-used)
    buy_bins = np.zeros(vp_rows)
    sell_bins = np.zeros(vp_rows)

    # Collect completed ranges for overlap management
    completed_ranges: List[_Range] = []

    # Running volume sum for ready-score normalisation
    vol_sum_window = np.zeros(n)
    vol_sum_window[0] = vol[0] if not np.isnan(vol[0]) else 0.0
    for i in range(1, n):
        v = vol[i] if not np.isnan(vol[i]) else 0.0
        vol_sum_window[i] = vol_sum_window[i - 1] + v

    def _vol_sum(start: int, end: int) -> float:
        """Sum of volume from start to end inclusive."""
        s = vol_sum_window[end]
        if start > 0:
            s -= vol_sum_window[start - 1]
        return max(s, 1.0)

    for i in range(n):
        atr_val = atr[i] if not np.isnan(atr[i]) else 0.0

        if i < length - 1 or np.isnan(range_width[i]) or atr_val <= 0:
            continue

        is_consolidating = range_width[i] < (sensitivity * atr_val)

        # ── Detect new range ─────────────────────────────────────
        if is_consolidating and not range_active:
            range_active = True
            r_high = highest_high[i]
            r_low = lowest_low[i]
            r_mid = (r_high + r_low) / 2.0
            r_start = max(0, i - length + 1)
            sweep_h_count = 0
            sweep_l_count = 0

            # Initialise bins with lookback bars
            buy_bins[:] = 0.0
            sell_bins[:] = 0.0
            net_delta_acc = 0.0
            bin_height = (
                (r_high - r_low) / vp_rows
                if r_high > r_low else 1e-10
            )

            for j in range(r_start, i + 1):
                bar_vol = vol[j] if not np.isnan(vol[j]) else 1.0
                is_bull = close[j] >= opn[j]
                min_bin = max(
                    0, min(vp_rows - 1, int(
                        np.floor((low[j] - r_low)
                                 / bin_height))))
                max_bin = max(
                    0, min(vp_rows - 1, int(
                        np.floor((high[j] - r_low)
                                 / bin_height))))
                bc = max_bin - min_bin + 1
                vs = bar_vol / bc if bc > 0 else 0.0
                for b in range(min_bin, max_bin + 1):
                    if is_bull:
                        buy_bins[b] += vs
                    else:
                        sell_bins[b] += vs
                net_delta_acc += (bar_vol if is_bull else -bar_vol)

            # Initial POC
            total_bins = buy_bins + sell_bins
            poc_idx = int(np.argmax(total_bins))
            poc_price = r_low + poc_idx * bin_height + bin_height / 2.0

            # Hide overlapping historical ranges
            if hide_overlaps and completed_ranges:
                completed_ranges = [
                    r for r in completed_ranges
                    if r_start > r.end_bar
                ]

        # ── Update active range ──────────────────────────────────
        if range_active:
            bin_height = (
                (r_high - r_low) / vp_rows
                if r_high > r_low else 1e-10
            )

            # Accumulate current bar volume
            bar_vol = vol[i] if not np.isnan(vol[i]) else 1.0
            is_bull = close[i] >= opn[i]
            min_bin = max(0, min(vp_rows - 1,
                                 int(np.floor((low[i] - r_low) / bin_height))))
            max_bin = max(
                0, min(vp_rows - 1, int(
                    np.floor((high[i] - r_low)
                             / bin_height))))
            bc = max_bin - min_bin + 1
            vs = bar_vol / bc if bc > 0 else 0.0
            for b in range(min_bin, max_bin + 1):
                if is_bull:
                    buy_bins[b] += vs
                else:
                    sell_bins[b] += vs
            net_delta_acc += (bar_vol if is_bull else -bar_vol)

            # Update POC
            total_bins = buy_bins + sell_bins
            poc_idx = int(np.argmax(total_bins))
            poc_price = r_low + poc_idx * bin_height + bin_height / 2.0

            # Liquidity sweeps (fakeouts)
            if high[i] > r_high and close[i] < r_high:
                out_sweep_high[i] = 1
                sweep_h_count += 1
            if low[i] < r_low and close[i] > r_low:
                out_sweep_low[i] = 1
                sweep_l_count += 1

            duration = i - r_start + 1

            # Ready score: composite of duration + delta imbalance
            recent_vol = _vol_sum(max(0, i - length + 1), i)
            intensity = min(100.0,
                            (duration / length) * 50.0
                            + (abs(net_delta_acc) / recent_vol) * 50.0)

            state_str = (
                "Accumulation"
                if net_delta_acc >= 0
                else "Distribution"
            )

            # Write row output
            out_active[i] = 1
            out_high[i] = r_high
            out_low[i] = r_low
            out_mid[i] = r_mid
            out_poc[i] = poc_price
            out_delta[i] = net_delta_acc
            out_state[i] = state_str
            out_ready[i] = round(intensity, 1)
            out_duration[i] = duration

            # ── Breakout check ───────────────────────────────────
            if close[i] > r_high:
                out_breakout[i] = 1   # Bullish breakout
                range_active = False

                # Build final profile
                profile, final_poc, total_vol, final_delta = \
                    _build_volume_profile(
                        high, low, opn, close, vol,
                        r_high, r_low, r_start, i, vp_rows,
                    )
                completed_ranges.append(_Range(
                    start_bar=r_start, end_bar=i,
                    range_high=r_high, range_low=r_low, mid=r_mid,
                    poc_price=final_poc,
                    total_volume=total_vol, net_delta=final_delta,
                    state=state_str,
                    breakout="Bullish",
                    ready_score=round(intensity, 1),
                    sweep_highs=sweep_h_count,
                    sweep_lows=sweep_l_count,
                    duration=duration,
                    profile=profile,
                ))

            elif close[i] < r_low:
                out_breakout[i] = -1  # Bearish breakout
                range_active = False

                profile, final_poc, total_vol, final_delta = \
                    _build_volume_profile(
                        high, low, opn, close, vol,
                        r_high, r_low, r_start, i, vp_rows,
                    )
                completed_ranges.append(_Range(
                    start_bar=r_start, end_bar=i,
                    range_high=r_high, range_low=r_low, mid=r_mid,
                    poc_price=final_poc,
                    total_volume=total_vol, net_delta=final_delta,
                    state=state_str,
                    breakout="Bearish",
                    ready_score=round(intensity, 1),
                    sweep_highs=sweep_h_count,
                    sweep_lows=sweep_l_count,
                    duration=duration,
                    profile=profile,
                ))

    # Assign output columns
    df[ri_active_col] = out_active
    df[ri_high_col] = out_high
    df[ri_low_col] = out_low
    df[ri_mid_col] = out_mid
    df[ri_poc_col] = out_poc
    df[ri_delta_col] = out_delta
    df[ri_state_col] = out_state
    df[ri_ready_col] = out_ready
    df[ri_sweep_high_col] = out_sweep_high
    df[ri_sweep_low_col] = out_sweep_low
    df[ri_breakout_col] = out_breakout
    df[ri_duration_col] = out_duration

    return df


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #
def range_intelligence(
    data: Union[PdDataFrame, PlDataFrame],
    length: int = 20,
    sensitivity: float = 4.0,
    vp_rows: int = 10,
    hide_overlaps: bool = True,
    open_column: str = "Open",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    volume_column: str = "Volume",
    ri_active_column: str = "ri_active",
    ri_high_column: str = "ri_high",
    ri_low_column: str = "ri_low",
    ri_mid_column: str = "ri_mid",
    ri_poc_column: str = "ri_poc",
    ri_delta_column: str = "ri_delta",
    ri_state_column: str = "ri_state",
    ri_ready_column: str = "ri_ready",
    ri_sweep_high_column: str = "ri_sweep_high",
    ri_sweep_low_column: str = "ri_sweep_low",
    ri_breakout_column: str = "ri_breakout",
    ri_duration_column: str = "ri_duration",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Detect consolidation ranges with volume profiling and sweep detection.

    This is a Python port of the "Range Intelligence Suite [LuxAlgo]"
    indicator.  It identifies periods of price compression by comparing
    the rolling range width to the ATR, then enriches each detected
    range with a volume profile, Point of Control, net delta, and
    liquidity sweep counts.

    Args:
        data: OHLCV DataFrame (pandas or polars).
        length: Lookback period for range detection and ATR (default 20).
        sensitivity: ATR multiplier threshold — lower values detect
            tighter ranges (default 4.0).
        vp_rows: Number of horizontal volume-profile bins (default 10).
        hide_overlaps: When True, discard earlier ranges that overlap
            with a newly detected range (default True).
        open_column: Name of the Open column.
        high_column: Name of the High column.
        low_column: Name of the Low column.
        close_column: Name of the Close column.
        volume_column: Name of the Volume column.

    Returns:
        DataFrame with added columns:

        - ``ri_active``      — 1 while inside a consolidation range
        - ``ri_high``        — upper boundary of the active range
        - ``ri_low``         — lower boundary of the active range
        - ``ri_mid``         — midpoint of the range
        - ``ri_poc``         — Point of Control price (highest volume bin)
        - ``ri_delta``       — cumulative net delta (buy − sell volume)
        - ``ri_state``       — "Accumulation" or "Distribution"
        - ``ri_ready``       — ready score 0–100 (breakout imminence)
        - ``ri_sweep_high``  — 1 when a high-side liquidity sweep occurs
        - ``ri_sweep_low``   — 1 when a low-side liquidity sweep occurs
        - ``ri_breakout``    — +1 bullish breakout, −1 bearish breakout
        - ``ri_duration``    — bars since range started

    Example::

        >>> from pyindicators import range_intelligence
        >>> df = range_intelligence(df, length=20, sensitivity=4.0)
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        pdf = _range_intelligence_pandas(
            pdf, length, sensitivity, vp_rows, hide_overlaps,
            high_column, low_column, open_column, close_column,
            volume_column,
            ri_active_column, ri_high_column, ri_low_column,
            ri_mid_column, ri_poc_column, ri_delta_column,
            ri_state_column, ri_ready_column,
            ri_sweep_high_column, ri_sweep_low_column,
            ri_breakout_column, ri_duration_column,
        )
        import polars as pl
        return pl.from_pandas(pdf)

    if isinstance(data, PdDataFrame):
        return _range_intelligence_pandas(
            data, length, sensitivity, vp_rows, hide_overlaps,
            high_column, low_column, open_column, close_column,
            volume_column,
            ri_active_column, ri_high_column, ri_low_column,
            ri_mid_column, ri_poc_column, ri_delta_column,
            ri_state_column, ri_ready_column,
            ri_sweep_high_column, ri_sweep_low_column,
            ri_breakout_column, ri_duration_column,
        )

    raise PyIndicatorException(
        "Input data must be a pandas or polars DataFrame."
    )


def range_intelligence_signal(
    data: Union[PdDataFrame, PlDataFrame],
    ri_breakout_column: str = "ri_breakout",
    signal_column: str = "ri_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Generate a directional breakout signal from Range Intelligence.

    Signal values:
        -  ``1``  — bullish breakout (close above range high)
        - ``-1``  — bearish breakout (close below range low)
        -  ``0``  — no breakout

    Args:
        data: DataFrame with ``ri_breakout`` column already computed.
        ri_breakout_column: Name of the breakout column.
        signal_column: Output signal column name.

    Returns:
        DataFrame with added signal column.
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        pdf[signal_column] = pdf[ri_breakout_column].fillna(0).astype(int)
        import polars as pl
        return pl.from_pandas(pdf)

    if isinstance(data, PdDataFrame):
        df = data.copy()
        df[signal_column] = df[ri_breakout_column].fillna(0).astype(int)
        return df

    raise PyIndicatorException(
        "Input data must be a pandas or polars DataFrame."
    )


def get_range_intelligence_stats(
    data: Union[PdDataFrame, PlDataFrame],
    ri_active_column: str = "ri_active",
    ri_breakout_column: str = "ri_breakout",
    ri_delta_column: str = "ri_delta",
    ri_state_column: str = "ri_state",
    ri_sweep_high_column: str = "ri_sweep_high",
    ri_sweep_low_column: str = "ri_sweep_low",
    ri_ready_column: str = "ri_ready",
    ri_duration_column: str = "ri_duration",
) -> Dict[str, object]:
    """
    Compute summary statistics for Range Intelligence detections.

    Args:
        data: DataFrame with Range Intelligence columns.

    Returns:
        Dictionary with keys:

        - ``total_ranges`` — total number of completed ranges
        - ``bullish_breakouts`` — count of bullish breakouts
        - ``bearish_breakouts`` — count of bearish breakouts
        - ``accumulation_ranges`` — ranges ending in accumulation state
        - ``distribution_ranges`` — ranges ending in distribution state
        - ``total_sweep_highs`` — total high-side liquidity sweeps
        - ``total_sweep_lows`` — total low-side liquidity sweeps
        - ``avg_ready_score`` — average ready score at breakout
        - ``avg_duration`` — average range duration (bars)
    """
    if isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
    elif isinstance(data, PdDataFrame):
        pdf = data
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )

    breakouts = pdf[pdf[ri_breakout_column] != 0]
    bull = int((breakouts[ri_breakout_column] == 1).sum())
    bear = int((breakouts[ri_breakout_column] == -1).sum())

    # State at breakout bars
    accum = int((breakouts[ri_state_column] == "Accumulation").sum())
    distrib = int((breakouts[ri_state_column] == "Distribution").sum())

    total_sweeps_h = int(pdf[ri_sweep_high_column].sum())
    total_sweeps_l = int(pdf[ri_sweep_low_column].sum())

    # Average ready score at breakout
    if len(breakouts) > 0:
        avg_ready = float(breakouts[ri_ready_column].mean())
        avg_dur = float(breakouts[ri_duration_column].mean())
    else:
        avg_ready = 0.0
        avg_dur = 0.0

    return {
        "total_ranges": bull + bear,
        "bullish_breakouts": bull,
        "bearish_breakouts": bear,
        "accumulation_ranges": accum,
        "distribution_ranges": distrib,
        "total_sweep_highs": total_sweeps_h,
        "total_sweep_lows": total_sweeps_l,
        "avg_ready_score": round(avg_ready, 1),
        "avg_duration": round(avg_dur, 1),
    }
