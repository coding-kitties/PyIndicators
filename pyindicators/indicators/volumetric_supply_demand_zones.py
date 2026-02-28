"""
Volumetric Supply and Demand Zones

Detects supply and demand zones using swing-point analysis with impulse
confirmation, enhanced with volume profiling and delta (buy/sell pressure)
analysis within each zone.

**Concept:**
    Supply and demand zones form at swing points where price made a
    significant impulse move.  A *demand zone* forms at a swing low
    followed by an upward impulse; a *supply zone* forms at a swing
    high followed by a downward impulse.  Each zone is enriched with
    a volume profile (volume distributed across price rows within the
    zone) and a delta profile (buy vs sell volume).

**Demand Zone:**
    Detected when a pivot low is confirmed (``swing_length`` bars on
    each side) and the subsequent price move upward exceeds
    ``impulse_mult * ATR``.  The zone boundaries are derived from the
    high/low of the ``base_lookback`` candles around the pivot.

**Supply Zone:**
    Detected when a pivot high is confirmed and the subsequent price
    move downward exceeds ``impulse_mult * ATR``.

**Volume Profile:**
    The zone is divided into ``profile_rows`` horizontal slices.
    Volume from the base candles is distributed proportionally across
    rows based on price overlap, producing a histogram.

**Point of Control (POC):**
    The price row within the zone that captured the most volume.

**Delta Profile:**
    Buy volume (bullish candles) vs sell volume (bearish candles)
    is allocated to the same rows.  The net delta per row shows
    where buying or selling pressure was concentrated.

**Zone Lifecycle:**
    * ``Fresh``   – newly created, price has not revisited
    * ``Tested``  – price has touched the zone at least once
    * ``Mitigated`` – price broke through the zone boundary
      (wick or close, per ``mitigation_type``)

**Merging:**
    Overlapping zones of the same type (both supply or both demand)
    can be merged, combining their volume and delta profiles.

**Signals:**
    * ``1``  – price enters a demand zone (potential long)
    * ``-1`` – price enters a supply zone (potential short)
    * ``0``  – no signal
"""
from typing import Union, Dict, List, Optional
from dataclasses import dataclass, field
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame
import polars as pl
import numpy as np

from pyindicators.exceptions import PyIndicatorException


# ------------------------------------------------------------------ #
#  Zone data structure                                                #
# ------------------------------------------------------------------ #
@dataclass
class _VolumeRow:
    price_low: float
    price_high: float
    volume: float
    width_pct: float
    is_poc: bool


@dataclass
class _DeltaRow:
    price_low: float
    price_high: float
    delta: float
    width_pct: float
    is_positive: bool


@dataclass
class _SDZone:
    is_demand: bool
    zone_top: float
    zone_bottom: float
    poc_price: float
    total_volume: float
    total_delta: float
    touch_count: int
    creation_bar: int
    status: str  # "Fresh", "Tested", "Mitigated"
    profile: List[_VolumeRow] = field(default_factory=list)
    delta_profile: List[_DeltaRow] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Helper: pivot detection                                            #
# ------------------------------------------------------------------ #
def _pivot_high(high: np.ndarray, length: int) -> np.ndarray:
    """Return array with pivot-high price at confirmation
    bar, NaN elsewhere."""
    n = len(high)
    result = np.full(n, np.nan)

    for i in range(length, n - length):
        val = high[i]
        is_pivot = True

        for j in range(1, length + 1):
            if high[i - j] >= val or high[i + j] >= val:
                is_pivot = False
                break

        if is_pivot:
            # Confirmed `length` bars later
            confirm_bar = i + length
            if confirm_bar < n:
                result[confirm_bar] = val

    return result


def _pivot_low(low: np.ndarray, length: int) -> np.ndarray:
    """Return array with pivot-low price at confirmation bar, NaN elsewhere."""
    n = len(low)
    result = np.full(n, np.nan)

    for i in range(length, n - length):
        val = low[i]
        is_pivot = True

        for j in range(1, length + 1):
            if low[i - j] <= val or low[i + j] <= val:
                is_pivot = False
                break

        if is_pivot:
            confirm_bar = i + length
            if confirm_bar < n:
                result[confirm_bar] = val

    return result


# ------------------------------------------------------------------ #
#  Helper: ATR                                                        #
# ------------------------------------------------------------------ #
def _compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Compute ATR as a simple rolling mean of True Range."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.empty(n)
    atr[:] = np.nan

    # Seed with simple mean then EMA-style for consistency
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ------------------------------------------------------------------ #
#  Helper: build volume profile                                       #
# ------------------------------------------------------------------ #
def _build_volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    opn: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    zone_top: float,
    zone_bottom: float,
    start_idx: int,
    end_idx: int,
    profile_rows: int,
):
    """
    Distribute volume across horizontal rows within the zone.

    Returns (profile_list, delta_list, poc_price, total_vol, total_delta).
    """
    zone_height = zone_top - zone_bottom
    if zone_height <= 0:
        zone_height = 1e-10
    row_height = zone_height / profile_rows

    actual_start = min(start_idx, end_idx)
    actual_end = max(start_idx, end_idx)

    # Accumulate per-row
    row_vol = np.zeros(profile_rows)
    row_buy = np.zeros(profile_rows)
    row_sell = np.zeros(profile_rows)

    for i in range(actual_start, actual_end + 1):
        if i < 0 or i >= len(high):
            continue

        bar_high = high[i]
        bar_low = low[i]
        bar_vol = volume[i] if not np.isnan(volume[i]) else 1.0
        bar_open = opn[i]
        bar_close = close[i]

        for r in range(profile_rows):
            row_low = zone_bottom + r * row_height
            row_high = row_low + row_height

            if bar_high >= row_low and bar_low <= row_high:
                overlap_low = max(bar_low, row_low)
                overlap_high = min(bar_high, row_high)
                bar_range = bar_high - bar_low
                overlap_pct = (
                    (overlap_high - overlap_low) / bar_range
                    if bar_range > 0
                    else 1.0
                )
                alloc_vol = bar_vol * overlap_pct
                row_vol[r] += alloc_vol

                # Delta: up-close = buy, down-close = sell, doji = split
                if bar_close > bar_open:
                    row_buy[r] += alloc_vol
                elif bar_close < bar_open:
                    row_sell[r] += alloc_vol
                else:
                    row_buy[r] += alloc_vol * 0.5
                    row_sell[r] += alloc_vol * 0.5

    max_vol = np.max(row_vol) if np.max(row_vol) > 0 else 1.0
    poc_idx = int(np.argmax(row_vol))
    poc_price = zone_bottom + poc_idx * row_height + row_height / 2
    total_vol = float(np.sum(row_vol))

    row_delta = row_buy - row_sell
    total_delta = float(np.sum(row_delta))
    max_abs_delta = (
        np.max(np.abs(row_delta))
        if np.max(np.abs(row_delta)) > 0
        else 1.0
    )

    # Build profile rows
    profile = []
    delta_profile = []

    for r in range(profile_rows):
        row_low_price = zone_bottom + r * row_height
        row_high_price = row_low_price + row_height
        w_pct = row_vol[r] / max_vol

        profile.append(_VolumeRow(
            price_low=row_low_price,
            price_high=row_high_price,
            volume=float(row_vol[r]),
            width_pct=float(w_pct),
            is_poc=(r == poc_idx),
        ))

        d = float(row_delta[r])
        d_pct = abs(d) / max_abs_delta

        delta_profile.append(_DeltaRow(
            price_low=row_low_price,
            price_high=row_high_price,
            delta=d,
            width_pct=float(d_pct),
            is_positive=(d >= 0),
        ))

    return profile, delta_profile, poc_price, total_vol, total_delta


# ------------------------------------------------------------------ #
#  Helper: check zone overlap (for merging)                           #
# ------------------------------------------------------------------ #
def _zones_overlap(
    top_a: float, btm_a: float,
    top_b: float, btm_b: float,
    gap: float,
) -> bool:
    """Return True if zones overlap with gap tolerance."""
    return (btm_a - gap) <= (top_b + gap) and (top_a + gap) >= (btm_b - gap)


# ------------------------------------------------------------------ #
#  Core pandas implementation                                         #
# ------------------------------------------------------------------ #
def _volumetric_supply_demand_zones_pandas(
    data: PdDataFrame,
    swing_length: int,
    impulse_mult: float,
    base_lookback: int,
    atr_length: int,
    max_zone_atr: float,
    max_zones: int,
    merge_zones: bool,
    merge_gap_atr: float,
    mitigation_type: str,
    profile_rows: int,
    high_col: str,
    low_col: str,
    open_col: str,
    close_col: str,
    volume_col: str,
    # Output columns
    demand_zone_col: str,
    supply_zone_col: str,
    zone_top_col: str,
    zone_bottom_col: str,
    zone_poc_col: str,
    zone_type_col: str,
    zone_volume_col: str,
    zone_delta_col: str,
    zone_status_col: str,
    zone_touches_col: str,
    signal_col: str,
) -> PdDataFrame:
    """Core implementation on pandas DataFrame."""
    df = data.copy()
    n = len(df)

    high = df[high_col].values.astype(float)
    low = df[low_col].values.astype(float)
    opn = df[open_col].values.astype(float)
    close = df[close_col].values.astype(float)
    volume = df[volume_col].values.astype(float)

    # ATR
    atr = _compute_atr(high, low, close, atr_length)

    # Pivot detection
    ph = _pivot_high(high, swing_length)
    pl_arr = _pivot_low(low, swing_length)

    # Output arrays
    demand_zone = np.zeros(n, dtype=int)
    supply_zone = np.zeros(n, dtype=int)
    zone_top = np.full(n, np.nan)
    zone_bottom = np.full(n, np.nan)
    zone_poc = np.full(n, np.nan)
    zone_type = np.zeros(n, dtype=int)  # 1=demand, -1=supply
    zone_vol = np.full(n, np.nan)
    zone_delta = np.full(n, np.nan)
    zone_status_arr = np.full(n, "", dtype=object)
    zone_touches = np.zeros(n, dtype=int)
    signal = np.zeros(n, dtype=int)

    active_zones: List[_SDZone] = []

    for i in range(n):
        atr_val = atr[i] if not np.isnan(atr[i]) else 0.0

        if atr_val <= 0:
            continue

        # ── Check for new demand zone ────────────────────────────
        if not np.isnan(pl_arr[i]):
            # Impulse check: price moved up from swing low
            pivot_bar = i - swing_length  # actual swing low bar
            if pivot_bar >= 0:
                move = close[i] - close[pivot_bar]
                if move >= atr_val * impulse_mult:
                    new_zone = _create_zone(
                        True, pivot_bar, base_lookback,
                        high, low, opn, close, volume,
                        atr_val, max_zone_atr, profile_rows, i,
                    )
                    if new_zone is not None:
                        merged = _try_merge(
                            new_zone, active_zones,
                            merge_zones, atr_val, merge_gap_atr,
                            max_zone_atr,
                            high, low, opn, close, volume,
                            profile_rows,
                        )
                        if not merged:
                            if len(active_zones) >= max_zones:
                                active_zones.pop(0)
                            active_zones.append(new_zone)
                        demand_zone[i] = 1

        # ── Check for new supply zone ────────────────────────────
        if not np.isnan(ph[i]):
            pivot_bar = i - swing_length
            if pivot_bar >= 0:
                move = close[pivot_bar] - close[i]
                if move >= atr_val * impulse_mult:
                    new_zone = _create_zone(
                        False, pivot_bar, base_lookback,
                        high, low, opn, close, volume,
                        atr_val, max_zone_atr, profile_rows, i,
                    )
                    if new_zone is not None:
                        merged = _try_merge(
                            new_zone, active_zones,
                            merge_zones, atr_val, merge_gap_atr,
                            max_zone_atr,
                            high, low, opn, close, volume,
                            profile_rows,
                        )
                        if not merged:
                            if len(active_zones) >= max_zones:
                                active_zones.pop(0)
                            active_zones.append(new_zone)
                        supply_zone[i] = 1

        # ── Manage active zones ──────────────────────────────────
        to_remove = []

        for zi, z in enumerate(active_zones):
            in_zone = high[i] >= z.zone_bottom and low[i] <= z.zone_top
            was_in_zone = (
                i > 0
                and high[i - 1] >= z.zone_bottom
                and low[i - 1] <= z.zone_top
            )

            # Touch detection
            if in_zone and not was_in_zone and z.status == "Fresh":
                z.status = "Tested"
                z.touch_count += 1
            elif in_zone and z.status == "Tested":
                z.touch_count += 1

            # Mitigation check
            mitigated = False
            if mitigation_type == "Wick":
                if z.is_demand:
                    mitigated = low[i] < z.zone_bottom
                else:
                    mitigated = high[i] > z.zone_top
            else:  # "Close"
                if z.is_demand:
                    mitigated = close[i] < z.zone_bottom
                else:
                    mitigated = close[i] > z.zone_top

            if mitigated:
                z.status = "Mitigated"
                to_remove.append(zi)

            # Signal: price enters a zone
            if in_zone and not was_in_zone and not mitigated:
                if z.is_demand:
                    signal[i] = 1
                else:
                    signal[i] = -1

        # Remove mitigated zones (reverse order)
        for idx in reversed(to_remove):
            active_zones.pop(idx)

        # ── Write zone state for the most recent active zone of
        #    each type (demand / supply) to the output arrays ─────
        # Find nearest demand and supply zones
        nearest_demand: Optional[_SDZone] = None
        nearest_supply: Optional[_SDZone] = None

        for z in reversed(active_zones):
            if z.is_demand and nearest_demand is None:
                nearest_demand = z
            elif not z.is_demand and nearest_supply is None:
                nearest_supply = z
            if nearest_demand is not None and nearest_supply is not None:
                break

        # Primary zone: prefer demand if price is near it, else
        # supply.  For simplicity output the most recently created.
        primary = None
        if nearest_demand is not None and nearest_supply is not None:
            # Output whichever is closer to current price
            mid = (high[i] + low[i]) / 2
            d_dist = abs(mid - (
                nearest_demand.zone_top
                + nearest_demand.zone_bottom) / 2)
            s_dist = abs(mid - (
                nearest_supply.zone_top
                + nearest_supply.zone_bottom) / 2)
            primary = nearest_demand if d_dist <= s_dist else nearest_supply
        elif nearest_demand is not None:
            primary = nearest_demand
        elif nearest_supply is not None:
            primary = nearest_supply

        if primary is not None:
            zone_top[i] = primary.zone_top
            zone_bottom[i] = primary.zone_bottom
            zone_poc[i] = primary.poc_price
            zone_type[i] = 1 if primary.is_demand else -1
            zone_vol[i] = primary.total_volume
            zone_delta[i] = primary.total_delta
            zone_status_arr[i] = primary.status
            zone_touches[i] = primary.touch_count

    # Write output columns
    df[demand_zone_col] = demand_zone
    df[supply_zone_col] = supply_zone
    df[zone_top_col] = zone_top
    df[zone_bottom_col] = zone_bottom
    df[zone_poc_col] = zone_poc
    df[zone_type_col] = zone_type
    df[zone_volume_col] = zone_vol
    df[zone_delta_col] = zone_delta
    df[zone_status_col] = zone_status_arr
    df[zone_touches_col] = zone_touches
    df[signal_col] = signal

    return df


# ------------------------------------------------------------------ #
#  Zone creation helper                                               #
# ------------------------------------------------------------------ #
def _create_zone(
    is_demand: bool,
    pivot_bar: int,
    base_lookback: int,
    high: np.ndarray,
    low: np.ndarray,
    opn: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr_val: float,
    max_zone_atr: float,
    profile_rows: int,
    current_bar: int,
) -> Optional[_SDZone]:
    """Create a supply or demand zone around a pivot bar."""
    end_idx = pivot_bar
    start_idx = min(pivot_bar + base_lookback, len(high) - 1)

    if end_idx < 0 or start_idx < 0:
        return None

    # Compute zone boundaries from the base candles
    raw_top = high[end_idx]
    raw_bottom = low[end_idx]

    for k in range(end_idx, start_idx + 1):
        if k < len(high):
            raw_top = max(raw_top, high[k])
            raw_bottom = min(raw_bottom, low[k])

    # Clamp zone height
    zone_height = raw_top - raw_bottom
    max_height = atr_val * max_zone_atr

    if zone_height > max_height:
        mid = (raw_top + raw_bottom) / 2
        raw_top = mid + max_height / 2
        raw_bottom = mid - max_height / 2

    # Build volume + delta profiles
    profile, delta_profile, poc_price, total_vol, total_delta = (
        _build_volume_profile(
            high, low, opn, close, volume,
            raw_top, raw_bottom,
            end_idx, start_idx,
            profile_rows,
        )
    )

    return _SDZone(
        is_demand=is_demand,
        zone_top=raw_top,
        zone_bottom=raw_bottom,
        poc_price=poc_price,
        total_volume=total_vol,
        total_delta=total_delta,
        touch_count=0,
        creation_bar=current_bar,
        status="Fresh",
        profile=profile,
        delta_profile=delta_profile,
    )


# ------------------------------------------------------------------ #
#  Merge helper                                                       #
# ------------------------------------------------------------------ #
def _try_merge(
    new_zone: _SDZone,
    active_zones: List[_SDZone],
    merge_enabled: bool,
    atr_val: float,
    merge_gap_atr: float,
    max_zone_atr: float,
    high: np.ndarray,
    low: np.ndarray,
    opn: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    profile_rows: int,
) -> bool:
    """Try to merge new_zone into an existing same-type zone."""
    if not merge_enabled or len(active_zones) == 0:
        return False

    gap = atr_val * merge_gap_atr

    for existing in active_zones:
        if existing.is_demand != new_zone.is_demand:
            continue

        if _zones_overlap(
            existing.zone_top, existing.zone_bottom,
            new_zone.zone_top, new_zone.zone_bottom,
            gap,
        ):
            new_top = max(existing.zone_top, new_zone.zone_top)
            new_bottom = min(existing.zone_bottom, new_zone.zone_bottom)

            # Clamp height
            zone_height = new_top - new_bottom
            max_height = atr_val * max_zone_atr

            if zone_height > max_height:
                mid = (new_top + new_bottom) / 2
                new_top = mid + max_height / 2
                new_bottom = mid - max_height / 2

            existing.zone_top = new_top
            existing.zone_bottom = new_bottom
            existing.total_volume += new_zone.total_volume
            existing.total_delta += new_zone.total_delta

            # Merge volume rows
            n_rows = min(len(existing.profile), len(new_zone.profile))
            max_vol = 0.0
            poc_idx = 0

            for r in range(n_rows):
                existing.profile[r].volume += new_zone.profile[r].volume
                if existing.profile[r].volume > max_vol:
                    max_vol = existing.profile[r].volume
                    poc_idx = r

            for r in range(len(existing.profile)):
                existing.profile[r].width_pct = (
                    existing.profile[r].volume
                    / max_vol
                    if max_vol > 0 else 0.5
                )
                existing.profile[r].is_poc = (r == poc_idx)

            # Merge delta rows
            for r in range(min(
                len(existing.delta_profile),
                len(new_zone.delta_profile),
            )):
                existing.delta_profile[r].delta += (
                    new_zone.delta_profile[r].delta
                )
                existing.delta_profile[r].is_positive = (
                    existing.delta_profile[r].delta >= 0
                )

            # Update POC
            row_height = (
                (existing.zone_top
                 - existing.zone_bottom)
                / len(existing.profile)
            )
            existing.poc_price = (
                existing.zone_bottom + poc_idx * row_height + row_height / 2
            )

            return True

    return False


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #
def volumetric_supply_demand_zones(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 8,
    impulse_mult: float = 1.2,
    base_lookback: int = 3,
    atr_length: int = 14,
    max_zone_atr: float = 4.0,
    max_zones: int = 10,
    merge_zones: bool = True,
    merge_gap_atr: float = 0.3,
    mitigation_type: str = "Wick",
    profile_rows: int = 10,
    high_column: str = "High",
    low_column: str = "Low",
    open_column: str = "Open",
    close_column: str = "Close",
    volume_column: str = "Volume",
    demand_zone_column: str = "vsdz_demand",
    supply_zone_column: str = "vsdz_supply",
    zone_top_column: str = "vsdz_zone_top",
    zone_bottom_column: str = "vsdz_zone_bottom",
    zone_poc_column: str = "vsdz_poc",
    zone_type_column: str = "vsdz_zone_type",
    zone_volume_column: str = "vsdz_volume",
    zone_delta_column: str = "vsdz_delta",
    zone_status_column: str = "vsdz_status",
    zone_touches_column: str = "vsdz_touches",
    signal_column: str = "vsdz_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Identify Volumetric Supply and Demand Zones with volume profiling.

    Detects supply (resistance) and demand (support) zones at
    significant swing points, enriched with volume distribution
    and buy/sell delta data within each zone.

    Args:
        data: pandas or polars DataFrame with OHLCV price data.
        swing_length: Pivot lookback period for detecting swing
            highs/lows (default: 8).
        impulse_mult: Minimum price move in ATR multiples required
            to form a zone (default: 1.2).
        base_lookback: Number of candles to include in base zone
            formation (default: 3).
        atr_length: ATR calculation period (default: 14).
        max_zone_atr: Maximum zone height as ATR multiple
            (default: 4.0).
        max_zones: Maximum active zones to track (default: 10).
        merge_zones: Merge overlapping same-type zones
            (default: True).
        merge_gap_atr: Gap tolerance in ATR for merging
            (default: 0.3).
        mitigation_type: ``"Wick"`` or ``"Close"`` — how zones
            are invalidated (default: ``"Wick"``).
        profile_rows: Number of volume distribution rows per zone
            (default: 10).
        high_column: Column name for highs (default: ``"High"``).
        low_column: Column name for lows (default: ``"Low"``).
        open_column: Column name for opens (default: ``"Open"``).
        close_column: Column name for closes (default: ``"Close"``).
        volume_column: Column name for volume (default: ``"Volume"``).

    Returns:
        DataFrame with added columns:

        - ``vsdz_demand``: 1 on demand zone formation bar
        - ``vsdz_supply``: 1 on supply zone formation bar
        - ``vsdz_zone_top``: Active zone top boundary
        - ``vsdz_zone_bottom``: Active zone bottom boundary
        - ``vsdz_poc``: Point of Control price level
        - ``vsdz_zone_type``: 1 (demand) or -1 (supply)
        - ``vsdz_volume``: Total volume in the zone
        - ``vsdz_delta``: Net buy-sell delta in the zone
        - ``vsdz_status``: Zone lifecycle status
        - ``vsdz_touches``: Number of times zone was tested
        - ``vsdz_signal``: Trading signal (+1/-1/0)

    Example:
        >>> from pyindicators import volumetric_supply_demand_zones
        >>> df = volumetric_supply_demand_zones(df)
    """
    if mitigation_type not in ("Wick", "Close"):
        raise PyIndicatorException(
            "mitigation_type must be 'Wick' or "
            f"'Close', got '{mitigation_type}'"
        )

    if isinstance(data, PdDataFrame):
        return _volumetric_supply_demand_zones_pandas(
            data, swing_length, impulse_mult, base_lookback,
            atr_length, max_zone_atr, max_zones,
            merge_zones, merge_gap_atr, mitigation_type,
            profile_rows,
            high_column, low_column, open_column,
            close_column, volume_column,
            demand_zone_column, supply_zone_column,
            zone_top_column, zone_bottom_column,
            zone_poc_column, zone_type_column,
            zone_volume_column, zone_delta_column,
            zone_status_column, zone_touches_column,
            signal_column,
        )
    elif isinstance(data, PlDataFrame):
        pdf = data.to_pandas()
        result = _volumetric_supply_demand_zones_pandas(
            pdf, swing_length, impulse_mult, base_lookback,
            atr_length, max_zone_atr, max_zones,
            merge_zones, merge_gap_atr, mitigation_type,
            profile_rows,
            high_column, low_column, open_column,
            close_column, volume_column,
            demand_zone_column, supply_zone_column,
            zone_top_column, zone_bottom_column,
            zone_poc_column, zone_type_column,
            zone_volume_column, zone_delta_column,
            zone_status_column, zone_touches_column,
            signal_column,
        )
        return pl.from_pandas(result)
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ------------------------------------------------------------------ #
#  Signal function                                                    #
# ------------------------------------------------------------------ #
def volumetric_supply_demand_zones_signal(
    data: Union[PdDataFrame, PlDataFrame],
    signal_column: str = "vsdz_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """
    Extract or confirm the trading signal column.

    The signal is already computed by ``volumetric_supply_demand_zones()``.
    This function exists for API consistency with the triple-export
    pattern used across all PyIndicators.

    Signal values:
        * ``1``  – price enters a demand zone (potential long)
        * ``-1`` – price enters a supply zone (potential short)
        * ``0``  – no signal

    Args:
        data: DataFrame with ``volumetric_supply_demand_zones()``
            columns already computed.
        signal_column: Column name for the signal
            (default: ``"vsdz_signal"``).

    Returns:
        DataFrame (unchanged — signal is already present).

    Example:
        >>> df = volumetric_supply_demand_zones(df)
        >>> df = volumetric_supply_demand_zones_signal(df)
        >>> buys = df[df['vsdz_signal'] == 1]
    """
    if isinstance(data, PdDataFrame):
        if signal_column not in data.columns:
            raise PyIndicatorException(
                f"Column '{signal_column}' not found. "
                f"Run volumetric_supply_demand_zones() first."
            )
        return data.copy()
    elif isinstance(data, PlDataFrame):
        if signal_column not in data.columns:
            raise PyIndicatorException(
                f"Column '{signal_column}' not found. "
                f"Run volumetric_supply_demand_zones() first."
            )
        return data.clone()
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ------------------------------------------------------------------ #
#  Stats function                                                     #
# ------------------------------------------------------------------ #
def get_volumetric_supply_demand_zones_stats(
    data: Union[PdDataFrame, PlDataFrame],
    demand_zone_column: str = "vsdz_demand",
    supply_zone_column: str = "vsdz_supply",
    zone_status_column: str = "vsdz_status",
    zone_touches_column: str = "vsdz_touches",
    zone_volume_column: str = "vsdz_volume",
    zone_delta_column: str = "vsdz_delta",
    signal_column: str = "vsdz_signal",
) -> Dict:
    """
    Compute summary statistics from supply/demand zone output.

    Args:
        data: DataFrame with ``volumetric_supply_demand_zones()``
            columns already computed.

    Returns:
        Dictionary with keys:

        - ``total_demand_zones``: Number of demand zones formed
        - ``total_supply_zones``: Number of supply zones formed
        - ``total_zones``: Total zones formed
        - ``bullish_signals``: Number of bullish (demand) signals
        - ``bearish_signals``: Number of bearish (supply) signals
        - ``total_signals``: Total signals
        - ``avg_zone_volume``: Average volume across active zones
        - ``avg_zone_delta``: Average delta across active zones
        - ``max_touches``: Maximum touch count observed

    Example:
        >>> df = volumetric_supply_demand_zones(df)
        >>> stats = get_volumetric_supply_demand_zones_stats(df)
        >>> print(stats)
    """
    if isinstance(data, PlDataFrame):
        data = data.to_pandas()

    demand_count = int(data[demand_zone_column].sum())
    supply_count = int(data[supply_zone_column].sum())
    total = demand_count + supply_count

    sig = data[signal_column]
    bullish_signals = int((sig == 1).sum())
    bearish_signals = int((sig == -1).sum())
    total_signals = bullish_signals + bearish_signals

    vol = data[zone_volume_column].dropna()
    avg_volume = float(vol.mean()) if len(vol) > 0 else 0.0

    delta = data[zone_delta_column].dropna()
    avg_delta = float(delta.mean()) if len(delta) > 0 else 0.0

    max_touches = int(data[zone_touches_column].max()) if len(data) > 0 else 0

    return {
        "total_demand_zones": demand_count,
        "total_supply_zones": supply_count,
        "total_zones": total,
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
        "total_signals": total_signals,
        "avg_zone_volume": round(avg_volume, 2),
        "avg_zone_delta": round(avg_delta, 2),
        "max_touches": max_touches,
    }
