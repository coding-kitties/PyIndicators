"""
Liquidity Levels / Voids (Volume Profile) Indicator

Uses volume-profile analysis between swing points to
identify price levels where little volume was traded — these are
*liquidity voids* that price tends to revisit.

Algorithm overview
------------------
1. Detect swing highs and swing lows using a configurable lookback
   period (``detection_length``).
2. Between consecutive swing points, divide the price range into
   ``sensitivity`` equally-spaced levels.
3. Build a volume profile by distributing each bar's volume across
   the levels it spans.
4. Levels whose volume is less than ``threshold`` percent of the
   maximum level's volume are marked as **liquidity voids** (low-
   volume zones).
5. Voids are *filled* when price subsequently crosses through the
   zone's midpoint.

The indicator tracks all active (unfilled) voids and reports per-bar
summary columns: nearest void boundaries, counts, formation and fill
events.
"""
from typing import Union, Optional
import math as _math

import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


# ── Swing detection helper ───────────────────────────────────────────

def _pivot_high(highs: np.ndarray, length: int, idx: int) -> bool:
    """Return True if *idx* is a pivot high over *length* bars each side."""
    if idx < length or idx >= len(highs) - length:
        return False
    val = highs[idx]
    for j in range(1, length + 1):
        if highs[idx - j] > val or highs[idx + j] > val:
            return False
    return True


def _pivot_low(lows: np.ndarray, length: int, idx: int) -> bool:
    """Return True if *idx* is a pivot low over *length* bars each side."""
    if idx < length or idx >= len(lows) - length:
        return False
    val = lows[idx]
    for j in range(1, length + 1):
        if lows[idx - j] < val or lows[idx + j] < val:
            return False
    return True


# ── Main indicator ───────────────────────────────────────────────────

def liquidity_levels_voids(
    data: Union[PdDataFrame, PlDataFrame],
    detection_length: int = 47,
    threshold: float = 0.21,
    sensitivity: int = 27,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    volume_column: Optional[str] = "Volume",
    void_formed_column: str = "liq_void_formed",
    void_filled_column: str = "liq_void_filled",
    void_count_column: str = "liq_void_count",
    void_nearest_top_column: str = "liq_void_nearest_top",
    void_nearest_bottom_column: str = "liq_void_nearest_bot",
    void_above_count_column: str = "liq_void_above_count",
    void_below_count_column: str = "liq_void_below_count",
) -> Union[PdDataFrame, PlDataFrame]:
    """Detect Liquidity Levels / Voids using volume-profile analysis.

    Identifies low-volume price zones between swing points that act as
    liquidity voids — areas where price moved through quickly and is
    likely to revisit.

    Args:
        data: pandas or polars DataFrame with OHLCV data.
        detection_length: Lookback/look-ahead period for swing
            detection (default: 47).
        threshold: Volume fraction below which a level is classified
            as a liquidity void (default: 0.21, i.e. 21 %).  A level
            is a void if its volume < ``threshold * max_volume_level``.
        sensitivity: Number of price levels to divide each swing
            range into (default: 27).  Higher values produce thinner,
            more granular zones.
        high_column: Column name for highs.
        low_column: Column name for lows.
        close_column: Column name for closes.
        volume_column: Column name for volume.  Set to ``None`` to
            use uniform volume (each bar = 1).
        void_formed_column: Output — 1 on bars where new voids are
            identified.
        void_filled_column: Output — 1 on bars where a void is
            filled.
        void_count_column: Output — total active unfilled voids.
        void_nearest_top_column: Output — top of the nearest
            unfilled void to the current close.
        void_nearest_bottom_column: Output — bottom of the nearest
            unfilled void to the current close.
        void_above_count_column: Output — unfilled voids above the
            current close.
        void_below_count_column: Output — unfilled voids below the
            current close.

    Returns:
        DataFrame with void columns added.
    """
    is_polars = isinstance(data, PlDataFrame)

    if is_polars:
        df = data.to_pandas()
    elif isinstance(data, PdDataFrame):
        df = data.copy()
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )

    highs = df[high_column].values.astype(float)
    lows = df[low_column].values.astype(float)
    closes = df[close_column].values.astype(float)
    has_vol = (
        volume_column is not None and volume_column in df.columns
    )
    volumes = (
        df[volume_column].values.astype(float)
        if has_vol else np.ones(len(df))
    )
    n = len(highs)

    # ── Output arrays ────────────────────────────────────────────
    out_formed = np.zeros(n, dtype=int)
    out_filled = np.zeros(n, dtype=int)
    out_count = np.zeros(n, dtype=int)
    out_nearest_top = np.full(n, np.nan)
    out_nearest_bot = np.full(n, np.nan)
    out_above = np.zeros(n, dtype=int)
    out_below = np.zeros(n, dtype=int)

    ppLen = detection_length
    vpLev = sensitivity
    liqT = threshold

    # Active unfilled voids: list of [top, bottom] pairs
    active_voids: list[list[float]] = []

    # Previous pivot bar index and previous HIGH/LOW values for
    # swing comparison.
    prev_pivot_bar: int = -1
    prev_pivot_bar_2: int = -1

    for bar in range(n):
        # ── 1. Detect pivots (confirmed ppLen bars ago) ──────────
        check_bar = bar - ppLen  # the bar being confirmed now

        new_pivot = False

        if check_bar >= ppLen:
            is_ph = _pivot_high(highs, ppLen, check_bar)
            is_pl = _pivot_low(lows, ppLen, check_bar)

            if is_ph or is_pl:
                new_pivot = True
                prev_pivot_bar_2 = prev_pivot_bar
                prev_pivot_bar = check_bar

        # ── 2. Build VP and detect voids between pivots ──────────
        if (new_pivot
                and prev_pivot_bar_2 >= 0
                and prev_pivot_bar > prev_pivot_bar_2):

            vp_start = prev_pivot_bar_2
            vp_end = prev_pivot_bar
            vp_len = vp_end - vp_start

            if vp_len > 0:
                # Find highest/lowest in the pivot range
                range_highs = highs[vp_start:vp_end + 1]
                range_lows = lows[vp_start:vp_end + 1]
                p_hst = float(np.max(range_highs))
                p_lst = float(np.min(range_lows))
                p_stp = (p_hst - p_lst) / vpLev

                if p_stp > 0:
                    # Build volume profile
                    vp_vol = np.zeros(vpLev, dtype=float)

                    for bi in range(vp_start, vp_end + 1):
                        h_bar = highs[bi]
                        l_bar = lows[bi]
                        v_bar = volumes[bi]
                        bar_range = h_bar - l_bar

                        for lev in range(vpLev):
                            lev_bot = p_lst + lev * p_stp
                            lev_top = lev_bot + p_stp

                            if h_bar >= lev_bot and l_bar < lev_top:
                                if bar_range > 0:
                                    vp_vol[lev] += (
                                        v_bar * p_stp / bar_range
                                    )
                                else:
                                    vp_vol[lev] += v_bar

                    max_vol = float(np.max(vp_vol))

                    if max_vol > 0:
                        new_count = 0
                        for lev in range(vpLev):
                            if vp_vol[lev] / max_vol < liqT:
                                lev_bot = p_lst + lev * p_stp
                                lev_top = lev_bot + p_stp

                                # Check if already filled in the
                                # interval from vp_start to current
                                # bar
                                filled_already = False
                                mid = (lev_bot + lev_top) / 2.0
                                fill_start = -1

                                for fi in range(
                                    vp_start, min(bar + 1, n)
                                ):
                                    if fi == 0:
                                        continue
                                    prev_c = closes[fi - 1]
                                    cur_h = highs[fi]
                                    cur_l = lows[fi]

                                    s_prev = _math.copysign(
                                        1, prev_c - mid
                                    )
                                    s_lo = _math.copysign(
                                        1, cur_l - mid
                                    )
                                    s_hi = _math.copysign(
                                        1, cur_h - mid
                                    )

                                    if (s_prev != s_lo
                                            or s_prev != s_hi):
                                        if fill_start < 0:
                                            # First cross sets the
                                            # void left boundary
                                            fill_start = fi
                                        else:
                                            # Second cross fills it
                                            filled_already = True
                                            if fi <= bar:
                                                out_filled[fi] = 1
                                            break

                                if not filled_already:
                                    active_voids.append(
                                        [lev_top, lev_bot]
                                    )
                                    new_count += 1

                        if new_count > 0:
                            out_formed[bar] = 1

        # ── 3. Check active voids against current bar ────────────
        fills_this_bar = 0
        i = len(active_voids) - 1
        while i >= 0:
            v = active_voids[i]
            mid = (v[0] + v[1]) / 2.0

            if bar > 0:
                prev_c = closes[bar - 1]
                cur_h = highs[bar]
                cur_l = lows[bar]

                s_prev = _math.copysign(1, prev_c - mid)
                s_lo = _math.copysign(1, cur_l - mid)
                s_hi = _math.copysign(1, cur_h - mid)

                if s_prev != s_lo or s_prev != s_hi:
                    active_voids.pop(i)
                    fills_this_bar += 1

            i -= 1

        if fills_this_bar > 0:
            out_filled[bar] = 1

        # ── 4. Compute per-bar summary ───────────────────────────
        c = closes[bar]
        out_count[bar] = len(active_voids)

        if active_voids:
            best_dist = float("inf")
            best_top = np.nan
            best_bot = np.nan
            above = 0
            below = 0

            for v in active_voids:
                mid = (v[0] + v[1]) / 2.0
                dist = abs(c - mid)

                if dist < best_dist:
                    best_dist = dist
                    best_top = v[0]
                    best_bot = v[1]

                if v[1] > c:
                    above += 1
                elif v[0] < c:
                    below += 1
                else:
                    # Price is inside the void
                    above += 1
                    below += 1

            out_nearest_top[bar] = best_top
            out_nearest_bot[bar] = best_bot
            out_above[bar] = above
            out_below[bar] = below

    # ── Write results ────────────────────────────────────────────
    df[void_formed_column] = out_formed
    df[void_filled_column] = out_filled
    df[void_count_column] = out_count
    df[void_nearest_top_column] = out_nearest_top
    df[void_nearest_bottom_column] = out_nearest_bot
    df[void_above_count_column] = out_above
    df[void_below_count_column] = out_below

    if is_polars:
        import polars as pl
        return pl.from_pandas(df)

    return df


# ── Signal helper ────────────────────────────────────────────────────

def liquidity_levels_voids_signal(
    data: Union[PdDataFrame, PlDataFrame],
    void_nearest_top_column: str = "liq_void_nearest_top",
    void_nearest_bottom_column: str = "liq_void_nearest_bot",
    close_column: str = "Close",
    signal_column: str = "liq_void_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """Generate a directional signal based on proximity to voids.

    Args:
        data: DataFrame containing void columns (output of
            :func:`liquidity_levels_voids`).
        void_nearest_top_column: Column with nearest void top.
        void_nearest_bottom_column: Column with nearest void bottom.
        close_column: Column with close prices.
        signal_column: Output column name.

    Returns:
        DataFrame with ``{signal_column}`` added:

        - ``1``  — price is below the nearest void (void acts as
          a magnet above → bullish bias)
        - ``-1`` — price is above the nearest void (void acts as
          a magnet below → bearish bias)
        - ``0``  — no active void or price is inside the void
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        c = data[close_column].values
        top = data[void_nearest_top_column].values
        bot = data[void_nearest_bottom_column].values

        signal = np.where(
            np.isnan(top),
            0,
            np.where(c < bot, 1, np.where(c > top, -1, 0)),
        )
        data[signal_column] = signal.astype(int)
        return data
    elif isinstance(data, PlDataFrame):
        import polars as pl

        return data.with_columns(
            pl.when(pl.col(void_nearest_top_column).is_null())
            .then(0)
            .when(pl.col(close_column) < pl.col(void_nearest_bottom_column))
            .then(1)
            .when(pl.col(close_column) > pl.col(void_nearest_top_column))
            .then(-1)
            .otherwise(0)
            .alias(signal_column)
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ── Stats helper ─────────────────────────────────────────────────────

def get_liquidity_levels_voids_stats(
    data: Union[PdDataFrame, PlDataFrame],
    void_formed_column: str = "liq_void_formed",
    void_filled_column: str = "liq_void_filled",
    void_count_column: str = "liq_void_count",
) -> dict:
    """Return summary statistics for liquidity levels / voids.

    Args:
        data: DataFrame containing void columns (output of
            :func:`liquidity_levels_voids`).

    Returns:
        Dictionary with keys:

        - ``total_formation_events`` — bars where new voids formed
        - ``total_fill_events`` — bars where voids were filled
        - ``active_voids_last_bar`` — unfilled voids on the last bar
        - ``max_active_voids`` — peak number of simultaneous voids
    """
    if isinstance(data, PlDataFrame):
        formed = int(data[void_formed_column].sum())
        filled = int(data[void_filled_column].sum())
        last_count = int(data[void_count_column][-1])
        max_count = int(data[void_count_column].max())
    elif isinstance(data, PdDataFrame):
        formed = int(data[void_formed_column].sum())
        filled = int(data[void_filled_column].sum())
        last_count = int(data[void_count_column].iloc[-1])
        max_count = int(data[void_count_column].max())
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )

    return {
        "total_formation_events": formed,
        "total_fill_events": filled,
        "active_voids_last_bar": last_count,
        "max_active_voids": max_count,
    }
