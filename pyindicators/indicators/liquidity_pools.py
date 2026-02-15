"""
Liquidity Pools Indicator

Faithful Python translation of the **Liquidity Pools [LuxAlgo]**
indicator from TradingView (Pine Script v5, Oct 16 2024, 332 lines).

Identifies liquidity pool zones by analysing high and low wicked
price areas, the number of contacts, and the frequency of visits.

Algorithm overview (matching Pine Script exactly)
-------------------------------------------------
1. Track a *highest* (``hst``) and *lowest* (``lst``) reference
   candle.  On each bar, adjust these references when price makes
   a new high/low with specific body-position conditions.

2. Count *wick contacts*: a bar where the wick extends beyond the
   reference body level but the body stays within.  Contacts must
   be separated by at least ``gap_bars`` bars and the reference
   body level must be stable (unchanged for 2 bars).

3. When enough contacts accumulate (``contact_count``) and the
   bars-since-last-wick counter crosses above ``confirmation_bars``
   (``ta.crossover(bs_hw, wait)`` in Pine), AND the close is on
   the correct side of the reference body level, a zone is created.

4. Zone creation uses a *running zone* pattern (``h_zn`` / ``l_zn``)
   with four merge branches:
   - Same origin bar -> adjust bounds of running zone.
   - Overlap from one side -> merge bounds and push to array.
   - Straddle -> merge bounds and push to array.
   - No overlap -> create a fresh running zone and push to array.

5. Zone extension: the most recent zone accumulates fractional volume
   from every overlapping candle (``get_civ`` in Pine).

6. Zone mitigation: when price closes beyond the zone boundary for
   two consecutive bars, the zone is removed.

Bearish pool  (resistance) -- body-top (``hst.t``) to highest high
(``hst.h``).
Bullish pool  (support) -- lowest low (``lst.l``) to body-bottom
(``lst.b``).
"""
from typing import Union, Optional

import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


# -- Helper: fractional candle volume inside a zone -------------------

def _get_civ(h: float, lo: float, v: float,
             zone_top: float, zone_bot: float) -> float:
    """Return the fraction of candle volume that lies inside *zone*.

    Mirrors Pine Script's ``get_civ`` helper:
    ``nz((_h - _l) / _r, 1) * volume`` where _h/_l are clamped to
    the zone boundaries.
    """
    r = h - lo
    if r <= 0:
        return v  # zero-range bar -> assign full volume (nz default)
    h_in = min(h, zone_top)
    l_in = max(lo, zone_bot)
    frac = max(0.0, (h_in - l_in) / r)
    return frac * v


# -- Main indicator ---------------------------------------------------

def liquidity_pools(
    data: Union[PdDataFrame, PlDataFrame],
    contact_count: int = 2,
    gap_bars: int = 5,
    confirmation_bars: int = 10,
    high_column: str = "High",
    low_column: str = "Low",
    open_column: str = "Open",
    close_column: str = "Close",
    volume_column: Optional[str] = "Volume",
    bull_pool_top_column: str = "liq_pool_bull_top",
    bull_pool_bottom_column: str = "liq_pool_bull_bottom",
    bear_pool_top_column: str = "liq_pool_bear_top",
    bear_pool_bottom_column: str = "liq_pool_bear_bottom",
    bull_pool_formed_column: str = "liq_pool_bull_formed",
    bear_pool_formed_column: str = "liq_pool_bear_formed",
    bull_pool_mitigated_column: str = "liq_pool_bull_mitigated",
    bear_pool_mitigated_column: str = "liq_pool_bear_mitigated",
    bull_pool_vol_column: str = "liq_pool_bull_vol",
    bear_pool_vol_column: str = "liq_pool_bear_vol",
) -> Union[PdDataFrame, PlDataFrame]:
    """Detect Liquidity Pool zones on OHLC(V) data.

    Faithful translation of:
    *Liquidity Pools [LuxAlgo]* -- Pine Script v5, Oct 16 2024.

    Args:
        data: pandas or polars DataFrame with OHLC(V) data.
        contact_count: Minimum wick contacts required to form a zone
            (Pine: ``cNum``, default 2).
        gap_bars: Minimum bars between successive contacts
            (Pine: ``gapCount``, default 5).
        confirmation_bars: Bars to wait before confirming a zone
            (Pine: ``wait``, default 10).
        high_column: Column name for highs.
        low_column: Column name for lows.
        open_column: Column name for opens.
        close_column: Column name for closes.
        volume_column: Column name for volume (``None`` to skip).
        bull_pool_top_column: Output -- top of most recent active
            bullish pool (= ``lst.b``, body bottom of reference).
        bull_pool_bottom_column: Output -- bottom of most recent
            active bullish pool (= ``lst.l``, lowest low).
        bear_pool_top_column: Output -- top of most recent active
            bearish pool (= ``hst.h``, highest high).
        bear_pool_bottom_column: Output -- bottom of most recent
            active bearish pool (= ``hst.t``, body top of reference).
        bull_pool_formed_column: Output -- 1 when a new bullish pool
            is confirmed.
        bear_pool_formed_column: Output -- 1 when a new bearish pool
            is confirmed.
        bull_pool_mitigated_column: Output -- 1 when a bullish pool
            is mitigated (close < bottom for 2 consecutive bars).
        bear_pool_mitigated_column: Output -- 1 when a bearish pool
            is mitigated (close > top for 2 consecutive bars).
        bull_pool_vol_column: Output -- accumulated volume inside the
            most recent active bullish pool.
        bear_pool_vol_column: Output -- accumulated volume inside the
            most recent active bearish pool.

    Returns:
        DataFrame with pool columns added.
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
    opens = df[open_column].values.astype(float)
    closes = df[close_column].values.astype(float)
    has_volume = (
        volume_column is not None and volume_column in df.columns
    )
    volumes = (
        df[volume_column].values.astype(float)
        if has_volume else np.zeros(len(df))
    )
    n = len(highs)

    # -- Output arrays ------------------------------------------------
    out_bull_top = np.full(n, np.nan)
    out_bull_bot = np.full(n, np.nan)
    out_bear_top = np.full(n, np.nan)
    out_bear_bot = np.full(n, np.nan)
    out_bull_formed = np.zeros(n, dtype=int)
    out_bear_formed = np.zeros(n, dtype=int)
    out_bull_mitigated = np.zeros(n, dtype=int)
    out_bear_mitigated = np.zeros(n, dtype=int)
    out_bull_vol = np.full(n, np.nan)
    out_bear_vol = np.full(n, np.nan)

    if n < 2:
        _write_output(
            df,
            out_bull_top, out_bull_bot, out_bear_top, out_bear_bot,
            out_bull_formed, out_bear_formed,
            out_bull_mitigated, out_bear_mitigated,
            out_bull_vol, out_bear_vol,
            bull_pool_top_column, bull_pool_bottom_column,
            bear_pool_top_column, bear_pool_bottom_column,
            bull_pool_formed_column, bear_pool_formed_column,
            bull_pool_mitigated_column, bear_pool_mitigated_column,
            bull_pool_vol_column, bear_pool_vol_column,
        )
        if is_polars:
            import polars as pl
            return pl.from_pandas(df)
        return df

    # -- Pine: var declarations (persistent across bars) --------------
    c_top_0 = max(opens[0], closes[0])
    c_bot_0 = min(opens[0], closes[0])

    # Reference candle data: {h, t, b, l, bi}
    # hst = highest reference, lst = lowest reference
    hst = {
        "h": highs[0], "t": c_top_0,
        "b": c_bot_0, "l": lows[0], "bi": 0,
    }
    lst = {
        "h": highs[0], "t": c_top_0,
        "b": c_bot_0, "l": lows[0], "bi": 0,
    }

    h_count = 0     # var int h_count = 0
    l_count = 0     # var int l_count = 0
    last_h_wick = 0  # var int last_h_wick = bar_index (=0)
    last_l_wick = 0  # var int last_l_wick = bar_index (=0)
    hi_vol = 0.0    # var float hi_vol = 0
    lo_vol = 0.0    # var float lo_vol = 0

    # Running zones (Pine: var zn h_zn, l_zn -- initially na)
    # Each: {top, bottom, state, vol, start_bar, right_bar} or None
    h_zn = None
    l_zn = None

    # Zone arrays (un-mitigated)
    bull_zones: list[dict] = []
    bear_zones: list[dict] = []

    # Pine: var int bs_hw = 0, bs_lw = 0
    bs_hw = 0
    bs_lw = 0

    # Previous-bar references for Pine [1] / [2] lookbacks
    prev_h_wick = False
    prev_l_wick = False
    hst_t_1ago = hst["t"]
    hst_t_2ago = float("nan")
    lst_b_1ago = lst["b"]
    lst_b_2ago = float("nan")
    prev_bs_hw = 0
    prev_bs_lw = 0

    # -- Bar-by-bar calculation ---------------------------------------
    for bar in range(1, n):
        h = highs[bar]
        lo = lows[bar]
        o = opens[bar]
        c = closes[bar]
        v = volumes[bar]
        c_top = max(o, c)
        c_bot = min(o, c)

        # -- 1. Adjusting High and Low Check Boundaries ---------------
        # Pine: if (high > hst.h) and ((c_top > hst.h) or
        #        (c_top < hst.t))
        if h > hst["h"] and (c_top > hst["h"] or c_top < hst["t"]):
            if h_count > 1:
                # Reset low reference to current candle
                lst = {
                    "h": h, "t": c_top,
                    "b": c_bot, "l": lo, "bi": bar,
                }
                lo_vol = 0.0
                l_count = 0
            # Set new high reference
            hst = {
                "h": h, "t": c_top,
                "b": c_bot, "l": lo, "bi": bar,
            }
            hi_vol = 0.0
            h_count = 1
            last_h_wick = bar

        # Pine: if (low < lst.l) and ((c_bot < lst.l) or
        #        (c_bot > lst.b))
        if lo < lst["l"] and (c_bot < lst["l"] or c_bot > lst["b"]):
            if l_count > 1:
                # Reset high reference to current candle
                hst = {
                    "h": h, "t": c_top,
                    "b": c_bot, "l": lo, "bi": bar,
                }
                hi_vol = 0.0
                h_count = 0
            # Set new low reference
            lst = {
                "h": h, "t": c_top,
                "b": c_bot, "l": lo, "bi": bar,
            }
            lo_vol = 0.0
            l_count = 1
            last_l_wick = bar

        # -- 2. Compute wicks for current bar -------------------------
        # Pine: h_wick = high > hst.t and c_top <= hst.t
        h_wick = (h > hst["t"]) and (c_top <= hst["t"])
        # Pine: l_wick = low < lst.b and c_bot >= lst.b
        l_wick = (lo < lst["b"]) and (c_bot >= lst["b"])

        # -- 3. Counting contacts (Pine uses [1] / [2] refs) ---------
        # Pine: if (h_wick[1] and (hst.t[1] == hst.t[2])
        #         and (bs_hw > gapCount))
        if (prev_h_wick
                and hst_t_1ago == hst_t_2ago
                and prev_bs_hw > gap_bars):
            h_count += 1
            last_h_wick = bar - 1

        # Pine: if (l_wick[1] and (lst.b[1] == lst.b[2])
        #         and (bs_lw > gapCount))
        if (prev_l_wick
                and lst_b_1ago == lst_b_2ago
                and prev_bs_lw > gap_bars):
            l_count += 1
            last_l_wick = bar - 1

        # -- 4. Bars since last wick ----------------------------------
        # Pine: bs_hw := math.abs(last_h_wick - bar_index)
        bs_hw = abs(last_h_wick - bar)
        bs_lw = abs(last_l_wick - bar)

        # -- 5. Track outer extremes ----------------------------------
        # Pine: if (high > hst.h)  hst.h := high
        if h > hst["h"]:
            hst["h"] = h
        # Pine: if (low < lst.l)  lst.l := low
        if lo < lst["l"]:
            lst["l"] = lo

        # -- 6. Volume tracking (reference-level volume) --------------
        # Pine: hst_vol = max((high - hst.t), 0) / (high-low) * vol
        h_range = h - lo
        if h_range > 0:
            hi_vol += max(h - hst["t"], 0) / h_range * v
            lo_vol += max(lst["b"] - lo, 0) / h_range * v

        # -- 7. Zone creation: ta.crossover(bs_hw/lw, wait) ----------
        hw_cross = (
            bs_hw > confirmation_bars
            and prev_bs_hw <= confirmation_bars
        )
        lw_cross = (
            bs_lw > confirmation_bars
            and prev_bs_lw <= confirmation_bars
        )

        # -- 7a. Bearish pool (resistance) from high contacts ---------
        # Pine: if (h_count >= cNum) and ta.crossover(bs_hw, wait)
        #        and (close < hst.t)
        if (h_count >= contact_count
                and hw_cross
                and c < hst["t"]):
            out_bear_formed[bar] = 1

            if h_zn is not None:
                zt = h_zn["top"]
                zb = h_zn["bottom"]

                if hst["bi"] == h_zn["start_bar"]:
                    # Same origin bar -> adjust running zone in-place
                    h_zn["top"] = max(hst["h"], zt)
                    h_zn["bottom"] = min(hst["t"], zt)

                elif (hst["h"] <= zt) and (hst["t"] >= zt):
                    # Overlap from below -> merge and push
                    h_zn["right_bar"] = bar
                    h_zn["vol"] += hi_vol
                    bear_zones.append(h_zn)
                    h_zn = None

                elif (hst["h"] > zt and hst["t"] < zt):
                    # Straddle -> merge bounds and push
                    h_zn["top"] = max(hst["t"], zt)
                    h_zn["bottom"] = min(hst["h"], zt)
                    h_zn["right_bar"] = bar
                    h_zn["vol"] += hi_vol
                    bear_zones.append(h_zn)
                    h_zn = None

                else:
                    # No overlap -> create fresh running zone
                    h_zn = {
                        "top": hst["h"],
                        "bottom": hst["t"],
                        "state": 0,
                        "vol": hi_vol,
                        "start_bar": hst["bi"],
                        "right_bar": bar,
                    }
                    bear_zones.append(h_zn)
            else:
                # First zone ever
                h_zn = {
                    "top": hst["h"],
                    "bottom": hst["t"],
                    "state": 0,
                    "vol": hi_vol,
                    "start_bar": hst["bi"],
                    "right_bar": bar,
                }
                bear_zones.append(h_zn)

        # -- 7b. Bullish pool (support) from low contacts -------------
        # Pine: if (l_count >= cNum) and ta.crossover(bs_lw, wait)
        #        and (close > lst.b)
        if (l_count >= contact_count
                and lw_cross
                and c > lst["b"]):
            out_bull_formed[bar] = 1

            if l_zn is not None:
                zt = l_zn["top"]
                zb = l_zn["bottom"]

                if lst["bi"] == l_zn["start_bar"]:
                    # Same origin bar -> adjust running zone in-place
                    l_zn["top"] = max(lst["b"], zt)
                    l_zn["bottom"] = min(lst["l"], zb)

                elif (lst["b"] <= zt) and (lst["l"] >= zb):
                    # Overlap within -> merge and push
                    l_zn["right_bar"] = bar
                    l_zn["vol"] += lo_vol
                    bull_zones.append(l_zn)
                    l_zn = None

                elif ((lst["b"] > zt and lst["l"] < zt)
                      or (lst["b"] > zb and lst["l"] < zb)
                      or (lst["b"] > zt and lst["l"] < zb)):
                    # Straddle -> merge bounds and push
                    l_zn["top"] = max(lst["b"], zt)
                    l_zn["bottom"] = min(lst["l"], zb)
                    l_zn["right_bar"] = bar
                    l_zn["vol"] += lo_vol
                    bull_zones.append(l_zn)
                    l_zn = None

                else:
                    # No overlap -> create fresh running zone
                    l_zn = {
                        "top": lst["b"],
                        "bottom": lst["l"],
                        "state": 0,
                        "vol": lo_vol,
                        "start_bar": lst["bi"],
                        "right_bar": bar,
                    }
                    bull_zones.append(l_zn)
            else:
                # First zone ever
                l_zn = {
                    "top": lst["b"],
                    "bottom": lst["l"],
                    "state": 0,
                    "vol": lo_vol,
                    "start_bar": lst["bi"],
                    "right_bar": bar,
                }
                bull_zones.append(l_zn)

        # -- 8. Zone extension & deletion -----------------------------
        # Bull zones
        i = len(bull_zones) - 1
        while i >= 0:
            z = bull_zones[i]
            zt = z["top"]
            zb = z["bottom"]

            # Most recent zone: accumulate candle volume if
            # overlapping (Pine get_civ) and extend line
            if i == len(bull_zones) - 1:
                if c > zt:
                    z["right_bar"] = bar
                # Volume accumulation when candle overlaps zone
                if ((h < zt and h > zb)
                        or (lo < zt and lo > zb)
                        or (h >= zt and lo <= zb)):
                    z["vol"] += _get_civ(h, lo, v, zt, zb)

            # Zone deletion: close < bottom for 2 consecutive bars
            if c < zb:
                if z["state"] < 0:
                    bull_zones.pop(i)
                    out_bull_mitigated[bar] = 1
                z["state"] -= 1
            else:
                z["state"] = 0
            i -= 1

        # Bear zones
        i = len(bear_zones) - 1
        while i >= 0:
            z = bear_zones[i]
            zt = z["top"]
            zb = z["bottom"]

            # Most recent zone: accumulate candle volume if
            # overlapping (Pine get_civ) and extend line
            if i == len(bear_zones) - 1:
                if c < zb:
                    z["right_bar"] = bar
                # Volume accumulation when candle overlaps zone
                if ((h < zt and h > zb)
                        or (lo < zt and lo > zb)
                        or (h >= zt and lo <= zb)):
                    z["vol"] += _get_civ(h, lo, v, zt, zb)

            # Zone deletion: close > top for 2 consecutive bars
            if c > zt:
                if z["state"] < 0:
                    bear_zones.pop(i)
                    out_bear_mitigated[bar] = 1
                z["state"] -= 1
            else:
                z["state"] = 0
            i -= 1

        # -- 9. Record output (most recent active zone) ---------------
        if bull_zones:
            out_bull_top[bar] = bull_zones[-1]["top"]
            out_bull_bot[bar] = bull_zones[-1]["bottom"]
            out_bull_vol[bar] = bull_zones[-1]["vol"]
        if bear_zones:
            out_bear_top[bar] = bear_zones[-1]["top"]
            out_bear_bot[bar] = bear_zones[-1]["bottom"]
            out_bear_vol[bar] = bear_zones[-1]["vol"]

        # -- 10. Save previous-bar values for [1]/[2] refs ------------
        prev_h_wick = h_wick
        prev_l_wick = l_wick
        hst_t_2ago = hst_t_1ago
        lst_b_2ago = lst_b_1ago
        hst_t_1ago = hst["t"]
        lst_b_1ago = lst["b"]
        prev_bs_hw = bs_hw
        prev_bs_lw = bs_lw

    # -- Write results ------------------------------------------------
    _write_output(
        df,
        out_bull_top, out_bull_bot, out_bear_top, out_bear_bot,
        out_bull_formed, out_bear_formed,
        out_bull_mitigated, out_bear_mitigated,
        out_bull_vol, out_bear_vol,
        bull_pool_top_column, bull_pool_bottom_column,
        bear_pool_top_column, bear_pool_bottom_column,
        bull_pool_formed_column, bear_pool_formed_column,
        bull_pool_mitigated_column, bear_pool_mitigated_column,
        bull_pool_vol_column, bear_pool_vol_column,
    )

    if is_polars:
        import polars as pl
        return pl.from_pandas(df)

    return df


def _write_output(
    df,
    bull_top, bull_bot, bear_top, bear_bot,
    bull_formed, bear_formed, bull_mit, bear_mit,
    bull_vol, bear_vol,
    bull_top_col, bull_bot_col, bear_top_col, bear_bot_col,
    bull_formed_col, bear_formed_col,
    bull_mit_col, bear_mit_col,
    bull_vol_col, bear_vol_col,
):
    """Write output arrays to DataFrame columns."""
    df[bull_top_col] = bull_top
    df[bull_bot_col] = bull_bot
    df[bear_top_col] = bear_top
    df[bear_bot_col] = bear_bot
    df[bull_formed_col] = bull_formed
    df[bear_formed_col] = bear_formed
    df[bull_mit_col] = bull_mit
    df[bear_mit_col] = bear_mit
    df[bull_vol_col] = bull_vol
    df[bear_vol_col] = bear_vol


# -- Signal helper ----------------------------------------------------

def liquidity_pool_signal(
    data: Union[PdDataFrame, PlDataFrame],
    bull_pool_formed_column: str = "liq_pool_bull_formed",
    bear_pool_formed_column: str = "liq_pool_bear_formed",
    signal_column: str = "liq_pool_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """Generate a combined signal from liquidity pool events.

    Args:
        data: DataFrame containing pool columns (output of
            :func:`liquidity_pools`).
        bull_pool_formed_column: Column with bullish pool formation
            flags.
        bear_pool_formed_column: Column with bearish pool formation
            flags.
        signal_column: Output column name.

    Returns:
        DataFrame with ``{signal_column}`` added:

        - ``1``  -- bullish pool formed (support detected)
        - ``-1`` -- bearish pool formed (resistance detected)
        - ``0``  -- no event
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        data[signal_column] = np.where(
            data[bull_pool_formed_column] == 1,
            1,
            np.where(data[bear_pool_formed_column] == 1, -1, 0),
        )
        return data
    elif isinstance(data, PlDataFrame):
        import polars as pl

        return data.with_columns(
            pl.when(pl.col(bull_pool_formed_column) == 1)
            .then(1)
            .when(pl.col(bear_pool_formed_column) == 1)
            .then(-1)
            .otherwise(0)
            .alias(signal_column)
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# -- Stats helper -----------------------------------------------------

def get_liquidity_pool_stats(
    data: Union[PdDataFrame, PlDataFrame],
    bull_pool_formed_column: str = "liq_pool_bull_formed",
    bear_pool_formed_column: str = "liq_pool_bear_formed",
    bull_pool_mitigated_column: str = "liq_pool_bull_mitigated",
    bear_pool_mitigated_column: str = "liq_pool_bear_mitigated",
) -> dict:
    """Return summary statistics for liquidity pools.

    Args:
        data: DataFrame containing pool columns (output of
            :func:`liquidity_pools`).

    Returns:
        Dictionary with keys:

        - ``total_bull_formed`` -- bullish pools formed
        - ``total_bear_formed`` -- bearish pools formed
        - ``total_formed`` -- total pools formed
        - ``total_bull_mitigated`` -- bullish pools mitigated
        - ``total_bear_mitigated`` -- bearish pools mitigated
        - ``total_mitigated`` -- total pools mitigated
        - ``bull_formed_ratio`` -- fraction that are bullish
        - ``bear_formed_ratio`` -- fraction that are bearish
    """
    if isinstance(data, PlDataFrame):
        bf = int(data[bull_pool_formed_column].sum())
        brf = int(data[bear_pool_formed_column].sum())
        bm = int(data[bull_pool_mitigated_column].sum())
        brm = int(data[bear_pool_mitigated_column].sum())
    elif isinstance(data, PdDataFrame):
        bf = int(data[bull_pool_formed_column].sum())
        brf = int(data[bear_pool_formed_column].sum())
        bm = int(data[bull_pool_mitigated_column].sum())
        brm = int(data[bear_pool_mitigated_column].sum())
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )

    total_formed = bf + brf

    return {
        "total_bull_formed": bf,
        "total_bear_formed": brf,
        "total_formed": total_formed,
        "total_bull_mitigated": bm,
        "total_bear_mitigated": brm,
        "total_mitigated": bm + brm,
        "bull_formed_ratio": bf / total_formed if total_formed else 0.0,
        "bear_formed_ratio": brf / total_formed if total_formed else 0.0,
    }
