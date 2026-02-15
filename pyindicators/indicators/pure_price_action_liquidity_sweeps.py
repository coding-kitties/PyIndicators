"""
Pure Price Action Liquidity Sweeps Indicator

Uses recursive fractal swing detection to identify
significant pivot levels and liquidity sweep events.

Unlike simple swing-based approaches, this indicator employs a
hierarchical pivot detection algorithm with configurable depth
("short", "intermediate", "long") to find progressively more
significant swing points.

A liquidity sweep occurs when price wicks through a pivot level
(high or low) without closing beyond it—indicating institutional
stop-hunting.  Levels are removed once price closes through them
(mitigated) or they exceed a configurable maximum age.
"""
from typing import Union

import numpy as np
from pandas import DataFrame as PdDataFrame
from polars import DataFrame as PlDataFrame

from pyindicators.exceptions import PyIndicatorException


# ── Recursive fractal swing detection ────────────────────────────────

def _recursive_swing_detect(vectors, mode, depth, swing_state):
    """
    Processes depth levels ``0 .. depth-1``.  At each level, if three
    swings are present and the middle one is the extreme (max for
    ``'bull'``, min for ``'bear'``), the middle point is either promoted
    to the next depth level or recorded as the final confirmed swing.

    Args:
        vectors: List of lists; each inner list holds up to 3
            ``(price, bar_index)`` tuples ordered newest-first.
        mode: ``'bull'`` or ``'bear'``.
        depth: Number of recursion levels.
        swing_state: ``(price, bar_index)`` of the persistent swing.

    Returns:
        Updated ``swing_state`` tuple.
    """
    for i in range(depth):
        v = vectors[i]

        if len(v) == 3:
            prices = (v[0][0], v[1][0], v[2][0])

            if mode == "bull":
                pivot_val = max(prices)
            else:
                pivot_val = min(prices)

            if pivot_val == v[1][0]:  # middle is the pivot
                if i < depth - 1:
                    # Promote to next depth level
                    vectors[i + 1].insert(0, v[1])

                    if len(vectors[i + 1]) > 3:
                        vectors[i + 1].pop()
                else:
                    # Final depth → confirmed swing level
                    swing_state = (v[1][0], v[1][1])

                # Keep only the newest element
                v.pop()
                v.pop()

    return swing_state


# ── Main indicator ───────────────────────────────────────────────────

def pure_price_action_liquidity_sweeps(
    data: Union[PdDataFrame, PlDataFrame],
    term: str = "long",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    max_level_age: int = 2000,
    bullish_sweep_column: str = "ppa_sweep_bullish",
    bearish_sweep_column: str = "ppa_sweep_bearish",
    sweep_high_column: str = "ppa_sweep_high",
    sweep_low_column: str = "ppa_sweep_low",
) -> Union[PdDataFrame, PlDataFrame]:
    """Detect Pure Price Action Liquidity Sweeps.

    Uses recursive fractal swing detection at the specified depth to
    find significant pivot highs and lows.  A sweep is flagged when
    price wicks through a pivot level without closing beyond it
    (smart-money stop-hunt).  Levels are removed once price closes
    through them (mitigated) or they exceed *max_level_age* bars.

    Args:
        data: pandas or polars DataFrame with OHLC data.
        term: Detection granularity – ``"short"`` (depth 1),
            ``"intermediate"`` (depth 2), or ``"long"`` (depth 3).
            Higher depth finds more significant swing points.
        high_column: Column name for highs.
        low_column: Column name for lows.
        close_column: Column name for closes.
        max_level_age: Maximum number of bars a level stays active
            before being discarded (default: 2000).
        bullish_sweep_column: Output column for bullish sweep flags.
        bearish_sweep_column: Output column for bearish sweep flags.
        sweep_high_column: Output column – price of the swept swing
            high (bearish sweep bars).
        sweep_low_column: Output column – price of the swept swing
            low (bullish sweep bars).

    Returns:
        DataFrame with sweep columns added:

        - ``{bullish_sweep_column}``: 1 when a bullish sweep is
          detected (sell-side liquidity grabbed below a pivot low).
        - ``{bearish_sweep_column}``: 1 when a bearish sweep is
          detected (buy-side liquidity grabbed above a pivot high).
        - ``{sweep_high_column}``: Price level of the swept swing high
          on bearish-sweep bars (NaN otherwise).
        - ``{sweep_low_column}``: Price level of the swept swing low
          on bullish-sweep bars (NaN otherwise).
    """
    # ── Resolve term to depth ────────────────────────────────────
    term_key = term.lower().replace(" ", "").replace("_", "")
    depth_map = {
        "short": 1,
        "shortterm": 1,
        "intermediate": 2,
        "intermediateterm": 2,
        "long": 3,
        "longterm": 3,
    }
    depth = depth_map.get(term_key)

    if depth is None:
        raise ValueError(
            f"Unsupported term '{term}'. "
            f"Use 'short', 'intermediate', or 'long'."
        )

    # ── DataFrame handling ───────────────────────────────────────
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
    n = len(highs)

    # ── Output arrays ────────────────────────────────────────────
    bull_sweep = np.zeros(n, dtype=int)
    bear_sweep = np.zeros(n, dtype=int)
    sweep_hi = np.full(n, np.nan)
    sweep_lo = np.full(n, np.nan)

    # Fractal vectors for highs and lows (one list per depth level)
    fh_vectors: list[list] = [[] for _ in range(depth)]
    fl_vectors: list[list] = [[] for _ in range(depth)]

    # Persistent swing states (price, bar_index)
    swing_high_state: tuple = (None, None)
    swing_low_state: tuple = (None, None)
    prev_ph = None
    prev_pl = None

    # Active pivot levels: each entry is
    # [price, origin_bar, mitigated, swept]
    active_highs: list[list] = []
    active_lows: list[list] = []

    for bar in range(n):
        h = highs[bar]
        lo = lows[bar]
        c = closes[bar]

        # ── Push current bar to depth-0 vectors ─────────────────
        fh_vectors[0].insert(0, (h, bar))
        fl_vectors[0].insert(0, (lo, bar))

        if len(fh_vectors[0]) > 3:
            fh_vectors[0].pop()

        if len(fl_vectors[0]) > 3:
            fl_vectors[0].pop()

        # ── Recursive swing detection ───────────────────────────
        swing_high_state = _recursive_swing_detect(
            fh_vectors, "bull", depth, swing_high_state
        )
        swing_low_state = _recursive_swing_detect(
            fl_vectors, "bear", depth, swing_low_state
        )

        ph = swing_high_state[0]
        pl = swing_low_state[0]

        # ── Register new pivot levels ───────────────────────────
        if ph is not None and ph > 0 and ph != prev_ph:
            active_highs.insert(
                0, [ph, swing_high_state[1], False, False]
            )

        if pl is not None and pl > 0 and pl != prev_pl:
            active_lows.insert(
                0, [pl, swing_low_state[1], False, False]
            )

        prev_ph = ph
        prev_pl = pl

        # ── Bearish sweeps (pivot highs) ────────────────────────
        remove_hi: list[int] = []

        for idx in range(len(active_highs) - 1, -1, -1):
            piv = active_highs[idx]
            prc = piv[0]
            origin_bar = piv[1]

            if not piv[2]:  # not mitigated
                if c > prc:
                    piv[2] = True  # mitigated

                if not piv[3]:  # not yet swept
                    if h > prc and c < prc:
                        bear_sweep[bar] = 1
                        sweep_hi[bar] = prc
                        piv[3] = True

            if (bar - origin_bar > max_level_age) or piv[2]:
                remove_hi.append(idx)

        for idx in sorted(remove_hi, reverse=True):
            active_highs.pop(idx)

        # ── Bullish sweeps (pivot lows) ─────────────────────────
        remove_lo: list[int] = []

        for idx in range(len(active_lows) - 1, -1, -1):
            piv = active_lows[idx]
            prc = piv[0]
            origin_bar = piv[1]

            if not piv[2]:  # not mitigated
                if c < prc:
                    piv[2] = True

                if not piv[3]:  # not yet swept
                    if lo < prc and c > prc:
                        bull_sweep[bar] = 1
                        sweep_lo[bar] = prc
                        piv[3] = True

            if (bar - origin_bar > max_level_age) or piv[2]:
                remove_lo.append(idx)

        for idx in sorted(remove_lo, reverse=True):
            active_lows.pop(idx)

    # ── Write results ────────────────────────────────────────────
    df[bullish_sweep_column] = bull_sweep
    df[bearish_sweep_column] = bear_sweep
    df[sweep_high_column] = sweep_hi
    df[sweep_low_column] = sweep_lo

    if is_polars:
        import polars as pl

        return pl.from_pandas(df)

    return df


# ── Signal helper ────────────────────────────────────────────────────

def pure_price_action_liquidity_sweep_signal(
    data: Union[PdDataFrame, PlDataFrame],
    bullish_sweep_column: str = "ppa_sweep_bullish",
    bearish_sweep_column: str = "ppa_sweep_bearish",
    signal_column: str = "ppa_sweep_signal",
) -> Union[PdDataFrame, PlDataFrame]:
    """Generate a combined signal from pure-PA sweep results.

    Args:
        data: DataFrame that already contains sweep columns (output
            of :func:`pure_price_action_liquidity_sweeps`).
        bullish_sweep_column: Column with bullish sweep flags.
        bearish_sweep_column: Column with bearish sweep flags.
        signal_column: Output column name.

    Returns:
        DataFrame with ``{signal_column}`` added:

        - ``1``  – bullish sweep (sell-side liquidity grabbed)
        - ``-1`` – bearish sweep (buy-side liquidity grabbed)
        - ``0``  – no sweep
    """
    if isinstance(data, PdDataFrame):
        data = data.copy()
        data[signal_column] = np.where(
            data[bullish_sweep_column] == 1,
            1,
            np.where(data[bearish_sweep_column] == 1, -1, 0),
        )
        return data
    elif isinstance(data, PlDataFrame):
        import polars as pl

        return data.with_columns(
            pl.when(pl.col(bullish_sweep_column) == 1)
            .then(1)
            .when(pl.col(bearish_sweep_column) == 1)
            .then(-1)
            .otherwise(0)
            .alias(signal_column)
        )
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )


# ── Stats helper ─────────────────────────────────────────────────────

def get_pure_price_action_liquidity_sweep_stats(
    data: Union[PdDataFrame, PlDataFrame],
    bullish_sweep_column: str = "ppa_sweep_bullish",
    bearish_sweep_column: str = "ppa_sweep_bearish",
) -> dict:
    """Return summary statistics for pure-PA liquidity sweeps.

    Args:
        data: DataFrame containing sweep columns (output of
            :func:`pure_price_action_liquidity_sweeps`).
        bullish_sweep_column: Column with bullish sweep flags.
        bearish_sweep_column: Column with bearish sweep flags.

    Returns:
        Dictionary with keys:

        - ``total_bullish`` – number of bullish sweeps
        - ``total_bearish`` – number of bearish sweeps
        - ``total_sweeps`` – total sweep count
        - ``bullish_ratio`` – fraction of sweeps that are bullish
        - ``bearish_ratio`` – fraction of sweeps that are bearish
    """
    if isinstance(data, PlDataFrame):
        bull = int(data[bullish_sweep_column].sum())
        bear = int(data[bearish_sweep_column].sum())
    elif isinstance(data, PdDataFrame):
        bull = int(data[bullish_sweep_column].sum())
        bear = int(data[bearish_sweep_column].sum())
    else:
        raise PyIndicatorException(
            "Input data must be a pandas or polars DataFrame."
        )

    total = bull + bear

    return {
        "total_bullish": bull,
        "total_bearish": bear,
        "total_sweeps": total,
        "bullish_ratio": bull / total if total > 0 else 0.0,
        "bearish_ratio": bear / total if total > 0 else 0.0,
    }
