---
title: "Trendline Breakout Navigator"
sidebar_position: 26
tags: [lagging]
---

:::info[Warmup Window]
**Minimum bars needed:** `2 × swing_long + 1` bars
  (default params: 121 bars (swing_long=60))

The indicator needs confirmed pivot highs and lows before constructing trendlines. After warmup, new trendlines and breakouts are detected on every incoming bar.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::caution[Lagging Indicator]
Pivot confirmation and trendline construction introduce lag.

| Event | Lag | Detail |
| --- | --- | --- |
| Pivot confirmed | **≈ `swing_length` bars** | Each timeframe waits for left/right bars to confirm the pivot |
| HH / LL trend flip | **≈ `swing_length` bars after pivot** | Trend changes once a Higher-High or Lower-Low is confirmed |
| Wick break detected | **0 bars after trendline exists** | Instant once the trendline is active |

:::

The Trendline Breakout Navigator is a multi-timeframe trendline detection indicator ported from the LuxAlgo PineScript indicator. It detects pivot highs and lows at three swing lengths (long, medium, short), constructs trendlines on HH/LL trend reversals, and tracks trendline breakouts and wick interactions.

**Concept:**
- **Pivot Detection** — confirmed pivot highs/lows using left/right bars at each swing length.
- **Trendlines** — bearish trendlines drawn from swing highs on a trend reversal to LL; bullish trendlines drawn from swing lows on a trend reversal to HH.
- **Wick Breaks** — bars where the wick crosses the trendline but the close does not (indicating a false break / liquidity grab).
- **Composite Score** — aggregated trend score from all three timeframes (range: −3 to +3 when all enabled).

```python
def trendline_breakout_navigator(
    data: Union[PdDataFrame, PlDataFrame],
    swing_long: int = 60,
    swing_medium: int = 30,
    swing_short: int = 10,
    enable_long: bool = True,
    enable_medium: bool = True,
    enable_short: bool = True,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
) -> Union[PdDataFrame, PlDataFrame]:
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | `DataFrame` | — | OHLCV DataFrame (pandas or polars) |
| `swing_long` | `int` | `60` | Swing length for the long timeframe |
| `swing_medium` | `int` | `30` | Swing length for the medium timeframe |
| `swing_short` | `int` | `10` | Swing length for the short timeframe |
| `enable_long` | `bool` | `True` | Enable the long timeframe |
| `enable_medium` | `bool` | `True` | Enable the medium timeframe |
| `enable_short` | `bool` | `True` | Enable the short timeframe |
| `high_column` | `str` | `"High"` | Name of the High column |
| `low_column` | `str` | `"Low"` | Name of the Low column |
| `close_column` | `str` | `"Close"` | Name of the Close column |

Returns the following columns:
- `tbn_trend_long` / `tbn_trend_medium` / `tbn_trend_short`: Trend direction per timeframe (1 = bullish, −1 = bearish, 0 = undetermined)
- `tbn_value_long` / `tbn_value_medium` / `tbn_value_short`: Projected trendline price per timeframe (NaN when no active line)
- `tbn_slope_long` / `tbn_slope_medium` / `tbn_slope_short`: Trendline slope per bar per timeframe
- `tbn_wick_bull`: 1 on bars with a bullish wick break (any timeframe)
- `tbn_wick_bear`: 1 on bars with a bearish wick break (any timeframe)
- `tbn_hh`: 1 on bars where a Higher High is confirmed (any timeframe)
- `tbn_ll`: 1 on bars where a Lower Low is confirmed (any timeframe)
- `tbn_composite_trend`: Sum of all enabled timeframe trends (−3 to +3 if all enabled)

Signal function:

```python
def trendline_breakout_navigator_signal(
    data: Union[PdDataFrame, PlDataFrame],
    composite_trend_column: str = "tbn_composite_trend",
    signal_column: str = "tbn_signal",
) -> Union[PdDataFrame, PlDataFrame]:
```

- `tbn_signal`: `1` = bullish (composite > 0), `-1` = bearish (composite < 0), `0` = neutral

Stats function:

```python
def get_trendline_breakout_navigator_stats(
    data: Union[PdDataFrame, PlDataFrame],
) -> Dict[str, object]:
```

Returns a dictionary with:
- `bullish_bars_long` / `bearish_bars_long` — bars where long trend == 1 / −1
- `bullish_bars_medium` / `bearish_bars_medium` — bars where medium trend == 1 / −1
- `bullish_bars_short` / `bearish_bars_short` — bars where short trend == 1 / −1
- `composite_bullish` / `composite_bearish` — bars where composite > 0 / < 0
- `composite_bullish_pct` / `composite_bearish_pct` — percentage of composite bullish/bearish bars
- `hh_count` / `ll_count` — total HH / LL detections
- `wick_bull_count` / `wick_bear_count` — total bullish / bearish wick breaks
- `trend_changes` — number of composite trend sign changes
- `active_trendline_bars` — bars where long trendline is active

Example

```python
from investing_algorithm_framework import download

from pyindicators import (
    trendline_breakout_navigator,
    trendline_breakout_navigator_signal,
    get_trendline_breakout_navigator_stats,
)

pd_df = download(
    symbol="btc/eur",
    market="bitvavo",
    time_frame="4h",
    start_date="2024-01-01",
    end_date="2024-06-01",
    pandas=True,
)

# Detect trendlines and breakouts
pd_df = trendline_breakout_navigator(pd_df, swing_long=60, swing_medium=30, swing_short=10)
pd_df = trendline_breakout_navigator_signal(pd_df)

# Get summary statistics
stats = get_trendline_breakout_navigator_stats(pd_df)
print(stats)

pd_df[["Close", "tbn_trend_long", "tbn_value_long", "tbn_composite_trend",
       "tbn_wick_bull", "tbn_wick_bear", "tbn_hh", "tbn_ll", "tbn_signal"]].tail(10)
```

![TRENDLINE_BREAKOUT_NAVIGATOR](/img/indicators/trendline_breakout_navigator.png)
