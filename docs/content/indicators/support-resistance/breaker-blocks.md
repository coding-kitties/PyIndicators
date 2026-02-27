---
title: "Breaker Blocks"
sidebar_position: 6
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** `2 × swing_length + 1` bars
  (default params: 11 bars (swing_length=5))

Like order blocks, breaker blocks need confirmed swing pivots first. After warmup, new breaker blocks appear in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
Breaker blocks appear after pivot confirmation, not after smoothing delay.

| Event | Lag | Detail |
| --- | --- | --- |
| Breaker block zone appears | **≈ `swing_length` bars after the pivot** | Failed order block detected via MSS; pivot needs swing_length bars to confirm |
| Signal fires when price returns to zone | **0 bars** | Instant once the zone exists |

:::

Breaker Blocks are failed Order Blocks that flip into opposite support/resistance zones after a Market Structure Shift (MSS). Inspired by the "Breaker Blocks with Signals [LuxAlgo]" indicator.

**Concept:**
- When a bullish MSS occurs (close breaks above the most recent swing high) after a confirmed lower-low pattern, the decisive bullish candle in the up-leg becomes a **Bullish Breaker Block (+BB)** — acting as future support.
- When a bearish MSS occurs (close breaks below the most recent swing low) after a confirmed higher-high pattern, the decisive bearish candle becomes a **Bearish Breaker Block (-BB)** — acting as future resistance.

**Signals:**
- **Entry Long (+BB):** Price opens between the center line and the top, then closes above the top (bounce confirmation)
- **Entry Short (-BB):** Price opens between the center line and the bottom, then closes below the bottom
- **Cancel:** Price closes past the center line without triggering an entry (invalidation)
- **Mitigated:** Price closes fully through the opposite side of the zone

```python
def breaker_blocks(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 5,
    use_body: bool = False,
    use_2_candles: bool = False,
    stop_at_first_center_break: bool = True,
    high_column: str = "High",
    low_column: str = "Low",
    open_column: str = "Open",
    close_column: str = "Close",
) -> Union[PdDataFrame, PlDataFrame]:
```

Returns the following columns:
- `bb_bullish` / `bb_bearish`: 1 when a Breaker Block is formed
- `bb_top` / `bb_bottom` / `bb_center`: Active BB zone boundaries (forward-filled)
- `bb_direction`: 1 for bullish BB, -1 for bearish BB, 0 when no BB is active
- `bb_entry_long` / `bb_entry_short`: 1 when an entry signal fires
- `bb_cancel`: 1 when the center line is broken (invalidation)
- `bb_mitigated`: 1 when the BB is fully mitigated

Signal function:
- `bb_signal`: `1` = long entry, `-1` = short entry, `0` = no signal

Example

```python
from investing_algorithm_framework import download

from pyindicators import (
    breaker_blocks,
    breaker_blocks_signal,
    get_breaker_blocks_stats,
)

pd_df = download(
    symbol="btc/eur",
    market="bitvavo",
    time_frame="4h",
    start_date="2024-01-01",
    end_date="2024-06-01",
    pandas=True,
)

# Detect Breaker Blocks
pd_df = breaker_blocks(pd_df, swing_length=5)
pd_df = breaker_blocks_signal(pd_df)

# Get summary statistics
stats = get_breaker_blocks_stats(pd_df)
print(stats)

pd_df[["Close", "bb_bullish", "bb_bearish", "bb_top", "bb_bottom",
       "bb_entry_long", "bb_entry_short", "bb_signal"]].tail(10)
```

![BREAKER_BLOCKS](/img/indicators/breaker_blocks.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `swing_length` | `5` |

:::

