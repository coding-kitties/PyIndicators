---
title: "Mitigation Blocks"
sidebar_position: 7
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** `2 × swing_length + 1` bars
  (default params: 11 bars (swing_length=5))

Requires confirmed swing pivots. After warmup, new mitigation blocks appear in real-time on every new bar.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
Mitigation blocks appear after pivot confirmation, not after smoothing delay.

| Event | Lag | Detail |
| --- | --- | --- |
| Mitigation block zone appears | **≈ `swing_length` bars after the pivot** | First same-direction candle of impulse to MSS; pivot needs swing_length bars |
| Signal fires when price returns to zone | **0 bars** | Instant once the zone exists |

:::

Mitigation Blocks identify the **first candle that initiates** an impulsive move leading to a Market Structure Shift — the origin of institutional order flow.

**Concept (ICT / Smart Money):**
- While an *Order Block* is the last **opposing** candle before an impulse, a *Mitigation Block* is the first **same-direction** candle that **starts** the move.
- **Bullish Mitigation Block:** After a bullish MSS (close breaks swing high with confirmed LL pattern), the first bullish candle (close > open) after the preceding swing low that kicked off the upward impulse.
- **Bearish Mitigation Block:** After a bearish MSS (close breaks swing low with confirmed HH pattern), the first bearish candle (close < open) after the preceding swing high that kicked off the downward impulse.
- When price returns to a Mitigation Block zone, institutional traders are "mitigating" (closing/adjusting) positions opened at that origin candle.

**Signals:**
- **Entry Long:** Price retraces into the bullish MB zone (potential long entry)
- **Entry Short:** Price retraces into the bearish MB zone (potential short entry)
- **Mitigated:** Price closes through the opposite side of the zone (block invalidated)

```python
def mitigation_blocks(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 5,
    use_body: bool = False,
    high_column: str = "High",
    low_column: str = "Low",
    open_column: str = "Open",
    close_column: str = "Close",
) -> Union[PdDataFrame, PlDataFrame]:
```

Returns the following columns:
- `mb_bullish` / `mb_bearish`: 1 when a Mitigation Block is established
- `mb_top` / `mb_bottom`: Active MB zone boundaries (forward-filled)
- `mb_direction`: 1 for bullish MB, -1 for bearish MB, 0 when no MB is active
- `mb_entry_long` / `mb_entry_short`: 1 when price enters the MB zone
- `mb_mitigated`: 1 when the MB zone is mitigated

Signal function:
- `mb_signal`: `1` = long entry, `-1` = short entry, `0` = no signal

Example

```python
from investing_algorithm_framework import download

from pyindicators import (
    mitigation_blocks,
    mitigation_blocks_signal,
    get_mitigation_blocks_stats,
)

pd_df = download(
    symbol="btc/eur",
    market="bitvavo",
    time_frame="4h",
    start_date="2024-01-01",
    end_date="2024-06-01",
    pandas=True,
)

# Detect Mitigation Blocks
pd_df = mitigation_blocks(pd_df, swing_length=5)
pd_df = mitigation_blocks_signal(pd_df)

# Get summary statistics
stats = get_mitigation_blocks_stats(pd_df)
print(stats)

pd_df[["Close", "mb_bullish", "mb_bearish", "mb_top",
       "mb_bottom", "mb_entry_long", "mb_entry_short",
       "mb_signal"]].tail(10)
```

![MITIGATION_BLOCKS](/img/indicators/mitigation_blocks.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `swing_length` | `5` |

:::

