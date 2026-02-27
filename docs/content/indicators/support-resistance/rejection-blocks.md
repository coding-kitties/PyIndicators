---
title: "Rejection Blocks"
sidebar_position: 8
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** `2 × swing_length + 1` bars
  (default params: 11 bars (swing_length=5))

Requires confirmed swing pivots. After warmup, rejection blocks appear in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
Rejection blocks appear after pivot confirmation, not after smoothing delay.

| Event | Lag | Detail |
| --- | --- | --- |
| Rejection block zone appears | **≈ `swing_length` bars after the pivot** | Wick-ratio candle at confirmed swing point; pivot needs swing_length bars |
| Signal fires when price returns to zone | **0 bars** | Instant once the zone exists |

:::

Rejection Blocks identify candles at swing extremes whose **disproportionately long wicks** signal institutional rejection of a price level, creating a tradeable zone.

**Concept (ICT / Smart Money):**
- A Rejection Block forms when a candle at a pivot point has a wick that is at least `wick_threshold` (default 50 %) of the total candle range.
- The long wick shows that price was driven to a level but was *rejected* — institutional participants absorbed the orders and pushed price back.
- **Bullish Rejection Block:** At a confirmed swing low, the candle's **lower wick** (Low → body bottom) is disproportionately large. The zone spans the lower wick area.
- **Bearish Rejection Block:** At a confirmed swing high, the candle's **upper wick** (body top → High) is disproportionately large. The zone spans the upper wick area.
- When price returns to this wick zone, the same institutional interest is expected — a potential trade entry.

**Signals:**
- **Entry Long:** Price retraces into the bullish RB zone (potential long entry)
- **Entry Short:** Price retraces into the bearish RB zone (potential short entry)
- **Mitigated:** Price closes through the opposite side of the zone (block invalidated)

```python
def rejection_blocks(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 5,
    wick_threshold: float = 0.5,
    high_column: str = "High",
    low_column: str = "Low",
    open_column: str = "Open",
    close_column: str = "Close",
) -> Union[PdDataFrame, PlDataFrame]:
```

Returns the following columns:
- `rb_bullish` / `rb_bearish`: 1 when a Rejection Block is established
- `rb_top` / `rb_bottom`: Active RB zone boundaries (forward-filled)
- `rb_direction`: 1 for bullish RB, -1 for bearish RB, 0 when no RB is active
- `rb_entry_long` / `rb_entry_short`: 1 when price enters the RB zone
- `rb_mitigated`: 1 when the RB zone is mitigated

Signal function:
- `rb_signal`: `1` = long entry, `-1` = short entry, `0` = no signal

Example

```python
from investing_algorithm_framework import download

from pyindicators import (
    rejection_blocks,
    rejection_blocks_signal,
    get_rejection_blocks_stats,
)

pd_df = download(
    symbol="btc/eur",
    market="bitvavo",
    time_frame="4h",
    start_date="2024-01-01",
    end_date="2024-06-01",
    pandas=True,
)

# Detect Rejection Blocks
pd_df = rejection_blocks(pd_df, swing_length=5, wick_threshold=0.5)
pd_df = rejection_blocks_signal(pd_df)

# Get summary statistics
stats = get_rejection_blocks_stats(pd_df)
print(stats)

pd_df[["Close", "rb_bullish", "rb_bearish", "rb_top",
       "rb_bottom", "rb_entry_long", "rb_entry_short",
       "rb_signal"]].tail(10)
```

![REJECTION_BLOCKS](/img/indicators/rejection_blocks.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `swing_length` | `5` |

:::

