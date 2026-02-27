---
title: "Liquidity Sweeps"
sidebar_position: 12
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** `2 × swing_length + 1` bars
  (default params: 11 bars (swing_length=5))

Swing points need `swing_length` bars on each side to be confirmed. After the first swings are established, sweep signals fire in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
Sweep signals fire instantly when price wicks through a confirmed swing and reverses.

| Event | Lag | Detail |
| --- | --- | --- |
| Swing high/low is confirmed | **≈ `swing_length` bars after the swing** | Swing needs swing_length bars on each side |
| Sweep signal fires | **0 bars after the sweep** | Instant: detected on the bar that wicks through and reverses |

:::

Liquidity Sweeps is a Smart Money Concept indicator that detects when price momentarily pierces a swing high or swing low—grabbing resting liquidity—before reversing. This behaviour is a hallmark of institutional order flow: stop-loss clusters sitting above swing highs (buyside liquidity) or below swing lows (sellside liquidity) get triggered, and price quickly snaps back.

Three detection modes are available:

- **Wicks** – the candle wick pierces the swing level but the close remains on the original side.
- **Outbreak / Retest** – price closes beyond the level, then a later candle retests it from the other side while closing back.
- **All** – combines both wick and outbreak/retest sweeps.

```python
def liquidity_sweeps(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 5,
    mode: str = "wicks",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    bullish_sweep_column: str = "liq_sweep_bullish",
    bearish_sweep_column: str = "liq_sweep_bearish",
    sweep_high_column: str = "liq_sweep_high",
    sweep_low_column: str = "liq_sweep_low",
    sweep_type_column: str = "liq_sweep_type",
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
import pandas as pd
from pyindicators import (
    liquidity_sweeps,
    liquidity_sweep_signal,
    get_liquidity_sweep_stats
)

# Create sample OHLC data
df = pd.DataFrame({
    'High': [...],
    'Low': [...],
    'Close': [...]
})

# Detect liquidity sweeps (wick-through mode)
df = liquidity_sweeps(df, swing_length=5, mode="wicks")
print(df[['liq_sweep_bullish', 'liq_sweep_bearish', 'liq_sweep_high', 'liq_sweep_low']])

# Generate trading signals
# 1 = bullish sweep, -1 = bearish sweep, 0 = no sweep
df = liquidity_sweep_signal(df)
bullish_sweeps = df[df['liq_sweep_signal'] == 1]

# Get statistics
stats = get_liquidity_sweep_stats(df)
print(f"Total bullish sweeps: {stats['total_bullish']}")
print(f"Total bearish sweeps: {stats['total_bearish']}")
```

The function returns:
- `liq_sweep_bullish`: 1 when a bullish liquidity sweep is detected (sell-side liquidity grabbed)
- `liq_sweep_bearish`: 1 when a bearish liquidity sweep is detected (buy-side liquidity grabbed)
- `liq_sweep_high`: Price level of the swept swing high
- `liq_sweep_low`: Price level of the swept swing low
- `liq_sweep_type`: Type of sweep (`"wick"` or `"outbreak_retest"`)

**Trading Strategy:**
- Bullish sweeps below swing lows indicate potential long entries (smart money accumulation)
- Bearish sweeps above swing highs indicate potential short entries (smart money distribution)
- Use the sweep level (`liq_sweep_high` / `liq_sweep_low`) as a reference for stop-loss placement

![LIQUIDITY_SWEEPS](/img/indicators/liquidity_sweeps.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `swing_length` | `5` |

:::

