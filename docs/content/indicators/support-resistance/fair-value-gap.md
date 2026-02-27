---
title: "Fair Value Gap (FVG)"
sidebar_position: 4
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** 3 bars
  (default params: 3 bars)

FVG is a 3-bar candlestick pattern. The first possible detection is on bar 3. No smoothing — fully real-time after that.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
FVGs are detected instantly on the current bar.

| Event | Lag | Detail |
| --- | --- | --- |
| FVG detected | **0 bars** | 3-bar candlestick pattern; detection uses only the current bar and 2 bars lookback |

:::

A Fair Value Gap (FVG) is a price imbalance that occurs when there's a gap between candlesticks, representing institutional order flow. These gaps often act as support/resistance zones where price tends to return.

**Bullish FVG (Gap Up):** Occurs when the low of the current candle is higher than the high of the candle 2 bars ago. This creates an upward gap that may act as future support.

**Bearish FVG (Gap Down):** Occurs when the high of the current candle is lower than the low of the candle 2 bars ago. This creates a downward gap that may act as future resistance.

```python
def fair_value_gap(
    data: Union[PdDataFrame, PlDataFrame],
    high_column: str = 'High',
    low_column: str = 'Low',
    bullish_fvg_column: str = 'bullish_fvg',
    bearish_fvg_column: str = 'bearish_fvg',
    bullish_fvg_top_column: str = 'bullish_fvg_top',
    bullish_fvg_bottom_column: str = 'bullish_fvg_bottom',
    bearish_fvg_top_column: str = 'bearish_fvg_top',
    bearish_fvg_bottom_column: str = 'bearish_fvg_bottom'
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
import pandas as pd
from pyindicators import fair_value_gap, fvg_signal, fvg_filled

# Create sample OHLC data
df = pd.DataFrame({
    'High': [100, 105, 115, 120, 118, 115],
    'Low': [95, 100, 102, 115, 113, 99],
    'Close': [98, 103, 110, 117, 115, 100]
})

# Detect Fair Value Gaps
df = fair_value_gap(df)
print(df[['bullish_fvg', 'bearish_fvg', 'bullish_fvg_top', 'bullish_fvg_bottom']])

# Generate signals when price enters an FVG zone
df = fvg_signal(df)
print(df['fvg_signal'])  # 1 = in bullish zone, -1 = in bearish zone, 0 = outside

# Detect when FVGs have been filled (mitigated)
df = fvg_filled(df)
print(df[['bullish_fvg_filled', 'bearish_fvg_filled']])
```

The `fvg_signal` function generates signals:
- **1**: Price is within a bullish FVG zone (potential long entry)
- **-1**: Price is within a bearish FVG zone (potential short entry)
- **0**: Price is outside any FVG zone

The `fvg_filled` function detects when FVGs have been mitigated:
- Bullish FVG filled: Price drops to reach the bottom of the gap
- Bearish FVG filled: Price rises to reach the top of the gap

![FAIR_VALUE_GAP](/img/indicators/fair_value_gap.png)
