---
title: "Market Structure Break"
sidebar_position: 10
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** `2 × pivot_length + 1` bars
  (default params: 15 bars (pivot_length=7))

Pivot points need `pivot_length` bars on each side. Once the first pivots are confirmed, break signals fire in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
Structure breaks fire after pivot confirmation, not after smoothing delay.

| Event | Lag | Detail |
| --- | --- | --- |
| Break signal fires | **≈ `pivot_length` bars after the pivot** | Pivot needs pivot_length bars on each side; signal fires when close breaks past the pivot |

:::

Market Structure Break (MSB) is a Smart Money Concept (SMC) indicator that detects when price breaks through significant pivot points, signaling potential trend changes. Combined with Order Block detection and quality scoring, this tool helps identify high-probability trading zones.

**Market Structure Break (MSB):**
- **Bullish MSB:** Price closes above the last pivot high, indicating potential bullish momentum
- **Bearish MSB:** Price closes below the last pivot low, indicating potential bearish momentum

**Order Block Quality Score (0-100):**
- Based on momentum z-score and volume percentile
- Score > 80 indicates a High Probability Zone (HPZ)

**Best Use Cases:**
- Pullback/retracement trading (enter at OB zones after MSB)
- Multi-timeframe analysis (use higher TF for bias, lower TF for entries)
- Supply & demand zone trading

```python
def market_structure_break(
    data: Union[PdDataFrame, PlDataFrame],
    pivot_length: int = 7,
    momentum_zscore_threshold: float = 0.5,
    high_column: str = 'High',
    low_column: str = 'Low',
    close_column: str = 'Close',
    volume_column: str = 'Volume',
    msb_bullish_column: str = 'msb_bullish',
    msb_bearish_column: str = 'msb_bearish',
    last_pivot_high_column: str = 'last_pivot_high',
    last_pivot_low_column: str = 'last_pivot_low',
    momentum_z_column: str = 'momentum_z'
) -> Union[PdDataFrame, PlDataFrame]:

def market_structure_ob(
    data: Union[PdDataFrame, PlDataFrame],
    pivot_length: int = 7,
    momentum_zscore_threshold: float = 0.5,
    max_active_obs: int = 10,
    ...
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
import pandas as pd
from pyindicators import (
    market_structure_break,
    market_structure_ob,
    get_market_structure_stats
)

# Create sample OHLC data
df = pd.DataFrame({
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
})

# Basic MSB detection
df = market_structure_break(df, pivot_length=5)
print(df[['msb_bullish', 'msb_bearish', 'last_pivot_high', 'last_pivot_low']])

# MSB with Order Block detection and quality scoring
df = market_structure_ob(df, pivot_length=5)
print(df[['msb_bullish', 'msb_bearish', 'ob_bullish', 'ob_bearish', 'ob_quality', 'ob_is_hpz']])

# Get statistics
stats = get_market_structure_stats(df)
print(f"Reliability: {stats['reliability']:.1f}%")
print(f"HPZ Count: {stats['hpz_count']}")
print(f"Bullish MSBs: {stats['bullish_msb_count']}")
print(f"Bearish MSBs: {stats['bearish_msb_count']}")
```

The `market_structure_break` function returns:
- `msb_bullish` / `msb_bearish`: 1 when MSB detected, 0 otherwise
- `last_pivot_high` / `last_pivot_low`: Most recent pivot levels
- `momentum_z`: Momentum z-score value

The `market_structure_ob` function additionally returns:
- `ob_bullish` / `ob_bearish`: 1 when Order Block detected at MSB
- `ob_top` / `ob_bottom`: Order Block zone boundaries
- `ob_quality`: Quality score (0-100)
- `ob_is_hpz`: True if quality > 80 (High Probability Zone)
- `ob_mitigated`: 1 when Order Block has been mitigated

**Recommended Parameters by Timeframe:**

| Timeframe | pivot_length | Use Case |
|-----------|-------------|----------|
| 1m-5m | 2-3 | Scalping entries |
| 15m | 3-5 | Day trading |
| 1H | 5-7 | Swing confirmation |
| 4H-Daily | 7-10 | Trend direction |

![MARKET_STRUCTURE](/img/indicators/market_structure_ob.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `pivot_length` | `7` |

:::

