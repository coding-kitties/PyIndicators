---
title: "Liquidity Pools"
sidebar_position: 15
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** ≥ `contact_count × gap_bars` bars
  (default params: Varies (contact_count=2))

Zone requires `contact_count` wick touches with sufficient spacing (`gap_bars`). The exact warmup depends on when enough contacts occur. After zones are formed, signals fire in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
Pool zones appear after enough wick contacts are observed.

| Event | Lag | Detail |
| --- | --- | --- |
| Pool zone created | **depends on contact_count + gap_bars** | Zone requires contact_count wick touches with sufficient spacing |
| Price enters pool zone (signal) | **0 bars** | Instant once the zone exists |

:::

Liquidity Pools is a Smart Money Concept indicator that identifies zones where resting orders cluster, detected by tracking areas where price repeatedly bounces (wicks) from a level.

A **bullish pool** (support) forms when price wicks below a body-bottom level multiple times without closing below it. A **bearish pool** (resistance) forms when price wicks above a body-top level multiple times without closing above it. Zones are mitigated (invalidated) when price closes through them on two consecutive bars.

Key parameters:

- **Contact Count** – minimum wick bounces required to form a pool (default: 2). Higher = fewer, more reliable zones.
- **Gap Bars** – minimum bars between contacts to prevent double-counting (default: 5).
- **Confirmation Bars** – bars price must stay away before confirming the zone (default: 10).

```python
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
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
import pandas as pd
from pyindicators import (
    liquidity_pools,
    liquidity_pool_signal,
    get_liquidity_pool_stats
)

# Create sample OHLC data
df = pd.DataFrame({
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...]
})

# Detect liquidity pools
df = liquidity_pools(df, contact_count=2, gap_bars=5, confirmation_bars=10)
print(df[['liq_pool_bull_top', 'liq_pool_bull_bottom',
          'liq_pool_bear_top', 'liq_pool_bear_bottom']])

# Generate trading signals
# 1 = bullish pool formed (support), -1 = bearish pool formed (resistance)
df = liquidity_pool_signal(df)
pool_events = df[df['liq_pool_signal'] != 0]

# Get statistics
stats = get_liquidity_pool_stats(df)
print(f"Bull pools formed: {stats['total_bull_formed']}")
print(f"Bear pools formed: {stats['total_bear_formed']}")
print(f"Total mitigated: {stats['total_mitigated']}")
```

The function returns:
- `liq_pool_bull_top` / `liq_pool_bull_bottom`: Boundaries of the most recent active bullish pool (NaN if none)
- `liq_pool_bear_top` / `liq_pool_bear_bottom`: Boundaries of the most recent active bearish pool (NaN if none)
- `liq_pool_bull_formed` / `liq_pool_bear_formed`: 1 when a new pool forms
- `liq_pool_bull_mitigated` / `liq_pool_bear_mitigated`: 1 when a pool is mitigated (broken)

**Trading Strategy:**
- Bullish pools are support zones where institutional buyers accumulate—look for long entries near the zone
- Bearish pools are resistance zones where institutional sellers distribute—look for short entries near the zone
- Mitigation signals a change in market structure; the zone is no longer valid
- Increase `contact_count` for higher-quality, more reliable zones

![LIQUIDITY_POOLS](/img/indicators/liquidity_pools.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `contact_count` | `2` |

:::

