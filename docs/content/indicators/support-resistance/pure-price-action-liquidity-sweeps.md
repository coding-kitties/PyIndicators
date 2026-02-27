---
title: "Pure Price Action Liquidity Sweeps"
sidebar_position: 14
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** depth-dependent (varies by fractal depth)
  (default params: Varies — deeper fractals need more bars)

Recursive fractal detection with configurable depth (short/intermediate/long). Deeper detection needs more bars. After warmup, sweep signals fire in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
Sweeps fire instantly once recursive fractal swings are confirmed.

| Event | Lag | Detail |
| --- | --- | --- |
| Fractal swing is confirmed | **depth-dependent** | Recursive fractal detection with configurable depth; deeper = more bars needed |
| Sweep signal fires | **0 bars after the sweep** | Instant once the swing is confirmed |

:::

Pure Price Action Liquidity Sweeps is a Smart Money Concept indicator that uses recursive fractal swing detection to identify significant pivot levels and detect liquidity sweep events.

Unlike simple swing-based approaches, this indicator employs a hierarchical pivot detection algorithm with configurable depth to find progressively more significant swing points. A liquidity sweep occurs when price wicks through a pivot level without closing beyond it—indicating institutional stop-hunting. Levels are automatically invalidated once price closes through them (mitigated).

Three detection granularities are available:

- **Short Term** (depth 1) – detects all basic swing pivots, yielding the most sweep signals.
- **Intermediate Term** (depth 2) – uses two levels of fractal filtering for moderately significant pivots.
- **Long Term** (depth 3) – three levels of recursion, producing only the most significant swing points and fewest sweeps.

```python
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
```

Example

```python
import pandas as pd
from pyindicators import (
    pure_price_action_liquidity_sweeps,
    pure_price_action_liquidity_sweep_signal,
    get_pure_price_action_liquidity_sweep_stats
)

# Create sample OHLC data
df = pd.DataFrame({
    'High': [...],
    'Low': [...],
    'Close': [...]
})

# Detect pure price action liquidity sweeps (long-term fractal depth)
df = pure_price_action_liquidity_sweeps(df, term="long")
print(df[['ppa_sweep_bullish', 'ppa_sweep_bearish', 'ppa_sweep_high', 'ppa_sweep_low']])

# Generate trading signals
# 1 = bullish sweep, -1 = bearish sweep, 0 = no sweep
df = pure_price_action_liquidity_sweep_signal(df)
bullish_sweeps = df[df['ppa_sweep_signal'] == 1]

# Get statistics
stats = get_pure_price_action_liquidity_sweep_stats(df)
print(f"Total bullish sweeps: {stats['total_bullish']}")
print(f"Total bearish sweeps: {stats['total_bearish']}")
```

The function returns:
- `ppa_sweep_bullish`: 1 when a bullish sweep is detected (sell-side liquidity grabbed below a pivot low)
- `ppa_sweep_bearish`: 1 when a bearish sweep is detected (buy-side liquidity grabbed above a pivot high)
- `ppa_sweep_high`: Price level of the swept swing high on bearish-sweep bars
- `ppa_sweep_low`: Price level of the swept swing low on bullish-sweep bars

**Trading Strategy:**
- Use the `term` parameter to match your trading timeframe (short for scalping, long for swing trading)
- Bullish sweeps at pivot lows suggest smart money accumulation—potential long entries
- Bearish sweeps at pivot highs suggest smart money distribution—potential short entries
- Higher-depth sweeps (long term) are rarer but more significant

![PURE_PRICE_ACTION_LIQUIDITY_SWEEPS](/img/indicators/pure_price_action_liquidity_sweeps.png)
