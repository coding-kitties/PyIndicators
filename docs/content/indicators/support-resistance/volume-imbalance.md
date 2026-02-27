---
title: "Volume Imbalance (VI)"
sidebar_position: 22
tags: [real-time]
---

:::info[Real-time Indicator]
Volume Imbalances are detected on the current bar by comparing it to the previous bar.

| Event | Lag | Detail |
| --- | --- | --- |
| Detection | **0 bars** | Detected as soon as the current bar's body creates a gap with the previous bar's body |

:::

Volume Imbalance detects gaps between consecutive candle **bodies** (open/close range) where no trading overlap occurred. Unlike a Fair Value Gap which spans three bars, a Volume Imbalance only needs two adjacent bars — making it a faster, more granular signal.

**How it works:**
1. Calculate body edges for each bar: `body_hi = max(Open, Close)`, `body_lo = min(Open, Close)`
2. **Bullish VI:** Previous bar's body top is below current bar's body bottom, previous bar's high is below current body bottom, and the current bar moves aggressively upward
3. **Bearish VI:** Current bar's body top is below previous bar's body bottom, previous bar's low is above current body top, and the current bar moves aggressively downward
4. Zone boundaries capture the gap between the two bodies

```python
def volume_imbalance(
    data: Union[PdDataFrame, PlDataFrame],
    open_column: str = "Open",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    bullish_vi_column: str = "bullish_vi",
    bearish_vi_column: str = "bearish_vi",
    bullish_vi_top_column: str = "bullish_vi_top",
    bullish_vi_bottom_column: str = "bullish_vi_bottom",
    bearish_vi_top_column: str = "bearish_vi_top",
    bearish_vi_bottom_column: str = "bearish_vi_bottom",
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from pyindicators import (
    volume_imbalance,
    volume_imbalance_signal,
    get_volume_imbalance_stats,
)

# Detect Volume Imbalances
df = volume_imbalance(df)

# Generate signal
df = volume_imbalance_signal(df)

# Get statistics
stats = get_volume_imbalance_stats(df)
print(f"Bullish VIs: {stats['total_bullish']}")
print(f"Bearish VIs: {stats['total_bearish']}")
```

**Output Columns:**
| Column | Description |
| --- | --- |
| `bullish_vi` | 1 on bars with a bullish Volume Imbalance, else 0 |
| `bearish_vi` | 1 on bars with a bearish Volume Imbalance, else 0 |
| `bullish_vi_top` | Top price of the bullish VI zone (NaN otherwise) |
| `bullish_vi_bottom` | Bottom price of the bullish VI zone (NaN otherwise) |
| `bearish_vi_top` | Top price of the bearish VI zone (NaN otherwise) |
| `bearish_vi_bottom` | Bottom price of the bearish VI zone (NaN otherwise) |

**Comparison with FVG:**

| Feature | Fair Value Gap (FVG) | Volume Imbalance (VI) |
| --- | --- | --- |
| Bars span | 3 bars | 2 bars |
| Measured from | Wick (high/low) | Body (open/close) |
| Sensitivity | Lower — requires larger gap | Higher — catches smaller gaps |

![Volume Imbalance](/img/indicators/volume_imbalance.png)
