---
title: "Opening Gap (OG)"
sidebar_position: 23
tags: [real-time]
---

:::info[Real-time Indicator]
Opening Gaps are detected on the current bar by comparing it to two bars earlier.

| Event | Lag | Detail |
| --- | --- | --- |
| Detection | **0 bars** | Detected as soon as the current bar prints |

:::

Opening Gap detects three-bar imbalance zones modelled after the *OG* mode in the RUDYINDICATOR Pine Script. It is a variant of a Fair Value Gap but uses a different comparison methodology:

**How it works:**
1. **Bullish OG:** `Low[i] > High[i-2]` — the current bar's low is entirely above the bar-two-back's high, creating an upside gap
2. **Bearish OG:** `High[i] < Low[i-2]` — the current bar's high is entirely below the bar-two-back's low, creating a downside gap
3. Zone boundaries capture the price gap between the two non-adjacent bars

```python
def opening_gap(
    data: Union[PdDataFrame, PlDataFrame],
    open_column: str = "Open",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    bullish_og_column: str = "bullish_og",
    bearish_og_column: str = "bearish_og",
    bullish_og_top_column: str = "bullish_og_top",
    bullish_og_bottom_column: str = "bullish_og_bottom",
    bearish_og_top_column: str = "bearish_og_top",
    bearish_og_bottom_column: str = "bearish_og_bottom",
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from pyindicators import (
    opening_gap,
    opening_gap_signal,
    get_opening_gap_stats,
)

# Detect Opening Gaps
df = opening_gap(df)

# Generate signal
df = opening_gap_signal(df)

# Get statistics
stats = get_opening_gap_stats(df)
print(f"Bullish OGs: {stats['total_bullish']}")
print(f"Bearish OGs: {stats['total_bearish']}")
```

**Output Columns:**
| Column | Description |
| --- | --- |
| `bullish_og` | 1 on bars with a bullish Opening Gap, else 0 |
| `bearish_og` | 1 on bars with a bearish Opening Gap, else 0 |
| `bullish_og_top` | Top price of the bullish OG zone (NaN otherwise) |
| `bullish_og_bottom` | Bottom price of the bullish OG zone (NaN otherwise) |
| `bearish_og_top` | Top price of the bearish OG zone (NaN otherwise) |
| `bearish_og_bottom` | Bottom price of the bearish OG zone (NaN otherwise) |

![Opening Gap](/img/indicators/opening_gap.png)
