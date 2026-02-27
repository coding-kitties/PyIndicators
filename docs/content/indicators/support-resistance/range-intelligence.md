---
title: "Range Intelligence Suite"
sidebar_position: 25
tags: [real-time]
---

:::info[Real-time Indicator]
Consolidation detection triggers as soon as rolling range width falls below the ATR-based threshold.

| Event | Lag | Detail |
| --- | --- | --- |
| Range detection | **≈ length bars** | Requires `length` bars of history to compute ATR and rolling extremes (default 20) |
| Volume profile | **0 bars** | Built incrementally from the bars inside the detected range |
| Sweep detection | **0 bars** | Checked in real-time as price pierces and reverts from range boundaries |
| Breakout | **0 bars** | Confirmed on the bar where close crosses range high/low |

**Formula for custom params:** `lag ≈ length` (ATR / range-width warm-up)

:::

Range Intelligence Suite detects periods of price consolidation by comparing the rolling range width to the Average True Range (ATR), then enriches each detected range with a volume profile, Point of Control (POC), net delta, liquidity sweep counts, and a breakout-readiness score.

**How it works:**
1. Compute ATR with EMA-style smoothing over `length` bars
2. Calculate rolling highest high and lowest low over `length` bars
3. A consolidation range starts when `range_width < sensitivity × ATR`
4. Distribute volume across `vp_rows` horizontal bins within the range (buy vs sell)
5. Track the Point of Control (POC) — the bin with the highest volume
6. Accumulate net delta (buy volume − sell volume) for state classification
7. Detect liquidity sweeps — price briefly pierces a boundary but closes back inside
8. Compute a ready score (0–100) based on duration and delta imbalance
9. Breakout confirmed when close crosses above range high (+1) or below range low (−1)
10. Optionally hide older ranges that overlap with newly detected ones

```python
def range_intelligence(
    data: Union[PdDataFrame, PlDataFrame],
    length: int = 20,
    sensitivity: float = 4.0,
    vp_rows: int = 10,
    hide_overlaps: bool = True,
    open_column: str = "Open",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    volume_column: str = "Volume",
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from pyindicators import (
    range_intelligence,
    range_intelligence_signal,
    get_range_intelligence_stats,
)

# Calculate Range Intelligence
df = range_intelligence(df, length=20, sensitivity=4.0, vp_rows=10)

# Extract breakout signal
df = range_intelligence_signal(df)

# Get statistics
stats = get_range_intelligence_stats(df)
print(f"Total ranges: {stats['total_ranges']}")
print(f"Bullish breakouts: {stats['bullish_breakouts']}")
print(f"Bearish breakouts: {stats['bearish_breakouts']}")
print(f"Avg ready score: {stats['avg_ready_score']}")
```

**Output Columns:**
- `ri_active`: 1 while inside a consolidation range, else 0
- `ri_high` / `ri_low`: Upper and lower boundary of the active range
- `ri_mid`: Midpoint of the range
- `ri_poc`: Point of Control — price level with the highest volume
- `ri_delta`: Cumulative net delta (buy − sell volume) within the range
- `ri_state`: "Accumulation" (positive delta) or "Distribution" (negative delta)
- `ri_ready`: Ready score 0–100 — gauge of breakout imminence
- `ri_sweep_high`: 1 when a high-side liquidity sweep occurs
- `ri_sweep_low`: 1 when a low-side liquidity sweep occurs
- `ri_breakout`: +1 bullish breakout, −1 bearish breakout, 0 neutral
- `ri_duration`: Bars since the current range started
- `ri_signal`: Trading signal from `range_intelligence_signal` (+1/−1/0)

**Parameters:**

| Parameter | Default | Description |
| --- | --- | --- |
| `length` | 20 | Lookback period for ATR and rolling extremes |
| `sensitivity` | 4.0 | ATR multiplier threshold — lower values detect tighter ranges |
| `vp_rows` | 10 | Number of horizontal volume-profile bins |
| `hide_overlaps` | True | Discard earlier ranges that overlap with a newly detected range |

**Range State:**

| State | Meaning |
| --- | --- |
| Accumulation | Net delta is positive — buying pressure dominates |
| Distribution | Net delta is negative — selling pressure dominates |

**Signal Values (from `range_intelligence_signal`):**
- `1`: Bullish breakout — close above range high
- `0`: Neutral — no breakout
- `-1`: Bearish breakout — close below range low

**Ready Score:**

The ready score (0–100) combines two components:
- **Duration component** (50%): `(duration / length) × 50` — longer consolidation increases readiness
- **Delta imbalance component** (50%): `(|net_delta| / recent_volume) × 50` — stronger imbalance suggests directional intent

Higher values indicate the range is more likely to produce an imminent breakout.

**Statistics (from `get_range_intelligence_stats`):**

| Key | Description |
| --- | --- |
| `total_ranges` | Total number of completed ranges |
| `bullish_breakouts` | Count of bullish breakouts |
| `bearish_breakouts` | Count of bearish breakouts |
| `accumulation_ranges` | Ranges ending in accumulation state |
| `distribution_ranges` | Ranges ending in distribution state |
| `total_sweep_highs` | Total high-side liquidity sweeps |
| `total_sweep_lows` | Total low-side liquidity sweeps |
| `avg_ready_score` | Average ready score at breakout |
| `avg_duration` | Average range duration in bars |

![RANGE_INTELLIGENCE](/img/indicators/range_intelligence.png)
