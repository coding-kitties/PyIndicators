---
title: "Internal & External Liquidity Zones"
sidebar_position: 17
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** `2 × external_pivot_length + 1` bars
  (default params: 21 bars (external_pivot_length=10))

Multi-timeframe pivot analysis waits for the longest pivot to confirm. After warmup, zone updates are real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
Zones appear after multi-timeframe pivot confirmation.

| Event | Lag | Detail |
| --- | --- | --- |
| Liquidity zone appears | **≈ `external_pivot_length` bars** | Multi-TF pivot analysis; delay from longest pivot confirmation |
| Zone state changes (active → swept/broken) | **0 bars** | Instant when price crosses the zone |

:::

Internal & External Liquidity Zones is a Smart Money Concept indicator that identifies internal and external liquidity zones based on multi-timeframe pivot analysis, sweep detection, and market structure (BOS / CHoCH).

![Internal & External Liquidity Zones](/img/indicators/internal_external_liquidity_zones.png)

**External Zones** are derived from longer-period pivots (`external_pivot_length`) and represent major liquidity pools. **Internal Zones** come from shorter-period pivots (`internal_pivot_length`) that reside within the external range.

Core concepts:

- **External Pivot** — a swing high/low confirmed over a longer lookback window. These define the outer liquidity range.
- **Internal Pivot** — a swing high/low confirmed over a shorter lookback window. Two modes are supported:
  - `"every_pivot"` — every internal pivot creates a zone.
  - `"equal_hl"` — only consecutive pivots within an ATR-based tolerance create a zone (equal-high/low logic).
- **Zone States** — each zone transitions through: 0 = active, 1 = swept (price touched but did not close through), 2 = broken (price closed through the zone).
- **Sweep Mode** — determines how a zone is swept / broken:
  - `"wick"` — any wick touch marks a sweep.
  - `"close"` — only a close through the zone counts as a sweep.
  - `"wick_close"` — wick touches sweep; closes break.
- **Structure (BOS / CHoCH)** — for both external and internal pivots, Break of Structure and Change of Character events are detected.

Key parameters:

- **internal_pivot_length** — lookback/look-ahead for internal pivots (default: 3).
- **external_pivot_length** — lookback/look-ahead for external pivots (default: 10).
- **internal_mode** — `"every_pivot"` or `"equal_hl"` (default: `"equal_hl"`).
- **eq_tolerance_atr** — tolerance for the equal-high/low test as a fraction of ATR (default: 0.25).
- **zone_size_atr** — half-height of each zone as a fraction of ATR (default: 0.40).
- **sweep_mode** — `"wick"`, `"close"`, or `"wick_close"` (default: `"wick"`).
- **atr_length** — period for ATR calculation (default: 14).

```python
def internal_external_liquidity_zones(
    data: Union[PdDataFrame, PlDataFrame],
    internal_pivot_length: int = 3,
    external_pivot_length: int = 10,
    internal_mode: str = "equal_hl",
    eq_tolerance_atr: float = 0.25,
    require_internal_inside: bool = True,
    reset_internal_on_external: bool = True,
    atr_length: int = 14,
    zone_size_atr: float = 0.40,
    sweep_mode: str = "wick",
    structure_lookback_external: int = 36,
    structure_lookback_internal: int = 2,
    use_closes_for_structure: bool = True,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    ...
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
import pandas as pd
from pyindicators import (
    internal_external_liquidity_zones,
    internal_external_liquidity_zones_signal,
    get_internal_external_liquidity_zones_stats
)

# Create sample OHLC data
df = pd.DataFrame({
    'High': [...],
    'Low': [...],
    'Close': [...]
})

# Detect internal and external liquidity zones
df = internal_external_liquidity_zones(
    df,
    internal_pivot_length=3,
    external_pivot_length=10,
    internal_mode="equal_hl",
    sweep_mode="wick"
)
print(df[[
    'ielz_ext_high', 'ielz_ext_low',
    'ielz_int_high', 'ielz_int_low',
    'ielz_ext_structure', 'ielz_int_structure'
]])

# Generate a combined signal from sweep results
# 1 = bullish sweep, -1 = bearish sweep, 0 = no sweep
df = internal_external_liquidity_zones_signal(df)
signals = df[df['ielz_signal'] != 0]

# Get summary statistics
stats = get_internal_external_liquidity_zones_stats(df)
print(f"External highs: {stats['total_ext_highs']}")
print(f"External lows: {stats['total_ext_lows']}")
print(f"Internal highs: {stats['total_int_highs']}")
print(f"Internal lows: {stats['total_int_lows']}")
print(f"External sweeps: {stats['total_ext_sweeps']}")
print(f"Internal sweeps: {stats['total_int_sweeps']}")
print(f"Bullish sweep ratio: {stats['bullish_sweep_ratio']}")
```

The function returns:
- `ielz_ext_high` / `ielz_ext_low`: 1 on bars where an external high/low zone is created
- `ielz_ext_high_price` / `ielz_ext_low_price`: Price level of the external pivot (NaN otherwise)
- `ielz_int_high` / `ielz_int_low`: 1 on bars where an internal high/low zone is created
- `ielz_int_high_price` / `ielz_int_low_price`: Price level of the internal pivot (NaN otherwise)
- `ielz_range_high` / `ielz_range_low`: Running external range boundaries
- `ielz_ext_sweep_bull` / `ielz_ext_sweep_bear`: 1 on bars with a bullish/bearish external sweep
- `ielz_int_sweep_bull` / `ielz_int_sweep_bear`: 1 on bars with a bullish/bearish internal sweep
- `ielz_ext_structure`: Structure label at external level (`"eBOS"`, `"eCHoCH"`, or `""`)
- `ielz_int_structure`: Structure label at internal level (`"iBOS"`, `"iCHoCH"`, or `""`)
