---
title: "Pulse Mean Accelerator (PMA)"
sidebar_position: 8
tags: [lagging]
---

:::info[Warmup Window]
**Minimum bars needed:** `max(ma_length, accel_lookback)` bars
  (default params: 32 bars (ma_length=20, accel_lookback=32))

Both the base moving average (`ma_length`) and the acceleration lookback need to fill before the first PMA value is valid. After warmup, the indicator updates in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::caution[Lagging Indicator]
The PMA line and trend signals lag behind price reversals.

| Event | Lag | Detail |
| --- | --- | --- |
| MA base line reacts to price reversal | **≈ `ma_length / 2` bars** | RMA has effective lag ≈ ma_length/2 |
| PMA line changes direction | **≈ `ma_length / 2` to `accel_lookback / 2` bars** | Acceleration lookback modulates the offset, adding a variable lag component |
| Trend flips bullish ↔ bearish | **≈ `ma_length / 2` to `accel_lookback / 2` bars** | Trend derived from PMA slope vs MA |

**Formula for custom params:** `lag ≈ ma_length / 2  (+ accel influence)`

:::

The Pulse Mean Accelerator is a trend-following overlay indicator
translated from the Pine Script® by MisinkoMaster.  It adds a
volatility- and momentum-scaled acceleration offset to a base moving
average.  The acceleration accumulates over a configurable lookback:
bars where source momentum exceeds MA momentum push the PMA further
from the MA, while bars where the MA leads source momentum pull it
back.  Multiple MA types (RMA, SMA, EMA, WMA, DEMA, TEMA, HMA),
volatility measures (ATR, Standard Deviation, MAD), and smoothing
modes are supported.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source_column` | str | `"Close"` | Source price column |
| `ma_type` | str | `"RMA"` | MA type: RMA, SMA, EMA, WMA, DEMA, TEMA, HMA |
| `ma_length` | int | `20` | Lookback for the base moving average |
| `accel_lookback` | int | `32` | Bars over which acceleration is accumulated |
| `max_accel` | float | `0.2` | Maximum absolute acceleration factor |
| `volatility_type` | str | `"Standard Deviation"` | Volatility: ATR, Standard Deviation, MAD |
| `smooth_type` | str | `"Double Moving Average"` | Smoothing: NONE, Exponential, Extra Moving Average, Double Moving Average |
| `use_confirmation` | bool | `True` | Require combined PMA+MA momentum to confirm trend flips |

**Output columns:** `pma`, `pma_ma`, `pma_trend`, `pma_long`, `pma_short`, `pma_acceleration`

```python
import pandas as pd
from pyindicators import (
    pulse_mean_accelerator,
    pulse_mean_accelerator_signal,
    get_pulse_mean_accelerator_stats,
)

# --- With pandas ---
df = pd.read_csv("data.csv")
df = pulse_mean_accelerator(
    df,
    ma_type="RMA",
    ma_length=20,
    accel_lookback=32,
    max_accel=0.2,
    volatility_type="Standard Deviation",
    smooth_type="Double Moving Average",
    use_confirmation=True,
)
df = pulse_mean_accelerator_signal(df)
stats = get_pulse_mean_accelerator_stats(df)
print(stats)
df[["Close", "pma", "pma_ma", "pma_trend", "pma_long", "pma_short"]].tail(10)
```

![PULSE_MEAN_ACCELERATOR](/img/indicators/pulse_mean_accelerator.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `ma_type` | `RMA` |
| `ma_length` | `20` |
| `accel_lookback` | `32` |
| `max_accel` | `0.2` |
| `volatility_type` | `Standard Deviation` |
| `smooth_type` | `Double Moving Average` |
| `use_confirmation` | `True` |

:::

