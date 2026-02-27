---
title: "Is Uptrend"
sidebar_position: 6
tags: [lagging]
---

:::info[Warmup Window]
**Minimum bars needed:** `slow_ema_period` bars
  (default params: 200 bars (slow_ema_period=200))

The slow EMA needs `slow_ema_period` bars to initialize. The fast EMA fills much sooner. After warmup, the trend check updates in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::caution[Lagging Indicator]
Uses EMA golden cross which has very high lag.

| Event | Lag | Detail |
| --- | --- | --- |
| Uptrend detected (fast EMA > slow EMA) | **≈ `(slow_ema_period − 1) / 2` bars** | Dominated by the slow EMA's smoothing lag |

**Formula for custom params:** `lag ≈ (slow_ema_period − 1) / 2`

:::

The is_up_trend function is used to determine if an uptrend occurred in the last N data points. It returns a boolean value indicating if an uptrend occurred in the last N data points. The function can be used to check for uptrends in a DataFrame that was previously calculated using the crossover function.

```python
def is_up_trend(
    data: Union[PdDataFrame, PlDataFrame],
    use_golden_cross: bool = True,
) -> bool:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import is_up_trend

pl_df = download(
    symbol="btc/eur",
    market="binance",
    time_frame="1d",
    start_date="2023-12-01",
    end_date="2023-12-25",
    save=True,
    storage_path="./data"
)
pd_df = download(
    symbol="btc/eur",
    market="binance",
    time_frame="1d",
    start_date="2023-12-01",
    end_date="2023-12-25",
    pandas=True,
    save=True,
    storage_path="./data"
)

print(is_up_trend(pl_df))
print(is_up_trend(pd_df))
```
