---
title: "Is Downtrend"
sidebar_position: 5
tags: [lagging]
---

:::info[Warmup Window]
**Minimum bars needed:** `slow_ema_period` bars
  (default params: 200 bars (slow_ema_period=200))

The slow EMA needs `slow_ema_period` bars to initialize. The fast EMA fills much sooner. After warmup, the trend check updates in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::caution[Lagging Indicator]
Uses EMA death cross which has very high lag.

| Event | Lag | Detail |
| --- | --- | --- |
| Downtrend detected (fast EMA < slow EMA) | **≈ `(slow_ema_period − 1) / 2` bars** | Dominated by the slow EMA's smoothing lag |

**Formula for custom params:** `lag ≈ (slow_ema_period − 1) / 2`

:::

The is_downtrend function is used to determine if a downtrend occurred in the last N data points. It returns a boolean value indicating if a downtrend occurred in the last N data points. The function can be used to check for downtrends in a DataFrame that was previously calculated using the crossover function.

```python
def is_down_trend(
    data: Union[PdDataFrame, PlDataFrame],
    use_death_cross: bool = True,
) -> bool:
```

Example

```python
from investing_algorithm_framework import CSVOHLCVMarketDataSource

from pyindicators import is_down_trend

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

print(is_down_trend(pl_df))
print(is_down_trend(pd_df))
```
