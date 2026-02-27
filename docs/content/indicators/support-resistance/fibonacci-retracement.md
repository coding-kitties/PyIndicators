---
title: "Fibonacci Retracement"
sidebar_position: 1
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** 2 bars
  (default params: 2 bars)

Only needs a high and a low to compute levels. No smoothing, no rolling window.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
Levels are computed instantly from the swing high/low of the dataset.

| Event | Lag | Detail |
| --- | --- | --- |
| Retracement levels appear | **0 bars** | Static calculation from dataset extremes; no smoothing |

:::

Fibonacci retracement levels are horizontal lines that indicate where support and resistance are likely to occur. They are based on Fibonacci numbers and are drawn between a swing high and swing low. The standard levels are 0.0 (0%), 0.236 (23.6%), 0.382 (38.2%), 0.5 (50%), 0.618 (61.8% - Golden Ratio), 0.786 (78.6%), and 1.0 (100%).

The calculation formula is:
```
Level Price = Swing High - (Swing High - Swing Low) × Fibonacci Ratio
```

```python
def fibonacci_retracement(
    data: Union[PdDataFrame, PlDataFrame],
    high_column: str = 'High',
    low_column: str = 'Low',
    levels: Optional[List[float]] = None,
    lookback_period: Optional[int] = None,
    swing_high: Optional[float] = None,
    swing_low: Optional[float] = None,
    result_prefix: str = 'fib'
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import fibonacci_retracement

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

# Calculate Fibonacci retracement for Polars DataFrame
pl_df = fibonacci_retracement(pl_df, high_column="High", low_column="Low")
pl_df.show(10)

# Calculate Fibonacci retracement for Pandas DataFrame
pd_df = fibonacci_retracement(pd_df, high_column="High", low_column="Low")
pd_df.tail(10)
```

![FIBONACCI_RETRACEMENT](/img/indicators/fibonacci_retracement.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `high_column` | `High` |
| `low_column` | `Low` |

:::

