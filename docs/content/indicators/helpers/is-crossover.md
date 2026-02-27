---
title: "Is Crossover"
sidebar_position: 2
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** 2 bars
  (default params: 2 bars)

Single-bar comparison — works from bar 2 onward.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
Boolean check is instant.

| Event | Lag | Detail |
| --- | --- | --- |
| Crossover condition detected | **0 bars** | Single-bar comparison |

:::

The is_crossover function is used to determine if a crossover occurred in the last N data points. It returns a boolean value indicating if a crossover occurred in the last N data points. The function can be used to check for crossovers in a DataFrame that was previously calculated using the crossover function.

```python
def is_crossover(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str = None,
    second_column: str = None,
    crossover_column: str = None,
    number_of_data_points: int = None,
    strict=True,
) -> bool:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import crossover, ema

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

# Calculate EMA and crossover for Polars DataFrame
pl_df = ema(pl_df, source_column="Close", period=200, result_column="EMA_200")
pl_df = ema(pl_df, source_column="Close", period=50, result_column="EMA_50")
pl_df = crossover(
    pl_df,
    first_column="EMA_50",
    second_column="EMA_200",
    result_column="Crossover_EMA"
)

# If you want the function to calculate the crossovors in the function
if is_crossover(
    pl_df, first_column="EMA_50", second_column="EMA_200", number_of_data_points=3
):
    print("Crossover detected in Pandas DataFrame in the last 3 data points")

# If you want to use the result of a previous crossover calculation
if is_crossover(pl_df, crossover_column="Crossover_EMA", number_of_data_points=3):
    print("Crossover detected in Pandas DataFrame in the last 3 data points")

# Calculate EMA and crossover for Pandas DataFrame
pd_df = ema(pd_df, source_column="Close", period=200, result_column="EMA_200")
pd_df = ema(pd_df, source_column="Close", period=50, result_column="EMA_50")
pd_df = crossover(
    pd_df,
    first_column="EMA_50",
    second_column="EMA_200",
    result_column="Crossover_EMA"
)

# If you want the function to calculate the crossovors in the function
if is_crossover(
    pd_df, first_column="EMA_50", second_column="EMA_200", number_of_data_points=3
):
    print("Crossover detected in Pandas DataFrame in the last 3 data points")

# If you want to use the result of a previous crossover calculation
if is_crossover(pd_df, crossover_column="Crossover_EMA", number_of_data_points=3):
    print("Crossover detected in Pandas DataFrame in the last 3 data points")
```
