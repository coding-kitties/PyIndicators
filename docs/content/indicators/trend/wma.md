---
title: "Weighted Moving Average (WMA)"
sidebar_position: 1
tags: [lagging]
---

:::info[Warmup Window]
**Minimum bars needed:** `period` bars
  (default params: 200 bars (period=200))

The first valid WMA value appears once `period` bars of close data are available. After the warmup, the indicator updates in real-time on every new bar.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::caution[Lagging Indicator]
The WMA line moves with a delay behind price.

| Event | Lag | Detail |
| --- | --- | --- |
| Line reacts to a price reversal | **≈ `(period − 1) / 3` bars** | Center of gravity of the linear-weight window sits at ⅓ of the period |
| Line crosses price (trend confirmation) | **≈ `(period − 1) / 3` bars** | Crossover inherits the same smoothing delay |

**Formula for custom params:** `lag ≈ (period − 1) / 3`

:::

A Weighted Moving Average (WMA) is a type of moving average that assigns greater importance to recent data points compared to older ones. This makes it more responsive to recent price changes compared to a Simple Moving Average (SMA), which treats all data points equally. The WMA does this by using linear weighting, where the most recent prices get the highest weight, and weights decrease linearly for older data points.

```python
def wma(
    data: Union[PandasDataFrame, PolarsDataFrame],
    source_column: str,
    period: int,
    result_column: Optional[str] = None
) -> Union[PandasDataFrame, PolarsDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import wma

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

# Calculate SMA for Polars DataFrame
pl_df = wma(pl_df, source_column="Close", period=200, result_column="WMA_200")
pl_df.show(10)

# Calculate SMA for Pandas DataFrame
pd_df = wma(pd_df, source_column="Close", period=200, result_column="WMA_200")
pd_df.tail(10)
```

![WMA](/img/indicators/wma.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `source_column` | `Close` |
| `period` | `200` |

:::

