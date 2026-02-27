---
title: "Exponential Moving Average (EMA)"
sidebar_position: 3
tags: [lagging]
---

:::info[Warmup Window]
**Minimum bars needed:** `period` bars
  (default params: 200 bars (period=200))

The first valid EMA value appears once `period` bars of close data are available (seeded from the first close). After the warmup, the indicator updates in real-time on every new bar.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::caution[Lagging Indicator]
The EMA line moves with a delay behind price.

| Event | Lag | Detail |
| --- | --- | --- |
| Line reacts to a price reversal | **≈ `(period − 1) / 2` bars** | Exponential decay weights recent bars more, but effective lag is still (period−1)/2 |
| Line crosses price (trend confirmation) | **≈ `(period − 1) / 2` bars** | Crossover inherits the smoothing delay |

**Formula for custom params:** `lag ≈ (period − 1) / 2`

:::

The Exponential Moving Average (EMA) is a type of moving average that gives more weight to recent prices, making it more responsive to price changes than a Simple Moving Average (SMA). It does this by using an exponential decay where the most recent prices get exponentially more weight.

```python
def ema(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str,
    period: int,
    result_column: str = None,
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import ema

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

# Calculate EMA for Polars DataFrame
pl_df = ema(pl_df, source_column="Close", period=200, result_column="EMA_200")
pl_df.show(10)

# Calculate EMA for Pandas DataFrame
pd_df = ema(pd_df, source_column="Close", period=200, result_column="EMA_200")
pd_df.tail(10)
```

![EMA](/img/indicators/ema.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `source_column` | `Close` |
| `period` | `200` |

:::

