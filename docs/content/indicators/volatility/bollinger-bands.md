---
title: "Bollinger Bands (BB)"
sidebar_position: 1
tags: [lagging]
---

:::info[Warmup Window]
**Minimum bars needed:** `period` bars
  (default params: 20 bars (period=20))

Both the SMA middle line and the standard deviation need `period` bars of data. After warmup, all three bands update in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::caution[Lagging Indicator]
The bands and middle line lag behind both price and volatility changes.

| Event | Lag | Detail |
| --- | --- | --- |
| Middle band (SMA) reacts to price reversal | **≈ `period / 2` bars** | SMA has lag ≈ period/2 |
| Bands widen/narrow after volatility change | **≈ `period / 2` bars** | Std dev computed over same rolling window |
| Price touches upper/lower band | **≈ `period / 2` bars** | Bands trail the actual volatility shift |

**Formula for custom params:** `lag ≈ period / 2`

:::

Bollinger Bands are a volatility indicator that consists of a middle band (SMA) and two outer bands (standard deviations). They help traders identify overbought and oversold conditions.

```python
def bollinger_bands(
    data: Union[PdDataFrame, PlDataFrame],
    source_column='Close',
    period=20,
    std_dev=2,
    middle_band_column_result_column='bollinger_middle',
    upper_band_column_result_column='bollinger_upper',
    lower_band_column_result_column='bollinger_lower'
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import bollinger_bands

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

# Calculate bollinger bands for Polars DataFrame
pl_df = bollinger_bands(pl_df, source_column="Close")
pl_df.show(10)

# Calculate bollinger bands for Pandas DataFrame
pd_df = bollinger_bands(pd_df, source_column="Close")
pd_df.tail(10)
```

![BOLLINGER_BANDS](/img/indicators/bollinger_bands.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `source_column` | `Close` |
| `period` | `20` |
| `std_dev` | `2` |

:::

