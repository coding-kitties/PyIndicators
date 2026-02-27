---
title: "Wilders Relative Strength Index (Wilders RSI)"
sidebar_position: 3
tags: [lagging]
---

:::info[Warmup Window]
**Minimum bars needed:** `period` bars
  (default params: 14 bars (period=14))

Like standard RSI, the initial average gain/loss needs `period` bars. However, the Wilder's RMA smoothing (alpha=1/period) means the effective lag is ~2× period. After warmup, the indicator updates in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::caution[Lagging Indicator]
Wilder's smoothing makes this RSI variant significantly slower than standard RSI.

| Event | Lag | Detail |
| --- | --- | --- |
| RSI reaches overbought / oversold | **≈ `2 × period` bars** | Wilder's RMA (alpha=1/period) is equivalent to EMA(2×period−1) |
| RSI crosses 50 (trend confirmation) | **≈ `2 × period` bars** | Same double-period effective lag |

**Formula for custom params:** `lag ≈ 2 × period`

:::

The Wilders Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. It moves between 0 and 100 and is used to identify overbought or oversold conditions in a market. The Wilders RSI uses a different calculation method than the standard RSI.

```python
def wilders_rsi(
    data: Union[pd.DataFrame, pl.DataFrame],
    source_column: str,
    period: int = 14,
    result_column: str = None,
) -> Union[pd.DataFrame, pl.DataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import wilders_rsi

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

# Calculate Wilders RSI for Polars DataFrame
pl_df = wilders_rsi(pl_df, source_column="Close", period=14, result_column="RSI_14")
pl_df.show(10)

# Calculate Wilders RSI for Pandas DataFrame
pd_df = wilders_rsi(pd_df, source_column="Close", period=14, result_column="RSI_14")
pd_df.tail(10)
```

![wilders_RSI](/img/indicators/wilders_rsi.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `source_column` | `Close` |
| `period` | `14` |

:::

