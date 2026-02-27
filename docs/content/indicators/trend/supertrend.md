---
title: "SuperTrend"
sidebar_position: 6
tags: [lagging]
---

:::info[Warmup Window]
**Minimum bars needed:** `atr_length` bars
  (default params: 10 bars (atr_length=10))

The ATR component needs `atr_length` bars before the first SuperTrend value is computed. After warmup, the indicator updates in real-time on every new bar.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::caution[Lagging Indicator]
Trend flips and buy/sell signals lag behind the actual price reversal.

| Event | Lag | Detail |
| --- | --- | --- |
| Trend flips bullish ↔ bearish | **≈ `atr_length / 2` bars** | ATR smoothing is the primary lag source |
| Buy / sell signal fires | **≈ `atr_length / 2` bars** | Signal fires on the bar the trend flips |

**Formula for custom params:** `lag ≈ atr_length / 2`

:::

The SuperTrend indicator uses a fixed ATR multiplier factor to create a trend-following trailing stop. When the price is above the SuperTrend line the trend is bullish; when below, bearish. Trend changes generate buy/sell signals.

```python
def supertrend(
    data: Union[PdDataFrame, PlDataFrame],
    atr_length: int = 10,
    factor: float = 3.0
) -> Union[PdDataFrame, PlDataFrame]:
```

Returns the following columns:
- `supertrend`: The SuperTrend trailing stop value
- `supertrend_trend`: Current trend (1=bullish, 0=bearish)
- `supertrend_upper`: Upper band
- `supertrend_lower`: Lower band
- `supertrend_signal`: 1=buy signal, -1=sell signal, 0=no signal

Example

```python
from investing_algorithm_framework import download

from pyindicators import supertrend

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

# Calculate SuperTrend
pd_df = supertrend(pd_df, atr_length=10, factor=3.0)
pd_df.tail(10)
```

![SUPERTREND](/img/indicators/supertrend.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `atr_length` | `10` |
| `factor` | `3.0` |

:::

