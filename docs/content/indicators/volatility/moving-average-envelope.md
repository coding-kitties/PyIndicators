---
title: "Moving Average Envelope (MAE)"
sidebar_position: 4
tags: [lagging]
---

:::info[Warmup Window]
**Minimum bars needed:** `period` bars
  (default params: 20 bars (period=20))

The base moving average needs `period` bars. Bands are a fixed % offset, so they have the same warmup. After warmup, all bands update in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::caution[Lagging Indicator]
The envelope bands lag behind price because they are offset from a moving average.

| Event | Lag | Detail |
| --- | --- | --- |
| Middle line reacts to price reversal | **≈ `period / 2` bars** | Base MA has lag ≈ period/2 |
| Bands shift after a price move | **≈ `period / 2` bars** | Bands are fixed % offset from the MA; they move in lockstep with the MA |

**Formula for custom params:** `lag ≈ period / 2`

:::

Moving Average Envelopes are percentage-based envelopes set above and below a moving average. The moving average forms the base, and the envelopes are set at a fixed percentage above and below. This indicator is useful for identifying overbought/oversold conditions, spotting trend direction, and finding support and resistance levels.

```python
def moving_average_envelope(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str = 'Close',
    period: int = 20,
    percentage: float = 2.5,
    ma_type: str = 'sma',
    middle_column: str = 'ma_envelope_middle',
    upper_column: str = 'ma_envelope_upper',
    lower_column: str = 'ma_envelope_lower'
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import moving_average_envelope

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

# Calculate Moving Average Envelope for Polars DataFrame
pl_df = moving_average_envelope(pl_df, source_column="Close", period=20, percentage=2.5)
pl_df.show(10)

# Calculate Moving Average Envelope for Pandas DataFrame
pd_df = moving_average_envelope(pd_df, source_column="Close", period=20, percentage=2.5)
pd_df.tail(10)
```

![MOVING_AVERAGE_ENVELOPE](/img/indicators/moving_average_envelope.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `source_column` | `Close` |
| `period` | `20` |
| `percentage` | `2.5` |

:::

