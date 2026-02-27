---
title: "Golden Zone Signal"
sidebar_position: 3
tags: [real-time]
---

:::info[Warmup Window]
**Minimum bars needed:** Same as Golden Zone (`length` bars)
  (default params: 60 bars (length=60))

Requires the Golden Zone to be computed first. Once zones are available, signals fire in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::tip[Real-time Indicator]
Signals fire instantly when price enters or exits a pre-computed golden zone.

| Event | Lag | Detail |
| --- | --- | --- |
| Signal fires when price enters zone | **0 bars** | Simple comparison of current close vs pre-computed zone levels |

:::

The Golden Zone Signal function generates trading signals based on whether the price is within the Golden Zone. It returns a signal value of 1 when the close price is between the upper (50%) and lower (61.8%) boundaries of the Golden Zone, and 0 when the price is outside the zone.

This can be used to identify potential support/resistance areas and generate trading signals when price enters or exits the Golden Zone.

> !Important: This function requires the Golden Zone columns to be present in the DataFrame. You must call the `golden_zone()` function first before using `golden_zone_signal()`.

Signal values:
- **1**: Price is within the Golden Zone (potential support/resistance area)
- **0**: Price is outside the Golden Zone

```python
def golden_zone_signal(
    data: Union[PdDataFrame, PlDataFrame],
    close_column: str = 'Close',
    upper_column: str = 'golden_zone_upper',
    lower_column: str = 'golden_zone_lower',
    signal_column: str = 'golden_zone_signal'
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import golden_zone, golden_zone_signal

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

# First calculate Golden Zone, then the signal for Polars DataFrame
pl_df = golden_zone(pl_df, high_column="High", low_column="Low", length=60)
pl_df = golden_zone_signal(pl_df)
pl_df.show(10)

# First calculate Golden Zone, then the signal for Pandas DataFrame
pd_df = golden_zone(pd_df, high_column="High", low_column="Low", length=60)
pd_df = golden_zone_signal(pd_df)
pd_df.tail(10)
```

![GOLDEN_ZONE_SIGNAL](/img/indicators/golden_zone_signal.png)
