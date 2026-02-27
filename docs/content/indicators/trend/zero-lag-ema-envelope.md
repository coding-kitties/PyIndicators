---
title: "Zero-Lag EMA Envelope (ZLEMA)"
sidebar_position: 4
tags: [lagging]
---

:::info[Warmup Window]
**Minimum bars needed:** `length` bars
  (default params: 200 bars (length=200))

The ZLEMA center line needs `length` bars to initialize. The ATR bands additionally need `atr_length` bars. After warmup, both update in real-time.

✅ **After the warmup window is filled, this indicator produces a new value on every incoming bar in real-time.**

:::

:::caution[Lagging Indicator]
The center line has near-zero lag; the ATR-based bands still lag.

| Event | Lag | Detail |
| --- | --- | --- |
| Center line reacts to price reversal | **≈ 0 bars** | ZLEMA compensates EMA lag via close + (close − close[lag]) |
| Upper/lower bands react to volatility change | **≈ `atr_length / 2` bars** | Bands are offset by ATR; ATR smoothing introduces lag of atr_length / 2 |

**Formula for custom params:** `center ≈ 0; bands ≈ atr_length / 2`

:::

The Zero-Lag EMA Envelope combines a Zero-Lag Exponential Moving Average (ZLEMA) with ATR-based bands and multi-bar swing confirmation. The ZLEMA compensates for the inherent lag of a standard EMA by using a lag-compensated source (`close + (close - close[lag])`). Trend state is confirmed when multiple consecutive bars close beyond a band while the ZLEMA slope agrees.

Calculation:
- `lag = floor((length - 1) / 2)`
- `compensated = close + (close - close[lag])`
- `ZLEMA = EMA(compensated, length)`
- `Upper = ZLEMA + ATR × mult`
- `Lower = ZLEMA - ATR × mult`
- Bull: close > Upper for N bars AND ZLEMA rising
- Bear: close < Lower for N bars AND ZLEMA falling

```python
def zero_lag_ema_envelope(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str = 'Close',
    length: int = 200,
    mult: float = 2.0,
    atr_length: int = 21,
    confirm_bars: int = 2,
    upper_column: str = 'zlema_upper',
    lower_column: str = 'zlema_lower',
    middle_column: str = 'zlema_middle',
    trend_column: str = 'zlema_trend',
    signal_column: str = 'zlema_signal',
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import zero_lag_ema_envelope

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

# Calculate Zero-Lag EMA Envelope for Polars DataFrame
pl_df = zero_lag_ema_envelope(pl_df, source_column="Close", length=200, mult=2.0)
pl_df.show(10)

# Calculate Zero-Lag EMA Envelope for Pandas DataFrame
pd_df = zero_lag_ema_envelope(pd_df, source_column="Close", length=200, mult=2.0)
pd_df.tail(10)
```

![ZERO_LAG_EMA_ENVELOPE](/img/indicators/zero_lag_ema_envelope.png)
:::info[Chart Parameters]
The image above uses the following parameters:

| Parameter | Value |
| --- | --- |
| `source_column` | `Close` |
| `length` | `200` |
| `mult` | `2.0` |

:::

