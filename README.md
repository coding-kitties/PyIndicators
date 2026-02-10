# PyIndicators

PyIndicators is a powerful and user-friendly Python library for financial technical analysis indicators, metrics and helper functions. Written entirely in Python, it requires no external dependencies, ensuring seamless integration and ease of use.

## Sponsors

<a href="https://www.finterion.com/" target="_blank">
    <picture style="height: 30px;">
    <source media="(prefers-color-scheme: dark)" srcset="static/sponsors/finterion-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="static/sponsors/finterion-light.png">
    <img src="static/sponsors/finterion-light.svg" alt="Finterion Logo" width="200px" height="50px">
    </picture>
</a>

## Installation

PyIndicators can be installed using pip:

```bash
pip install pyindicators
```

## Features

* Native Python implementation, no external dependencies needed except for Polars or Pandas
* Dataframe first approach, with support for both pandas dataframes and polars dataframes
* Supports python version 3.10 and above.
* [Trend indicators](#trend-indicators)
  * [Weighted Moving Average (WMA)](#weighted-moving-average-wma)
  * [Simple Moving Average (SMA)](#simple-moving-average-sma)
  * [Exponential Moving Average (EMA)](#exponential-moving-average-ema)
  * [SuperTrend Clustering](#supertrend-clustering)
  * [SuperTrend Basic](#supertrend-basic)
* [Momentum and Oscillators](#momentum-and-oscillators)
  * [Moving Average Convergence Divergence (MACD)](#moving-average-convergence-divergence-macd)
  * [Relative Strength Index (RSI)](#relative-strength-index-rsi)
  * [Relative Strength Index Wilders method (Wilders RSI)](#wilders-relative-strength-index-wilders-rsi)
  * [Williams %R](#williams-r)
  * [Average Directional Index (ADX)](#average-directional-index-adx)
  * [Stochastic Oscillator (STO)](#stochastic-oscillator-sto)
  * [Momentum Confluence](#momentum-confluence)
* [Volatility indicators](#volatility-indicators)
  * [Bollinger Bands (BB)](#bollinger-bands-bb)
  * [Bollinger Bands Overshoot](#bollinger-bands-overshoot)
  * [Average True Range (ATR)](#average-true-range-atr)
  * [Moving Average Envelope (MAE)](#moving-average-envelope-mae)
  * [Nadaraya-Watson Envelope (NWE)](#nadaraya-watson-envelope-nwe)
* [Support and Resistance](#support-and-resistance)
  * [Fibonacci Retracement](#fibonacci-retracement)
  * [Golden Zone](#golden-zone)
  * [Golden Zone Signal](#golden-zone-signal)
  * [Fair Value Gap (FVG)](#fair-value-gap-fvg)
  * [Order Blocks](#order-blocks)
  * [Market Structure Break](#market-structure-break)
  * [Market Structure CHoCH/BOS](#market-structure-chochbos)
* [Pattern recognition](#pattern-recognition)
  * [Detect Peaks](#detect-peaks)
  * [Detect Bullish Divergence](#detect-bullish-divergence)
  * [Detect Bearish Divergence](#detect-bearish-divergence)
* [Indicator helpers](#indicator-helpers)
  * [Crossover](#crossover)
  * [Is Crossover](#is-crossover)
  * [Crossunder](#crossunder)
  * [Is Crossunder](#is-crossunder)
  * [Is Downtrend](#is-downtrend)
  * [Is Uptrend](#is-uptrend)
  * [has_any_lower_then_threshold](#has_any_lower_then_threshold)

## Indicators

### Trend Indicators

Indicators that help to determine the direction of the market (uptrend, downtrend, or sideways) and confirm if a trend is in place.

#### Weighted Moving Average (WMA)

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

![WMA](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/wma.png)

#### Simple Moving Average (SMA)

A Simple Moving Average (SMA) is the average of the last N data points, recalculated as new data comes in. Unlike the Weighted Moving Average (WMA), SMA treats all values equally, giving them the same weight.

```python
def sma(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str,
    period: int,
    result_column: str = None,
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import sma

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
pl_df = sma(pl_df, source_column="Close", period=200, result_column="SMA_200")
pl_df.show(10)

# Calculate SMA for Pandas DataFrame
pd_df = sma(pd_df, source_column="Close", period=200, result_column="SMA_200")
pd_df.tail(10)
```

![SMA](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/sma.png)

#### Exponential Moving Average (EMA)

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

![EMA](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/ema.png)

### Momentum and Oscillators

Indicators that measure the strength and speed of price movements rather than the direction.

#### Moving Average Convergence Divergence (MACD)

The Moving Average Convergence Divergence (MACD) is used to identify trend direction, strength, and potential reversals. It is based on the relationship between two Exponential Moving Averages (EMAs) and includes a histogram to visualize momentum.

```python
def macd(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str,
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
    macd_column: str = "macd",
    signal_column: str = "macd_signal",
    histogram_column: str = "macd_histogram"
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import macd

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

# Calculate MACD for Polars DataFrame
pl_df = macd(pl_df, source_column="Close", short_period=12, long_period=26, signal_period=9)

# Calculate MACD for Pandas DataFrame
pd_df = macd(pd_df, source_column="Close", short_period=12, long_period=26, signal_period=9)

pl_df.show(10)
pd_df.tail(10)
```

![MACD](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/macd.png)

#### Relative Strength Index (RSI)

The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. It moves between 0 and 100 and is used to identify overbought or oversold conditions in a market.

```python
def rsi(
    data: Union[pd.DataFrame, pl.DataFrame],
    source_column: str,
    period: int = 14,
    result_column: str = None,
) -> Union[pd.DataFrame, pl.DataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import rsi

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

# Calculate RSI for Polars DataFrame
pl_df = rsi(pl_df, source_column="Close", period=14, result_column="RSI_14")
pl_df.show(10)

# Calculate RSI for Pandas DataFrame
pd_df = rsi(pd_df, source_column="Close", period=14, result_column="RSI_14")
pd_df.tail(10)
```

![RSI](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/rsi.png)

#### Wilders Relative Strength Index (Wilders RSI)

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

![wilders_RSI](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/wilders_rsi.png)

#### Williams %R

Williams %R (Williams Percent Range) is a momentum indicator used in technical analysis to measure overbought and oversold conditions in a market. It moves between 0 and -100 and helps traders identify potential reversal points.

```python
def willr(
    data: Union[pd.DataFrame, pl.DataFrame],
    period: int = 14,
    result_column: str = None,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close"
) -> Union[pd.DataFrame, pl.DataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import willr

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

pl_df = data_source.get_data()
pd_df = data_source.get_data(pandas=True)

# Calculate Williams%R for Polars DataFrame
pl_df = willr(pl_df, result_column="WILLR")
pl_df.show(10)

# Calculate Williams%R for Pandas DataFrame
pd_df = willr(pd_df, result_column="WILLR")
pd_df.tail(10)
```

![williams %R](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/willr.png)

#### Average Directional Index (ADX)

The Average Directional Index (ADX) is a trend strength indicator that helps traders identify the strength of a trend, regardless of its direction. It is derived from the Positive Directional Indicator (+DI) and Negative Directional Indicator (-DI) and moves between 0 and 100.

```python
def adx(
    data: Union[PdDataFrame, PlDataFrame],
    period=14,
    adx_result_column="ADX",
    di_plus_result_column="+DI",
    di_minus_result_column="-DI",
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import adx

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

# Calculate ADX for Polars DataFrame
pl_df = adx(pl_df)
pl_df.show(10)

# Calculate ADX for Pandas DataFrame
pd_df = adx(pd_df)
pd_df.tail(10)
```

![ADX](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/adx.png)

#### Stochastic Oscillator (STO)
The Stochastic Oscillator (STO) is a momentum indicator that compares a particular closing price of an asset to a range of its prices over a certain period. It is used to identify overbought or oversold conditions in a market. The STO consists of two lines: %K and %D, where %K is the main line and %D is the signal line.

```python
def stochastic_oscillator(
    data: Union[pd.DataFrame, pl.DataFrame],
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    k_period: int = 14,
    k_slowing: int = 3,
    d_period: int = 3,
    result_column: Optional[str] = None
) -> Union[pd.DataFrame, pl.DataFrame]:
```

Example

```python
from investing_algorithm_framework import download
from pyindicators import stochastic_oscillator
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
# Calculate Stochastic Oscillator for Polars DataFrame
pl_df = stochastic_oscillator(pl_df, high_column="High", low_column="Low", close_column="Close", k_period=14, k_slowing=3, d_period=3, result_column="STO")
pl_df.show(10)
# Calculate Stochastic Oscillator for Pandas DataFrame
pd_df = stochastic_oscillator(pd_df, high_column="High", low_column="Low", close_column="Close", k_period=14, k_slowing=3, d_period=3, result_column="STO")
pd_df.tail(10)
```

![STO](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/sto.png)

#### Momentum Confluence

Momentum Confluence is a comprehensive multi-component oscillator that combines multiple technical analysis components to provide a powerful trend following and reversal detection system.

**Components:**
1. **Money Flow**: Measures buying/selling liquidity entering the market (-100 to +100)
2. **Thresholds**: Dynamic levels showing significant buying/selling activity
3. **Overflow**: Detects excess buying/selling that predicts reversals
4. **Trend Wave**: A highly reactive trend-following oscillator (0-100)
5. **Real-Time Divergences**: Price vs oscillator divergence detection
6. **Reversal Signals**: High-frequency (small dots) and strong (arrows) reversal signals
7. **Confluence**: Combined signal strength from all components (-100 to +100)

```python
def momentum_confluence(
    data: Union[PdDataFrame, PlDataFrame],
    money_flow_length: int = 14,
    trend_wave_length: int = 10,
    threshold_mult: float = 1.5,
    overflow_threshold: float = 0.8,
    divergence_lookback: int = 5,
    high_column: str = 'High',
    low_column: str = 'Low',
    close_column: str = 'Close',
    volume_column: str = 'Volume',
    ...
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from pyindicators import (
    momentum_confluence,
    momentum_confluence_signal,
    get_momentum_confluence_stats
)

# Calculate Momentum Confluence
df = momentum_confluence(df)

# Generate trading signals
df = momentum_confluence_signal(df)

# Get statistics
stats = get_momentum_confluence_stats(df)
print(f"Strong bullish reversals: {stats['strong_reversal_bullish_count']}")
print(f"Divergences detected: {stats['divergence_bullish_count']}")
```

**Output Columns:**
- `money_flow`: Money flow oscillator (-100 to +100)
- `mf_upper_threshold` / `mf_lower_threshold`: Dynamic threshold levels
- `overflow_bullish` / `overflow_bearish`: Excess buying/selling (0 or 1)
- `trend_wave`: Trend oscillator (0-100)
- `trend_wave_signal`: Trend direction (1=bullish, -1=bearish, 0=neutral)
- `divergence_bullish` / `divergence_bearish`: Divergence detection (0 or 1)
- `reversal_bullish` / `reversal_bearish`: High-frequency reversal signals (0 or 1)
- `reversal_strong_bullish` / `reversal_strong_bearish`: Strong reversal signals (0 or 1)
- `confluence`: Combined signal strength (-100 to +100)
- `mc_trend`: Overall trend direction (1=bullish, -1=bearish, 0=neutral)

**Signal Values (from momentum_confluence_signal):**
- `2`: Strong bullish reversal signal
- `1`: Bullish confluence
- `0`: Neutral
- `-1`: Bearish confluence
- `-2`: Strong bearish reversal signal

![MOMENTUM_CONFLUENCE](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/momentum_confluence.png)


### Volatility indicators

Indicators that measure the rate of price movement, regardless of direction. They help to identify
periods of high and low volatility in the market.

#### Bollinger Bands (BB)

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

![BOLLINGER_BANDS](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/bollinger_bands.png)

#### Bollinger Bands Overshoot

Bollinger Bands Overshoot measures how far the price has exceeded the upper or lower Bollinger Band, expressed as a percentage of the half-band width (distance from middle to upper/lower band). This indicator helps identify extreme price movements and potential mean reversion opportunities.

**Calculation:**
- When price > upper band (bullish overshoot): `((Price - Upper Band) / (Upper Band - Middle Band)) × 100`
- When price < lower band (bearish overshoot): `((Price - Lower Band) / (Middle Band - Lower Band)) × 100`
- When price is within bands: `0%`

**Interpretation:**
- Positive values indicate overbought conditions (price above upper band)
- Negative values indicate oversold conditions (price below lower band)
- High overshoots (e.g., 40%) indicate increased risk of mean reversion

```python
def bollinger_overshoot(
    data: Union[PdDataFrame, PlDataFrame],
    source_column='Close',
    period=20,
    std_dev=2,
    result_column='bollinger_overshoot'
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import bollinger_overshoot

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

# Calculate Bollinger Bands Overshoot for Polars DataFrame
pl_df = bollinger_overshoot(pl_df, source_column="Close")
pl_df.show(10)

# Calculate Bollinger Bands Overshoot for Pandas DataFrame
pd_df = bollinger_overshoot(pd_df, source_column="Close")
pd_df.tail(10)
```

![BOLLINGER_OVERSHOOT](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/bollinger_overshoot.png)

#### Average True Range (ATR)

The Average True Range (ATR) is a volatility indicator that measures the average range between the high and low prices over a specified period. It helps traders identify potential price fluctuations and adjust their strategies accordingly.

```python
def atr(
    data: Union[PdDataFrame, PlDataFrame],
    source_column="Close",
    period=14,
    result_column="ATR"
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import atr

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

# Calculate average true range for Polars DataFrame
pl_df = atr(pl_df, source_column="Close")
pl_df.show(10)

# Calculate average true range for Pandas DataFrame
pd_df = atr(pd_df, source_column="Close")
pd_df.tail(10)
```

![ATR](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/atr.png)

#### Moving Average Envelope (MAE)

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

![MOVING_AVERAGE_ENVELOPE](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/moving_average_envelope.png)

#### Nadaraya-Watson Envelope (NWE)

The Nadaraya-Watson Envelope uses Gaussian kernel regression to create a smoothed price estimate, then adds an envelope based on the mean absolute error (MAE) scaled by a multiplier. This is a non-repainting (endpoint) implementation inspired by the TradingView "Nadaraya-Watson Envelope [LuxAlgo]" indicator. It is useful for identifying overbought/oversold zones and mean-reversion opportunities.

Calculation:
- Kernel weights: `w(i) = exp(-i² / (2 × h²))` for `i = 0..lookback-1`
- Smoothed value: `sum(src[t-i] × w(i)) / sum(w(i))`
- MAE: SMA of `|src - smoothed|` over the lookback period
- Upper: `smoothed + mult × MAE`
- Lower: `smoothed - mult × MAE`

```python
def nadaraya_watson_envelope(
    data: Union[PdDataFrame, PlDataFrame],
    source_column: str = 'Close',
    bandwidth: float = 8.0,
    mult: float = 3.0,
    lookback: int = 500,
    upper_column: str = 'nwe_upper',
    lower_column: str = 'nwe_lower',
    middle_column: str = 'nwe_middle',
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import nadaraya_watson_envelope

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

# Calculate Nadaraya-Watson Envelope for Polars DataFrame
pl_df = nadaraya_watson_envelope(pl_df, source_column="Close", bandwidth=8.0, mult=3.0)
pl_df.show(10)

# Calculate Nadaraya-Watson Envelope for Pandas DataFrame
pd_df = nadaraya_watson_envelope(pd_df, source_column="Close", bandwidth=8.0, mult=3.0)
pd_df.tail(10)
```

![NADARAYA_WATSON_ENVELOPE](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/nadaraya_watson_envelope.png)

### Trend Following

Indicators that combine trend detection with adaptive trailing stops.

#### SuperTrend Clustering

The SuperTrend Clustering indicator uses K-means clustering to optimize the ATR multiplier factor for the SuperTrend calculation. It computes multiple SuperTrend variations with different factors, evaluates their performance, and clusters them into "best", "average", and "worst" groups. The best-performing factor is then used to generate an adaptive trailing stop with buy/sell signals.

Based on the LuxAlgo SuperTrend AI indicator concept.

```python
def supertrend_clustering(
    data: Union[PdDataFrame, PlDataFrame],
    atr_length: int = 10,
    min_mult: float = 1.0,
    max_mult: float = 5.0,
    step: float = 0.5,
    perf_alpha: float = 10.0,
    from_cluster: str = 'best',
    max_iter: int = 1000,
    max_data: int = 10000
) -> Union[PdDataFrame, PlDataFrame]:
```

Returns the following columns:
- `supertrend`: The optimized SuperTrend trailing stop
- `supertrend_trend`: Current trend (1=bullish, 0=bearish)
- `supertrend_ama`: Adaptive moving average of SuperTrend
- `supertrend_perf_idx`: Performance index (0–1 scale)
- `supertrend_factor`: Currently used ATR factor
- `supertrend_signal`: 1=buy signal, -1=sell signal, 0=no signal

Example

```python
from investing_algorithm_framework import download

from pyindicators import supertrend_clustering, get_supertrend_stats

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

# Calculate SuperTrend Clustering
pd_df = supertrend_clustering(
    pd_df,
    atr_length=14,
    min_mult=2.0,
    max_mult=6.0,
    step=0.5,
    perf_alpha=14.0,
    from_cluster='best',
    max_data=500
)

# Get statistics
stats = get_supertrend_stats(pd_df)
print(stats)
pd_df.tail(10)
```

![SUPERTREND_CLUSTERING](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/supertrend_clustering.png)

#### SuperTrend Basic

The basic SuperTrend indicator uses a fixed ATR multiplier factor to create a trend-following trailing stop. When the price is above the SuperTrend line the trend is bullish; when below, bearish. Trend changes generate buy/sell signals.

```python
def supertrend_basic(
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

from pyindicators import supertrend_basic

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

# Calculate basic SuperTrend
pd_df = supertrend_basic(pd_df, atr_length=10, factor=3.0)
pd_df.tail(10)
```

### Support and Resistance

Indicators that help identify potential support and resistance levels in the market.

#### Fibonacci Retracement

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

![FIBONACCI_RETRACEMENT](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/fibonacci_retracement.png)

#### Golden Zone

The Golden Zone indicator calculates Fibonacci retracement levels based on the highest high and lowest low over a specified rolling period. The "Golden Zone" refers to the area between the 50% and 61.8% Fibonacci retracement levels, which is often considered a key area for potential price reversals or continuations.

This indicator plots dynamic support/resistance levels that update with each bar, making it useful for identifying potential entry and exit points in trending markets.

The calculation formula is:
```
Highest High (HH) = Rolling maximum of high prices over `length` bars
Lowest Low (LL) = Rolling minimum of low prices over `length` bars
Diff = HH - LL
Upper Level = HH - (Diff × 0.5)      # 50% retracement
Lower Level = HH - (Diff × 0.618)    # 61.8% retracement
```

```python
def golden_zone(
    data: Union[PdDataFrame, PlDataFrame],
    high_column: str = 'High',
    low_column: str = 'Low',
    length: int = 60,
    retracement_level_1: float = 0.5,
    retracement_level_2: float = 0.618,
    upper_column: str = 'golden_zone_upper',
    lower_column: str = 'golden_zone_lower',
    hh_column: str = 'golden_zone_hh',
    ll_column: str = 'golden_zone_ll'
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import golden_zone

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

# Calculate Golden Zone for Polars DataFrame
pl_df = golden_zone(pl_df, high_column="High", low_column="Low", length=60)
pl_df.show(10)

# Calculate Golden Zone for Pandas DataFrame
pd_df = golden_zone(pd_df, high_column="High", low_column="Low", length=60)
pd_df.tail(10)
```

![GOLDEN_ZONE](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/golden_zone.png)

#### Golden Zone Signal

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

![GOLDEN_ZONE_SIGNAL](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/golden_zone_signal.png)

#### Fair Value Gap (FVG)

A Fair Value Gap (FVG) is a price imbalance that occurs when there's a gap between candlesticks, representing institutional order flow. These gaps often act as support/resistance zones where price tends to return.

**Bullish FVG (Gap Up):** Occurs when the low of the current candle is higher than the high of the candle 2 bars ago. This creates an upward gap that may act as future support.

**Bearish FVG (Gap Down):** Occurs when the high of the current candle is lower than the low of the candle 2 bars ago. This creates a downward gap that may act as future resistance.

```python
def fair_value_gap(
    data: Union[PdDataFrame, PlDataFrame],
    high_column: str = 'High',
    low_column: str = 'Low',
    bullish_fvg_column: str = 'bullish_fvg',
    bearish_fvg_column: str = 'bearish_fvg',
    bullish_fvg_top_column: str = 'bullish_fvg_top',
    bullish_fvg_bottom_column: str = 'bullish_fvg_bottom',
    bearish_fvg_top_column: str = 'bearish_fvg_top',
    bearish_fvg_bottom_column: str = 'bearish_fvg_bottom'
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
import pandas as pd
from pyindicators import fair_value_gap, fvg_signal, fvg_filled

# Create sample OHLC data
df = pd.DataFrame({
    'High': [100, 105, 115, 120, 118, 115],
    'Low': [95, 100, 102, 115, 113, 99],
    'Close': [98, 103, 110, 117, 115, 100]
})

# Detect Fair Value Gaps
df = fair_value_gap(df)
print(df[['bullish_fvg', 'bearish_fvg', 'bullish_fvg_top', 'bullish_fvg_bottom']])

# Generate signals when price enters an FVG zone
df = fvg_signal(df)
print(df['fvg_signal'])  # 1 = in bullish zone, -1 = in bearish zone, 0 = outside

# Detect when FVGs have been filled (mitigated)
df = fvg_filled(df)
print(df[['bullish_fvg_filled', 'bearish_fvg_filled']])
```

The `fvg_signal` function generates signals:
- **1**: Price is within a bullish FVG zone (potential long entry)
- **-1**: Price is within a bearish FVG zone (potential short entry)
- **0**: Price is outside any FVG zone

The `fvg_filled` function detects when FVGs have been mitigated:
- Bullish FVG filled: Price drops to reach the bottom of the gap
- Bearish FVG filled: Price rises to reach the top of the gap

![FAIR_VALUE_GAP](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/fair_value_gap.png)

#### Order Blocks

Order Blocks are zones where institutional traders (banks, hedge funds) placed large orders, causing significant price moves. They represent areas of supply and demand imbalance that often act as support/resistance when price returns.

**Bullish Order Block:** The last bearish candle before a strong upward move. When price returns to this zone, it often bounces up (support).

**Bearish Order Block:** The last bullish candle before a strong downward move. When price returns to this zone, it often reverses down (resistance).

**Breaker Blocks:** When an Order Block is broken (invalidated), it becomes a breaker block and may act as the opposite type of support/resistance.

```python
def order_blocks(
    data: Union[PdDataFrame, PlDataFrame],
    swing_length: int = 10,
    use_body: bool = False,
    high_column: str = 'High',
    low_column: str = 'Low',
    open_column: str = 'Open',
    close_column: str = 'Close',
    bullish_ob_column: str = 'bullish_ob',
    bearish_ob_column: str = 'bearish_ob',
    bullish_ob_top_column: str = 'bullish_ob_top',
    bullish_ob_bottom_column: str = 'bullish_ob_bottom',
    bearish_ob_top_column: str = 'bearish_ob_top',
    bearish_ob_bottom_column: str = 'bearish_ob_bottom',
    bullish_breaker_column: str = 'bullish_breaker',
    bearish_breaker_column: str = 'bearish_breaker'
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
import pandas as pd
from pyindicators import order_blocks, ob_signal, get_active_order_blocks

# Create sample OHLC data
df = pd.DataFrame({
    'Open': [100, 102, 101, 105, 110, 108, 112, 115, 113, 118],
    'High': [103, 104, 106, 112, 115, 112, 118, 120, 117, 122],
    'Low': [99, 100, 100, 104, 108, 106, 110, 113, 111, 116],
    'Close': [102, 101, 105, 110, 108, 110, 115, 113, 116, 120]
})

# Detect Order Blocks
df = order_blocks(df, swing_length=5)
print(df[['bullish_ob', 'bearish_ob', 'bullish_ob_top', 'bullish_ob_bottom']])

# Generate signals when price enters an OB zone
df = ob_signal(df)
print(df['ob_signal'])  # 1 = in bullish zone, -1 = in bearish zone, 0 = outside

# Get currently active Order Blocks
active = get_active_order_blocks(df, max_bullish=3, max_bearish=3)
print(f"Active bullish OBs: {len(active['bullish'])}")
print(f"Active bearish OBs: {len(active['bearish'])}")
```

The function returns columns for:
- `bullish_ob` / `bearish_ob`: 1 when Order Block is detected, 0 otherwise
- `bullish_ob_top` / `bullish_ob_bottom`: Zone boundaries for bullish OBs
- `bearish_ob_top` / `bearish_ob_bottom`: Zone boundaries for bearish OBs
- `bullish_breaker` / `bearish_breaker`: 1 when OB is broken (becomes breaker block)

The `ob_signal` function generates signals:
- **1**: Price is within a bullish OB zone (potential long entry)
- **-1**: Price is within a bearish OB zone (potential short entry)
- **0**: Price is outside any OB zone

![ORDER_BLOCKS](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/order_blocks.png)

#### Market Structure Break

Market Structure Break (MSB) is a Smart Money Concept (SMC) indicator that detects when price breaks through significant pivot points, signaling potential trend changes. Combined with Order Block detection and quality scoring, this tool helps identify high-probability trading zones.

**Market Structure Break (MSB):**
- **Bullish MSB:** Price closes above the last pivot high, indicating potential bullish momentum
- **Bearish MSB:** Price closes below the last pivot low, indicating potential bearish momentum

**Order Block Quality Score (0-100):**
- Based on momentum z-score and volume percentile
- Score > 80 indicates a High Probability Zone (HPZ)

**Best Use Cases:**
- Pullback/retracement trading (enter at OB zones after MSB)
- Multi-timeframe analysis (use higher TF for bias, lower TF for entries)
- Supply & demand zone trading

```python
def market_structure_break(
    data: Union[PdDataFrame, PlDataFrame],
    pivot_length: int = 7,
    momentum_zscore_threshold: float = 0.5,
    high_column: str = 'High',
    low_column: str = 'Low',
    close_column: str = 'Close',
    volume_column: str = 'Volume',
    msb_bullish_column: str = 'msb_bullish',
    msb_bearish_column: str = 'msb_bearish',
    last_pivot_high_column: str = 'last_pivot_high',
    last_pivot_low_column: str = 'last_pivot_low',
    momentum_z_column: str = 'momentum_z'
) -> Union[PdDataFrame, PlDataFrame]:

def market_structure_ob(
    data: Union[PdDataFrame, PlDataFrame],
    pivot_length: int = 7,
    momentum_zscore_threshold: float = 0.5,
    max_active_obs: int = 10,
    ...
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
import pandas as pd
from pyindicators import (
    market_structure_break,
    market_structure_ob,
    get_market_structure_stats
)

# Create sample OHLC data
df = pd.DataFrame({
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
})

# Basic MSB detection
df = market_structure_break(df, pivot_length=5)
print(df[['msb_bullish', 'msb_bearish', 'last_pivot_high', 'last_pivot_low']])

# MSB with Order Block detection and quality scoring
df = market_structure_ob(df, pivot_length=5)
print(df[['msb_bullish', 'msb_bearish', 'ob_bullish', 'ob_bearish', 'ob_quality', 'ob_is_hpz']])

# Get statistics
stats = get_market_structure_stats(df)
print(f"Reliability: {stats['reliability']:.1f}%")
print(f"HPZ Count: {stats['hpz_count']}")
print(f"Bullish MSBs: {stats['bullish_msb_count']}")
print(f"Bearish MSBs: {stats['bearish_msb_count']}")
```

The `market_structure_break` function returns:
- `msb_bullish` / `msb_bearish`: 1 when MSB detected, 0 otherwise
- `last_pivot_high` / `last_pivot_low`: Most recent pivot levels
- `momentum_z`: Momentum z-score value

The `market_structure_ob` function additionally returns:
- `ob_bullish` / `ob_bearish`: 1 when Order Block detected at MSB
- `ob_top` / `ob_bottom`: Order Block zone boundaries
- `ob_quality`: Quality score (0-100)
- `ob_is_hpz`: True if quality > 80 (High Probability Zone)
- `ob_mitigated`: 1 when Order Block has been mitigated

**Recommended Parameters by Timeframe:**

| Timeframe | pivot_length | Use Case |
|-----------|-------------|----------|
| 1m-5m | 2-3 | Scalping entries |
| 15m | 3-5 | Day trading |
| 1H | 5-7 | Swing confirmation |
| 4H-Daily | 7-10 | Trend direction |

![MARKET_STRUCTURE](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/market_structure_ob.png)

#### Market Structure CHoCH/BOS

Market Structure CHoCH/BOS (Fractal) is a Smart Money Concept indicator that uses fractal detection to identify swing points and distinguishes between two types of structure breaks:

**CHoCH (Change of Character):** A trend reversal signal that occurs when price breaks a swing point in the **opposite direction** of the current trend.
- Bullish CHoCH: Trend was bearish, price breaks above swing high (reversal to bullish)
- Bearish CHoCH: Trend was bullish, price breaks below swing low (reversal to bearish)

**BOS (Break of Structure):** A trend continuation signal that occurs when price breaks a swing point in the **same direction** as the current trend.
- Bullish BOS: Trend is bullish, price breaks above swing high (continuation)
- Bearish BOS: Trend is bearish, price breaks below swing low (continuation)

This indicator also tracks dynamic support and resistance levels based on the swing structure.

```python
def market_structure_choch_bos(
    data: Union[PdDataFrame, PlDataFrame],
    length: int = 5,
    high_column: str = 'High',
    low_column: str = 'Low',
    close_column: str = 'Close',
    choch_bullish_column: str = 'choch_bullish',
    choch_bearish_column: str = 'choch_bearish',
    bos_bullish_column: str = 'bos_bullish',
    bos_bearish_column: str = 'bos_bearish',
    support_column: str = 'support_level',
    resistance_column: str = 'resistance_level',
    support_broken_column: str = 'support_broken',
    resistance_broken_column: str = 'resistance_broken',
    trend_column: str = 'market_trend'
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
import pandas as pd
from pyindicators import (
    market_structure_choch_bos,
    choch_bos_signal,
    get_choch_bos_stats
)

# Create sample OHLC data
df = pd.DataFrame({
    'High': [...],
    'Low': [...],
    'Close': [...]
})

# Detect CHoCH and BOS signals
df = market_structure_choch_bos(df, length=5)
print(df[['choch_bullish', 'choch_bearish', 'bos_bullish', 'bos_bearish', 'market_trend']])

# Generate trading signals
# 2 = bullish CHoCH (strong reversal), 1 = bullish BOS (continuation)
# -1 = bearish BOS (continuation), -2 = bearish CHoCH (strong reversal)
df = choch_bos_signal(df)
reversal_signals = df[abs(df['structure_signal']) == 2]

# Get statistics
stats = get_choch_bos_stats(df)
print(f"Total reversals (CHoCH): {stats['total_choch']}")
print(f"Total continuations (BOS): {stats['total_bos']}")
```

The function returns:
- `choch_bullish` / `choch_bearish`: 1 when CHoCH detected (trend reversal)
- `bos_bullish` / `bos_bearish`: 1 when BOS detected (trend continuation)
- `support_level` / `resistance_level`: Current S/R level prices
- `support_broken` / `resistance_broken`: 1 when S/R level is broken
- `market_trend`: Current trend direction (1=bullish, -1=bearish, 0=neutral)

**Trading Strategy:**
- CHoCH signals are stronger (trend reversals) - good for counter-trend entries
- BOS signals are trend confirmations - good for trend-following entries
- Use support/resistance levels for stop loss placement

![MARKET_STRUCTURE_CHOCH_BOS](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/market_structure_choch_bos.png)

### Pattern Recognition

#### Detect Peaks

The detect_peaks function is used to identify peaks and lows in a given column of a DataFrame. It returns a DataFrame with two additional columns: one for higher highs and another for lower lows. The function can be used to detect peaks and lows in a DataFrame. It identifies local maxima and minima based on the specified order of neighboring points. The function can also filter out peaks and lows based on a minimum number of consecutive occurrences. This allows you to focus on significant peaks and lows that are more likely to be relevant for analysis.

> There is always a delay between an actual peak and the detection of that peak. This is determined by the `number_of_neighbors_to_compare` parameter. For example
> if for a given column you set `number_of_neighbors_to_compare=5`, the function will look at the 5 previous and 5 next data points to determine if the current point is a peak or a low. This means that the peak or low will only be detected after the 5th data point has been processed. So say you have OHLCV data of 15 minute intervals, and you set `number_of_neighbors_to_compare=5`, the function will only detect the peak or low after the 5th data point has been processed, which means that there will be a delay of 75 minutes (5 * 15 minutes) before the peak or low is detected.

```python
def detect_peaks(
    data: Union[PdDataFrame, PlDataFrame],
    column: str,
    number_of_neighbors_to_compare: int = 5,
    min_consecutive: int = 2
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download
from pyindicators import detect_peaks

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

# Calculate peaks and lows for Polars DataFrame, with a neighbour comparison of 4 and minimum of 2 consecutive peaks
pl_df = detect_peaks(pl_df, source_column="Close", number_of_neighbors_to_compare=4, min_consecutive=2)
pl_df.show(10)

# Calculate peaks and lows for Pandas DataFrame, with a neighbour comparison of 4 and minimum of 2 consecutive peaks
pd_df = detect_peaks(pd_df, source_column="Close", number_of_neighbors_to_compare=4, min_consecutive=2)
pd_df.tail(10)
```

![PEAKS](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/detect_peaks.png)

#### Detect Bullish Divergence

The detect_bullish_divergence function is used to identify bullish divergences between two columns in a DataFrame. It checks for bullish divergences based on the peaks and lows detected in the specified columns. The function returns a DataFrame with additional columns indicating the presence of bullish divergences.

A bullish divergence occurs when the price makes a lower low while the indicator makes a higher low. This suggests that the downward momentum is weakening, and a potential reversal to the upside may occur.

> !Important: This function expects that for two given columns there will be corresponding peaks and lows columns. This means that before you can use this function, you must first call the detect_peaks function on both columns. For example: if you want to detect bullish divergence between the "Close" column and the "RSI_14" column, you must first call detect_peaks on both columns.
> If no corresponding {column}_peaks and {column}_lows columns are found, the function will raise a PyIndicatorException.

```python
def bullish_divergence(
    data: Union[pd.DataFrame, pl.DataFrame],
    first_column: str,
    second_column: str,
    window_size=1,
    result_column: str = "bullish_divergence",
    number_of_neighbors_to_compare: int = 5,
    min_consecutive: int = 2
) -> Union[pd.DataFrame, pl.DataFrame]:
```

Example

```python
from investing_algorithm_framework import download
from pyindicators import bullish_divergence
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

# Calculate bearish divergence for Polars DataFrame, treat first_column always as the indicator column
pl_df = bearish_divergence(pl_df, first_column="RSI_14", second_column="Close", window_size=8)
pl_df.show(10)

# Calculate bearish divergence for Pandas DataFrame
pd_df = bearish_divergence(pd_df, first_column="RSI_14", second_column="Close", window_size=8)
pd_df.tail(10)
```

![BULLISH_DIVERGENCE](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/bullish_divergence.png)

#### Detect Bearish Divergence

The detect_bearish_divergence function is used to identify bearish divergences between two columns in a DataFrame. It checks for bearish divergences based on the peaks and lows detected in the specified columns. The function returns a DataFrame with additional columns indicating the presence of bearish divergences.

A bearish divergence occurs when the price makes a higher high while the indicator makes a lower high. This suggests that the upward momentum is weakening, and a potential reversal to the downside may occur.

```python
def bearish_divergence(
    data: Union[pd.DataFrame, pl.DataFrame],
    first_column: str,
    second_column: str,
    window_size=1,
    result_column: str = "bearish_divergence",
    number_of_neighbors_to_compare: int = 5,
    min_consecutive: int = 2
) -> Union[pd.DataFrame, pl.DataFrame]:
```

Example

```python
from investing_algorithm_framework import download
from pyindicators import bearish_divergence
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

# Calculate bearish divergence for Polars DataFrame, treat first_column always as the indicator column
pl_df = bearish_divergence(pl_df, first_column="RSI_14", second_column="Close", window_size=8)
pl_df.show(10)

# Calculate bearish divergence for Pandas DataFrame, treat first_column always as the indicator column
pd_df = bearish_divergence(pd_df, first_column="RSI_14", second_column="Close", window_size=8)
pd_df.tail(10)
```

![BEARISH_DIVERGENCE](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/bearish_divergence.png)

### Indicator helpers

#### Crossover

The crossover function is used to calculate the crossover between two columns in a DataFrame. It returns a new DataFrame with an additional column that contains the crossover values. A crossover occurs when the first column crosses above or below the second column. This can happen in two ways, a strict crossover or a non-strict crossover. In a strict crossover, the first column must cross above or below the second column. In a non-strict crossover, the first column must cross above or below the second column, but the values can be equal.

```python
def crossover(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str,
    second_column: str,
    result_column="crossover",
    number_of_data_points: int = None,
    strict: bool = True,
) -> Union[PdDataFrame, PlDataFrame]:
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
pl_df.show(10)

# Calculate EMA and crossover for Pandas DataFrame
pd_df = ema(pd_df, source_column="Close", period=200, result_column="EMA_200")
pd_df = ema(pd_df, source_column="Close", period=50, result_column="EMA_50")
pd_df = crossover(
    pd_df,
    first_column="EMA_50",
    second_column="EMA_200",
    result_column="Crossover_EMA"
)
pd_df.tail(10)
```

![CROSSOVER](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/crossover.png)

#### Is Crossover

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

#### Crossunder

The crossunder function is used to calculate the crossunder between two columns in a DataFrame. It returns a new DataFrame with an additional column that contains the crossunder values. A crossunder occurs when the first column crosses below the second column. This can happen in two ways, a strict crossunder or a non-strict crossunder. In a strict crossunder, the first column must cross below the second column. In a non-strict crossunder, the first column must cross below the second column, but the values can be equal.

```python
def crossunder(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str,
    second_column: str,
    result_column="crossunder",
    number_of_data_points: int = None,
    strict: bool = True,
) -> Union[PdDataFrame, PlDataFrame]:
```

Example

```python
from investing_algorithm_framework import download
from pyindicators import crossunder, ema

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

# Calculate EMA and crossunder for Polars DataFrame
pl_df = ema(pl_df, source_column="Close", period=200, result_column="EMA_200")
pl_df = ema(pl_df, source_column="Close", period=50, result_column="EMA_50")
pl_df = crossunder(
    pl_df,
    first_column="EMA_50",
    second_column="EMA_200",
    result_column="Crossunder_EMA"
)
pl_df.show(10)

# Calculate EMA and crossunder for Pandas DataFrame
pd_df = ema(pd_df, source_column="Close", period=200, result_column="EMA_200")
pd_df = ema(pd_df, source_column="Close", period=50, result_column="EMA_50")
pd_df = crossunder(
    pd_df,
    first_column="EMA_50",
    second_column="EMA_200",
    result_column="Crossunder_EMA"
)
pd_df.tail(10)
```

![CROSSUNDER](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/crossunder.png)

#### Is Crossunder

The is_crossunder function is used to determine if a crossunder occurred in the last N data points. It returns a boolean value indicating if a crossunder occurred in the last N data points. The function can be used to check for crossunders in a DataFrame that was previously calculated using the crossunder function.

```python
def is_crossunder(
    data: Union[PdDataFrame, PlDataFrame],
    first_column: str = None,
    second_column: str = None,
    crossunder_column: str = None,
    number_of_data_points: int = None,
    strict: bool = True,
) -> bool:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import crossunder, ema, is_crossunder

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

# Calculate EMA and crossunders for Polars DataFrame
pl_df = ema(pl_df, source_column="Close", period=200, result_column="EMA_200")
pl_df = ema(pl_df, source_column="Close", period=50, result_column="EMA_50")
pl_df = crossunder(
    pl_df,
    first_column="EMA_50",
    second_column="EMA_200",
    result_column="Crossunder_EMA"
)

# If you want the function to calculate the crossunders in the function
if is_crossunder(
    pl_df, first_column="EMA_50", second_column="EMA_200", number_of_data_points=3
):
    print("Crossunder detected in Pandas DataFrame in the last 3 data points")

# If you want to use the result of a previous crossunders calculation
if is_crossunder(pl_df, crossunder_column="Crossunder_EMA", number_of_data_points=3):
    print("Crossunder detected in Pandas DataFrame in the last 3 data points")

# Calculate EMA and crossunders for Pandas DataFrame
pd_df = ema(pd_df, source_column="Close", period=200, result_column="EMA_200")
pd_df = ema(pd_df, source_column="Close", period=50, result_column="EMA_50")

# If you want the function to calculate the crossunders in the function
if is_crossunder(
    pd_df, first_column="EMA_50", second_column="EMA_200", number_of_data_points=3
):
    print("Crossunders detected in Pandas DataFrame in the last 3 data points")

# If you want to use the result of a previous crossover calculation
if is_crossunder(pd_df, crossunder_column="Crossunder_EMA", number_of_data_points=3):
    print("Crossunder detected in Pandas DataFrame in the last 3 data points")
```

#### Is Downtrend

The is_downtrend function is used to determine if a downtrend occurred in the last N data points. It returns a boolean value indicating if a downtrend occurred in the last N data points. The function can be used to check for downtrends in a DataFrame that was previously calculated using the crossover function.

```python
def is_down_trend(
    data: Union[PdDataFrame, PlDataFrame],
    use_death_cross: bool = True,
) -> bool:
```

Example

```python
from investing_algorithm_framework import CSVOHLCVMarketDataSource

from pyindicators import is_down_trend

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

print(is_down_trend(pl_df))
print(is_down_trend(pd_df))
```

#### Is Uptrend

The is_up_trend function is used to determine if an uptrend occurred in the last N data points. It returns a boolean value indicating if an uptrend occurred in the last N data points. The function can be used to check for uptrends in a DataFrame that was previously calculated using the crossover function.

```python
def is_up_trend(
    data: Union[PdDataFrame, PlDataFrame],
    use_golden_cross: bool = True,
) -> bool:
```

Example

```python
from investing_algorithm_framework import download

from pyindicators import is_up_trend

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

print(is_up_trend(pl_df))
print(is_up_trend(pd_df))
```

#### has_any_lower_then_threshold

The `has_any_lower_then_threshold` function checks if any value in a given column is lower than a specified threshold within the last N data points. This is useful for detecting when an indicator or price falls below a critical level.

```python
def has_any_lower_then_threshold(
    data: Union[pd.DataFrame, pl.DataFrame],
    column,
    threshold,
    strict=True,
    number_of_data_points=1
) -> bool:
    ...
```

Example

```python
import pandas as pd
from pyindicators.indicators.utils import has_any_lower_then_threshold

# Example DataFrame
prices = pd.DataFrame({
    'Close': [100, 98, 97, 99, 96, 95, 97, 98, 99, 100]
})

# Check if any of the last 5 closes are below 97
result = has_any_lower_then_threshold(prices, column='Close', threshold=97, number_of_data_points=5)
print(result)  # Output: True
```

Below is a chart showing the threshold and the points where the condition is met:

![has_any_lower_then_threshold](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/has_any_lower_then_threshold.png)

In this chart, the red line represents the threshold, and the highlighted points are where the `Close` value is below the threshold in the last N data points.
