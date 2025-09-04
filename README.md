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
* [Momentum and Oscillators](#momentum-and-oscillators)
  * [Moving Average Convergence Divergence (MACD)](#moving-average-convergence-divergence-macd)
  * [Relative Strength Index (RSI)](#relative-strength-index-rsi)
  * [Relative Strength Index Wilders method (Wilders RSI)](#wilders-relative-strength-index-wilders-rsi)
  * [Williams %R](#williams-r)
  * [Average Directional Index (ADX)](#average-directional-index-adx)
  * [Stochastic Oscillator (STO)](#stochastic-oscillator-sto)
* [Volatility indicators](#volatility-indicators)
  * [Bollinger Bands (BB)](#bollinger-bands-bb)
  * [Average True Range (ATR)](#average-true-range-atr)
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
