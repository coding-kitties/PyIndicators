# PyIndicators

PyIndicators is a powerful and user-friendly Python library for technical analysis indicators, metrics and helper functions. Written entirely in Python, it requires no external dependencies, ensuring seamless integration and ease of use.

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
* Supports python version 3.9 and above.
* [Trend indicators](#trend-indicators)
  * [Weighted Moving Average (WMA)](#weighted-moving-average-wma)
  * [Simple Moving Average (SMA)](#simple-moving-average-sma)
  * [Exponential Moving Average (EMA)](#exponential-moving-average-ema)
* [Momentum indicators](#momentum-indicators)
  * [Moving Average Convergence Divergence (MACD)](#moving-average-convergence-divergence-macd)
  * [Relative Strength Index (RSI)](#relative-strength-index-rsi)
  * [Relative Strength Index Wilders method (Wilders RSI)](#wilders-relative-strength-index-wilders-rsi)
  * [Williams %R](#williams-r)
* [Indicator helpers](#indicator-helpers)
  * [Crossover](#crossover)
  * [Is Crossover](#is-crossover)

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
from investing_algorithm_framework import CSVOHLCVMarketDataSource

from pyindicators import wma

# For this example the investing algorithm framework is used for dataframe creation,
csv_path = "./tests/test_data/OHLCV_BTC-EUR_BINANCE_15m_2023-12-01:00:00_2023-12-25:00:00.csv"
data_source = CSVOHLCVMarketDataSource(csv_file_path=csv_path)

pl_df = data_source.get_data()
pd_df = data_source.get_data(pandas=True)

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
from investing_algorithm_framework import CSVOHLCVMarketDataSource

from pyindicators import sma

# For this example the investing algorithm framework is used for dataframe creation,
csv_path = "./tests/test_data/OHLCV_BTC-EUR_BINANCE_15m_2023-12-01:00:00_2023-12-25:00:00.csv"
data_source = CSVOHLCVMarketDataSource(csv_file_path=csv_path)

pl_df = data_source.get_data()
pd_df = data_source.get_data(pandas=True)

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
from investing_algorithm_framework import CSVOHLCVMarketDataSource

from pyindicators import ema

# For this example the investing algorithm framework is used for dataframe creation,
csv_path = "./tests/test_data/OHLCV_BTC-EUR_BINANCE_15m_2023-12-01:00:00_2023-12-25:00:00.csv"
data_source = CSVOHLCVMarketDataSource(csv_file_path=csv_path)

pl_df = data_source.get_data()
pd_df = data_source.get_data(pandas=True)

# Calculate EMA for Polars DataFrame
pl_df = ema(pl_df, source_column="Close", period=200, result_column="EMA_200")
pl_df.show(10)

# Calculate EMA for Pandas DataFrame
pd_df = ema(pd_df, source_column="Close", period=200, result_column="EMA_200")
pd_df.tail(10)
```

![EMA](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/ema.png)

### Momentum Indicators

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
from investing_algorithm_framework import CSVOHLCVMarketDataSource

from pyindicators import macd

# For this example the investing algorithm framework is used for dataframe creation,
csv_path = "./tests/test_data/OHLCV_BTC-EUR_BINANCE_15m_2023-12-01:00:00_2023-12-25:00:00.csv"
data_source = CSVOHLCVMarketDataSource(csv_file_path=csv_path)

pl_df = data_source.get_data()
pd_df = data_source.get_data(pandas=True)

# Calculate MACD for Polars DataFrame
pl_df = macd(pl_df, source_column="Close", short_period=12, long_period=26, signal_period=9)

# Calculate MACD for Pandas DataFrame
pd_df = macd(pd_df, source_column="Close", short_period=12, long_period=26, signal_period=9)

pl_df.show(10)
pd_df.tail(10)
```

![MACD](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/macd.png)

#### Relative Strength Index (RSI)

```python
from investing_algorithm_framework import CSVOHLCVMarketDataSource

from pyindicators import rsi

# For this example the investing algorithm framework is used for dataframe creation,
csv_path = "./tests/test_data/OHLCV_BTC-EUR_BINANCE_15m_2023-12-01:00:00_2023-12-25:00:00.csv"
data_source = CSVOHLCVMarketDataSource(csv_file_path=csv_path)

pl_df = data_source.get_data()
pd_df = data_source.get_data(pandas=True)

# Calculate RSI for Polars DataFrame
pl_df = rsi(pl_df, source_column="Close", period=14, result_column="RSI_14")
pl_df.show(10)

# Calculate RSI for Pandas DataFrame
pd_df = rsi(pd_df, source_column="Close", period=14, result_column="RSI_14")
pd_df.tail(10)
```

![RSI](https://github.com/coding-kitties/PyIndicators/blob/main/static/images/indicators/rsi.png)

#### Wilders Relative Strength Index (Wilders RSI)

```python
from investing_algorithm_framework import CSVOHLCVMarketDataSource

from pyindicators import wilders_rsi

# For this example the investing algorithm framework is used for dataframe creation,
csv_path = "./tests/test_data/OHLCV_BTC-EUR_BINANCE_15m_2023-12-01:00:00_2023-12-25:00:00.csv"
data_source = CSVOHLCVMarketDataSource(csv_file_path=csv_path)

pl_df = data_source.get_data()
pd_df = data_source.get_data(pandas=True)

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
from investing_algorithm_framework import CSVOHLCVMarketDataSource

from pyindicators import willr

# For this example the investing algorithm framework is used for dataframe creation,
csv_path = "./tests/test_data/OHLCV_BTC-EUR_BINANCE_15m_2023-12-01:00:00_2023-12-25:00:00.csv"
data_source = CSVOHLCVMarketDataSource(csv_file_path=csv_path)

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

### Indicator helpers

#### Crossover

```python
from polars import DataFrame as plDataFrame
from pandas import DataFrame as pdDataFrame

from investing_algorithm_framework import CSVOHLCVMarketDataSource
from pyindicators import crossover, ema

# For this example the investing algorithm framework is used for dataframe creation,
csv_path = "./tests/test_data/OHLCV_BTC-EUR_BINANCE_15m_2023-12-01:00:00_2023-12-25:00:00.csv"
data_source = CSVOHLCVMarketDataSource(csv_file_path=csv_path)

pl_df = data_source.get_data()
pd_df = data_source.get_data(pandas=True)

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

```python
from polars import DataFrame as plDataFrame
from pandas import DataFrame as pdDataFrame

from investing_algorithm_framework import CSVOHLCVMarketDataSource
from pyindicators import crossover, ema

# For this example the investing algorithm framework is used for dataframe creation,
csv_path = "./tests/test_data/OHLCV_BTC-EUR_BINANCE_15m_2023-12-01:00:00_2023-12-25:00:00.csv"
data_source = CSVOHLCVMarketDataSource(csv_file_path=csv_path)

pl_df = data_source.get_data()
pd_df = data_source.get_data(pandas=True)

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
    pl_df, first_column="EMA_50", second_column="EMA_200", data_points=3
):
    print("Crossover detected in Pandas DataFrame in the last 3 data points")

# If you want to use the result of a previous crossover calculation
if is_crossover(pl_df, crossover_column="Crossover_EMA", data_points=3):
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
    pd_df, first_column="EMA_50", second_column="EMA_200", data_points=3
):
    print("Crossover detected in Pandas DataFrame in the last 3 data points")

# If you want to use the result of a previous crossover calculation
if is_crossover(pd_df, crossover_column="Crossover_EMA", data_points=3):
    print("Crossover detected in Pandas DataFrame in the last 3 data points")
```
