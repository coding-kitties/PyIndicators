# PyIndicators

PyIndicators is a powerful and user-friendly Python library for technical analysis indicators and metrics. Written entirely in Python, it requires no external dependencies, ensuring seamless integration and ease of use.

## Features

* Native Python implementation
* Dataframe first approach, with support for both pandas dataframes and polars dataframes

## Indicators

### Trend Indicators

#### Simple Moving Average (SMA)

```python
from polars import DataFrame as plDataFrame
from pandas import DataFrame as pdDataFrame

from pyindicators import sma

# Polars DataFrame
pl_df = plDataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# Pandas DataFrame
pd_df = pdDataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# Calculate SMA for Polars DataFrame
pl_df = sma(pl_df, "close", 3)
pl_df.show(10)

# Calculate SMA for Pandas DataFrame
pd_df = sma(pd_df, "close", 3)
print(pd_df)
```

#### Exponential Moving Average (EMA)

```python
from polars import DataFrame as plDataFrame
from pandas import DataFrame as pdDataFrame

from pyindicators import ema

# Polars DataFrame
pl_df = plDataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
# Pandas DataFrame
pd_df = pdDataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# Calculate EMA for Polars DataFrame
pl_df = ema(pl_df, "close", 3)
pl_df.show(10)

# Calculate EMA for Pandas DataFrame
pd_df = ema(pd_df, "close", 3)
print(pd_df)
```

### Momentum Indicators

#### Relative Strength Index (RSI)

```python
from polars import DataFrame as plDataFrame
from pandas import DataFrame as pdDataFrame

from pyindicators import rsi

# Polars DataFrame
pl_df = plDataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
# Pandas DataFrame
pd_df = pdDataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# Calculate RSI for Polars DataFrame
pl_df = rsi(pl_df, "close", 14)
pl_df.show(10)

# Calculate RSI for Pandas DataFrame
pd_df = rsi(pd_df, "close", 14)
print(pd_df)
```
