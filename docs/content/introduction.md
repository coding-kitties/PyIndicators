---
title: Introduction
sidebar_position: 1
slug: /
---

# PyIndicators

PyIndicators is a powerful and user-friendly Python library for financial technical analysis indicators, metrics and helper functions. Written entirely in Python, it requires no external dependencies, ensuring seamless integration and ease of use.

## Marketplace

We support [Finterion](https://www.finterion.com/) as our go-to marketplace for quantitative trading and trading bots.

## Works with the Investing Algorithm Framework

PyIndicators works natively with the [Investing Algorithm Framework](https://github.com/coding-kitties/investing-algorithm-framework) for creating trading bots. All indicators accept the DataFrame format returned by the framework, so you can go from market data to trading signals without any conversion or glue code.

```python
from investing_algorithm_framework import download
from pyindicators import ema, rsi, supertrend

# Download data directly into a DataFrame
df = download(
    symbol="btc/eur",
    market="binance",
    time_frame="1d",
    start_date="2024-01-01",
    end_date="2024-06-01",
    pandas=True,
    save=True,
    storage_path="./data"
)

# Apply indicators — no conversion needed
df = ema(df, source_column="Close", period=200)
df = rsi(df, source_column="Close")
df = supertrend(df, atr_length=10, factor=3.0)
```

## Features

* Native Python implementation, no external dependencies needed except for Polars or Pandas
* Dataframe first approach, with support for both pandas dataframes and polars dataframes
* Supports python version 3.10 and above
* Over 45 technical indicators covering trend, momentum, volatility, support/resistance, and pattern recognition
* Smart Money Concepts (SMC) / ICT indicators including Order Blocks, Breaker Blocks, Fair Value Gaps, Liquidity Sweeps, and more
