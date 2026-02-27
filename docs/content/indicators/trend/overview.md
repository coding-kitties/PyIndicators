---
title: "Trend Indicators"
sidebar_label: "Overview"
sidebar_position: 0
---

# Trend Indicators

Trend indicators help you identify the direction and strength of a market move. They smooth out price noise so you can focus on the underlying trajectory. Because they rely on historical averages, **all trend indicators are lagging** — they confirm a trend rather than predict it. Use them to align your trades with the dominant direction and to filter out counter-trend noise.

<div className="indicator-grid">

<a href="wma" className="indicator-card">

![Weighted Moving Average (WMA)](/img/indicators/wma.png)

Weighted Moving Average (WMA)

</a>

<a href="sma" className="indicator-card">

![Simple Moving Average (SMA)](/img/indicators/sma.png)

Simple Moving Average (SMA)

</a>

<a href="ema" className="indicator-card">

![Exponential Moving Average (EMA)](/img/indicators/ema.png)

Exponential Moving Average (EMA)

</a>

<a href="zero-lag-ema-envelope" className="indicator-card">

![Zero-Lag EMA Envelope (ZLEMA)](/img/indicators/zero_lag_ema_envelope.png)

Zero-Lag EMA Envelope (ZLEMA)

</a>

<a href="ema-trend-ribbon" className="indicator-card">

![EMA Trend Ribbon](/img/indicators/ema_trend_ribbon.png)

EMA Trend Ribbon

</a>

<a href="supertrend" className="indicator-card">

![SuperTrend](/img/indicators/supertrend.png)

SuperTrend

</a>

<a href="supertrend-clustering" className="indicator-card">

![SuperTrend Clustering](/img/indicators/supertrend_clustering.png)

SuperTrend Clustering

</a>

<a href="pulse-mean-accelerator" className="indicator-card">

![Pulse Mean Accelerator (PMA)](/img/indicators/pulse_mean_accelerator.png)

Pulse Mean Accelerator (PMA)

</a>

<a href="volume-weighted-trend" className="indicator-card">

![Volume Weighted Trend (VWT)](/img/indicators/volume_weighted_trend.png)

Volume Weighted Trend (VWT)

</a>

</div>

## Indicators at a glance

| Indicator | Type | Warmup | Lag | When to use |
| --- | --- | --- | --- | --- |
| [Weighted Moving Average (WMA)](wma) | 🔴 Lagging | `period` bars | ≈ `(period − 1) / 3` bars | When you need a moving average that reacts faster than SMA because it gives more weight to recent prices. Good for short-term trend following where responsiveness matters more than smoothness. |
| [Simple Moving Average (SMA)](sma) | 🔴 Lagging | `period` bars | ≈ `(period − 1) / 2` bars | The baseline moving average. Use it for straightforward trend detection (e.g. price above SMA 200 = bullish bias), as the foundation for Bollinger Bands, or as a benchmark to compare other MAs against. |
| [Exponential Moving Average (EMA)](ema) | 🔴 Lagging | `period` bars | ≈ `(period − 1) / 2` bars | The most popular moving average for active trading. Reacts faster than SMA to recent price changes, making it ideal for crossover systems, dynamic support/resistance, and as an input to other indicators like MACD and SuperTrend. |
| [Zero-Lag EMA Envelope (ZLEMA)](zero-lag-ema-envelope) | 🔴 Lagging | `length` bars | ≈ 0 bars | When standard EMA lag is too high. ZLEMA compensates for the inherent EMA delay, giving you an almost zero-lag centre line with ATR-based bands for volatility envelopes. |
| [EMA Trend Ribbon](ema-trend-ribbon) | 🔴 Lagging | `ema_max` bars | ≈ `(ema_min − 1) / 2` bars | Visualize trend strength at a glance using a ribbon of multiple EMAs. When the ribbon fans out, the trend is strong; when it compresses, consolidation or reversal may be coming. Great for swing trading dashboards. |
| [SuperTrend](supertrend) | 🔴 Lagging | `atr_length` bars | ≈ `atr_length / 2` bars | A trailing stop and trend filter in one. Use it to determine trend direction and as a dynamic stop-loss level. Particularly effective on trending markets with clear directional moves. |
| [SuperTrend Clustering](supertrend-clustering) | 🔴 Lagging | `atr_length` bars | ≈ `atr_length / 2` bars | When you want the optimal SuperTrend factor selected automatically. K-means clustering finds the best multiplier from price data, removing guesswork from parameter tuning. |
| [Pulse Mean Accelerator (PMA)](pulse-mean-accelerator) | 🔴 Lagging | `max(ma_length, accel_lookback)` bars | ≈ `ma_length / 2` bars | An acceleration-aware moving average that adapts its offset based on price momentum. Use it when you want a trend line that tightens during acceleration and loosens during deceleration. |
| [Volume Weighted Trend (VWT)](volume-weighted-trend) | 🔴 Lagging | `vwma_length` bars | ≈ `vwma_length / 2` bars | Combines price trend with volume confirmation. Use it when you want trend signals that are validated by volume — useful for filtering out low-conviction moves. |

## Detailed descriptions

### [Weighted Moving Average (WMA)](wma)

> 🔴 **Lagging** — ≈ `(period − 1) / 3` bars
>
> **Warmup:** `period` bars (default: 200 bars (period=200))

When you need a moving average that reacts faster than SMA because it gives more weight to recent prices. Good for short-term trend following where responsiveness matters more than smoothness.

### [Simple Moving Average (SMA)](sma)

> 🔴 **Lagging** — ≈ `(period − 1) / 2` bars
>
> **Warmup:** `period` bars (default: 200 bars (period=200))

The baseline moving average. Use it for straightforward trend detection (e.g. price above SMA 200 = bullish bias), as the foundation for Bollinger Bands, or as a benchmark to compare other MAs against.

### [Exponential Moving Average (EMA)](ema)

> 🔴 **Lagging** — ≈ `(period − 1) / 2` bars
>
> **Warmup:** `period` bars (default: 200 bars (period=200))

The most popular moving average for active trading. Reacts faster than SMA to recent price changes, making it ideal for crossover systems, dynamic support/resistance, and as an input to other indicators like MACD and SuperTrend.

### [Zero-Lag EMA Envelope (ZLEMA)](zero-lag-ema-envelope)

> 🔴 **Lagging** — ≈ 0 bars
>
> **Warmup:** `length` bars (default: 200 bars (length=200))

When standard EMA lag is too high. ZLEMA compensates for the inherent EMA delay, giving you an almost zero-lag centre line with ATR-based bands for volatility envelopes.

### [EMA Trend Ribbon](ema-trend-ribbon)

> 🔴 **Lagging** — ≈ `(ema_min − 1) / 2` bars
>
> **Warmup:** `ema_max` bars (default: 60 bars (ema_max=60))

Visualize trend strength at a glance using a ribbon of multiple EMAs. When the ribbon fans out, the trend is strong; when it compresses, consolidation or reversal may be coming. Great for swing trading dashboards.

### [SuperTrend](supertrend)

> 🔴 **Lagging** — ≈ `atr_length / 2` bars
>
> **Warmup:** `atr_length` bars (default: 10 bars (atr_length=10))

A trailing stop and trend filter in one. Use it to determine trend direction and as a dynamic stop-loss level. Particularly effective on trending markets with clear directional moves.

### [SuperTrend Clustering](supertrend-clustering)

> 🔴 **Lagging** — ≈ `atr_length / 2` bars
>
> **Warmup:** `atr_length` bars (default: 14 bars (atr_length=14))

When you want the optimal SuperTrend factor selected automatically. K-means clustering finds the best multiplier from price data, removing guesswork from parameter tuning.

### [Pulse Mean Accelerator (PMA)](pulse-mean-accelerator)

> 🔴 **Lagging** — ≈ `ma_length / 2` bars
>
> **Warmup:** `max(ma_length, accel_lookback)` bars (default: 32 bars (ma_length=20, accel_lookback=32))

An acceleration-aware moving average that adapts its offset based on price momentum. Use it when you want a trend line that tightens during acceleration and loosens during deceleration.

### [Volume Weighted Trend (VWT)](volume-weighted-trend)

> 🔴 **Lagging** — ≈ `vwma_length / 2` bars
>
> **Warmup:** `vwma_length` bars (default: 34 bars (vwma_length=34))

Combines price trend with volume confirmation. Use it when you want trend signals that are validated by volume — useful for filtering out low-conviction moves.

