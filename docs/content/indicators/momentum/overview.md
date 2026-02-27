---
title: "Momentum & Oscillators"
sidebar_label: "Overview"
sidebar_position: 0
---

# Momentum & Oscillators

Momentum indicators measure the speed and strength of price movements. They oscillate between extremes and are particularly useful for identifying overbought/oversold conditions, momentum divergence, and trend exhaustion. All momentum indicators are **lagging** due to their smoothing calculations, but they excel at confirming the quality of a trend and signaling when it may be losing steam.

<div className="indicator-grid">

<a href="macd" className="indicator-card">

![Moving Average Convergence Divergence (MACD)](/img/indicators/macd.png)

Moving Average Convergence Divergence (MACD)

</a>

<a href="rsi" className="indicator-card">

![Relative Strength Index (RSI)](/img/indicators/rsi.png)

Relative Strength Index (RSI)

</a>

<a href="wilders-rsi" className="indicator-card">

![Wilders Relative Strength Index (Wilders RSI)](/img/indicators/wilders_rsi.png)

Wilders Relative Strength Index (Wilders RSI)

</a>

<a href="williams-r" className="indicator-card">

![Williams %R](/img/indicators/willr.png)

Williams %R

</a>

<a href="adx" className="indicator-card">

![Average Directional Index (ADX)](/img/indicators/adx.png)

Average Directional Index (ADX)

</a>

<a href="stochastic-oscillator" className="indicator-card">

![Stochastic Oscillator (STO)](/img/indicators/sto.png)

Stochastic Oscillator (STO)

</a>

<a href="momentum-confluence" className="indicator-card">

![Momentum Confluence](/img/indicators/momentum_confluence.png)

Momentum Confluence

</a>

</div>

## Indicators at a glance

| Indicator | Type | Warmup | Lag | When to use |
| --- | --- | --- | --- | --- |
| [Moving Average Convergence Divergence (MACD)](macd) | 🔴 Lagging | `long_period + signal_period` bars | ≈ `long_period / 2` bars | The workhorse momentum indicator. Use it for trend direction (MACD above zero = bullish), momentum shifts (signal line crossovers), and divergence detection. Works best on daily and higher timeframes. |
| [Relative Strength Index (RSI)](rsi) | 🔴 Lagging | `period` bars | ≈ `period` bars | The most widely used oscillator. Use it to spot overbought (>70) and oversold (<30) conditions, detect momentum divergences, and confirm trend strength. Versatile across all timeframes. |
| [Wilders Relative Strength Index (Wilders RSI)](wilders-rsi) | 🔴 Lagging | `period` bars | ≈ `2 × period` bars | Wilder's original RSI with heavier smoothing. Use it when you want fewer but more reliable overbought/oversold signals. The extra lag filters out short-lived extremes. |
| [Williams %R](williams-r) | 🔴 Lagging | `period` bars | ≈ `period / 2` bars | A fast oscillator ideal for timing entries. Use it in ranging markets to spot when price is near the top or bottom of its recent range. Works well in combination with a trend filter. |
| [Average Directional Index (ADX)](adx) | 🔴 Lagging | `2 × period` bars | ≈ `2 × period` bars | Measures trend strength without regard to direction. Use ADX > 25 to confirm a strong trend is in place (and avoid range-bound strategies), or ADX < 20 to identify consolidation (and avoid trend-following strategies). |
| [Stochastic Oscillator (STO)](stochastic-oscillator) | 🔴 Lagging | `k_period + k_slowing + d_period` bars | ≈ `k_period / 2 + k_slowing / 2` bars | Compares closing price to its recent range. Use it for overbought/oversold signals in sideways markets, and for %K/%D crossovers as entry triggers. Combine with a trend filter for best results. |
| [Momentum Confluence](momentum-confluence) | 🔴 Lagging | `max(money_flow_length, trend_wave_length)` bars | ≈ `max(money_flow_length, trend_wave_length)` bars | A composite score that merges RSI, Stochastic, and EMA-based momentum into a single value. Use it when you want one number to summarize overall momentum across multiple sub-indicators. |

## Detailed descriptions

### [Moving Average Convergence Divergence (MACD)](macd)

> 🔴 **Lagging** — ≈ `long_period / 2` bars
>
> **Warmup:** `long_period + signal_period` bars (default: 35 bars (long_period=26, signal_period=9))

The workhorse momentum indicator. Use it for trend direction (MACD above zero = bullish), momentum shifts (signal line crossovers), and divergence detection. Works best on daily and higher timeframes.

### [Relative Strength Index (RSI)](rsi)

> 🔴 **Lagging** — ≈ `period` bars
>
> **Warmup:** `period` bars (default: 14 bars (period=14))

The most widely used oscillator. Use it to spot overbought (>70) and oversold (<30) conditions, detect momentum divergences, and confirm trend strength. Versatile across all timeframes.

### [Wilders Relative Strength Index (Wilders RSI)](wilders-rsi)

> 🔴 **Lagging** — ≈ `2 × period` bars
>
> **Warmup:** `period` bars (default: 14 bars (period=14))

Wilder's original RSI with heavier smoothing. Use it when you want fewer but more reliable overbought/oversold signals. The extra lag filters out short-lived extremes.

### [Williams %R](williams-r)

> 🔴 **Lagging** — ≈ `period / 2` bars
>
> **Warmup:** `period` bars (default: 14 bars (period=14))

A fast oscillator ideal for timing entries. Use it in ranging markets to spot when price is near the top or bottom of its recent range. Works well in combination with a trend filter.

### [Average Directional Index (ADX)](adx)

> 🔴 **Lagging** — ≈ `2 × period` bars
>
> **Warmup:** `2 × period` bars (default: 28 bars (period=14))

Measures trend strength without regard to direction. Use ADX > 25 to confirm a strong trend is in place (and avoid range-bound strategies), or ADX < 20 to identify consolidation (and avoid trend-following strategies).

### [Stochastic Oscillator (STO)](stochastic-oscillator)

> 🔴 **Lagging** — ≈ `k_period / 2 + k_slowing / 2` bars
>
> **Warmup:** `k_period + k_slowing + d_period` bars (default: 20 bars (k_period=14, k_slowing=3, d_period=3))

Compares closing price to its recent range. Use it for overbought/oversold signals in sideways markets, and for %K/%D crossovers as entry triggers. Combine with a trend filter for best results.

### [Momentum Confluence](momentum-confluence)

> 🔴 **Lagging** — ≈ `max(money_flow_length, trend_wave_length)` bars
>
> **Warmup:** `max(money_flow_length, trend_wave_length)` bars (default: 14 bars (money_flow_length=14))

A composite score that merges RSI, Stochastic, and EMA-based momentum into a single value. Use it when you want one number to summarize overall momentum across multiple sub-indicators.

