---
title: "Indicator Helpers"
sidebar_label: "Overview"
sidebar_position: 0
---

# Indicator Helpers

Helper functions are utility tools that work alongside indicators to generate signals and evaluate conditions. Most are **real-time** with no lag — they simply compare values on the current bar. Use them to build composite trading rules from individual indicator outputs.

<div className="indicator-grid">

<a href="crossover" className="indicator-card">

![Crossover](/img/indicators/crossover.png)

Crossover

</a>

<a href="crossunder" className="indicator-card">

![Crossunder](/img/indicators/crossunder.png)

Crossunder

</a>

<a href="has-any-lower-then-threshold" className="indicator-card">

![has_any_lower_then_threshold](/img/indicators/has_any_lower_then_threshold.png)

has_any_lower_then_threshold

</a>

</div>

## Indicators at a glance

| Indicator | Type | Warmup | Lag | When to use |
| --- | --- | --- | --- | --- |
| [Crossover](crossover) | 🟢 Real-time | 2 bars | 0 bars | Detects when one series crosses above another (e.g. SMA 50 crosses above SMA 200 — a golden cross). Use it to generate entry signals from any two overlapping indicators. |
| [Is Crossover](is-crossover) | 🟢 Real-time | 2 bars | 0 bars | A boolean check: did a crossover happen on the current bar? Use it in conditional logic when you only need a True/False answer. |
| [Crossunder](crossunder) | 🟢 Real-time | 2 bars | 0 bars | Detects when one series crosses below another (e.g. SMA 50 crosses below SMA 200 — a death cross). Use it to generate exit or short signals. |
| [Is Crossunder](is-crossunder) | 🟢 Real-time | 2 bars | 0 bars | A boolean check: did a crossunder happen on the current bar? Use it in conditional logic when you only need a True/False answer. |
| [Is Downtrend](is-downtrend) | 🔴 Lagging | `slow_ema_period` bars | ≈ `(slow_ema_period − 1) / 2` bars | Checks if the market is in a downtrend using EMA 50 / EMA 200 death cross. Use it as a directional filter to avoid buying in a bear market. |
| [Is Uptrend](is-uptrend) | 🔴 Lagging | `slow_ema_period` bars | ≈ `(slow_ema_period − 1) / 2` bars | Checks if the market is in an uptrend using EMA 50 / EMA 200 golden cross. Use it as a directional filter to avoid selling in a bull market. |
| [has_any_lower_then_threshold](has-any-lower-then-threshold) | 🟢 Real-time | 1 bar | 0 bars | Checks if any value in a lookback window is below a threshold (e.g. RSI < 30). Use it for conditional rules like 'has RSI been oversold recently?'. |

## Detailed descriptions

### [Crossover](crossover)

> 🟢 **Real-time**
>
> **Warmup:** 2 bars (default: 2 bars)

Detects when one series crosses above another (e.g. SMA 50 crosses above SMA 200 — a golden cross). Use it to generate entry signals from any two overlapping indicators.

### [Is Crossover](is-crossover)

> 🟢 **Real-time**
>
> **Warmup:** 2 bars (default: 2 bars)

A boolean check: did a crossover happen on the current bar? Use it in conditional logic when you only need a True/False answer.

### [Crossunder](crossunder)

> 🟢 **Real-time**
>
> **Warmup:** 2 bars (default: 2 bars)

Detects when one series crosses below another (e.g. SMA 50 crosses below SMA 200 — a death cross). Use it to generate exit or short signals.

### [Is Crossunder](is-crossunder)

> 🟢 **Real-time**
>
> **Warmup:** 2 bars (default: 2 bars)

A boolean check: did a crossunder happen on the current bar? Use it in conditional logic when you only need a True/False answer.

### [Is Downtrend](is-downtrend)

> 🔴 **Lagging** — ≈ `(slow_ema_period − 1) / 2` bars
>
> **Warmup:** `slow_ema_period` bars (default: 200 bars (slow_ema_period=200))

Checks if the market is in a downtrend using EMA 50 / EMA 200 death cross. Use it as a directional filter to avoid buying in a bear market.

### [Is Uptrend](is-uptrend)

> 🔴 **Lagging** — ≈ `(slow_ema_period − 1) / 2` bars
>
> **Warmup:** `slow_ema_period` bars (default: 200 bars (slow_ema_period=200))

Checks if the market is in an uptrend using EMA 50 / EMA 200 golden cross. Use it as a directional filter to avoid selling in a bull market.

### [has_any_lower_then_threshold](has-any-lower-then-threshold)

> 🟢 **Real-time**
>
> **Warmup:** 1 bar (default: 1 bar)

Checks if any value in a lookback window is below a threshold (e.g. RSI < 30). Use it for conditional rules like 'has RSI been oversold recently?'.

