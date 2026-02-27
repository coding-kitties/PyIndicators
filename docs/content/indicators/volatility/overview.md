---
title: "Volatility Indicators"
sidebar_label: "Overview"
sidebar_position: 0
---

# Volatility Indicators

Volatility indicators measure how much price is moving, regardless of direction. They help you size positions, set stop-losses, and identify when markets are unusually quiet (potential breakout) or loud (potential exhaustion). All volatility indicators are **lagging** because they smooth historical price ranges.

<div className="indicator-grid">

<a href="bollinger-bands" className="indicator-card">

![Bollinger Bands (BB)](/img/indicators/bollinger_bands.png)

Bollinger Bands (BB)

</a>

<a href="bollinger-overshoot" className="indicator-card">

![Bollinger Bands Overshoot](/img/indicators/bollinger_overshoot.png)

Bollinger Bands Overshoot

</a>

<a href="atr" className="indicator-card">

![Average True Range (ATR)](/img/indicators/atr.png)

Average True Range (ATR)

</a>

<a href="moving-average-envelope" className="indicator-card">

![Moving Average Envelope (MAE)](/img/indicators/moving_average_envelope.png)

Moving Average Envelope (MAE)

</a>

<a href="nadaraya-watson-envelope" className="indicator-card">

![Nadaraya-Watson Envelope (NWE)](/img/indicators/nadaraya_watson_envelope.png)

Nadaraya-Watson Envelope (NWE)

</a>

</div>

## Indicators at a glance

| Indicator | Type | Warmup | Lag | When to use |
| --- | --- | --- | --- | --- |
| [Bollinger Bands (BB)](bollinger-bands) | 🔴 Lagging | `period` bars | ≈ `period / 2` bars | The standard volatility envelope. Bands widen in volatile markets and contract in quiet ones. Use them for mean-reversion entries (price touching outer band), breakout detection (squeeze), and dynamic support/resistance. |
| [Bollinger Bands Overshoot](bollinger-overshoot) | 🔴 Lagging | `period` bars | ≈ `period / 2` bars | Measures how far price extends beyond the Bollinger Bands. Use it to quantify overshoot extremes and identify high-probability mean-reversion setups. |
| [Average True Range (ATR)](atr) | 🔴 Lagging | `period` bars | ≈ `period / 2` bars | The go-to measure of absolute volatility. Use ATR for position sizing (risk per trade), trailing stop distances, and as a building block for other indicators like SuperTrend. |
| [Moving Average Envelope (MAE)](moving-average-envelope) | 🔴 Lagging | `period` bars | ≈ `period / 2` bars | Fixed-percentage bands around a moving average. Simpler than Bollinger Bands — use it when you want consistent band width for mean-reversion or breakout signals. |
| [Nadaraya-Watson Envelope (NWE)](nadaraya-watson-envelope) | 🔴 Lagging | `lookback` bars | bandwidth-dependent | A non-parametric kernel-regression envelope that adapts to the shape of price data. Use it when standard MA-based envelopes don't capture complex price curves well. |

## Detailed descriptions

### [Bollinger Bands (BB)](bollinger-bands)

> 🔴 **Lagging** — ≈ `period / 2` bars
>
> **Warmup:** `period` bars (default: 20 bars (period=20))

The standard volatility envelope. Bands widen in volatile markets and contract in quiet ones. Use them for mean-reversion entries (price touching outer band), breakout detection (squeeze), and dynamic support/resistance.

### [Bollinger Bands Overshoot](bollinger-overshoot)

> 🔴 **Lagging** — ≈ `period / 2` bars
>
> **Warmup:** `period` bars (default: 20 bars (period=20))

Measures how far price extends beyond the Bollinger Bands. Use it to quantify overshoot extremes and identify high-probability mean-reversion setups.

### [Average True Range (ATR)](atr)

> 🔴 **Lagging** — ≈ `period / 2` bars
>
> **Warmup:** `period` bars (default: 14 bars (period=14))

The go-to measure of absolute volatility. Use ATR for position sizing (risk per trade), trailing stop distances, and as a building block for other indicators like SuperTrend.

### [Moving Average Envelope (MAE)](moving-average-envelope)

> 🔴 **Lagging** — ≈ `period / 2` bars
>
> **Warmup:** `period` bars (default: 20 bars (period=20))

Fixed-percentage bands around a moving average. Simpler than Bollinger Bands — use it when you want consistent band width for mean-reversion or breakout signals.

### [Nadaraya-Watson Envelope (NWE)](nadaraya-watson-envelope)

> 🔴 **Lagging** — bandwidth-dependent
>
> **Warmup:** `lookback` bars (default: 500 bars (lookback=500))

A non-parametric kernel-regression envelope that adapts to the shape of price data. Use it when standard MA-based envelopes don't capture complex price curves well.

