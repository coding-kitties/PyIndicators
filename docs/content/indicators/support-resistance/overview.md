---
title: "Support & Resistance"
sidebar_label: "Overview"
sidebar_position: 0
---

# Support & Resistance

Support and resistance indicators identify key price levels where buying or selling pressure is likely to appear. This category includes both classical techniques (Fibonacci, Golden Zone) and Smart Money Concepts (SMC) / ICT methods (Order Blocks, Liquidity Sweeps, etc.). Most of these are **real-time** — they react to structural price action rather than smoothing history — though they often have a confirmation delay while waiting for swing points to be validated.

<div className="indicator-grid">

<a href="fibonacci-retracement" className="indicator-card">

![Fibonacci Retracement](/img/indicators/fibonacci_retracement.png)

Fibonacci Retracement

</a>

<a href="golden-zone" className="indicator-card">

![Golden Zone](/img/indicators/golden_zone.png)

Golden Zone

</a>

<a href="golden-zone-signal" className="indicator-card">

![Golden Zone Signal](/img/indicators/golden_zone_signal.png)

Golden Zone Signal

</a>

<a href="fair-value-gap" className="indicator-card">

![Fair Value Gap (FVG)](/img/indicators/fair_value_gap.png)

Fair Value Gap (FVG)

</a>

<a href="order-blocks" className="indicator-card">

![Order Blocks](/img/indicators/order_blocks.png)

Order Blocks

</a>

<a href="breaker-blocks" className="indicator-card">

![Breaker Blocks](/img/indicators/breaker_blocks.png)

Breaker Blocks

</a>

<a href="mitigation-blocks" className="indicator-card">

![Mitigation Blocks](/img/indicators/mitigation_blocks.png)

Mitigation Blocks

</a>

<a href="rejection-blocks" className="indicator-card">

![Rejection Blocks](/img/indicators/rejection_blocks.png)

Rejection Blocks

</a>

<a href="optimal-trade-entry" className="indicator-card">

![Optimal Trade Entry (OTE)](/img/indicators/optimal_trade_entry.png)

Optimal Trade Entry (OTE)

</a>

<a href="market-structure-break" className="indicator-card">

![Market Structure Break](/img/indicators/market_structure_ob.png)

Market Structure Break

</a>

<a href="market-structure-choch-bos" className="indicator-card">

![Market Structure CHoCH/BOS](/img/indicators/market_structure_choch_bos.png)

Market Structure CHoCH/BOS

</a>

<a href="liquidity-sweeps" className="indicator-card">

![Liquidity Sweeps](/img/indicators/liquidity_sweeps.png)

Liquidity Sweeps

</a>

<a href="buyside-sellside-liquidity" className="indicator-card">

![Buyside & Sellside Liquidity](/img/indicators/buy_side_sell_side_liquidity.png)

Buyside & Sellside Liquidity

</a>

<a href="pure-price-action-liquidity-sweeps" className="indicator-card">

![Pure Price Action Liquidity Sweeps](/img/indicators/pure_price_action_liquidity_sweeps.png)

Pure Price Action Liquidity Sweeps

</a>

<a href="liquidity-pools" className="indicator-card">

![Liquidity Pools](/img/indicators/liquidity_pools.png)

Liquidity Pools

</a>

<a href="liquidity-levels-voids" className="indicator-card">

![Liquidity Levels / Voids (VP)](/img/indicators/liquidity_levels_voids.png)

Liquidity Levels / Voids (VP)

</a>

<a href="internal-external-liquidity-zones" className="indicator-card">

![Internal & External Liquidity Zones](/img/indicators/internal_external_liquidity_zones.png)

Internal & External Liquidity Zones

</a>

<a href="premium-discount-zones" className="indicator-card">

![Premium / Discount Zones](/img/indicators/premium_discount_zones.png)

Premium / Discount Zones

</a>

</div>

## Indicators at a glance

| Indicator | Type | Warmup | Lag | When to use |
| --- | --- | --- | --- | --- |
| [Fibonacci Retracement](fibonacci-retracement) | 🟢 Real-time | 2 bars | 0 bars | Compute static retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) between a swing high and low. Use it to identify potential support/resistance zones during pullbacks in a trending market. |
| [Golden Zone](golden-zone) | 🔴 Lagging | `length` bars | ≈ `length / 2` bars | Highlights the 61.8%-78.6% Fibonacci zone — the strongest retracement area. Use it as a high-probability entry zone during pullbacks. |
| [Golden Zone Signal](golden-zone-signal) | 🟢 Real-time | Same as Golden Zone (`length` bars) | 0 bars | Generates buy/sell signals when price enters or exits the Golden Zone. Use it as a trigger alongside the Golden Zone overlay for systematic entries. |
| [Fair Value Gap (FVG)](fair-value-gap) | 🟢 Real-time | 3 bars | 0 bars | Detects 3-candle imbalance patterns where institutional order flow left a gap. Use FVGs as high-probability zones where price tends to return to rebalance. |
| [Order Blocks](order-blocks) | 🟢 Real-time | `2 × swing_length + 1` bars | ≈ `swing_length` bars after the pivot | Identifies the last opposing candle before a strong move — the footprint of institutional orders. Use order blocks as support/resistance zones for entries with tight stops. |
| [Breaker Blocks](breaker-blocks) | 🟢 Real-time | `2 × swing_length + 1` bars | ≈ `swing_length` bars after the pivot | Former order blocks that have been broken and flipped. When bulls fail, their order block becomes bearish resistance (and vice versa). Use them for continuation entries after a market-structure shift. |
| [Mitigation Blocks](mitigation-blocks) | 🟢 Real-time | `2 × swing_length + 1` bars | ≈ `swing_length` bars after the pivot | The first same-direction candle in an impulse leg leading to a market-structure shift. Use them as precision entry zones — tighter than order blocks but with a higher hit rate. |
| [Rejection Blocks](rejection-blocks) | 🟢 Real-time | `2 × swing_length + 1` bars | ≈ `swing_length` bars after the pivot | Candles with large wicks at confirmed swing points, showing price rejection. Use them to identify levels where price was strongly pushed back and may react again. |
| [Optimal Trade Entry (OTE)](optimal-trade-entry) | 🟢 Real-time | `2 × swing_length + 1` bars | ≈ `swing_length` bars after the pivot | Fibonacci retracement of an impulse leg after a market-structure shift. Use OTE to time entries at the best risk/reward zone within a confirmed move. |
| [Market Structure Break](market-structure-break) | 🟢 Real-time | `2 × pivot_length + 1` bars | ≈ `pivot_length` bars after the pivot | Detects when price breaks a confirmed pivot high/low with momentum. Use it to identify trend changes and potential reversal points. |
| [Market Structure CHoCH/BOS](market-structure-choch-bos) | 🟢 Real-time | `2 × length + 1` bars | ≈ `length` bars after the fractal | Distinguishes between Change of Character (CHoCH) and Break of Structure (BOS). CHoCH signals a potential reversal; BOS confirms trend continuation. Essential for SMC/ICT trading. |
| [Liquidity Sweeps](liquidity-sweeps) | 🟢 Real-time | `2 × swing_length + 1` bars | ≈ `swing_length` bars after the swing | Detects when price wicks through a swing point and reverses — a classic liquidity grab. Use it to spot institutional stop hunts and trade the reversal. |
| [Buyside & Sellside Liquidity](buyside-sellside-liquidity) | 🟢 Real-time | `2 × detection_length + 1` bars | ≈ `detection_length` bars after the pivot | Maps clusters of resting liquidity above highs (buyside) and below lows (sellside). Use it to anticipate where price is likely to be drawn toward next. |
| [Pure Price Action Liquidity Sweeps](pure-price-action-liquidity-sweeps) | 🟢 Real-time | depth-dependent (varies by fractal depth) | depth-dependent | A multi-depth fractal approach to liquidity sweep detection. Use it when you want to detect sweeps across different structural depths. |
| [Liquidity Pools](liquidity-pools) | 🟢 Real-time | ≥ `contact_count × gap_bars` bars | depends on contact_count + gap_bars | Zones where price wicks have touched multiple times, indicating resting orders. Use them to identify high-probability reversal or acceleration zones. |
| [Liquidity Levels / Voids (VP)](liquidity-levels-voids) | 🟢 Real-time | `detection_length` bars | ≈ `detection_length` bars | Highlights volume-profile voids — price areas with little trading activity. Price tends to move quickly through voids. Use them to spot potential fast-move zones. |
| [Internal & External Liquidity Zones](internal-external-liquidity-zones) | 🟢 Real-time | `2 × external_pivot_length + 1` bars | ≈ `external_pivot_length` bars | Distinguishes between internal (range-bound) and external (breakout) liquidity. Use it to understand whether price is targeting internal or external levels. |
| [Premium / Discount Zones](premium-discount-zones) | 🟢 Real-time | `2 × swing_length + 1` bars | ≈ `swing_length` bars after the swing | Divides the current range into premium (upper) and discount (lower) zones. Buy in discount, sell in premium — the core SMC/ICT framework for directional bias. |

## Detailed descriptions

### [Fibonacci Retracement](fibonacci-retracement)

> 🟢 **Real-time**
>
> **Warmup:** 2 bars (default: 2 bars)

Compute static retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) between a swing high and low. Use it to identify potential support/resistance zones during pullbacks in a trending market.

### [Golden Zone](golden-zone)

> 🔴 **Lagging** — ≈ `length / 2` bars
>
> **Warmup:** `length` bars (default: 60 bars (length=60))

Highlights the 61.8%-78.6% Fibonacci zone — the strongest retracement area. Use it as a high-probability entry zone during pullbacks.

### [Golden Zone Signal](golden-zone-signal)

> 🟢 **Real-time**
>
> **Warmup:** Same as Golden Zone (`length` bars) (default: 60 bars (length=60))

Generates buy/sell signals when price enters or exits the Golden Zone. Use it as a trigger alongside the Golden Zone overlay for systematic entries.

### [Fair Value Gap (FVG)](fair-value-gap)

> 🟢 **Real-time**
>
> **Warmup:** 3 bars (default: 3 bars)

Detects 3-candle imbalance patterns where institutional order flow left a gap. Use FVGs as high-probability zones where price tends to return to rebalance.

### [Order Blocks](order-blocks)

> 🟢 **Real-time** — ≈ `swing_length` bars after the pivot
>
> **Warmup:** `2 × swing_length + 1` bars (default: 21 bars (swing_length=10))

Identifies the last opposing candle before a strong move — the footprint of institutional orders. Use order blocks as support/resistance zones for entries with tight stops.

### [Breaker Blocks](breaker-blocks)

> 🟢 **Real-time** — ≈ `swing_length` bars after the pivot
>
> **Warmup:** `2 × swing_length + 1` bars (default: 11 bars (swing_length=5))

Former order blocks that have been broken and flipped. When bulls fail, their order block becomes bearish resistance (and vice versa). Use them for continuation entries after a market-structure shift.

### [Mitigation Blocks](mitigation-blocks)

> 🟢 **Real-time** — ≈ `swing_length` bars after the pivot
>
> **Warmup:** `2 × swing_length + 1` bars (default: 11 bars (swing_length=5))

The first same-direction candle in an impulse leg leading to a market-structure shift. Use them as precision entry zones — tighter than order blocks but with a higher hit rate.

### [Rejection Blocks](rejection-blocks)

> 🟢 **Real-time** — ≈ `swing_length` bars after the pivot
>
> **Warmup:** `2 × swing_length + 1` bars (default: 11 bars (swing_length=5))

Candles with large wicks at confirmed swing points, showing price rejection. Use them to identify levels where price was strongly pushed back and may react again.

### [Optimal Trade Entry (OTE)](optimal-trade-entry)

> 🟢 **Real-time** — ≈ `swing_length` bars after the pivot
>
> **Warmup:** `2 × swing_length + 1` bars (default: 11 bars (swing_length=5))

Fibonacci retracement of an impulse leg after a market-structure shift. Use OTE to time entries at the best risk/reward zone within a confirmed move.

### [Market Structure Break](market-structure-break)

> 🟢 **Real-time** — ≈ `pivot_length` bars after the pivot
>
> **Warmup:** `2 × pivot_length + 1` bars (default: 15 bars (pivot_length=7))

Detects when price breaks a confirmed pivot high/low with momentum. Use it to identify trend changes and potential reversal points.

### [Market Structure CHoCH/BOS](market-structure-choch-bos)

> 🟢 **Real-time** — ≈ `length` bars after the fractal
>
> **Warmup:** `2 × length + 1` bars (default: 11 bars (length=5))

Distinguishes between Change of Character (CHoCH) and Break of Structure (BOS). CHoCH signals a potential reversal; BOS confirms trend continuation. Essential for SMC/ICT trading.

### [Liquidity Sweeps](liquidity-sweeps)

> 🟢 **Real-time** — ≈ `swing_length` bars after the swing
>
> **Warmup:** `2 × swing_length + 1` bars (default: 11 bars (swing_length=5))

Detects when price wicks through a swing point and reverses — a classic liquidity grab. Use it to spot institutional stop hunts and trade the reversal.

### [Buyside & Sellside Liquidity](buyside-sellside-liquidity)

> 🟢 **Real-time** — ≈ `detection_length` bars after the pivot
>
> **Warmup:** `2 × detection_length + 1` bars (default: 15 bars (detection_length=7))

Maps clusters of resting liquidity above highs (buyside) and below lows (sellside). Use it to anticipate where price is likely to be drawn toward next.

### [Pure Price Action Liquidity Sweeps](pure-price-action-liquidity-sweeps)

> 🟢 **Real-time** — depth-dependent
>
> **Warmup:** depth-dependent (varies by fractal depth) (default: Varies — deeper fractals need more bars)

A multi-depth fractal approach to liquidity sweep detection. Use it when you want to detect sweeps across different structural depths.

### [Liquidity Pools](liquidity-pools)

> 🟢 **Real-time** — depends on contact_count + gap_bars
>
> **Warmup:** ≥ `contact_count × gap_bars` bars (default: Varies (contact_count=2))

Zones where price wicks have touched multiple times, indicating resting orders. Use them to identify high-probability reversal or acceleration zones.

### [Liquidity Levels / Voids (VP)](liquidity-levels-voids)

> 🟢 **Real-time** — ≈ `detection_length` bars
>
> **Warmup:** `detection_length` bars (default: Depends on detection_length)

Highlights volume-profile voids — price areas with little trading activity. Price tends to move quickly through voids. Use them to spot potential fast-move zones.

### [Internal & External Liquidity Zones](internal-external-liquidity-zones)

> 🟢 **Real-time** — ≈ `external_pivot_length` bars
>
> **Warmup:** `2 × external_pivot_length + 1` bars (default: 21 bars (external_pivot_length=10))

Distinguishes between internal (range-bound) and external (breakout) liquidity. Use it to understand whether price is targeting internal or external levels.

### [Premium / Discount Zones](premium-discount-zones)

> 🟢 **Real-time** — ≈ `swing_length` bars after the swing
>
> **Warmup:** `2 × swing_length + 1` bars (default: 21 bars (swing_length=10))

Divides the current range into premium (upper) and discount (lower) zones. Buy in discount, sell in premium — the core SMC/ICT framework for directional bias.

