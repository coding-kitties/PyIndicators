"""
Generate Docusaurus documentation pages from the PyIndicators README.md.

Usage:
    python scripts/generate_docs.py

This script parses README.md and creates individual .md files in the
docs/content/ directory, preserving the indicator grouping structure.
"""

import re
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DOCS_CONTENT = ROOT / "docs" / "content"

# Mapping: (category_folder, slug, sidebar_position) keyed by H4 title
INDICATOR_MAP = {
    # ── Trend ──
    "Weighted Moving Average (WMA)": ("indicators/trend", "wma", 1),
    "Simple Moving Average (SMA)": ("indicators/trend", "sma", 2),
    "Exponential Moving Average (EMA)": ("indicators/trend", "ema", 3),
    "Zero-Lag EMA Envelope (ZLEMA)": (
        "indicators/trend", "zero-lag-ema-envelope", 4
    ),
    "EMA Trend Ribbon": ("indicators/trend", "ema-trend-ribbon", 5),
    "SuperTrend": ("indicators/trend", "supertrend", 6),
    "SuperTrend Clustering": ("indicators/trend", "supertrend-clustering", 7),
    "Pulse Mean Accelerator (PMA)": (
        "indicators/trend", "pulse-mean-accelerator", 8
    ),
    "Volume Weighted Trend (VWT)": (
        "indicators/trend", "volume-weighted-trend", 9
    ),
    # ── Momentum ──
    "Moving Average Convergence Divergence (MACD)": (
        "indicators/momentum", "macd", 1
    ),
    "Relative Strength Index (RSI)": (
        "indicators/momentum", "rsi", 2
    ),
    "Wilders Relative Strength Index (Wilders RSI)": (
        "indicators/momentum", "wilders-rsi", 3
    ),
    "Williams %R": ("indicators/momentum", "williams-r", 4),
    "Average Directional Index (ADX)": ("indicators/momentum", "adx", 5),
    "Stochastic Oscillator (STO)": (
        "indicators/momentum", "stochastic-oscillator", 6
    ),
    "Momentum Confluence": (
        "indicators/momentum", "momentum-confluence", 7
    ),
    # ── Volatility ──
    "Bollinger Bands (BB)": (
        "indicators/volatility", "bollinger-bands", 1
    ),
    "Bollinger Bands Overshoot": (
        "indicators/volatility", "bollinger-overshoot", 2
    ),
    "Average True Range (ATR)": ("indicators/volatility", "atr", 3),
    "Moving Average Envelope (MAE)": (
        "indicators/volatility", "moving-average-envelope", 4
    ),
    "Nadaraya-Watson Envelope (NWE)": (
        "indicators/volatility", "nadaraya-watson-envelope", 5
    ),
    # ── Support & Resistance ──
    "Fibonacci Retracement": (
        "indicators/support-resistance", "fibonacci-retracement", 1
    ),
    "Golden Zone": (
        "indicators/support-resistance", "golden-zone", 2
    ),
    "Golden Zone Signal": (
        "indicators/support-resistance", "golden-zone-signal", 3
    ),
    "Fair Value Gap (FVG)": (
        "indicators/support-resistance", "fair-value-gap", 4
    ),
    "Order Blocks": (
        "indicators/support-resistance", "order-blocks", 5
    ),
    "Breaker Blocks": (
        "indicators/support-resistance", "breaker-blocks", 6
    ),
    "Mitigation Blocks": (
        "indicators/support-resistance", "mitigation-blocks", 7
    ),
    "Rejection Blocks": (
        "indicators/support-resistance", "rejection-blocks", 8
    ),
    "Optimal Trade Entry (OTE)": (
        "indicators/support-resistance", "optimal-trade-entry", 9
    ),
    "Market Structure Break": (
        "indicators/support-resistance", "market-structure-break", 10
    ),
    "Market Structure CHoCH/BOS": (
        "indicators/support-resistance", "market-structure-choch-bos", 11
    ),
    "Liquidity Sweeps": (
        "indicators/support-resistance", "liquidity-sweeps", 12
    ),
    "Buyside & Sellside Liquidity": (
        "indicators/support-resistance", "buyside-sellside-liquidity", 13
    ),
    "Pure Price Action Liquidity Sweeps": (
        "indicators/support-resistance",
        "pure-price-action-liquidity-sweeps", 14
    ),
    "Liquidity Pools": (
        "indicators/support-resistance", "liquidity-pools", 15
    ),
    "Liquidity Levels / Voids (VP)": (
        "indicators/support-resistance", "liquidity-levels-voids", 16
    ),
    "Internal & External Liquidity Zones": (
        "indicators/support-resistance",
        "internal-external-liquidity-zones", 17
    ),
    "Premium / Discount Zones": (
        "indicators/support-resistance", "premium-discount-zones", 18
    ),
    "Trendline Breakout Navigator": (
        "indicators/support-resistance",
        "trendline-breakout-navigator", 19
    ),
    # ── Pattern Recognition ──
    "Detect Peaks": (
        "indicators/pattern-recognition", "detect-peaks", 1
    ),
    "Detect Bullish Divergence": (
        "indicators/pattern-recognition", "bullish-divergence", 2
    ),
    "Detect Bearish Divergence": (
        "indicators/pattern-recognition", "bearish-divergence", 3
    ),
    # ── Helpers ──
    "Crossover": ("indicators/helpers", "crossover", 1),
    "Is Crossover": ("indicators/helpers", "is-crossover", 2),
    "Crossunder": ("indicators/helpers", "crossunder", 3),
    "Is Crossunder": ("indicators/helpers", "is-crossunder", 4),
    "Is Downtrend": ("indicators/helpers", "is-downtrend", 5),
    "Is Uptrend": ("indicators/helpers", "is-uptrend", 6),
    "has_any_lower_then_threshold": (
        "indicators/helpers", "has-any-lower-then-threshold", 7
    ),
}

# Map image filenames referenced in README to Docusaurus static path
IMAGE_RE = re.compile(
    r'!\[([^\]]*)\]\(static/images/indicators/([^)]+)\)'
)

# Lag / real-time classification **and warmup requirements** for every
# indicator.
#
# Each entry is a dict:
#   "type":      "Lagging" | "Real-time"
#   "summary":   one-line plain-English summary
#   "warmup":    dict describing bars needed before first valid value
#       "bars":       formula string (parameter-relative)
#       "default":    concrete bar count with the default parameters
#       "explanation": why this many bars are needed
#   "events":    list of (event_description, lag_bars, explanation)
#       event_description: what the quant developer is watching for
#       lag_bars:          formula / range (parameter-relative!)
#       explanation:       why this lag exists
#   "formula":   (optional) how to compute lag for custom params
#   "realtime_after_warmup": bool — True when the indicator updates
#       on every new bar once the warmup window is filled
#
INDICATOR_LAG = {
    # ── Trend ──
    "Weighted Moving Average (WMA)": {
        "type": "Lagging",
        "summary": "The WMA line moves with a delay behind price.",
        "warmup": {
            "bars": "`period` bars",
            "default": "200 bars (period=200)",
            "explanation": (
                "The first valid WMA value appears once "
                "`period` bars of close data are available. "
                "After the warmup, the indicator updates in "
                "real-time on every new bar."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Line reacts to a price reversal",
             "≈ `(period − 1) / 3` bars",
             "Center of gravity of the linear-weight window "
             "sits at ⅓ of the period"),
            ("Line crosses price (trend confirmation)",
             "≈ `(period − 1) / 3` bars",
             "Crossover inherits the same smoothing delay"),
        ],
        "formula": "lag ≈ (period − 1) / 3",
    },
    "Simple Moving Average (SMA)": {
        "type": "Lagging",
        "summary": "The SMA line moves with a delay behind price.",
        "warmup": {
            "bars": "`period` bars",
            "default": "200 bars (period=200)",
            "explanation": (
                "The first valid SMA value appears once "
                "`period` bars of close data are available. "
                "After the warmup, the indicator updates in "
                "real-time on every new bar."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Line reacts to a price reversal",
             "≈ `(period − 1) / 2` bars",
             "Uniform rolling window; center of gravity "
             "sits at the midpoint of the window"),
            ("Line crosses price (trend confirmation)",
             "≈ `(period − 1) / 2` bars",
             "Crossover inherits the same smoothing delay"),
        ],
        "formula": "lag ≈ (period − 1) / 2",
    },
    "Exponential Moving Average (EMA)": {
        "type": "Lagging",
        "summary": "The EMA line moves with a delay behind price.",
        "warmup": {
            "bars": "`period` bars",
            "default": "200 bars (period=200)",
            "explanation": (
                "The first valid EMA value appears once "
                "`period` bars of close data are available "
                "(seeded from the first close). After the "
                "warmup, the indicator updates in real-time "
                "on every new bar."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Line reacts to a price reversal",
             "≈ `(period − 1) / 2` bars",
             "Exponential decay weights recent bars more, "
             "but effective lag is still (period−1)/2"),
            ("Line crosses price (trend confirmation)",
             "≈ `(period − 1) / 2` bars",
             "Crossover inherits the smoothing delay"),
        ],
        "formula": "lag ≈ (period − 1) / 2",
    },
    "Zero-Lag EMA Envelope (ZLEMA)": {
        "type": "Lagging",
        "summary": "The center line has near-zero lag; the ATR-based "
                   "bands still lag.",
        "warmup": {
            "bars": "`length` bars",
            "default": "200 bars (length=200)",
            "explanation": (
                "The ZLEMA center line needs `length` bars "
                "to initialize. The ATR bands additionally "
                "need `atr_length` bars. After warmup, both "
                "update in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Center line reacts to price reversal",
             "≈ 0 bars",
             "ZLEMA compensates EMA lag via close + "
             "(close − close[lag])"),
            ("Upper/lower bands react to volatility change",
             "≈ `atr_length / 2` bars",
             "Bands are offset by ATR; ATR smoothing "
             "introduces lag of atr_length / 2"),
        ],
        "formula": "center ≈ 0; bands ≈ atr_length / 2",
    },
    "EMA Trend Ribbon": {
        "type": "Lagging",
        "summary": "The ribbon fans out / contracts with a delay. "
                   "Lag is dominated by the slowest EMA.",
        "warmup": {
            "bars": "`ema_max` bars",
            "default": "60 bars (ema_max=60)",
            "explanation": (
                "The slowest EMA in the ribbon defines the "
                "warmup. All faster EMAs will have valid "
                "values earlier, but the full ribbon "
                "requires `ema_max` bars. After warmup, "
                "all ribbon lines update in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Fastest EMA reacts to price reversal",
             "≈ `(ema_min − 1) / 2` bars",
             "Fastest EMA (default 8): lag ≈ (8−1)/2 ≈ 4 bars"),
            ("Slowest EMA reacts to price reversal",
             "≈ `(ema_max − 1) / 2` bars",
             "Slowest EMA (default 60): lag ≈ (60−1)/2 ≈ 30 bars"),
            ("Ribbon flips bullish ↔ bearish",
             "≈ `(ema_max − 1) / 2` bars",
             "Trend determined by slope of slowest EMA; "
             "smoothing_period adds minor extra delay"),
        ],
        "formula": "fastest ≈ (ema_min − 1) / 2; "
                   "slowest ≈ (ema_max − 1) / 2",
    },
    "SuperTrend": {
        "type": "Lagging",
        "summary": "Trend flips and buy/sell signals lag behind "
                   "the actual price reversal.",
        "warmup": {
            "bars": "`atr_length` bars",
            "default": "10 bars (atr_length=10)",
            "explanation": (
                "The ATR component needs `atr_length` bars "
                "before the first SuperTrend value is "
                "computed. After warmup, the indicator "
                "updates in real-time on every new bar."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Trend flips bullish ↔ bearish",
             "≈ `atr_length / 2` bars",
             "ATR smoothing is the primary lag source"),
            ("Buy / sell signal fires",
             "≈ `atr_length / 2` bars",
             "Signal fires on the bar the trend flips"),
        ],
        "formula": "lag ≈ atr_length / 2",
    },
    "SuperTrend Clustering": {
        "type": "Lagging",
        "summary": "Same lag as SuperTrend; K-means selects the "
                   "optimal factor but does not change the lag.",
        "warmup": {
            "bars": "`atr_length` bars",
            "default": "14 bars (atr_length=14)",
            "explanation": (
                "Same as SuperTrend — the ATR component "
                "needs `atr_length` bars. The K-means "
                "clustering runs over the full dataset but "
                "does not add to the warmup requirement. "
                "After warmup, the indicator updates in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Trend flips bullish ↔ bearish",
             "≈ `atr_length / 2` bars",
             "ATR smoothing creates the lag"),
            ("Buy / sell signal fires",
             "≈ `atr_length / 2` bars",
             "Signal fires on the bar the trend flips"),
        ],
        "formula": "lag ≈ atr_length / 2",
    },
    "Pulse Mean Accelerator (PMA)": {
        "type": "Lagging",
        "summary": "The PMA line and trend signals lag behind "
                   "price reversals.",
        "warmup": {
            "bars": "`max(ma_length, accel_lookback)` bars",
            "default": "32 bars (ma_length=20, accel_lookback=32)",
            "explanation": (
                "Both the base moving average (`ma_length`) "
                "and the acceleration lookback need to fill "
                "before the first PMA value is valid. After "
                "warmup, the indicator updates in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("MA base line reacts to price reversal",
             "≈ `ma_length / 2` bars",
             "RMA has effective lag ≈ ma_length/2"),
            ("PMA line changes direction",
             "≈ `ma_length / 2` to `accel_lookback / 2` bars",
             "Acceleration lookback modulates the offset, "
             "adding a variable lag component"),
            ("Trend flips bullish ↔ bearish",
             "≈ `ma_length / 2` to `accel_lookback / 2` bars",
             "Trend derived from PMA slope vs MA"),
        ],
        "formula": "lag ≈ ma_length / 2  (+ accel influence)",
    },
    "Volume Weighted Trend (VWT)": {
        "type": "Lagging",
        "summary": "The VWMA center line and trend signals lag "
                   "behind price reversals.",
        "warmup": {
            "bars": "`vwma_length` bars",
            "default": "34 bars (vwma_length=34)",
            "explanation": (
                "The VWMA needs `vwma_length` bars of price "
                "and volume data. The ATR band also uses "
                "the same period. After warmup, the "
                "indicator updates in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("VWMA center line reacts to price reversal",
             "≈ `vwma_length / 2` bars",
             "VWMA has lag ≈ vwma_length/2"),
            ("Upper/lower bands react",
             "≈ `vwma_length / 2` bars",
             "Bands use ATR with same period as VWMA"),
            ("Trend flips bullish ↔ bearish",
             "≈ `vwma_length / 2` bars",
             "Trend determined by close vs VWMA"),
            ("Buy / sell signal fires",
             "≈ `vwma_length / 2` bars",
             "Signal fires on the bar the trend flips"),
        ],
        "formula": "lag ≈ vwma_length / 2",
    },
    # ── Momentum ──
    "Moving Average Convergence Divergence (MACD)": {
        "type": "Lagging",
        "summary": "MACD line, histogram, and signal line all "
                   "lag behind price moves.",
        "warmup": {
            "bars": "`long_period + signal_period` bars",
            "default": "35 bars (long_period=26, signal_period=9)",
            "explanation": (
                "The slow EMA needs `long_period` bars, "
                "then the signal line EMA needs "
                "`signal_period` additional bars on top. "
                "After warmup, all MACD components update "
                "in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("MACD line crosses zero (trend change)",
             "≈ `long_period / 2` bars",
             "Dominated by the slow EMA"),
            ("MACD histogram changes sign",
             "≈ `long_period / 2` bars",
             "Histogram = MACD − Signal; inherits MACD lag"),
            ("Signal line crossover (buy/sell trigger)",
             "≈ `long_period / 2 + signal_period / 2` bars",
             "Signal is EMA of MACD; adds extra smoothing "
             "on top of MACD lag"),
        ],
        "formula": "MACD ≈ long_period / 2; "
                   "signal ≈ long_period / 2 + signal_period / 2",
    },
    "Relative Strength Index (RSI)": {
        "type": "Lagging",
        "summary": "RSI readings lag behind the actual momentum "
                   "shift in price.",
        "warmup": {
            "bars": "`period` bars",
            "default": "14 bars (period=14)",
            "explanation": (
                "RSI needs `period` bars to compute the "
                "initial average gain/loss. After the "
                "warmup, RSI updates in real-time on every "
                "new bar."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("RSI reaches overbought (>70) / oversold (<30)",
             "≈ `period` bars",
             "Rolling mean of gains/losses over the "
             "specified period"),
            ("RSI crosses 50 (trend confirmation)",
             "≈ `period` bars",
             "Same smoothing window applies"),
        ],
        "formula": "lag ≈ period",
    },
    "Wilders Relative Strength Index (Wilders RSI)": {
        "type": "Lagging",
        "summary": "Wilder's smoothing makes this RSI variant "
                   "significantly slower than standard RSI.",
        "warmup": {
            "bars": "`period` bars",
            "default": "14 bars (period=14)",
            "explanation": (
                "Like standard RSI, the initial average "
                "gain/loss needs `period` bars. However, the "
                "Wilder's RMA smoothing (alpha=1/period) "
                "means the effective lag is ~2× period. "
                "After warmup, the indicator updates in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("RSI reaches overbought / oversold",
             "≈ `2 × period` bars",
             "Wilder's RMA (alpha=1/period) is equivalent "
             "to EMA(2×period−1)"),
            ("RSI crosses 50 (trend confirmation)",
             "≈ `2 × period` bars",
             "Same double-period effective lag"),
        ],
        "formula": "lag ≈ 2 × period",
    },
    "Williams %R": {
        "type": "Lagging",
        "summary": "Williams %R readings lag behind price extremes.",
        "warmup": {
            "bars": "`period` bars",
            "default": "14 bars (period=14)",
            "explanation": (
                "Needs `period` bars to establish the "
                "highest-high and lowest-low window. After "
                "warmup, the indicator updates in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Oscillator reaches overbought (>−20) / "
             "oversold (<−80)",
             "≈ `period / 2` bars",
             "Rolling highest-high / lowest-low over "
             "the specified period"),
            ("Oscillator crosses −50 midline",
             "≈ `period / 2` bars",
             "Same rolling window applies"),
        ],
        "formula": "lag ≈ period / 2",
    },
    "Average Directional Index (ADX)": {
        "type": "Lagging",
        "summary": "ADX is double-smoothed, making it one of the "
                   "slowest momentum indicators.",
        "warmup": {
            "bars": "`2 × period` bars",
            "default": "28 bars (period=14)",
            "explanation": (
                "First Wilder's smoothing on +DI/−DI needs "
                "`period` bars, then the second smoothing "
                "for ADX itself needs another `period` bars. "
                "After warmup, the indicator updates in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("ADX rises above 25 (trend strengthening)",
             "≈ `2 × period` bars",
             "Double Wilder's smoothing: first on DI, "
             "then on ADX"),
            ("+DI / −DI crossover (direction change)",
             "≈ `period` bars",
             "DI lines have single Wilder's smoothing"),
        ],
        "formula": "DI ≈ period; ADX ≈ 2 × period",
    },
    "Stochastic Oscillator (STO)": {
        "type": "Lagging",
        "summary": "The %K and %D lines lag behind price momentum.",
        "warmup": {
            "bars": "`k_period + k_slowing + d_period` bars",
            "default": "20 bars (k_period=14, k_slowing=3, d_period=3)",
            "explanation": (
                "The raw %K needs `k_period` bars, then "
                "SMA smoothing adds `k_slowing`, then %D "
                "adds `d_period`. After warmup, both lines "
                "update in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("%K reaches overbought (>80) / oversold (<20)",
             "≈ `k_period / 2 + k_slowing / 2` bars",
             "Rolling HH/LL over k_period, then SMA "
             "smoothing with k_slowing"),
            ("%K / %D crossover (buy/sell signal)",
             "≈ `k_period / 2 + k_slowing / 2 + d_period / 2` bars",
             "d_period adds extra smoothing on top of %K"),
        ],
        "formula": "lag ≈ k_period / 2 + k_slowing / 2  "
                   "(+ d_period / 2 for %D)",
    },
    "Momentum Confluence": {
        "type": "Lagging",
        "summary": "Confluence score and reversal signals lag "
                   "behind the actual momentum shift.",
        "warmup": {
            "bars": "`max(money_flow_length, trend_wave_length)` bars",
            "default": "14 bars (money_flow_length=14)",
            "explanation": (
                "The warmup is determined by the slowest "
                "sub-indicator (RSI, Stochastic, or money "
                "flow). After warmup, the composite score "
                "updates in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Confluence crosses zero (trend change)",
             "≈ `max(money_flow_length, trend_wave_length)` bars",
             "Composite of RSI, Stochastic, and EMA "
             "components; dominated by the slowest"),
            ("Strong reversal signal fires",
             "≈ `max(money_flow_length, trend_wave_length)` bars",
             "Requires multiple sub-indicators to agree"),
            ("Money flow crosses threshold",
             "≈ `money_flow_length` bars",
             "Money flow uses its own smoothing period"),
        ],
        "formula": "lag ≈ max(money_flow_length, "
                   "trend_wave_length)",
    },
    # ── Volatility ──
    "Bollinger Bands (BB)": {
        "type": "Lagging",
        "summary": "The bands and middle line lag behind both "
                   "price and volatility changes.",
        "warmup": {
            "bars": "`period` bars",
            "default": "20 bars (period=20)",
            "explanation": (
                "Both the SMA middle line and the standard "
                "deviation need `period` bars of data. "
                "After warmup, all three bands update in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Middle band (SMA) reacts to price reversal",
             "≈ `period / 2` bars",
             "SMA has lag ≈ period/2"),
            ("Bands widen/narrow after volatility change",
             "≈ `period / 2` bars",
             "Std dev computed over same rolling window"),
            ("Price touches upper/lower band",
             "≈ `period / 2` bars",
             "Bands trail the actual volatility shift"),
        ],
        "formula": "lag ≈ period / 2",
    },
    "Bollinger Bands Overshoot": {
        "type": "Lagging",
        "summary": "Inherits the same lag as Bollinger Bands.",
        "warmup": {
            "bars": "`period` bars",
            "default": "20 bars (period=20)",
            "explanation": (
                "Built on Bollinger Bands — same warmup as "
                "BB. After warmup, the overshoot value "
                "updates in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Overshoot value reacts to price move",
             "≈ `period / 2` bars",
             "Inherits SMA + std dev lag from BB"),
            ("Overshoot crosses zero",
             "≈ `period / 2` bars",
             "Same underlying Bollinger Bands smoothing"),
        ],
        "formula": "lag ≈ period / 2",
    },
    "Average True Range (ATR)": {
        "type": "Lagging",
        "summary": "ATR readings lag behind actual volatility "
                   "changes.",
        "warmup": {
            "bars": "`period` bars",
            "default": "14 bars (period=14)",
            "explanation": (
                "Wilder's RMA needs `period` bars for the "
                "initial average true range. After warmup, "
                "ATR updates in real-time on every new bar."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("ATR reacts to a volatility spike",
             "≈ `period / 2` bars",
             "Wilder's RMA smoothing over the specified "
             "period"),
            ("ATR reacts to volatility contraction",
             "≈ `period / 2` bars",
             "Same smoothing; contractions are also "
             "detected late"),
        ],
        "formula": "lag ≈ period / 2",
    },
    "Moving Average Envelope (MAE)": {
        "type": "Lagging",
        "summary": "The envelope bands lag behind price because "
                   "they are offset from a moving average.",
        "warmup": {
            "bars": "`period` bars",
            "default": "20 bars (period=20)",
            "explanation": (
                "The base moving average needs `period` "
                "bars. Bands are a fixed % offset, so they "
                "have the same warmup. After warmup, all "
                "bands update in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Middle line reacts to price reversal",
             "≈ `period / 2` bars",
             "Base MA has lag ≈ period/2"),
            ("Bands shift after a price move",
             "≈ `period / 2` bars",
             "Bands are fixed % offset from the MA; "
             "they move in lockstep with the MA"),
        ],
        "formula": "lag ≈ period / 2",
    },
    "Nadaraya-Watson Envelope (NWE)": {
        "type": "Lagging",
        "summary": "The kernel regression line and bands lag "
                   "behind price. Bandwidth controls the tradeoff.",
        "warmup": {
            "bars": "`lookback` bars",
            "default": "500 bars (lookback=500)",
            "explanation": (
                "The Gaussian kernel regression uses a "
                "rolling window of `lookback` bars. Before "
                "that many bars are available, the "
                "regression is computed over fewer points. "
                "After warmup, the indicator updates in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("NWE line reacts to price reversal",
             "bandwidth-dependent",
             "Higher bandwidth = more smoothing = "
             "more lag; no fixed bar count"),
            ("Bands widen/narrow after volatility change",
             "bandwidth-dependent",
             "Bands derived from ATR-scaled offsets of "
             "the regression line"),
        ],
        "formula": "lag depends on bandwidth (default 8.0); "
                   "higher = smoother = more lag",
    },
    # ── Support & Resistance ──
    "Fibonacci Retracement": {
        "type": "Real-time",
        "summary": "Levels are computed instantly from the swing "
                   "high/low of the dataset.",
        "warmup": {
            "bars": "2 bars",
            "default": "2 bars",
            "explanation": (
                "Only needs a high and a low to compute "
                "levels. No smoothing, no rolling window."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Retracement levels appear",
             "0 bars",
             "Static calculation from dataset extremes; "
             "no smoothing"),
        ],
    },
    "Golden Zone": {
        "type": "Lagging",
        "summary": "The golden zone boundaries trail price because "
                   "they use a rolling window.",
        "warmup": {
            "bars": "`length` bars",
            "default": "60 bars (length=60)",
            "explanation": (
                "The rolling highest-high / lowest-low "
                "needs `length` bars to fill the window. "
                "After warmup, zone boundaries update in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Zone boundaries shift after new high/low",
             "≈ `length / 2` bars",
             "Rolling highest-high / lowest-low over "
             "the specified length"),
        ],
        "formula": "lag ≈ length / 2",
    },
    "Golden Zone Signal": {
        "type": "Real-time",
        "summary": "Signals fire instantly when price enters or "
                   "exits a pre-computed golden zone.",
        "warmup": {
            "bars": "Same as Golden Zone (`length` bars)",
            "default": "60 bars (length=60)",
            "explanation": (
                "Requires the Golden Zone to be computed "
                "first. Once zones are available, signals "
                "fire in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Signal fires when price enters zone",
             "0 bars",
             "Simple comparison of current close vs "
             "pre-computed zone levels"),
        ],
    },
    "Fair Value Gap (FVG)": {
        "type": "Real-time",
        "summary": "FVGs are detected instantly on the current bar.",
        "warmup": {
            "bars": "3 bars",
            "default": "3 bars",
            "explanation": (
                "FVG is a 3-bar candlestick pattern. "
                "The first possible detection is on bar 3. "
                "No smoothing — fully real-time after that."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("FVG detected",
             "0 bars",
             "3-bar candlestick pattern; detection uses "
             "only the current bar and 2 bars lookback"),
        ],
    },
    "Order Blocks": {
        "type": "Real-time",
        "summary": "Order blocks appear after pivot confirmation, "
                   "not after smoothing delay.",
        "warmup": {
            "bars": "`2 × swing_length + 1` bars",
            "default": "21 bars (swing_length=10)",
            "explanation": (
                "Swing pivots need `swing_length` bars on "
                "each side to be confirmed. After the first "
                "pivot is confirmed, new order blocks can "
                "appear in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Order block zone appears",
             "≈ `swing_length` bars after the pivot",
             "Swing pivots need swing_length bars on "
             "each side to be confirmed"),
            ("Signal fires when price returns to zone",
             "0 bars",
             "Once the zone exists, the signal fires "
             "instantly when price enters it"),
        ],
    },
    "Breaker Blocks": {
        "type": "Real-time",
        "summary": "Breaker blocks appear after pivot confirmation, "
                   "not after smoothing delay.",
        "warmup": {
            "bars": "`2 × swing_length + 1` bars",
            "default": "11 bars (swing_length=5)",
            "explanation": (
                "Like order blocks, breaker blocks need "
                "confirmed swing pivots first. After warmup, "
                "new breaker blocks appear in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Breaker block zone appears",
             "≈ `swing_length` bars after the pivot",
             "Failed order block detected via MSS; "
             "pivot needs swing_length bars to confirm"),
            ("Signal fires when price returns to zone",
             "0 bars",
             "Instant once the zone exists"),
        ],
    },
    "Mitigation Blocks": {
        "type": "Real-time",
        "summary": "Mitigation blocks appear after pivot "
                   "confirmation, not after smoothing delay.",
        "warmup": {
            "bars": "`2 × swing_length + 1` bars",
            "default": "11 bars (swing_length=5)",
            "explanation": (
                "Requires confirmed swing pivots. After "
                "warmup, new mitigation blocks appear in "
                "real-time on every new bar."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Mitigation block zone appears",
             "≈ `swing_length` bars after the pivot",
             "First same-direction candle of impulse "
             "to MSS; pivot needs swing_length bars"),
            ("Signal fires when price returns to zone",
             "0 bars",
             "Instant once the zone exists"),
        ],
    },
    "Rejection Blocks": {
        "type": "Real-time",
        "summary": "Rejection blocks appear after pivot "
                   "confirmation, not after smoothing delay.",
        "warmup": {
            "bars": "`2 × swing_length + 1` bars",
            "default": "11 bars (swing_length=5)",
            "explanation": (
                "Requires confirmed swing pivots. After "
                "warmup, rejection blocks appear in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Rejection block zone appears",
             "≈ `swing_length` bars after the pivot",
             "Wick-ratio candle at confirmed swing point; "
             "pivot needs swing_length bars"),
            ("Signal fires when price returns to zone",
             "0 bars",
             "Instant once the zone exists"),
        ],
    },
    "Optimal Trade Entry (OTE)": {
        "type": "Real-time",
        "summary": "OTE zones appear after swing confirmation, "
                   "not after smoothing delay.",
        "warmup": {
            "bars": "`2 × swing_length + 1` bars",
            "default": "11 bars (swing_length=5)",
            "explanation": (
                "Requires confirmed swing pivots and a "
                "market structure shift. After warmup, "
                "OTE zones appear in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("OTE zone appears",
             "≈ `swing_length` bars after the pivot",
             "Fibonacci retracement of impulse leg after "
             "MSS; depends on swing/zigzag confirmation"),
            ("Signal fires when price enters OTE zone",
             "0 bars",
             "Instant once the zone exists"),
        ],
    },
    "Market Structure Break": {
        "type": "Real-time",
        "summary": "Structure breaks fire after pivot confirmation, "
                   "not after smoothing delay.",
        "warmup": {
            "bars": "`2 × pivot_length + 1` bars",
            "default": "15 bars (pivot_length=7)",
            "explanation": (
                "Pivot points need `pivot_length` bars on "
                "each side. Once the first pivots are "
                "confirmed, break signals fire in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Break signal fires",
             "≈ `pivot_length` bars after the pivot",
             "Pivot needs pivot_length bars on each side; "
             "signal fires when close breaks past the pivot"),
        ],
    },
    "Market Structure CHoCH/BOS": {
        "type": "Real-time",
        "summary": "CHoCH and BOS signals fire after fractal "
                   "confirmation, not after smoothing delay.",
        "warmup": {
            "bars": "`2 × length + 1` bars",
            "default": "11 bars (length=5)",
            "explanation": (
                "Fractal swing points need `length` bars "
                "on each side. After the first fractals "
                "are confirmed, CHoCH/BOS signals fire in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("CHoCH / BOS signal fires",
             "≈ `length` bars after the fractal",
             "Fractal swing points need length bars on "
             "each side; signal fires when close breaks "
             "past the confirmed fractal"),
        ],
    },
    "Liquidity Sweeps": {
        "type": "Real-time",
        "summary": "Sweep signals fire instantly when price wicks "
                   "through a confirmed swing and reverses.",
        "warmup": {
            "bars": "`2 × swing_length + 1` bars",
            "default": "11 bars (swing_length=5)",
            "explanation": (
                "Swing points need `swing_length` bars on "
                "each side to be confirmed. After the first "
                "swings are established, sweep signals fire "
                "in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Swing high/low is confirmed",
             "≈ `swing_length` bars after the swing",
             "Swing needs swing_length bars on each side"),
            ("Sweep signal fires",
             "0 bars after the sweep",
             "Instant: detected on the bar that wicks "
             "through and reverses"),
        ],
    },
    "Buyside & Sellside Liquidity": {
        "type": "Real-time",
        "summary": "Liquidity levels appear after pivot "
                   "confirmation, not after smoothing delay.",
        "warmup": {
            "bars": "`2 × detection_length + 1` bars",
            "default": "15 bars (detection_length=7)",
            "explanation": (
                "Cluster detection of swing pivots needs "
                "`detection_length` bars on each side. "
                "After warmup, new levels appear in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Liquidity level appears",
             "≈ `detection_length` bars after the pivot",
             "Cluster detection of swing pivots"),
            ("Level is swept (signal fires)",
             "0 bars",
             "Instant when price crosses the level"),
        ],
    },
    "Pure Price Action Liquidity Sweeps": {
        "type": "Real-time",
        "summary": "Sweeps fire instantly once recursive fractal "
                   "swings are confirmed.",
        "warmup": {
            "bars": "depth-dependent (varies by fractal depth)",
            "default": "Varies — deeper fractals need more bars",
            "explanation": (
                "Recursive fractal detection with "
                "configurable depth (short/intermediate/"
                "long). Deeper detection needs more bars. "
                "After warmup, sweep signals fire in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Fractal swing is confirmed",
             "depth-dependent",
             "Recursive fractal detection with configurable "
             "depth; deeper = more bars needed"),
            ("Sweep signal fires",
             "0 bars after the sweep",
             "Instant once the swing is confirmed"),
        ],
    },
    "Liquidity Pools": {
        "type": "Real-time",
        "summary": "Pool zones appear after enough wick contacts "
                   "are observed.",
        "warmup": {
            "bars": "≥ `contact_count × gap_bars` bars",
            "default": "Varies (contact_count=2)",
            "explanation": (
                "Zone requires `contact_count` wick touches "
                "with sufficient spacing (`gap_bars`). "
                "The exact warmup depends on when enough "
                "contacts occur. After zones are formed, "
                "signals fire in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Pool zone created",
             "depends on contact_count + gap_bars",
             "Zone requires contact_count wick touches "
             "with sufficient spacing"),
            ("Price enters pool zone (signal)",
             "0 bars",
             "Instant once the zone exists"),
        ],
    },
    "Liquidity Levels / Voids (VP)": {
        "type": "Real-time",
        "summary": "Levels and voids appear after swing "
                   "confirmation.",
        "warmup": {
            "bars": "`detection_length` bars",
            "default": "Depends on detection_length",
            "explanation": (
                "Volume profile is computed between "
                "confirmed swing points. After the first "
                "pair of swings, zones appear in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Level / void zone appears",
             "≈ `detection_length` bars",
             "Volume profile computed between confirmed "
             "swing points"),
            ("Price enters a void (signal)",
             "0 bars",
             "Instant once the zone exists"),
        ],
    },
    "Internal & External Liquidity Zones": {
        "type": "Real-time",
        "summary": "Zones appear after multi-timeframe pivot "
                   "confirmation.",
        "warmup": {
            "bars": "`2 × external_pivot_length + 1` bars",
            "default": "21 bars (external_pivot_length=10)",
            "explanation": (
                "Multi-timeframe pivot analysis waits for "
                "the longest pivot to confirm. After warmup, "
                "zone updates are real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Liquidity zone appears",
             "≈ `external_pivot_length` bars",
             "Multi-TF pivot analysis; delay from "
             "longest pivot confirmation"),
            ("Zone state changes (active → swept/broken)",
             "0 bars",
             "Instant when price crosses the zone"),
        ],
    },
    "Premium / Discount Zones": {
        "type": "Real-time",
        "summary": "Zones update after swing confirmation, "
                   "not after smoothing delay.",
        "warmup": {
            "bars": "`2 × swing_length + 1` bars",
            "default": "21 bars (swing_length=10)",
            "explanation": (
                "Swing high/low confirmation needs "
                "`swing_length` bars on each side. After "
                "warmup, zone boundaries update in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Zone boundaries update",
             "≈ `swing_length` bars after the swing",
             "Zones computed from most recent confirmed "
             "swing high/low"),
            ("Price enters premium / discount zone",
             "0 bars",
             "Instant comparison of close vs zone levels"),
        ],
    },
    # ── Pattern Recognition ──
    "Detect Peaks": {
        "type": "Real-time",
        "summary": "Peaks are confirmed after comparing with "
                   "neighboring bars.",
        "warmup": {
            "bars": "`2 × number_of_neighbors_to_compare + 1` bars",
            "default": "11 bars (number_of_neighbors_to_compare=5)",
            "explanation": (
                "Needs `number_of_neighbors_to_compare` "
                "bars on each side of a candidate peak. "
                "After warmup, new peaks are detected in "
                "real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Peak / trough confirmed",
             "≈ `number_of_neighbors_to_compare` bars",
             "Needs the specified number of bars "
             "on each side to confirm a local extremum"),
        ],
    },
    "Detect Bullish Divergence": {
        "type": "Real-time",
        "summary": "Divergence signals fire after peak "
                   "confirmation.",
        "warmup": {
            "bars": "`2 × number_of_neighbors_to_compare + 1` bars",
            "default": "11 bars (number_of_neighbors_to_compare=5)",
            "explanation": (
                "Requires confirmed peaks in both price "
                "and indicator, inheriting the peak "
                "detection warmup. After warmup, "
                "divergences are detected in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Bullish divergence signal fires",
             "≈ `number_of_neighbors_to_compare` bars "
             "after the peak",
             "Requires confirmed peaks in both price and "
             "indicator; inherits peak detection delay"),
        ],
    },
    "Detect Bearish Divergence": {
        "type": "Real-time",
        "summary": "Divergence signals fire after peak "
                   "confirmation.",
        "warmup": {
            "bars": "`2 × number_of_neighbors_to_compare + 1` bars",
            "default": "11 bars (number_of_neighbors_to_compare=5)",
            "explanation": (
                "Requires confirmed peaks in both price "
                "and indicator, inheriting the peak "
                "detection warmup. After warmup, "
                "divergences are detected in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Bearish divergence signal fires",
             "≈ `number_of_neighbors_to_compare` bars "
             "after the peak",
             "Requires confirmed peaks in both price and "
             "indicator; inherits peak detection delay"),
        ],
    },
    # ── Helpers ──
    "Crossover": {
        "type": "Real-time",
        "summary": "Crossover detection is instant — no "
                   "additional smoothing is applied.",
        "warmup": {
            "bars": "2 bars",
            "default": "2 bars",
            "explanation": (
                "Compares current bar vs previous bar. "
                "No rolling window — works from bar 2 "
                "onward."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Crossover detected",
             "0 bars",
             "Compares current vs previous bar values"),
        ],
    },
    "Is Crossover": {
        "type": "Real-time",
        "summary": "Boolean check is instant.",
        "warmup": {
            "bars": "2 bars",
            "default": "2 bars",
            "explanation": (
                "Single-bar comparison — works from bar 2 "
                "onward."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Crossover condition detected",
             "0 bars",
             "Single-bar comparison"),
        ],
    },
    "Crossunder": {
        "type": "Real-time",
        "summary": "Crossunder detection is instant — no "
                   "additional smoothing is applied.",
        "warmup": {
            "bars": "2 bars",
            "default": "2 bars",
            "explanation": (
                "Compares current bar vs previous bar. "
                "No rolling window — works from bar 2 "
                "onward."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Crossunder detected",
             "0 bars",
             "Compares current vs previous bar values"),
        ],
    },
    "Is Crossunder": {
        "type": "Real-time",
        "summary": "Boolean check is instant.",
        "warmup": {
            "bars": "2 bars",
            "default": "2 bars",
            "explanation": (
                "Single-bar comparison — works from bar 2 "
                "onward."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Crossunder condition detected",
             "0 bars",
             "Single-bar comparison"),
        ],
    },
    "Is Downtrend": {
        "type": "Lagging",
        "summary": "Uses EMA death cross which has very high lag.",
        "warmup": {
            "bars": "`slow_ema_period` bars",
            "default": "200 bars (slow_ema_period=200)",
            "explanation": (
                "The slow EMA needs `slow_ema_period` bars "
                "to initialize. The fast EMA fills much "
                "sooner. After warmup, the trend check "
                "updates in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Downtrend detected (fast EMA < slow EMA)",
             "≈ `(slow_ema_period − 1) / 2` bars",
             "Dominated by the slow EMA's smoothing lag"),
        ],
        "formula": "lag ≈ (slow_ema_period − 1) / 2",
    },
    "Is Uptrend": {
        "type": "Lagging",
        "summary": "Uses EMA golden cross which has very high lag.",
        "warmup": {
            "bars": "`slow_ema_period` bars",
            "default": "200 bars (slow_ema_period=200)",
            "explanation": (
                "The slow EMA needs `slow_ema_period` bars "
                "to initialize. The fast EMA fills much "
                "sooner. After warmup, the trend check "
                "updates in real-time."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Uptrend detected (fast EMA > slow EMA)",
             "≈ `(slow_ema_period − 1) / 2` bars",
             "Dominated by the slow EMA's smoothing lag"),
        ],
        "formula": "lag ≈ (slow_ema_period − 1) / 2",
    },
    "has_any_lower_then_threshold": {
        "type": "Real-time",
        "summary": "Simple value comparison — no smoothing.",
        "warmup": {
            "bars": "1 bar",
            "default": "1 bar",
            "explanation": (
                "Direct comparison of values vs threshold. "
                "No rolling window needed."
            ),
        },
        "realtime_after_warmup": True,
        "events": [
            ("Threshold condition detected",
             "0 bars",
             "Direct comparison of recent values vs "
             "threshold"),
        ],
    },
}

# ── Chart parameters used for each indicator image ────────────
# Keyed by H4 title.  Each value is a list of tuples:
#   (param_name, param_value)
# Only user-facing parameters are listed (result_column etc. omitted).
CHART_PARAMS: dict[str, list[tuple[str, str]]] = {
    # ── Trend ──
    "Weighted Moving Average (WMA)": [
        ("source_column", "Close"),
        ("period", "200"),
    ],
    "Simple Moving Average (SMA)": [
        ("source_column", "Close"),
        ("period", "200"),
    ],
    "Exponential Moving Average (EMA)": [
        ("source_column", "Close"),
        ("period", "200"),
    ],
    "Zero-Lag EMA Envelope (ZLEMA)": [
        ("source_column", "Close"),
        ("length", "200"),
        ("mult", "2.0"),
    ],
    "EMA Trend Ribbon": [
        ("source_column", "Close"),
    ],
    "SuperTrend": [
        ("atr_length", "10"),
        ("factor", "3.0"),
    ],
    "SuperTrend Clustering": [
        ("atr_length", "14"),
        ("min_mult", "2.0"),
        ("max_mult", "6.0"),
        ("step", "0.5"),
        ("perf_alpha", "14.0"),
        ("from_cluster", "best"),
        ("max_data", "500"),
    ],
    "Pulse Mean Accelerator (PMA)": [
        ("ma_type", "RMA"),
        ("ma_length", "20"),
        ("accel_lookback", "32"),
        ("max_accel", "0.2"),
        ("volatility_type", "Standard Deviation"),
        ("smooth_type", "Double Moving Average"),
        ("use_confirmation", "True"),
    ],
    "Volume Weighted Trend (VWT)": [
        ("vwma_length", "34"),
        ("atr_multiplier", "1.5"),
    ],
    # ── Momentum ──
    "Moving Average Convergence Divergence (MACD)": [
        ("source_column", "Close"),
        ("short_period", "12"),
        ("long_period", "26"),
        ("signal_period", "9"),
    ],
    "Relative Strength Index (RSI)": [
        ("source_column", "Close"),
        ("period", "14"),
    ],
    "Wilders Relative Strength Index (Wilders RSI)": [
        ("source_column", "Close"),
        ("period", "14"),
    ],
    "Williams %R": [
        ("period", "14"),
    ],
    "Average Directional Index (ADX)": [
        ("period", "14"),
    ],
    "Stochastic Oscillator (STO)": [
        ("k_period", "14"),
        ("k_slowing", "3"),
        ("d_period", "3"),
    ],
    "Momentum Confluence": [],
    # ── Volatility ──
    "Bollinger Bands (BB)": [
        ("source_column", "Close"),
        ("period", "20"),
        ("std_dev", "2"),
    ],
    "Bollinger Bands Overshoot": [
        ("source_column", "Close"),
        ("period", "20"),
        ("std_dev", "2"),
    ],
    "Average True Range (ATR)": [
        ("source_column", "Close"),
        ("period", "14"),
    ],
    "Moving Average Envelope (MAE)": [
        ("source_column", "Close"),
        ("period", "20"),
        ("percentage", "2.5"),
    ],
    "Nadaraya-Watson Envelope (NWE)": [
        ("source_column", "Close"),
        ("bandwidth", "8.0"),
        ("mult", "3.0"),
        ("lookback", "500"),
    ],
    # ── Support & Resistance ──
    "Fibonacci Retracement": [
        ("high_column", "High"),
        ("low_column", "Low"),
    ],
    "Golden Zone": [],
    "Golden Zone Signal": [],
    "Fair Value Gap (FVG)": [],
    "Order Blocks": [
        ("swing_length", "10"),
    ],
    "Breaker Blocks": [
        ("swing_length", "5"),
    ],
    "Mitigation Blocks": [
        ("swing_length", "5"),
    ],
    "Rejection Blocks": [
        ("swing_length", "5"),
    ],
    "Optimal Trade Entry (OTE)": [
        ("swing_length", "5"),
    ],
    "Market Structure Break": [
        ("pivot_length", "7"),
    ],
    "Market Structure CHoCH/BOS": [
        ("length", "5"),
    ],
    "Liquidity Sweeps": [
        ("swing_length", "5"),
    ],
    "Buyside & Sellside Liquidity": [
        ("detection_length", "7"),
    ],
    "Pure Price Action Liquidity Sweeps": [],
    "Liquidity Pools": [
        ("contact_count", "2"),
    ],
    "Liquidity Levels / Voids (VP)": [],
    "Internal & External Liquidity Zones": [],
    "Premium / Discount Zones": [
        ("swing_length", "10"),
    ],
    # ── Pattern Recognition ──
    "Detect Peaks": [
        ("source_column", "Close"),
        ("number_of_neighbors_to_compare", "5"),
    ],
    "Detect Bullish Divergence": [
        ("first_column", "Close"),
        ("second_column", "RSI_14"),
    ],
    "Detect Bearish Divergence": [
        ("first_column", "Close"),
        ("second_column", "RSI_14"),
    ],
    # ── Helpers ──
    "Crossover": [
        ("first_column", "SMA_50"),
        ("second_column", "SMA_200"),
    ],
    "Crossunder": [
        ("first_column", "SMA_50"),
        ("second_column", "SMA_200"),
    ],
    "has_any_lower_then_threshold": [
        ("source_column", "RSI_14"),
        ("threshold", "30"),
    ],
}

# Maps doc slugs to image filenames for indicators where the mapping
# is not a simple hyphen→underscore conversion.  Only exceptions are
# listed; for all others the slug is converted automatically.
SLUG_TO_IMAGE = {
    "williams-r": "willr.png",
    "stochastic-oscillator": "sto.png",
    "market-structure-break": "market_structure_ob.png",
    "buyside-sellside-liquidity": "buy_side_sell_side_liquidity.png",
}

# Slugs that have no chart image at all
NO_IMAGE_SLUGS = {
    "is-crossover", "is-crossunder", "is-downtrend", "is-uptrend",
}


def slug_to_image(slug: str) -> str | None:
    """Return the image filename for a doc slug, or None if absent."""
    if slug in NO_IMAGE_SLUGS:
        return None
    if slug in SLUG_TO_IMAGE:
        return SLUG_TO_IMAGE[slug]
    return slug.replace("-", "_") + ".png"


def rewrite_images(content: str) -> str:
    """Rewrite image paths from static/images/… to /img/indicators/…"""
    return IMAGE_RE.sub(r'![\1](/img/indicators/\2)', content)


def escape_jsx_braces(content: str) -> str:
    """Escape {word} patterns outside fenced code blocks and inline code.

    MDX interprets curly braces as JSX expressions.  We wrap bare
    occurrences in inline-code backticks so Docusaurus doesn't choke.
    Patterns already inside ```…``` blocks or `…` spans are left alone.
    """
    fenced = re.compile(r'(```[\s\S]*?```)')
    parts = fenced.split(content)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside a fenced code block – keep as-is
            result.append(part)
        else:
            # Outside fenced code – escape {word} not in inline code
            part = re.sub(
                r'(?<!`)(\{[A-Za-z_]\w*\})(?!`)',
                r'`\1`',
                part,
            )
            result.append(part)
    return ''.join(result)


def parse_sections(readme_text: str):
    """Split README into H4 sections → {title: body_text}."""
    lines = readme_text.split('\n')
    sections = {}
    current_title = None
    current_lines = []

    for line in lines:
        m = re.match(r'^#{4}\s+(.+)$', line)
        if m:
            if current_title is not None:
                sections[current_title] = '\n'.join(current_lines)
            current_title = m.group(1).strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        sections[current_title] = '\n'.join(current_lines)

    return sections


def make_frontmatter(title: str, slug: str, position: int,
                     lag_type: str | None = None) -> str:
    tags = ""
    if lag_type:
        tag = "lagging" if lag_type == "Lagging" else "real-time"
        tags = f"tags: [{tag}]\n"
    return (
        f"---\n"
        f"title: \"{title}\"\n"
        f"sidebar_position: {position}\n"
        f"{tags}"
        f"---\n\n"
    )


def make_warmup_admonition(title: str) -> str:
    """Return a Docusaurus admonition block for warmup window."""
    info = INDICATOR_LAG.get(title)
    if info is None:
        return ""

    warmup = info.get("warmup")
    if warmup is None:
        return ""

    realtime = info.get("realtime_after_warmup", False)

    lines = [
        ":::info[Warmup Window]",
        f"**Minimum bars needed:** {warmup['bars']}",
        f"  (default params: {warmup['default']})",
        "",
        warmup["explanation"],
        "",
    ]

    if realtime:
        lines.append(
            "✅ **After the warmup window is filled, this indicator "
            "produces a new value on every incoming bar in real-time.**"
        )
        lines.append("")

    lines.append(":::")
    return "\n".join(lines) + "\n\n"


def make_lag_admonition(title: str) -> str:
    """Return a Docusaurus admonition block describing lag behaviour."""
    info = INDICATOR_LAG.get(title)
    if info is None:
        return ""

    lag_type = info["type"]
    summary = info["summary"]
    events = info.get("events", [])
    formula = info.get("formula")

    if lag_type == "Lagging":
        lines = [
            ":::caution[Lagging Indicator]",
            f"{summary}",
            "",
        ]
    else:
        lines = [
            ":::tip[Real-time Indicator]",
            f"{summary}",
            "",
        ]

    # Event table
    if events:
        lines.append("| Event | Lag | Detail |")
        lines.append("| --- | --- | --- |")
        for event, lag_bars, explanation in events:
            lines.append(f"| {event} | **{lag_bars}** | {explanation} |")
        lines.append("")

    # Formula for custom params
    if formula:
        lines.append(f"**Formula for custom params:** `{formula}`")
        lines.append("")

    lines.append(":::")
    return "\n".join(lines) + "\n\n"


def make_params_table(title: str) -> str:
    """Return a markdown table of chart parameters for this indicator."""
    params = CHART_PARAMS.get(title)
    if params is None or len(params) == 0:
        return ""

    lines = [
        ":::info[Chart Parameters]",
        "The image above uses the following parameters:",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
    ]
    for name, value in params:
        lines.append(f"| `{name}` | `{value}` |")
    lines.append("")
    lines.append(":::")
    return "\n".join(lines) + "\n\n"


def write_doc(folder: str, slug: str, title: str,
              position: int, body: str):
    out_dir = DOCS_CONTENT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{slug}.md"

    # Clean up body: strip leading/trailing blank lines
    body = body.strip()
    body = rewrite_images(body)
    body = escape_jsx_braces(body)

    lag_info = INDICATOR_LAG.get(title)
    lag_type = lag_info["type"] if lag_info else None
    warmup_admonition = make_warmup_admonition(title)
    lag_admonition = make_lag_admonition(title)
    params_table = make_params_table(title)

    content = (
        make_frontmatter(title, slug, position, lag_type)
        + warmup_admonition
        + lag_admonition
        + body
        + "\n"
        + params_table
    )
    filepath.write_text(content, encoding="utf-8")
    print(f"  Created: {filepath.relative_to(ROOT)}")


def main():
    print("Reading README.md ...")
    readme_text = README.read_text(encoding="utf-8")
    sections = parse_sections(readme_text)

    print(f"Found {len(sections)} H4 sections")
    print(f"Mapping covers {len(INDICATOR_MAP)} indicators\n")

    written = 0
    missing = []

    for title, (folder, slug, position) in INDICATOR_MAP.items():
        if title in sections:
            write_doc(folder, slug, title, position, sections[title])
            written += 1
        else:
            missing.append(title)
            print(f"  WARNING: Section not found: '{title}'")

    print(f"\nWrote {written} doc pages")
    if missing:
        print(f"Missing sections: {missing}")

    # Create introduction page
    intro = DOCS_CONTENT / "introduction.md"
    intro.write_text(
        "---\n"
        "title: Introduction\n"
        "sidebar_position: 1\n"
        "slug: /\n"
        "---\n\n"
        "# PyIndicators\n\n"
        "PyIndicators is a powerful and user-friendly Python library for "
        "financial technical analysis indicators, metrics and helper "
        "functions. Written entirely in Python, it requires no external "
        "dependencies, ensuring seamless integration and ease of use.\n\n"
        "## Marketplace\n\n"
        "We support [Finterion](https://www.finterion.com/) as our go-to "
        "marketplace for quantitative trading and trading bots.\n\n"
        "## Works with the Investing Algorithm Framework\n\n"
        "PyIndicators works natively with the "
        "[Investing Algorithm Framework]"
        "(https://github.com/coding-kitties/investing-algorithm-framework) "
        "for creating trading bots. All indicators accept the DataFrame "
        "format returned by the framework, so you can go from market data "
        "to trading signals without any conversion or glue code.\n\n"
        "```python\n"
        "from investing_algorithm_framework import download\n"
        "from pyindicators import ema, rsi, supertrend\n\n"
        "# Download data directly into a DataFrame\n"
        "df = download(\n"
        "    symbol=\"btc/eur\",\n"
        "    market=\"binance\",\n"
        "    time_frame=\"1d\",\n"
        "    start_date=\"2024-01-01\",\n"
        "    end_date=\"2024-06-01\",\n"
        "    pandas=True,\n"
        "    save=True,\n"
        "    storage_path=\"./data\"\n"
        ")\n\n"
        "# Apply indicators — no conversion needed\n"
        "df = ema(df, source_column=\"Close\", period=200)\n"
        "df = rsi(df, source_column=\"Close\")\n"
        "df = supertrend(df, atr_length=10, factor=3.0)\n"
        "```\n\n"
        "## Features\n\n"
        "* Native Python implementation, no external dependencies needed "
        "except for Polars or Pandas\n"
        "* Dataframe first approach, with support for both pandas "
        "dataframes and polars dataframes\n"
        "* Supports python version 3.10 and above\n"
        "* Over 45 technical indicators covering trend, momentum, "
        "volatility, support/resistance, and pattern recognition\n"
        "* Smart Money Concepts (SMC) / ICT indicators including "
        "Order Blocks, Breaker Blocks, Fair Value Gaps, "
        "Liquidity Sweeps, and more\n",
        encoding="utf-8"
    )
    print("  Created: docs/content/introduction.md")

    # Create installation page
    install = DOCS_CONTENT / "installation.md"
    install.write_text(
        "---\n"
        "title: Installation\n"
        "sidebar_position: 2\n"
        "---\n\n"
        "# Installation\n\n"
        "PyIndicators can be installed using pip:\n\n"
        "```bash\n"
        "pip install pyindicators\n"
        "```\n\n"
        "## Requirements\n\n"
        "- Python 3.10 or above\n"
        "- pandas >= 2.0.0 or polars >= 1.0.0\n"
        "- numpy >= 1.26.4\n"
        "- scipy >= 1.15.1\n",
        encoding="utf-8"
    )
    print("  Created: docs/content/installation.md")

    # Create category overview pages
    write_category_overviews()


# ──────────────────────────────────────────────────────────────
#  Category overview pages
# ──────────────────────────────────────────────────────────────

# Each entry: (folder, filename, sidebar_label, title, intro, indicators)
# indicators is a list of (name, slug, when_to_use)
CATEGORY_OVERVIEWS = [
    (
        "indicators/trend",
        "overview",
        "Overview",
        "Trend Indicators",
        (
            "Trend indicators help you identify the direction and "
            "strength of a market move. They smooth out price noise "
            "so you can focus on the underlying trajectory. Because "
            "they rely on historical averages, **all trend indicators "
            "are lagging** — they confirm a trend rather than predict "
            "it. Use them to align your trades with the dominant "
            "direction and to filter out counter-trend noise."
        ),
        [
            (
                "Weighted Moving Average (WMA)", "wma",
                "When you need a moving average that reacts faster "
                "than SMA because it gives more weight to recent "
                "prices. Good for short-term trend following where "
                "responsiveness matters more than smoothness."
            ),
            (
                "Simple Moving Average (SMA)", "sma",
                "The baseline moving average. Use it for "
                "straightforward trend detection (e.g. price above "
                "SMA 200 = bullish bias), as the foundation for "
                "Bollinger Bands, or as a benchmark to compare "
                "other MAs against."
            ),
            (
                "Exponential Moving Average (EMA)", "ema",
                "The most popular moving average for active trading. "
                "Reacts faster than SMA to recent price changes, "
                "making it ideal for crossover systems, dynamic "
                "support/resistance, and as an input to other "
                "indicators like MACD and SuperTrend."
            ),
            (
                "Zero-Lag EMA Envelope (ZLEMA)",
                "zero-lag-ema-envelope",
                "When standard EMA lag is too high. ZLEMA "
                "compensates for the inherent EMA delay, giving you "
                "an almost zero-lag centre line with ATR-based "
                "bands for volatility envelopes."
            ),
            (
                "EMA Trend Ribbon", "ema-trend-ribbon",
                "Visualize trend strength at a glance using a "
                "ribbon of multiple EMAs. When the ribbon fans out, "
                "the trend is strong; when it compresses, "
                "consolidation or reversal may be coming. Great for "
                "swing trading dashboards."
            ),
            (
                "SuperTrend", "supertrend",
                "A trailing stop and trend filter in one. Use it "
                "to determine trend direction and as a dynamic "
                "stop-loss level. Particularly effective on "
                "trending markets with clear directional moves."
            ),
            (
                "SuperTrend Clustering", "supertrend-clustering",
                "When you want the optimal SuperTrend factor "
                "selected automatically. K-means clustering finds "
                "the best multiplier from price data, removing "
                "guesswork from parameter tuning."
            ),
            (
                "Pulse Mean Accelerator (PMA)",
                "pulse-mean-accelerator",
                "An acceleration-aware moving average that adapts "
                "its offset based on price momentum. Use it when "
                "you want a trend line that tightens during "
                "acceleration and loosens during deceleration."
            ),
            (
                "Volume Weighted Trend (VWT)",
                "volume-weighted-trend",
                "Combines price trend with volume confirmation. "
                "Use it when you want trend signals that are "
                "validated by volume — useful for filtering out "
                "low-conviction moves."
            ),
        ],
    ),
    (
        "indicators/momentum",
        "overview",
        "Overview",
        "Momentum & Oscillators",
        (
            "Momentum indicators measure the speed and strength of "
            "price movements. They oscillate between extremes and "
            "are particularly useful for identifying overbought/"
            "oversold conditions, momentum divergence, and trend "
            "exhaustion. All momentum indicators are **lagging** due "
            "to their smoothing calculations, but they excel at "
            "confirming the quality of a trend and signaling when "
            "it may be losing steam."
        ),
        [
            (
                "Moving Average Convergence Divergence (MACD)",
                "macd",
                "The workhorse momentum indicator. Use it for trend "
                "direction (MACD above zero = bullish), momentum "
                "shifts (signal line crossovers), and divergence "
                "detection. Works best on daily and higher "
                "timeframes."
            ),
            (
                "Relative Strength Index (RSI)", "rsi",
                "The most widely used oscillator. Use it to spot "
                "overbought (>70) and oversold (<30) conditions, "
                "detect momentum divergences, and confirm trend "
                "strength. Versatile across all timeframes."
            ),
            (
                "Wilders Relative Strength Index (Wilders RSI)",
                "wilders-rsi",
                "Wilder's original RSI with heavier smoothing. "
                "Use it when you want fewer but more reliable "
                "overbought/oversold signals. The extra lag filters "
                "out short-lived extremes."
            ),
            (
                "Williams %R", "williams-r",
                "A fast oscillator ideal for timing entries. Use "
                "it in ranging markets to spot when price is near "
                "the top or bottom of its recent range. Works well "
                "in combination with a trend filter."
            ),
            (
                "Average Directional Index (ADX)", "adx",
                "Measures trend strength without regard to "
                "direction. Use ADX > 25 to confirm a strong trend "
                "is in place (and avoid range-bound strategies), "
                "or ADX < 20 to identify consolidation (and avoid "
                "trend-following strategies)."
            ),
            (
                "Stochastic Oscillator (STO)",
                "stochastic-oscillator",
                "Compares closing price to its recent range. Use "
                "it for overbought/oversold signals in sideways "
                "markets, and for %K/%D crossovers as entry "
                "triggers. Combine with a trend filter for best "
                "results."
            ),
            (
                "Momentum Confluence", "momentum-confluence",
                "A composite score that merges RSI, Stochastic, "
                "and EMA-based momentum into a single value. Use "
                "it when you want one number to summarize overall "
                "momentum across multiple sub-indicators."
            ),
        ],
    ),
    (
        "indicators/volatility",
        "overview",
        "Overview",
        "Volatility Indicators",
        (
            "Volatility indicators measure how much price is "
            "moving, regardless of direction. They help you "
            "size positions, set stop-losses, and identify when "
            "markets are unusually quiet (potential breakout) or "
            "loud (potential exhaustion). All volatility indicators "
            "are **lagging** because they smooth historical "
            "price ranges."
        ),
        [
            (
                "Bollinger Bands (BB)", "bollinger-bands",
                "The standard volatility envelope. Bands widen in "
                "volatile markets and contract in quiet ones. Use "
                "them for mean-reversion entries (price touching "
                "outer band), breakout detection (squeeze), and "
                "dynamic support/resistance."
            ),
            (
                "Bollinger Bands Overshoot", "bollinger-overshoot",
                "Measures how far price extends beyond the "
                "Bollinger Bands. Use it to quantify overshoot "
                "extremes and identify high-probability "
                "mean-reversion setups."
            ),
            (
                "Average True Range (ATR)", "atr",
                "The go-to measure of absolute volatility. Use "
                "ATR for position sizing (risk per trade), "
                "trailing stop distances, and as a building block "
                "for other indicators like SuperTrend."
            ),
            (
                "Moving Average Envelope (MAE)",
                "moving-average-envelope",
                "Fixed-percentage bands around a moving average. "
                "Simpler than Bollinger Bands — use it when you "
                "want consistent band width for mean-reversion "
                "or breakout signals."
            ),
            (
                "Nadaraya-Watson Envelope (NWE)",
                "nadaraya-watson-envelope",
                "A non-parametric kernel-regression envelope that "
                "adapts to the shape of price data. Use it when "
                "standard MA-based envelopes don't capture complex "
                "price curves well."
            ),
        ],
    ),
    (
        "indicators/support-resistance",
        "overview",
        "Overview",
        "Support & Resistance",
        (
            "Support and resistance indicators identify key price "
            "levels where buying or selling pressure is likely to "
            "appear. This category includes both classical "
            "techniques (Fibonacci, Golden Zone) and Smart Money "
            "Concepts (SMC) / ICT methods (Order Blocks, Liquidity "
            "Sweeps, etc.). Most of these are **real-time** — they "
            "react to structural price action rather than smoothing "
            "history — though they often have a confirmation delay "
            "while waiting for swing points to be validated."
        ),
        [
            (
                "Fibonacci Retracement", "fibonacci-retracement",
                "Compute static retracement levels (23.6%, 38.2%, "
                "50%, 61.8%, 78.6%) between a swing high and low. "
                "Use it to identify potential support/resistance "
                "zones during pullbacks in a trending market."
            ),
            (
                "Golden Zone", "golden-zone",
                "Highlights the 61.8%-78.6% Fibonacci zone — the "
                "strongest retracement area. Use it as a "
                "high-probability entry zone during pullbacks."
            ),
            (
                "Golden Zone Signal", "golden-zone-signal",
                "Generates buy/sell signals when price enters or "
                "exits the Golden Zone. Use it as a trigger "
                "alongside the Golden Zone overlay for systematic "
                "entries."
            ),
            (
                "Fair Value Gap (FVG)", "fair-value-gap",
                "Detects 3-candle imbalance patterns where "
                "institutional order flow left a gap. Use FVGs as "
                "high-probability zones where price tends to "
                "return to rebalance."
            ),
            (
                "Order Blocks", "order-blocks",
                "Identifies the last opposing candle before a "
                "strong move — the footprint of institutional "
                "orders. Use order blocks as support/resistance "
                "zones for entries with tight stops."
            ),
            (
                "Breaker Blocks", "breaker-blocks",
                "Former order blocks that have been broken and "
                "flipped. When bulls fail, their order block "
                "becomes bearish resistance (and vice versa). Use "
                "them for continuation entries after a "
                "market-structure shift."
            ),
            (
                "Mitigation Blocks", "mitigation-blocks",
                "The first same-direction candle in an impulse leg "
                "leading to a market-structure shift. Use them as "
                "precision entry zones — tighter than order blocks "
                "but with a higher hit rate."
            ),
            (
                "Rejection Blocks", "rejection-blocks",
                "Candles with large wicks at confirmed swing "
                "points, showing price rejection. Use them to "
                "identify levels where price was strongly pushed "
                "back and may react again."
            ),
            (
                "Optimal Trade Entry (OTE)", "optimal-trade-entry",
                "Fibonacci retracement of an impulse leg after a "
                "market-structure shift. Use OTE to time entries "
                "at the best risk/reward zone within a confirmed "
                "move."
            ),
            (
                "Market Structure Break", "market-structure-break",
                "Detects when price breaks a confirmed pivot "
                "high/low with momentum. Use it to identify trend "
                "changes and potential reversal points."
            ),
            (
                "Market Structure CHoCH/BOS",
                "market-structure-choch-bos",
                "Distinguishes between Change of Character (CHoCH) "
                "and Break of Structure (BOS). CHoCH signals a "
                "potential reversal; BOS confirms trend "
                "continuation. Essential for SMC/ICT trading."
            ),
            (
                "Liquidity Sweeps", "liquidity-sweeps",
                "Detects when price wicks through a swing point "
                "and reverses — a classic liquidity grab. Use it "
                "to spot institutional stop hunts and trade the "
                "reversal."
            ),
            (
                "Buyside & Sellside Liquidity",
                "buyside-sellside-liquidity",
                "Maps clusters of resting liquidity above highs "
                "(buyside) and below lows (sellside). Use it to "
                "anticipate where price is likely to be drawn "
                "toward next."
            ),
            (
                "Pure Price Action Liquidity Sweeps",
                "pure-price-action-liquidity-sweeps",
                "A multi-depth fractal approach to liquidity sweep "
                "detection. Use it when you want to detect sweeps "
                "across different structural depths."
            ),
            (
                "Liquidity Pools", "liquidity-pools",
                "Zones where price wicks have touched multiple "
                "times, indicating resting orders. Use them to "
                "identify high-probability reversal or "
                "acceleration zones."
            ),
            (
                "Liquidity Levels / Voids (VP)",
                "liquidity-levels-voids",
                "Highlights volume-profile voids — price areas "
                "with little trading activity. Price tends to move "
                "quickly through voids. Use them to spot potential "
                "fast-move zones."
            ),
            (
                "Internal & External Liquidity Zones",
                "internal-external-liquidity-zones",
                "Distinguishes between internal (range-bound) and "
                "external (breakout) liquidity. Use it to "
                "understand whether price is targeting internal "
                "or external levels."
            ),
            (
                "Premium / Discount Zones",
                "premium-discount-zones",
                "Divides the current range into premium (upper) "
                "and discount (lower) zones. Buy in discount, "
                "sell in premium — the core SMC/ICT framework "
                "for directional bias."
            ),
        ],
    ),
    (
        "indicators/pattern-recognition",
        "overview",
        "Overview",
        "Pattern Recognition",
        (
            "Pattern recognition indicators automatically detect "
            "recurring price structures such as peaks, troughs, "
            "and divergences. They are **real-time** but require "
            "a confirmation delay of several bars to validate "
            "each pattern. Use them for systematic scanning of "
            "setups that would be time-consuming to spot manually."
        ),
        [
            (
                "Detect Peaks", "detect-peaks",
                "Identifies local highs (Higher Highs / Lower "
                "Highs) and lows (Higher Lows / Lower Lows) in "
                "price data. Use it as a building block for trend "
                "analysis, divergence detection, and swing "
                "structure mapping."
            ),
            (
                "Detect Bullish Divergence", "bullish-divergence",
                "Detects when price makes lower lows but an "
                "oscillator (e.g. RSI) makes higher lows — a "
                "classic reversal signal. Use it to spot potential "
                "bottoms in a downtrend."
            ),
            (
                "Detect Bearish Divergence", "bearish-divergence",
                "Detects when price makes higher highs but an "
                "oscillator makes lower highs. Use it to spot "
                "potential tops in an uptrend."
            ),
        ],
    ),
    (
        "indicators/helpers",
        "overview",
        "Overview",
        "Indicator Helpers",
        (
            "Helper functions are utility tools that work alongside "
            "indicators to generate signals and evaluate conditions. "
            "Most are **real-time** with no lag — they simply "
            "compare values on the current bar. Use them to build "
            "composite trading rules from individual indicator "
            "outputs."
        ),
        [
            (
                "Crossover", "crossover",
                "Detects when one series crosses above another "
                "(e.g. SMA 50 crosses above SMA 200 — a golden "
                "cross). Use it to generate entry signals from "
                "any two overlapping indicators."
            ),
            (
                "Is Crossover", "is-crossover",
                "A boolean check: did a crossover happen on the "
                "current bar? Use it in conditional logic when "
                "you only need a True/False answer."
            ),
            (
                "Crossunder", "crossunder",
                "Detects when one series crosses below another "
                "(e.g. SMA 50 crosses below SMA 200 — a death "
                "cross). Use it to generate exit or short signals."
            ),
            (
                "Is Crossunder", "is-crossunder",
                "A boolean check: did a crossunder happen on the "
                "current bar? Use it in conditional logic when "
                "you only need a True/False answer."
            ),
            (
                "Is Downtrend", "is-downtrend",
                "Checks if the market is in a downtrend using "
                "EMA 50 / EMA 200 death cross. Use it as a "
                "directional filter to avoid buying in a "
                "bear market."
            ),
            (
                "Is Uptrend", "is-uptrend",
                "Checks if the market is in an uptrend using "
                "EMA 50 / EMA 200 golden cross. Use it as a "
                "directional filter to avoid selling in a "
                "bull market."
            ),
            (
                "has_any_lower_then_threshold",
                "has-any-lower-then-threshold",
                "Checks if any value in a lookback window is "
                "below a threshold (e.g. RSI < 30). Use it for "
                "conditional rules like 'has RSI been oversold "
                "recently?'."
            ),
        ],
    ),
]


def write_category_overviews():
    """Generate an overview .md page for each indicator category."""
    print("\nWriting category overviews ...")

    for folder, filename, label, title, intro, indicators in \
            CATEGORY_OVERVIEWS:
        out_dir = DOCS_CONTENT / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"{filename}.md"

        lines = [
            "---",
            f"title: \"{title}\"",
            "sidebar_label: \"Overview\"",
            "sidebar_position: 0",
            "---",
            "",
            f"# {title}",
            "",
            intro,
            "",
        ]

        # ── Image grid ──
        lines.append('<div className="indicator-grid">')
        lines.append("")
        for name, slug, _when in indicators:
            img = slug_to_image(slug)
            if img is None:
                continue
            lines.append(f'<a href="{slug}" className="indicator-card">')
            lines.append("")
            lines.append(f"![{name}](/img/indicators/{img})")
            lines.append("")
            lines.append(f"{name}")
            lines.append("")
            lines.append("</a>")
            lines.append("")
        lines.append("</div>")
        lines.append("")

        # ── Detail table ──
        lines.extend([
            "## Indicators at a glance",
            "",
            "| Indicator | Type | Warmup | Lag | When to use |",
            "| --- | --- | --- | --- | --- |",
        ])

        for name, slug, when in indicators:
            lag_info = INDICATOR_LAG.get(name)
            if lag_info:
                lag_type = lag_info["type"]
                # Build a short lag summary from the first event
                events = lag_info.get("events", [])
                if events:
                    lag_desc = events[0][1]  # lag_bars of first event
                else:
                    lag_desc = "—"
                badge = (
                    "🔴 Lagging" if lag_type == "Lagging"
                    else "🟢 Real-time"
                )
                # Warmup column
                warmup = lag_info.get("warmup")
                warmup_desc = warmup["bars"] if warmup else "—"
            else:
                badge = "—"
                lag_desc = "—"
                warmup_desc = "—"

            # Link to the indicator page (relative)
            link = f"[{name}]({slug})"
            # Escape pipes in when_to_use
            when_safe = when.replace("|", "\\|")
            lines.append(
                f"| {link} | {badge} | {warmup_desc} | {lag_desc} "
                f"| {when_safe} |"
            )

        lines.append("")
        lines.append("## Detailed descriptions")
        lines.append("")

        for name, slug, when in indicators:
            lag_info = INDICATOR_LAG.get(name)
            if lag_info:
                lag_type = lag_info["type"]
                events = lag_info.get("events", [])
                first_lag = events[0][1] if events else "—"
                warmup = lag_info.get("warmup")
                if lag_type == "Lagging":
                    badge = f"🔴 **Lagging** — {first_lag}"
                elif first_lag != "0 bars":
                    badge = f"🟢 **Real-time** — {first_lag}"
                else:
                    badge = "🟢 **Real-time**"
                warmup_line = (
                    f"**Warmup:** {warmup['bars']} "
                    f"(default: {warmup['default']})"
                ) if warmup else ""
            else:
                badge = ""
                warmup_line = ""

            lines.append(f"### [{name}]({slug})")
            lines.append("")
            if badge:
                lines.append(f"> {badge}")
                if warmup_line:
                    lines.append(f">")
                    lines.append(f"> {warmup_line}")
                lines.append("")
            lines.append(when)
            lines.append("")

        content = "\n".join(lines) + "\n"
        filepath.write_text(content, encoding="utf-8")
        print(f"  Created: {filepath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
