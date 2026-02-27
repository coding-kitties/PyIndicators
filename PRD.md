# PyIndicators — Product Requirements Document (PRD)

**Version:** 0.19.0
**Date:** 2026-02-27
**Author:** Carlos (Lead), requested by Marc (Project Owner)
**Status:** Draft — initial comprehensive audit

---

## 1. Vision Statement

PyIndicators is a powerful, DataFrame-first Python library for financial technical analysis. It provides a broad catalog of indicators — from classic trend/momentum tools to advanced ICT-style price-action concepts — that work natively with both **Pandas** and **Polars** DataFrames. The library integrates seamlessly with the [Investing Algorithm Framework](https://github.com/coding-kitties/investing-algorithm-framework), enabling users to go from raw market data to actionable trading signals with zero glue code.

The long-term goal is to become the most complete, well-tested, and well-documented open-source Python indicator library for quantitative and discretionary traders alike.

---

## 2. Current State Summary

| Metric | Count |
|---|---|
| Indicator source modules (`.py` files in `pyindicators/indicators/`) | **56** (incl. utils) |
| Distinct public indicators / indicator groups | **49** (see table below) |
| Helper/utility modules | **7** (crossover, crossunder, is_down_trend, is_up_trend, up_and_down_trends, divergence, utils) |
| Test files in `tests/indicators/` | **43** |
| **Indicators missing tests** | **14** |
| Documentation pages in `docs/content/indicators/` | **~60 .md files across 6 categories** |
| **Indicators missing docs** | **7** |
| Analysis notebooks in `analysis/indicators/` | **22 .ipynb files** |
| **Indicators missing analysis notebooks** | **many** (see table) |
| Python version | ≥ 3.10 |
| Dependencies | pandas ≥2.0, polars ≥1.0, numpy ≥1.26.4, scipy ≥1.15.1, pyarrow ≥12.0, scikit-learn ≥1.3 |

### Key Observations

1. **Test coverage gap:** 14 indicator modules have no corresponding test file — primarily ATR, CCI, ROC, trend helpers, and several liquidity/price-action indicators.
2. **Documentation is strong** but 7 modules lack doc pages — notably Volume Gated Trend Ribbon, Equal Highs/Lows, Swing Structure, Trendline Breakout Navigator, CCI, ROC, and Up & Downtrends.
3. **README is stale:** At least 16 implemented indicators are not listed in the README features section (see §4).
4. **Metadata discrepancy:** The README and pyproject.toml description claim "no external dependencies," but the library depends on pandas, polars, numpy, scipy, pyarrow, and scikit-learn.

---

## 3. Gap Analysis — Full Indicator Matrix

Legend:
✅ = present | ❌ = missing | ➖ = not applicable / covered elsewhere

### 3.1 Trend Indicators

| # | Indicator | Source File | Has Test | Has Docs | Has Notebook | In README |
|---|-----------|------------|----------|----------|--------------|-----------|
| 1 | Simple Moving Average (SMA) | `simple_moving_average.py` | ✅ `test_simple_moving_average` | ✅ `trend/sma.md` | ❌ | ✅ |
| 2 | Weighted Moving Average (WMA) | `weighted_moving_average.py` | ✅ `test_wma` | ✅ `trend/wma.md` | ❌ | ✅ |
| 3 | Exponential Moving Average (EMA) | `exponential_moving_average.py` | ✅ `test_ema` | ✅ `trend/ema.md` | ❌ | ✅ |
| 4 | Zero-Lag EMA Envelope | `zero_lag_ema_envelope.py` | ✅ `test_zero_lag_ema_envelope` | ✅ `trend/zero-lag-ema-envelope.md` | ✅ | ✅ |
| 5 | EMA Trend Ribbon | `ema_trend_ribbon.py` | ✅ `test_ema_trend_ribbon` | ✅ `trend/ema-trend-ribbon.md` | ✅ `trend_ribbon.ipynb` | ✅ |
| 6 | Volume Gated Trend Ribbon | `volume_gated_trend_ribbon.py` | ✅ `test_volume_gated_trend_ribbon` | ❌ | ❌ | ❌ |
| 7 | SuperTrend (incl. Clustering) | `supertrend.py` | ✅ `test_supertrend` | ✅ `trend/supertrend.md`, `supertrend-clustering.md` | ❌ | ✅ |
| 8 | Pulse Mean Accelerator | `pulse_mean_accelerator.py` | ❌ | ✅ `trend/pulse-mean-accelerator.md` | ✅ | ❌ |
| 9 | Volume Weighted Trend | `volume_weighted_trend.py` | ❌ | ✅ `trend/volume-weighted-trend.md` | ✅ | ✅ |

### 3.2 Momentum & Oscillators

| # | Indicator | Source File | Has Test | Has Docs | Has Notebook | In README |
|---|-----------|------------|----------|----------|--------------|-----------|
| 10 | MACD | `macd.py` | ✅ | ✅ `momentum/macd.md` | ❌ | ✅ |
| 11 | RSI | `rsi.py` | ✅ `test_rsi` | ✅ `momentum/rsi.md` | ❌ | ✅ |
| 12 | Wilders RSI | `rsi.py` | ✅ `test_wilders_rsi` | ✅ `momentum/wilders-rsi.md` | ❌ | ✅ |
| 13 | Williams %R | `williams_percent_range.py` | ✅ `test_williams_r` | ✅ `momentum/williams-r.md` | ❌ | ✅ |
| 14 | ADX | `adx.py` | ✅ | ✅ `momentum/adx.md` | ❌ | ✅ |
| 15 | Stochastic Oscillator | `stochastic_oscillator.py` | ✅ | ✅ `momentum/stochastic-oscillator.md` | ❌ | ✅ |
| 16 | Momentum Confluence | `momentum_confluence.py` | ✅ | ✅ `momentum/momentum-confluence.md` | ❌ | ✅ |
| 17 | Momentum Cycle Sentry | `momentum_cycle_sentry.py` | ✅ | ✅ `momentum/momentum-cycle-sentry.md` | ✅ | ❌ |
| 18 | Z-Score Predictive Zones | `z_score_predictive_zones.py` | ✅ | ✅ `momentum/z-score-predictive-zones.md` | ✅ | ❌ |
| 19 | Commodity Channel Index (CCI) | `commodity_channel_index.py` | ❌ | ❌ | ❌ | ❌ |
| 20 | Rate of Change (ROC) | `rate_of_change.py` | ❌ | ❌ | ❌ | ❌ |

### 3.3 Volatility Indicators

| # | Indicator | Source File | Has Test | Has Docs | Has Notebook | In README |
|---|-----------|------------|----------|----------|--------------|-----------|
| 21 | Bollinger Bands (incl. Width, Overshoot) | `bollinger_bands.py` | ✅ | ✅ `volatility/bollinger-bands.md`, `bollinger-overshoot.md` | ❌ | ✅ |
| 22 | Average True Range (ATR) | `average_true_range.py` | ❌ | ✅ `volatility/atr.md` | ❌ | ✅ |
| 23 | Moving Average Envelope | `moving_average_envelope.py` | ✅ | ✅ `volatility/moving-average-envelope.md` | ❌ | ✅ |
| 24 | Nadaraya-Watson Envelope | `nadaraya_watson_envelope.py` | ✅ | ✅ `volatility/nadaraya-watson-envelope.md` | ❌ | ✅ |

### 3.4 Support & Resistance

| # | Indicator | Source File | Has Test | Has Docs | Has Notebook | In README |
|---|-----------|------------|----------|----------|--------------|-----------|
| 25 | Fibonacci Retracement (incl. Extension) | `fibonacci_retracement.py` | ✅ | ✅ `support-resistance/fibonacci-retracement.md` | ❌ | ✅ |
| 26 | Golden Zone (incl. Signal) | `golden_zone.py` | ✅ | ✅ `support-resistance/golden-zone.md`, `golden-zone-signal.md` | ❌ | ✅ |
| 27 | Fair Value Gap | `fair_value_gap.py` | ✅ | ✅ `support-resistance/fair-value-gap.md` | ❌ | ✅ |
| 28 | Order Blocks | `order_blocks.py` | ✅ | ✅ `support-resistance/order-blocks.md` | ❌ | ✅ |
| 29 | Market Structure (Break + CHoCH/BOS) | `market_structure.py` | ✅ | ✅ `support-resistance/market-structure-break.md`, `market-structure-choch-bos.md` | ❌ | ✅ |
| 30 | Breaker Blocks | `breaker_blocks.py` | ✅ | ✅ `support-resistance/breaker-blocks.md` | ✅ | ✅ |
| 31 | Mitigation Blocks | `mitigation_blocks.py` | ✅ | ✅ `support-resistance/mitigation-blocks.md` | ✅ | ✅ |
| 32 | Rejection Blocks | `rejection_blocks.py` | ✅ | ✅ `support-resistance/rejection-blocks.md` | ✅ | ✅ |
| 33 | Optimal Trade Entry | `optimal_trade_entry.py` | ✅ | ✅ `support-resistance/optimal-trade-entry.md` | ✅ | ✅ |
| 34 | Liquidity Sweeps | `liquidity_sweeps.py` | ❌ | ✅ `support-resistance/liquidity-sweeps.md` | ✅ | ✅ |
| 35 | Buyside/Sellside Liquidity | `buyside_sellside_liquidity.py` | ❌ | ✅ `support-resistance/buyside-sellside-liquidity.md` | ✅ | ✅ |
| 36 | Pure Price Action Liquidity Sweeps | `pure_price_action_liquidity_sweeps.py` | ❌ | ✅ `support-resistance/pure-price-action-liquidity-sweeps.md` | ✅ | ✅ |
| 37 | Liquidity Pools | `liquidity_pools.py` | ❌ | ✅ `support-resistance/liquidity-pools.md` | ✅ | ✅ |
| 38 | Liquidity Levels / Voids | `liquidity_levels_voids.py` | ❌ | ✅ `support-resistance/liquidity-levels-voids.md` | ✅ | ✅ |
| 39 | Internal/External Liquidity Zones | `internal_external_liquidity_zones.py` | ✅ | ✅ `support-resistance/internal-external-liquidity-zones.md` | ✅ | ✅ |
| 40 | Premium / Discount Zones | `premium_discount_zones.py` | ✅ | ✅ `support-resistance/premium-discount-zones.md` | ✅ | ✅ |
| 41 | Equal Highs / Lows | `equal_highs_lows.py` | ❌ | ❌ | ✅ | ❌ |
| 42 | Swing Structure | `swing_structure.py` | ✅ | ❌ | ✅ `swing_structure_scanner.ipynb` | ❌ |
| 43 | Volumetric Supply/Demand Zones | `volumetric_supply_demand_zones.py` | ✅ | ✅ `support-resistance/volumetric-supply-demand-zones.md` | ✅ | ❌ |
| 44 | Accumulation Distribution Zones | `accumulation_distribution_zones.py` | ✅ | ✅ `support-resistance/accumulation-distribution-zones.md` | ❌ | ❌ |
| 45 | Volume Imbalance | `volume_imbalance.py` | ✅ | ✅ `support-resistance/volume-imbalance.md` | ❌ | ❌ |
| 46 | Opening Gap | `opening_gap.py` | ✅ | ✅ `support-resistance/opening-gap.md` | ❌ | ❌ |
| 47 | Strong / Weak High / Low | `strong_weak_high_low.py` | ✅ | ✅ `support-resistance/strong-weak-high-low.md` | ❌ | ❌ |
| 48 | Range Intelligence | `range_intelligence.py` | ✅ | ✅ `support-resistance/range-intelligence.md` | ✅ | ❌ |
| 49 | Trendline Breakout Navigator | `trendline_breakout_navigator.py` | ✅ | ❌ | ❌ | ❌ |

### 3.5 Pattern Recognition

| # | Indicator | Source File | Has Test | Has Docs | Has Notebook | In README |
|---|-----------|------------|----------|----------|--------------|-----------|
| 50 | Divergence (Peaks, Bearish, Bullish) | `divergence.py` | ✅ | ✅ `pattern-recognition/detect-peaks.md`, `bearish-divergence.md`, `bullish-divergence.md` | ❌ | ✅ |

### 3.6 Helpers & Utilities

| # | Module | Source File | Has Test | Has Docs | Has Notebook | In README |
|---|--------|------------|----------|----------|--------------|-----------|
| 51 | Crossover / Is Crossover | `crossover.py` | ✅ | ✅ `helpers/crossover.md`, `is-crossover.md` | ➖ | ✅ |
| 52 | Crossunder / Is Crossunder | `crossunder.py` | ✅ | ✅ `helpers/crossunder.md`, `is-crossunder.md` | ➖ | ✅ |
| 53 | Is Downtrend | `is_down_trend.py` | ❌ | ✅ `helpers/is-downtrend.md` | ➖ | ✅ |
| 54 | Is Uptrend | `is_up_trend.py` | ❌ | ✅ `helpers/is-uptrend.md` | ➖ | ✅ |
| 55 | Up and Downtrends | `up_and_down_trends.py` | ❌ | ❌ | ➖ | ❌ |
| 56 | Utils | `utils.py` | ✅ | ✅ (partial: `has-any-lower-then-threshold.md`) | ➖ | ✅ (partial) |

---

## 4. Gaps Summary

### 4.1 Indicators Missing Tests (14)

| Priority | Indicator | Module |
|----------|-----------|--------|
| **High** | Average True Range (ATR) | `average_true_range.py` |
| **High** | Commodity Channel Index (CCI) | `commodity_channel_index.py` |
| **High** | Rate of Change (ROC) | `rate_of_change.py` |
| **High** | Liquidity Sweeps | `liquidity_sweeps.py` |
| **High** | Buyside/Sellside Liquidity | `buyside_sellside_liquidity.py` |
| **High** | Pure Price Action Liquidity Sweeps | `pure_price_action_liquidity_sweeps.py` |
| **High** | Liquidity Pools | `liquidity_pools.py` |
| **High** | Liquidity Levels / Voids | `liquidity_levels_voids.py` |
| **Medium** | Pulse Mean Accelerator | `pulse_mean_accelerator.py` |
| **Medium** | Equal Highs / Lows | `equal_highs_lows.py` |
| **Medium** | Volume Weighted Trend | `volume_weighted_trend.py` |
| **Low** | Is Downtrend | `is_down_trend.py` |
| **Low** | Is Uptrend | `is_up_trend.py` |
| **Low** | Up and Downtrends | `up_and_down_trends.py` |

### 4.2 Indicators Missing Documentation (7)

| Indicator | Category |
|-----------|----------|
| Volume Gated Trend Ribbon | Trend |
| Commodity Channel Index (CCI) | Momentum |
| Rate of Change (ROC) | Momentum |
| Equal Highs / Lows | Support & Resistance |
| Swing Structure | Support & Resistance |
| Trendline Breakout Navigator | Support & Resistance |
| Up and Downtrends | Helpers |

### 4.3 Indicators Missing from README Features List (16+)

These indicators are implemented and exported but **not featured in the README**:

- Volume Gated Trend Ribbon
- Pulse Mean Accelerator
- Momentum Cycle Sentry
- Z-Score Predictive Zones
- Commodity Channel Index (CCI)
- Rate of Change (ROC)
- Equal Highs / Lows
- Swing Structure
- Volumetric Supply/Demand Zones
- Accumulation Distribution Zones
- Volume Imbalance
- Opening Gap
- Strong / Weak High / Low
- Range Intelligence
- Trendline Breakout Navigator
- Bollinger Width (separate from Bollinger Bands entry)

### 4.4 Metadata Discrepancy

The README and `pyproject.toml` description state:

> *"Written entirely in Python, it requires no external dependencies"*

However, `pyproject.toml` lists **6 runtime dependencies**: `pandas`, `polars`, `numpy`, `scipy`, `pyarrow`, `scikit-learn`. The description should be updated to reflect the actual dependency posture (e.g., "requires only common scientific Python packages").

---

## 5. Prioritized Roadmap

### Phase 1 — Test Coverage (Priority: Critical)

Close the 14-indicator test gap. Every public function must have at least basic tests covering:
- Pandas DataFrame input/output
- Polars DataFrame input/output
- Edge cases (empty data, insufficient rows, NaN handling)

| Work Item | Effort |
|-----------|--------|
| Write tests for ATR, CCI, ROC | Small (classic indicators, simple I/O) |
| Write tests for Liquidity Sweeps, Buyside/Sellside, Pure PA Sweeps, Liquidity Pools, Liquidity Levels/Voids | Medium (complex price-action logic) |
| Write tests for Pulse Mean Accelerator, Equal Highs/Lows, Volume Weighted Trend | Medium |
| Write tests for Is Downtrend, Is Uptrend, Up and Downtrends | Small |

### Phase 2 — Documentation Completeness (Priority: High)

Add the 7 missing doc pages and update the README.

| Work Item | Effort |
|-----------|--------|
| Create doc pages for CCI, ROC, Volume Gated Trend Ribbon, Equal Highs/Lows, Swing Structure, Trendline Breakout Navigator, Up and Downtrends | Medium |
| Update README features list to include all 16+ missing indicators | Small |
| Fix "no external dependencies" claim in README and pyproject.toml description | Small |
| Add missing utils docs (only `has_any_lower_then_threshold` is documented; other utils like `is_below`, `is_above`, `get_slope`, etc. are not) | Small |

### Phase 3 — README & Onboarding (Priority: Medium)

| Work Item | Effort |
|-----------|--------|
| Restructure README: reduce inline API docs, link to Docusaurus site instead | Medium |
| Add a "Quick Start" section with a minimal end-to-end example | Small |
| Add badges (PyPI version, test status, docs link) | Small |
| Add CONTRIBUTING.md with indicator authoring guide | Medium |

### Phase 4 — Analysis Notebooks (Priority: Low)

Many classic indicators lack analysis notebooks. These are nice-to-have for demonstrating real-world usage.

**Indicators without analysis notebooks:**
SMA, WMA, EMA, SuperTrend, MACD, RSI, Wilders RSI, Williams %R, ADX, Stochastic Oscillator, Momentum Confluence, CCI, ROC, Bollinger Bands, ATR, Moving Average Envelope, Nadaraya-Watson Envelope, Fibonacci Retracement, Golden Zone, Fair Value Gap, Order Blocks, Market Structure, Divergence, Accumulation Distribution Zones, Volume Imbalance, Opening Gap, Strong/Weak High/Low, Volume Gated Trend Ribbon, Trendline Breakout Navigator.

### Phase 5 — Future Feature Candidates (Priority: Backlog)

| Feature | Notes |
|---------|-------|
| Ichimoku Cloud | Commonly requested trend indicator |
| VWAP (Volume Weighted Average Price) | Essential for intraday analysis |
| Pivot Points (Standard, Camarilla, Woodie) | Classic S/R levels |
| Heikin-Ashi candles | Trend-smoothing candle type |
| Keltner Channels | Volatility envelope alternative |
| Donchian Channels | Breakout detection |
| Parabolic SAR | Trend reversal indicator |
| On-Balance Volume (OBV) | Volume-based trend confirmation |
| Chaikin Money Flow | Volume/momentum hybrid |
| Type annotations audit | Ensure all public APIs have full type hints |
| Performance benchmarks | Comparative pandas vs polars benchmarks |

---

## 6. Non-Goals & Constraints

1. **No real-time / streaming support.** PyIndicators is batch-oriented and operates on complete DataFrames. Streaming/tick-level processing is out of scope.
2. **No built-in charting.** The library computes indicators; visualization is the user's responsibility (analysis notebooks use plotly for demos but this isn't part of the library).
3. **No data fetching.** Market data acquisition is handled by external tools (e.g., Investing Algorithm Framework). PyIndicators only transforms DataFrames.
4. **Pandas + Polars only.** No plans to support other DataFrame libraries (e.g., Dask, Vaex, cuDF).
5. **Python ≥ 3.10 only.** No backport effort for older Python versions.
6. **No C/Rust extensions.** Performance improvements should come from algorithmic optimization and leveraging Polars' native speed, not from compiled extensions.

---

## 7. Technical Standards

### 7.1 Indicator Implementation Pattern

Every indicator module should follow this established pattern:

```
pyindicators/indicators/<indicator_name>.py
```

**Standard structure:**

```python
from typing import Union, Optional
import pandas as pd
import polars as pl

def indicator_name(
    data: Union[pd.DataFrame, pl.DataFrame],
    source_column: str = 'Close',
    # ... indicator-specific parameters ...
    result_column: Optional[str] = None,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Compute the indicator and add result column(s) to the DataFrame."""
    ...
```

**Key conventions:**
- Accept both `pd.DataFrame` and `pl.DataFrame`; return the same type as input.
- Use `source_column` for the primary input series.
- Use `result_column` (with sensible default) for output column naming.
- Complex indicators provide a triplet: `indicator()`, `indicator_signal()`, `get_indicator_stats()`.
- All public functions must be exported in `pyindicators/indicators/__init__.py` and listed in `__all__`.

### 7.2 Testing Pattern

```
tests/indicators/test_<indicator_name>.py
```

**Standard structure:**

```python
import unittest
import pandas as pd
import polars as pl

class Test<IndicatorName>(unittest.TestCase):
    def setUp(self):
        # Load or create test data (both pd and pl)
        ...

    def test_pandas_basic(self):
        # Test with pandas DataFrame
        ...

    def test_polars_basic(self):
        # Test with polars DataFrame
        ...

    def test_edge_cases(self):
        # Empty data, insufficient rows, NaN values
        ...
```

**Key conventions:**
- Every indicator must have both Pandas and Polars tests.
- Use realistic OHLCV data from `tests/test_data/` or `tests/resources/`.
- Assert output shape, column existence, and value correctness.

### 7.3 Documentation Pattern

```
docs/content/indicators/<category>/<indicator-name>.md
```

**Standard structure:**
- Title and one-paragraph description
- Function signature
- Parameters table
- Usage example with code
- Visual output (chart image or reference to analysis notebook)

Documentation is built with **Docusaurus** (see `docs/docusaurus.config.js`). New pages must be registered in `docs/sidebars.js`.

### 7.4 Analysis Notebook Pattern

```
analysis/indicators/<indicator_name>.ipynb
```

**Purpose:** Demonstrate the indicator on real market data with plotly charts. These are for exploration and validation, not part of the published package.

---

## 8. Architecture Notes

```
pyindicators/
├── __init__.py            # Re-exports from indicators/
├── date_range.py          # Date range utilities
├── exceptions.py          # Custom exceptions
└── indicators/
    ├── __init__.py        # Central export hub (294 lines, ~95 exports)
    └── *.py               # One file per indicator (56 files)
```

- **No class hierarchy** — all indicators are pure functions.
- **No internal state** — each call is stateless; the DataFrame is the only state carrier.
- **Flat module structure** — all indicators live in a single `indicators/` directory (no sub-packages by category). This keeps imports simple (`from pyindicators import ema`) but may become unwieldy as the catalog grows.

---

## 9. Acceptance Criteria for "Done"

An indicator is considered **complete** when it has:

- [x] Implementation in `pyindicators/indicators/` with Pandas + Polars support
- [x] Export in `__init__.py` and `__all__`
- [x] Unit tests covering both DataFrame types and edge cases
- [x] Documentation page in `docs/content/indicators/`
- [x] Entry in the README features list
- [ ] *(Nice to have)* Analysis notebook in `analysis/indicators/`

---

## 10. Current Indicator Completeness Scorecard

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Fully complete (impl + test + docs + README) | **28** | 50% |
| ⚠️ Missing one artifact (test OR docs OR README entry) | **14** | 25% |
| 🔴 Missing two or more artifacts | **14** | 25% |

**Target for v1.0:** 100% of indicators at ✅ status (all four artifacts present).

---

*End of PRD — Generated 2026-02-27 by automated audit of the PyIndicators repository.*
