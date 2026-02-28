# PyIndicators — Work Items Decomposition

> Decomposed from [PRD.md](../PRD.md) by Carlos (Lead) on 2026-02-27.

## Summary

| Phase | Total Items | P0 | P1 | P2 | P3 |
|-------|-------------|----|----|----|----|
| Phase 1 — Test Coverage | 14 | 8 | 3 | 3 | 0 |
| Phase 2 — Documentation | 11 | 0 | 9 | 2 | 0 |
| Phase 3 — README & Onboarding | 5 | 1 | 2 | 2 | 0 |
| Phase 4 — Analysis Notebooks | 29 | 0 | 0 | 0 | 29 |
| Phase 5 — New Indicators | 11 | 0 | 0 | 5 | 6 |
| **Total** | **70** | **9** | **14** | **12** | **35** |

---

## Phase 1 — Test Coverage (Critical)

Every indicator must have tests covering Pandas input/output, Polars input/output, and edge cases.

| ID | Title | Owner | Priority | Dependencies | Complexity | Notes |
|----|-------|-------|----------|--------------|------------|-------|
| 1.01 | Write tests for `average_true_range.py` (ATR) | ChaosAgent | P0 | None | S | Classic volatility indicator, simple I/O |
| 1.02 | Write tests for `commodity_channel_index.py` (CCI) | ChaosAgent | P0 | None | S | Classic momentum indicator |
| 1.03 | Write tests for `rate_of_change.py` (ROC) | ChaosAgent | P0 | None | S | Classic momentum indicator |
| 1.04 | Write tests for `liquidity_sweeps.py` | ChaosAgent | P0 | None | M | Complex price-action logic, needs realistic OHLCV data |
| 1.05 | Write tests for `buyside_sellside_liquidity.py` | ChaosAgent | P0 | None | M | Complex price-action logic |
| 1.06 | Write tests for `pure_price_action_liquidity_sweeps.py` | ChaosAgent | P0 | None | M | Complex price-action logic |
| 1.07 | Write tests for `liquidity_pools.py` | ChaosAgent | P0 | None | M | Complex price-action logic |
| 1.08 | Write tests for `liquidity_levels_voids.py` | ChaosAgent | P0 | None | M | Complex price-action logic |
| 1.09 | Write tests for `pulse_mean_accelerator.py` | ChaosAgent | P1 | None | M | Medium-priority trend indicator |
| 1.10 | Write tests for `equal_highs_lows.py` | ChaosAgent | P1 | None | M | Medium-priority S/R indicator |
| 1.11 | Write tests for `volume_weighted_trend.py` | ChaosAgent | P1 | None | M | Medium-priority trend indicator |
| 1.12 | Write tests for `is_down_trend.py` | ChaosAgent | P2 | None | S | Simple helper utility |
| 1.13 | Write tests for `is_up_trend.py` | ChaosAgent | P2 | None | S | Simple helper utility |
| 1.14 | Write tests for `up_and_down_trends.py` | ChaosAgent | P2 | None | S | Simple helper utility |

---

## Phase 2 — Documentation Completeness (High)

Add missing doc pages and ensure sidebar registration.

| ID | Title | Owner | Priority | Dependencies | Complexity | Notes |
|----|-------|-------|----------|--------------|------------|-------|
| 2.01 | Create doc page for Volume Gated Trend Ribbon | Doc Vader | P1 | None | S | Category: Trend. Follow `docs/content/indicators/trend/` pattern |
| 2.02 | Create doc page for Commodity Channel Index (CCI) | Doc Vader | P1 | None | S | Category: Momentum |
| 2.03 | Create doc page for Rate of Change (ROC) | Doc Vader | P1 | None | S | Category: Momentum |
| 2.04 | Create doc page for Equal Highs / Lows | Doc Vader | P1 | None | S | Category: Support & Resistance |
| 2.05 | Create doc page for Swing Structure | Doc Vader | P1 | None | S | Category: Support & Resistance |
| 2.06 | Create doc page for Trendline Breakout Navigator | Doc Vader | P1 | None | S | Category: Support & Resistance |
| 2.07 | Create doc page for Up and Downtrends | Doc Vader | P1 | None | S | Category: Helpers |
| 2.08 | Register all 7 new doc pages in `docs/sidebars.js` | Doc Vader | P1 | 2.01–2.07 | S | Must update sidebar config |
| 2.09 | Document missing utility functions (`is_below`, `is_above`, `get_slope`, etc.) | Doc Vader | P1 | None | S | Only `has_any_lower_then_threshold` currently documented |
| 2.10 | Update README features list to include all 16 missing indicators | Doc Vader | P2 | None | S | See PRD §4.3 list |
| 2.11 | Fix "no external dependencies" claim in README and `pyproject.toml` | Doc Vader | P2 | None | S | Replace with accurate description |

---

## Phase 3 — README & Onboarding (Medium)

| ID | Title | Owner | Priority | Dependencies | Complexity | Notes |
|----|-------|-------|----------|--------------|------------|-------|
| 3.01 | Fix "no external dependencies" metadata discrepancy | Doc Vader | P0 | None | S | Misleads users; fix in README + pyproject.toml description |
| 3.02 | Add "Quick Start" section with minimal end-to-end example | Doc Vader | P1 | None | S | Show install → import → compute → inspect |
| 3.03 | Restructure README: reduce inline API docs, link to Docusaurus site | Doc Vader | P1 | 2.01–2.09 | M | Depends on docs being complete first |
| 3.04 | Add badges (PyPI version, test status, docs link) | Doc Vader | P2 | None | S | Standard OSS project hygiene |
| 3.05 | Add CONTRIBUTING.md with indicator authoring guide | Doc Vader | P2 | None | M | Guide contributors on patterns from PRD §7 |

---

## Phase 4 — Analysis Notebooks (Low / Nice-to-have)

All items are P3. No dependencies unless noted. Owner: Doc Vader. Complexity: S each.

| ID | Title | Owner | Priority | Dependencies | Complexity | Notes |
|----|-------|-------|----------|--------------|------------|-------|
| 4.01 | Create notebook for Simple Moving Average (SMA) | Doc Vader | P3 | None | S | Classic trend |
| 4.02 | Create notebook for Weighted Moving Average (WMA) | Doc Vader | P3 | None | S | Classic trend |
| 4.03 | Create notebook for Exponential Moving Average (EMA) | Doc Vader | P3 | None | S | Classic trend |
| 4.04 | Create notebook for SuperTrend | Doc Vader | P3 | None | S | Popular trend indicator |
| 4.05 | Create notebook for Volume Gated Trend Ribbon | Doc Vader | P3 | None | S | |
| 4.06 | Create notebook for MACD | Doc Vader | P3 | None | S | High-value classic |
| 4.07 | Create notebook for RSI | Doc Vader | P3 | None | S | High-value classic |
| 4.08 | Create notebook for Wilders RSI | Doc Vader | P3 | None | S | |
| 4.09 | Create notebook for Williams %R | Doc Vader | P3 | None | S | |
| 4.10 | Create notebook for ADX | Doc Vader | P3 | None | S | |
| 4.11 | Create notebook for Stochastic Oscillator | Doc Vader | P3 | None | S | |
| 4.12 | Create notebook for Momentum Confluence | Doc Vader | P3 | None | S | |
| 4.13 | Create notebook for Commodity Channel Index (CCI) | Doc Vader | P3 | 1.02 | S | Needs working impl + tests first |
| 4.14 | Create notebook for Rate of Change (ROC) | Doc Vader | P3 | 1.03 | S | Needs working impl + tests first |
| 4.15 | Create notebook for Bollinger Bands | Doc Vader | P3 | None | S | High-value classic |
| 4.16 | Create notebook for Average True Range (ATR) | Doc Vader | P3 | 1.01 | S | |
| 4.17 | Create notebook for Moving Average Envelope | Doc Vader | P3 | None | S | |
| 4.18 | Create notebook for Nadaraya-Watson Envelope | Doc Vader | P3 | None | S | |
| 4.19 | Create notebook for Fibonacci Retracement | Doc Vader | P3 | None | S | |
| 4.20 | Create notebook for Golden Zone | Doc Vader | P3 | None | S | |
| 4.21 | Create notebook for Fair Value Gap | Doc Vader | P3 | None | S | |
| 4.22 | Create notebook for Order Blocks | Doc Vader | P3 | None | S | |
| 4.23 | Create notebook for Market Structure | Doc Vader | P3 | None | S | |
| 4.24 | Create notebook for Divergence | Doc Vader | P3 | None | S | |
| 4.25 | Create notebook for Accumulation Distribution Zones | Doc Vader | P3 | None | S | |
| 4.26 | Create notebook for Volume Imbalance | Doc Vader | P3 | None | S | |
| 4.27 | Create notebook for Opening Gap | Doc Vader | P3 | None | S | |
| 4.28 | Create notebook for Strong / Weak High / Low | Doc Vader | P3 | None | S | |
| 4.29 | Create notebook for Trendline Breakout Navigator | Doc Vader | P3 | None | S | |

---

## Phase 5 — New Indicators (Backlog)

| ID | Title | Owner | Priority | Dependencies | Complexity | Notes |
|----|-------|-------|----------|--------------|------------|-------|
| 5.01 | Implement VWAP (Volume Weighted Average Price) | DevMeister3000 | P2 | Phase 1 done | M | Essential for intraday; high user demand |
| 5.02 | Implement Ichimoku Cloud | DevMeister3000 | P2 | Phase 1 done | L | Commonly requested; multiple output lines |
| 5.03 | Implement Pivot Points (Standard, Camarilla, Woodie) | DevMeister3000 | P2 | Phase 1 done | M | Classic S/R levels |
| 5.04 | Implement Keltner Channels | DevMeister3000 | P2 | 1.01 (ATR tests) | M | Depends on ATR being fully tested |
| 5.05 | Implement Donchian Channels | DevMeister3000 | P2 | Phase 1 done | S | Simple breakout detection |
| 5.06 | Implement Parabolic SAR | DevMeister3000 | P3 | Phase 1 done | M | Trend reversal |
| 5.07 | Implement Heikin-Ashi Candles | DevMeister3000 | P3 | Phase 1 done | S | Trend-smoothing candle type |
| 5.08 | Implement On-Balance Volume (OBV) | DevMeister3000 | P3 | Phase 1 done | S | Volume-based trend confirmation |
| 5.09 | Implement Chaikin Money Flow | DevMeister3000 | P3 | Phase 1 done | S | Volume/momentum hybrid |
| 5.10 | Run type annotations audit across all public APIs | DevMeister3000 | P3 | Phase 1 done | M | Ensure full type hints |
| 5.11 | Create performance benchmark suite (pandas vs polars) | DevMeister3000 | P3 | Phase 1 done | M | Comparative benchmarks |

---

## Execution Order Recommendation

1. **Immediate (P0):** Items 1.01–1.08 (8 high-priority test gaps) + 3.01 (metadata fix) — these block quality confidence.
2. **Next sprint (P1):** Remaining tests (1.09–1.11) + all doc pages (2.01–2.09) + README quick start (3.02) + README restructure (3.03).
3. **Following sprint (P2):** README features update (2.10–2.11) + badges (3.04) + CONTRIBUTING.md (3.05) + top new indicators (5.01–5.05).
4. **Backlog (P3):** Phase 4 notebooks + remaining Phase 5 indicators + type audit + benchmarks.

---

*Generated by Carlos (Lead) — 2026-02-27*
