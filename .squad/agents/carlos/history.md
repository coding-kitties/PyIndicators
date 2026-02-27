# Carlos — History

## Project Context

- **Project:** PyIndicators — Python library for financial technical analysis
- **Owner:** marcvanduyn
- **Stack:** Python 3.12+, pandas, polars, unittest, Docusaurus docs
- **Key patterns:** Each indicator has three public functions (`indicator()`, `indicator_signal()`, `get_indicator_stats()`), supports both pandas/polars, registered in `pyindicators/indicators/__init__.py` and `pyindicators/__init__.py`

## Learnings

- Team formed 2026-02-27. Roster: Carlos (Lead), DevMeister3000 (Core Dev), ChaosAgent (Tester), Doc Vader (DevRel).
- 2026-02-27: Decomposed PRD into 70 work items across 5 phases. Breakdown: 9 P0, 14 P1, 12 P2, 35 P3. Phase 1 (test coverage) has 14 items — 8 at P0 targeting the liquidity/price-action cluster and classic indicators (ATR, CCI, ROC). Phase 2 (docs) has 11 items covering 7 missing doc pages + sidebar + utils + README updates. Phase 5 (new indicators) has 11 items, with VWAP, Ichimoku, and Pivot Points as highest priority new features. Priority ordering: tests first to establish quality baseline, then docs, then README polish, then notebooks and new features.
- 2026-02-27: Issue #3 — Produced detailed chart plan for Trendline Breakout Navigator. TBN is a multi-timeframe indicator with 3 trendline timeframes, composite score, HH/LL events, and wick breaks. Chart plan uses 3-row layout (price+trendlines, composite trend, volume). Key insight: need ~365 days of 4h data to get enough long-timeframe pivots. Trendline values (`tbn_value_*`) are the projected prices — these should be drawn as lines on the price chart, colored by the corresponding `tbn_trend_*` direction. Decision written to `.squad/decisions/inbox/carlos-tbn-chart-plan.md`.
