# Carlos — History

## Project Context

- **Project:** PyIndicators — Python library for financial technical analysis
- **Owner:** marcvanduyn
- **Stack:** Python 3.12+, pandas, polars, unittest, Docusaurus docs
- **Key patterns:** Each indicator has three public functions (`indicator()`, `indicator_signal()`, `get_indicator_stats()`), supports both pandas/polars, registered in `pyindicators/indicators/__init__.py` and `pyindicators/__init__.py`

## Learnings

- Team formed 2026-02-27. Roster: Carlos (Lead), DevMeister3000 (Core Dev), ChaosAgent (Tester), Doc Vader (DevRel).
- 2026-02-27: Decomposed PRD into 70 work items across 5 phases. Breakdown: 9 P0, 14 P1, 12 P2, 35 P3. Phase 1 (test coverage) has 14 items — 8 at P0 targeting the liquidity/price-action cluster and classic indicators (ATR, CCI, ROC). Phase 2 (docs) has 11 items covering 7 missing doc pages + sidebar + utils + README updates. Phase 5 (new indicators) has 11 items, with VWAP, Ichimoku, and Pivot Points as highest priority new features. Priority ordering: tests first to establish quality baseline, then docs, then README polish, then notebooks and new features.
