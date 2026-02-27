# Decision: PRD Decomposition Approach

**Date:** 2026-02-27
**By:** Carlos (Lead)
**Status:** Proposed

## What

Decomposed the PRD (v0.19.0) into 70 specific, actionable work items across 5 phases and wrote them to `.squad/work-items.md`.

## Approach

- Grouped items strictly by PRD phase (1–5).
- Cross-referenced the PRD gap analysis matrix to enumerate every single missing artifact (test, doc page, README entry, notebook) rather than using vague batch items.
- Assigned ownership by role: ChaosAgent owns all tests (Phase 1), Doc Vader owns all docs/README/notebooks (Phases 2–4), DevMeister3000 owns new indicator implementations (Phase 5).
- Priority ordering follows a quality-first strategy: **tests (P0) → docs (P1) → README (P0/P1/P2) → notebooks (P3) → new features (P2/P3)**.

## Priority Rationale

1. **P0 = test gaps for shipped indicators.** Users already depend on these; untested code is a liability. The 8 high-priority items target complex price-action indicators (liquidity sweeps, pools, levels/voids, buyside/sellside) and missing classic indicator tests (ATR, CCI, ROC).
2. **P0 = metadata fix.** The "no external dependencies" claim is factually wrong and should be fixed immediately.
3. **P1 = documentation + remaining tests.** Docs are the second biggest gap; 7 missing pages is manageable. Remaining tests (pulse mean accelerator, equal highs/lows, volume weighted trend) are medium priority since these indicators see less usage.
4. **P2 = README polish + top new indicators.** README updates depend on doc completion. New indicators (VWAP, Ichimoku, Pivot Points, Keltner, Donchian) chosen based on common user requests.
5. **P3 = notebooks + remaining new indicators.** Nice-to-have; 29 notebooks and 6 backlog indicators.

## Why

Ensures the team has a clear, granular backlog with no ambiguity about what needs doing, who does it, and in what order. Prevents "boil the ocean" paralysis by giving ChaosAgent an immediate P0 queue to start on.
