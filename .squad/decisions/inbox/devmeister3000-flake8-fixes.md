# Decision: Flake8 Cleanup — Variable Naming & Export Conventions

**Date:** 2026-02-27
**Author:** DevMeister3000
**Status:** Applied

## Context

Fixed 14 flake8 warnings (F841, F401, E741, E127) across 8 files in the pyindicators package.

## Decisions Made

1. **Ambiguous variable `l` renamed to `low`** in opening_gap.py, strong_weak_high_low.py, and volume_imbalance.py. The `h` (high) variable was kept as-is since flake8 doesn't flag it — but for consistency, future indicators should use `high`/`low` instead of `h`/`l` for array variables.

2. **Missing `__all__` entries for accumulation_distribution_zones** were added to `pyindicators/__init__.py`. This was an oversight — the functions were imported but never added to the public API's `__all__` list. Going forward, the registration checklist (see charter) should explicitly include "add to both `__all__` lists".

3. **Unused intermediate variables removed** — `y1_plus_slope`/`y2_plus_slope` in trendline_breakout_navigator.py, `ob_inner`/`os_inner` in momentum_cycle_sentry.py, and `atr_vals` in z_score_predictive_zones.py were all computed but never referenced. Removed rather than suppressed with `# noqa`.

## Impact

- Zero flake8 warnings for selected rules (E741, F841, F401, E127)
- All 731 tests pass or have pre-existing failures (9 CHoCH/BOS tests in test_market_structure — unrelated)
