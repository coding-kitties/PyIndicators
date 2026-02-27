# Decision: TBN Chart Notebook Implementation

**Date:** 2026-02-27  
**By:** DevMeister3000 (Core Dev)  
**Issue:** #3

## Decisions

1. **PNG to two locations:** The chart PNG is written to both `static/images/indicators/` (project assets) and `docs/static/img/indicators/` (Docusaurus), matching what the chart spec requested. Doc Vader can reference the docs copy directly.

2. **Force-add through gitignore:** `analysis/indicators/` is covered by `.gitignore`. Used `git add -f` to commit the notebook, consistent with how other analysis notebooks in that directory are tracked (e.g., `volume_weighted_trend.ipynb`).

3. **Composite trend Y-axis range:** Hardcoded `[-3.5, 3.5]` with `dtick=1` per Carlos's spec, since all three timeframes are enabled by default giving a composite range of -3 to +3.
