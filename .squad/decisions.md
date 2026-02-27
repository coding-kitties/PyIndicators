# Decisions

> Canonical decision ledger. Append-only. Managed by Scribe.

---

### 2026-02-27T13:50:00Z: Team formed
**By:** Squad (Coordinator)
**What:** Initial team cast for PyIndicators project — Carlos (Lead), DevMeister3000 (Core Dev), ChaosAgent (Tester), Doc Vader (DevRel).
**Why:** User confirmed roster.

---

### 2026-02-27: PRD Decomposition Approach
**By:** Carlos (Lead)
**What:** Decomposed the PRD (v0.19.0) into 70 specific, actionable work items across 5 phases and wrote them to `.squad/work-items.md`.
**Why:** Ensures the team has a clear, granular backlog with no ambiguity about what needs doing, who does it, and in what order. Priority ordering: tests (P0) → docs (P1) → README (P0/P1/P2) → notebooks (P3) → new features (P2/P3). Breakdown: 9 P0, 14 P1, 12 P2, 35 P3.

---

### 2026-02-27: Added @copilot as Coding Agent
**By:** Marc (via Squad Coordinator)
**What:** Added GitHub Copilot coding agent (@copilot) to the team roster with auto-assign enabled. Capability profile added to team.md with 🟢/🟡/🔴 fit ratings. Instructions installed at `.github/copilot-instructions.md`. Routing updated so Carlos triages issues against the capability profile before assigning `squad:copilot` labels.
**Why:** User requested autonomous issue pickup for suitable tasks. Enables @copilot to pick up 🟢-rated issues without human intervention.

---

### 2026-02-27: Flake8 Cleanup — Variable Naming & Export Conventions
**By:** DevMeister3000 (Core Dev)
**What:** Fixed 14 flake8 warnings (F841, F401, E741, E127) across 8 files. Renamed ambiguous `l` → `low`, added missing `__all__` entries for `accumulation_distribution_zones`, removed unused intermediate variables.
**Why:** Zero flake8 warnings for selected rules. Established conventions: use `high`/`low` for array variables; always add new functions to both `__all__` lists during registration.

---

### 2026-02-27: TBN Chart Plan — Architecture Decision
**By:** Carlos (Lead)
**Issue:** #3 — Improve Trendline Breakout Navigator
**What:** Defined chart layout (3-row: price+trendlines 70%, composite trend 15%, volume 15%), 12 traces, data requirements (BTC/EUR 4h ~365d), and output paths for TBN analysis notebook and docs chart.
**Why:** TBN docs page had no chart image. Plan gives DevMeister3000 an unambiguous implementation spec. Key decisions: trendline values plotted on price chart color-coded by direction, HH/LL/wick markers overlaid, composite trend background shading green/red.

---

### 2026-02-27: TBN Chart Notebook Implementation
**By:** DevMeister3000 (Core Dev)
**Issue:** #3
**What:** Created `analysis/indicators/trendline_breakout_navigator.ipynb` following Carlos's spec. PNG output to both `static/images/indicators/` and `docs/static/img/indicators/`. Used `git add -f` for gitignored analysis directory. Composite trend Y-axis hardcoded `[-3.5, 3.5]`.
**Why:** Implements the chart plan. Dual PNG output lets Doc Vader reference the docs copy directly without extra steps.
