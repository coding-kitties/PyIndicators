# PyIndicators — Squad Team

## Project Context

- **Project:** PyIndicators
- **Owner:** marcvanduyn
- **Description:** A powerful and user-friendly Python library for financial technical analysis indicators, metrics and helper functions for pandas and polars dataframes. Written entirely in Python with no external dependencies.
- **Stack:** Python, pandas, polars, Docusaurus (docs)
- **Repo:** CodingKitties/PyIndicators

## Members

| Name | Role | Specialty | Emoji |
|------|------|-----------|-------|
| Marc | Owner / PM | Project owner, direction, final say | 👤 Human |
| Carlos | Lead | Architecture, code review, decisions | 🏗️ |
| DevMeister3000 | Core Dev | Indicator implementations, Python | 🔧 |
| ChaosAgent | Tester | Tests, quality, edge cases | 🧪 |
| Doc Vader | DevRel | Docusaurus docs, examples, guides | 📝 |
| @copilot | Coding Agent | Autonomous issue pickup, PRs | 🤖 |
| Scribe | (silent) | Memory, decisions, session logs | 📋 |
| Ralph | (monitor) | Work queue, backlog, keep-alive | 🔄 |

<!-- copilot-auto-assign: true -->

## Issue Source

- **Repository:** `coding-kitties/PyIndicators`
- **Connected:** 2026-02-27
- **Filters:** All open issues

## PRD

- **Source:** `/Users/marcvanduyn/Projects/CodingKitties/PyIndicators/PRD.md`
- **Ingested:** 2026-02-27
- **Status:** Decomposed → 70 work items in `.squad/work-items.md`

## Coding Agent — Capabilities

| Task Type | Fit | Notes |
|-----------|-----|-------|
| Single-file indicator implementation | 🟢 | Follows established pattern |
| Writing unittest test suites | 🟢 | Straightforward from existing examples |
| Single-file bug fixes | 🟢 | Scoped, low risk |
| Documentation pages (Docusaurus md) | 🟢 | Template-driven |
| README updates | 🟢 | Text edits |
| Multi-file refactors | 🟡 | Needs squad review |
| New indicator with complex math (porting PineScript) | 🟡 | Logic correctness needs review |
| Architecture changes (new module structure) | 🔴 | Requires Carlos (Lead) decision |
| Docusaurus config / sidebar changes | 🔴 | Risk of breaking docs build |
| Release management / versioning | 🔴 | Needs human approval |

## Tech Stack

- **Language:** Python 3.12+
- **DataFrame support:** pandas, polars
- **Testing:** unittest
- **Documentation:** Docusaurus
- **Build:** pyproject.toml
- **No external dependencies** for core library
