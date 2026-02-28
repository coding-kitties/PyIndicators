# Routing Rules

## Default Routes

| Domain | Primary | Backup |
|--------|---------|--------|
| Architecture, design decisions, code review | Carlos | — |
| Indicator implementation, Python code, porting PineScript | DevMeister3000 | Carlos |
| Tests, edge cases, quality assurance, unittest | ChaosAgent | DevMeister3000 |
| Documentation, Docusaurus, examples, README | Doc Vader | Carlos |
| Session logging, decisions, memory | Scribe | — |
| Work queue, backlog, monitoring | Ralph | — |

## Keyword Routes

| Keywords | Route to |
|----------|----------|
| indicator, implement, port, pine, pinescript, algorithm, ema, sma, rsi, macd, pivot, swing, trendline, breakout | DevMeister3000 |
| test, unittest, assert, edge case, coverage, quality | ChaosAgent |
| docs, documentation, docusaurus, markdown, guide, example, tutorial, installation | Doc Vader |
| architecture, design, pattern, refactor, api, structure, review, approve | Carlos |
| backlog, issues, status, board, queue, monitor | Ralph |
| autonomous, copilot, simple fix, single-file, boilerplate | @copilot |

## Copilot Agent Routing

When triaging issues, Carlos evaluates against the capability profile in `team.md`:
- **🟢 issues** with `squad:copilot` label → @copilot picks up autonomously
- **🟡 issues** → @copilot works, but PR needs squad review before merge
- **🔴 issues** → route to appropriate squad agent instead

## Multi-Domain

When a task spans multiple domains (e.g., "add indicator with tests and docs"), fan out to all relevant agents in parallel.

## Review Gate

- Carlos reviews architecture decisions and API changes.
- ChaosAgent reviews all new indicator implementations (test coverage).
- Doc Vader reviews documentation accuracy.
