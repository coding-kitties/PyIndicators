# Doc Vader — History

## Project Context

- **Project:** PyIndicators — Python library for financial technical analysis
- **Owner:** marcvanduyn
- **Docs framework:** Docusaurus
- **Docs location:** `docs/content/indicators/` for indicator pages
- **Site config:** `docs/docusaurus.config.js`

## Learnings

- Team formed 2026-02-27.
- Documentation site is at `docs/` with standard Docusaurus structure.
- Indicator docs live in `docs/content/indicators/`.
- Analysis notebooks in `analysis/indicators/` serve as visual examples.
- Chart images in docs use UPPER_SNAKE_CASE alt text and path `/img/indicators/<name>.png`, placed immediately after the Example code block's closing fence.
- Added TBN chart image reference to `docs/content/indicators/support-resistance/trendline-breakout-navigator.md` (Issue #3).- Verified TBN docs page after chart improvement (Issue #3, 2026-02-28): image ref `![TRENDLINE_BREAKOUT_NAVIGATOR](/img/indicators/trendline_breakout_navigator.png)` correct, PNG exists at both `docs/static/img/indicators/` and `static/images/indicators/` (227 966 bytes), function signatures/params/return columns/signal logic/stats keys all match source. No changes needed.