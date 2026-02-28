## New Indicator Request

### Indicator Name
<!-- e.g., VWAP (Volume Weighted Average Price) -->


### Category
<!-- Pick one: Trend | Momentum | Volatility | Support & Resistance | Pattern Recognition | Volume | Helpers -->


### Description
<!-- What does this indicator do? How is it used in trading? -->


### Reference Chart(s)
<!-- Paste or drag & drop chart screenshots below showing the expected output -->
<!-- Include charts from TradingView, PineScript, or any other platform -->


### Chart Description
<!-- Describe what the charts show — agents may not always be able to see images -->
<!-- Include: timeframe, asset, what lines/zones/signals are visible, colors, key behavior -->

- **Timeframe:**
- **Asset:**
- **What's shown:**


### Parameters
<!-- List the indicator's parameters -->

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | int | 14 | Lookback period |
| `source_column` | str | 'Close' | Source column |


### Source / Reference
<!-- Link to PineScript source, academic paper, or reference implementation -->


### Expected Output Columns
<!-- What columns should the indicator add to the DataFrame? -->

- `indicator_value`:
- `indicator_signal`:


### Deliverables Checklist

- [ ] Implementation in `pyindicators/indicators/` (pandas + polars support)
- [ ] Exports in `__init__.py` and `__all__`
- [ ] Unit tests in `tests/indicators/` (pandas, polars, edge cases)
- [ ] Documentation page in `docs/content/indicators/` with chart image
- [ ] Sidebar registration in `docs/sidebars.js`
- [ ] Entry in README.md features list
- [ ] Analysis notebook in `analysis/indicators/` with plotly chart

### Additional Context
<!-- Optional: edge cases, related indicators, priority notes -->
