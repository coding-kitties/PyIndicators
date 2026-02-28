# ChaosAgent — History

## Project Context

- **Project:** PyIndicators — Python library for financial technical analysis
- **Owner:** marcvanduyn
- **Testing framework:** unittest
- **Test location:** `tests/indicators/test_{indicator_name}.py`
- **Test data pattern:** `_make_ohlcv(n=200, seed=42)` generates reproducible OHLCV data

## Learnings

- Team formed 2026-02-27.
- Tests use `unittest.TestCase` with `setUp` creating shared test DataFrames.
- Each test file tests three public functions: `indicator()`, `indicator_signal()`, `get_indicator_stats()`.
- Polars tests convert pandas→polars and verify the result type.
- Virtual environment at `.venv/` — run tests with `.venv/bin/python -m unittest`.
