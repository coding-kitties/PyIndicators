import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    range_intelligence,
    range_intelligence_signal,
    get_range_intelligence_stats,
)
from pyindicators.exceptions import PyIndicatorException


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate OHLCV data with trending and consolidation phases."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    high = close + np.abs(np.random.randn(n)) * 2.0
    low = close - np.abs(np.random.randn(n)) * 2.0
    opn = close - np.random.randn(n) * 0.5
    volume = np.abs(np.random.randn(n) * 1000) + 500
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    })


def _make_tight_range() -> pd.DataFrame:
    """Create data with a guaranteed consolidation range followed by breakout."""
    n = 60
    # First 20 bars: trending up
    prices_up = np.linspace(100, 120, 20)
    # Next 25 bars: tight range around 120 (±1)
    prices_flat = 120 + np.random.RandomState(0).uniform(-0.5, 0.5, 25)
    # Last 15 bars: breakout up
    prices_break = np.linspace(122, 140, 15)

    close = np.concatenate([prices_up, prices_flat, prices_break])
    high = close + 0.8
    low = close - 0.8
    opn = close - 0.2
    volume = np.ones(n) * 1000

    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    })


class TestRangeIntelligence(unittest.TestCase):
    """Tests for range_intelligence()."""

    def test_returns_pandas(self):
        df = _make_ohlcv()
        result = range_intelligence(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars(self):
        df = pl.from_pandas(_make_ohlcv())
        result = range_intelligence(df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_columns(self):
        result = range_intelligence(_make_ohlcv())
        expected_cols = [
            "ri_active", "ri_high", "ri_low", "ri_mid",
            "ri_poc", "ri_delta", "ri_state", "ri_ready",
            "ri_sweep_high", "ri_sweep_low", "ri_breakout",
            "ri_duration",
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_preserves_row_count(self):
        df = _make_ohlcv()
        result = range_intelligence(df)
        self.assertEqual(len(result), len(df))

    def test_active_is_binary(self):
        result = range_intelligence(_make_ohlcv())
        self.assertTrue(result["ri_active"].isin([0, 1]).all())

    def test_sweep_flags_are_binary(self):
        result = range_intelligence(_make_ohlcv())
        self.assertTrue(result["ri_sweep_high"].isin([0, 1]).all())
        self.assertTrue(result["ri_sweep_low"].isin([0, 1]).all())

    def test_breakout_values(self):
        result = range_intelligence(_make_ohlcv())
        self.assertTrue(result["ri_breakout"].isin([-1, 0, 1]).all())

    def test_tight_range_detected(self):
        df = _make_tight_range()
        result = range_intelligence(df, length=10, sensitivity=3.0)
        # Should detect at least one active range bar
        self.assertGreater(result["ri_active"].sum(), 0)

    def test_range_boundaries_valid(self):
        df = _make_ohlcv()
        result = range_intelligence(df)
        active = result[result["ri_active"] == 1]
        if len(active) > 0:
            # ri_high should be >= ri_low
            self.assertTrue((active["ri_high"] >= active["ri_low"]).all())
            # ri_mid should be between high and low
            self.assertTrue(
                ((active["ri_mid"] >= active["ri_low"])
                 & (active["ri_mid"] <= active["ri_high"])).all()
            )

    def test_poc_within_range(self):
        df = _make_ohlcv()
        result = range_intelligence(df)
        active = result[result["ri_active"] == 1].dropna(subset=["ri_poc"])
        if len(active) > 0:
            self.assertTrue(
                ((active["ri_poc"] >= active["ri_low"])
                 & (active["ri_poc"] <= active["ri_high"])).all()
            )

    def test_state_values(self):
        df = _make_ohlcv()
        result = range_intelligence(df)
        active = result[result["ri_active"] == 1]
        if len(active) > 0:
            valid_states = {"Accumulation", "Distribution"}
            unique_states = set(active["ri_state"].unique())
            self.assertTrue(unique_states.issubset(valid_states))

    def test_ready_score_range(self):
        df = _make_ohlcv()
        result = range_intelligence(df)
        active = result[result["ri_active"] == 1]
        if len(active) > 0:
            self.assertTrue((active["ri_ready"] >= 0).all())
            self.assertTrue((active["ri_ready"] <= 100).all())

    def test_invalid_input_raises(self):
        with self.assertRaises(PyIndicatorException):
            range_intelligence("not a dataframe")

    def test_custom_column_names(self):
        df = _make_ohlcv()
        result = range_intelligence(
            df,
            ri_active_column="custom_active",
            ri_breakout_column="custom_breakout",
        )
        self.assertIn("custom_active", result.columns)
        self.assertIn("custom_breakout", result.columns)


class TestRangeIntelligenceSignal(unittest.TestCase):
    """Tests for range_intelligence_signal()."""

    def test_signal_column_added(self):
        df = range_intelligence(_make_ohlcv())
        result = range_intelligence_signal(df)
        self.assertIn("ri_signal", result.columns)

    def test_signal_values(self):
        df = range_intelligence(_make_ohlcv())
        result = range_intelligence_signal(df)
        self.assertTrue(result["ri_signal"].isin([-1, 0, 1]).all())

    def test_signal_matches_breakout(self):
        df = range_intelligence(_make_ohlcv())
        result = range_intelligence_signal(df)
        # Signal should equal breakout
        breakout = result["ri_breakout"].fillna(0).astype(int)
        signal = result["ri_signal"]
        self.assertTrue((breakout == signal).all())

    def test_polars_input(self):
        df = range_intelligence(_make_ohlcv())
        pl_df = pl.from_pandas(df)
        result = range_intelligence_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)


class TestRangeIntelligenceStats(unittest.TestCase):
    """Tests for get_range_intelligence_stats()."""

    def test_stats_keys(self):
        df = range_intelligence(_make_ohlcv())
        stats = get_range_intelligence_stats(df)
        expected_keys = {
            "total_ranges", "bullish_breakouts", "bearish_breakouts",
            "accumulation_ranges", "distribution_ranges",
            "total_sweep_highs", "total_sweep_lows",
            "avg_ready_score", "avg_duration",
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_stats_consistency(self):
        df = range_intelligence(_make_ohlcv())
        stats = get_range_intelligence_stats(df)
        self.assertEqual(
            stats["total_ranges"],
            stats["bullish_breakouts"] + stats["bearish_breakouts"],
        )

    def test_stats_non_negative(self):
        df = range_intelligence(_make_ohlcv())
        stats = get_range_intelligence_stats(df)
        for key in ["total_ranges", "bullish_breakouts", "bearish_breakouts",
                     "total_sweep_highs", "total_sweep_lows"]:
            self.assertGreaterEqual(stats[key], 0)

    def test_polars_input(self):
        df = range_intelligence(_make_ohlcv())
        pl_df = pl.from_pandas(df)
        stats = get_range_intelligence_stats(pl_df)
        self.assertIn("total_ranges", stats)


if __name__ == "__main__":
    unittest.main()
