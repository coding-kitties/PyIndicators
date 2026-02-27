"""Tests for the Trendline Breakout Navigator indicator."""
import unittest
import pandas as pd
import polars as pl
import numpy as np
from pyindicators import (
    trendline_breakout_navigator,
    trendline_breakout_navigator_signal,
    get_trendline_breakout_navigator_stats,
)


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic random OHLCV data with clear trends."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    open_prices = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n)
    return pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


def _make_trending_data(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Generate data with clear up-then-down trend for HH/LL detection."""
    np.random.seed(seed)
    # Up trend for first half, down trend for second
    half = n // 2
    up = np.linspace(100, 200, half) + np.random.randn(half) * 3
    down = np.linspace(200, 100, n - half) + np.random.randn(n - half) * 3
    close = np.concatenate([up, down])
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    open_prices = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n)
    return pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


class TestTrendlineBreakoutNavigator(unittest.TestCase):
    """Test cases for trendline_breakout_navigator function."""

    def setUp(self):
        self.df = _make_ohlcv()
        self.trending_df = _make_trending_data()

    def test_returns_pandas_dataframe(self):
        result = trendline_breakout_navigator(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars_dataframe(self):
        pl_df = pl.from_pandas(self.df)
        result = trendline_breakout_navigator(pl_df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_all_expected_columns(self):
        result = trendline_breakout_navigator(self.df)
        expected = [
            "tbn_trend_long", "tbn_trend_medium", "tbn_trend_short",
            "tbn_value_long", "tbn_value_medium", "tbn_value_short",
            "tbn_slope_long", "tbn_slope_medium", "tbn_slope_short",
            "tbn_wick_bull", "tbn_wick_bear",
            "tbn_hh", "tbn_ll",
            "tbn_composite_trend",
        ]
        for col in expected:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_row_count_unchanged(self):
        result = trendline_breakout_navigator(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_trend_values(self):
        """Trend columns must only contain -1, 0, or 1."""
        result = trendline_breakout_navigator(self.df)
        for col in ["tbn_trend_long", "tbn_trend_medium", "tbn_trend_short"]:
            unique = set(result[col].unique())
            self.assertTrue(
                unique.issubset({-1, 0, 1}),
                f"{col} has unexpected values: {unique}"
            )

    def test_composite_range(self):
        """Composite trend should be in [-3, 3] when all TFs enabled."""
        result = trendline_breakout_navigator(self.df)
        comp = result["tbn_composite_trend"]
        self.assertTrue((comp >= -3).all())
        self.assertTrue((comp <= 3).all())

    def test_wick_values_binary(self):
        """Wick break columns must be 0 or 1."""
        result = trendline_breakout_navigator(self.df)
        for col in ["tbn_wick_bull", "tbn_wick_bear"]:
            unique = set(result[col].unique())
            self.assertTrue(
                unique.issubset({0, 1}),
                f"{col} has unexpected values: {unique}"
            )

    def test_hh_ll_values_binary(self):
        """HH/LL columns must be 0 or 1."""
        result = trendline_breakout_navigator(self.df)
        for col in ["tbn_hh", "tbn_ll"]:
            unique = set(result[col].unique())
            self.assertTrue(
                unique.issubset({0, 1}),
                f"{col} has unexpected values: {unique}"
            )

    def test_disable_timeframes(self):
        """Disabled timeframes should produce zero trend."""
        result = trendline_breakout_navigator(
            self.df, enable_long=False, enable_medium=False)
        self.assertTrue((result["tbn_trend_long"] == 0).all())
        self.assertTrue((result["tbn_trend_medium"] == 0).all())

    def test_custom_column_names(self):
        result = trendline_breakout_navigator(
            self.df,
            trend_long_column="my_trend",
            value_long_column="my_value",
        )
        self.assertIn("my_trend", result.columns)
        self.assertIn("my_value", result.columns)

    def test_different_swing_lengths_produce_different_output(self):
        """Different swing lengths should produce different results."""
        r1 = trendline_breakout_navigator(
            self.df, swing_long=60, enable_medium=False, enable_short=False)
        r2 = trendline_breakout_navigator(
            self.df, swing_long=20, enable_medium=False, enable_short=False)
        self.assertFalse(r1["tbn_trend_long"].equals(r2["tbn_trend_long"]))

    def test_trending_data_detects_hh_or_ll(self):
        """With trending data, at least some HH or LL should be detected."""
        result = trendline_breakout_navigator(
            self.trending_df,
            swing_long=20, swing_medium=10, swing_short=5)
        total_hh = result["tbn_hh"].sum()
        total_ll = result["tbn_ll"].sum()
        self.assertGreater(
            total_hh + total_ll, 0,
            "Expected at least some HH or LL detections in trending data"
        )

    def test_composite_equals_sum_of_trends(self):
        """Composite trend should equal the sum of individual trends."""
        result = trendline_breakout_navigator(self.df)
        expected = (result["tbn_trend_long"]
                    + result["tbn_trend_medium"]
                    + result["tbn_trend_short"])
        np.testing.assert_array_equal(
            result["tbn_composite_trend"].values,
            expected.values
        )

    def test_invalid_input_raises(self):
        """Non-DataFrame input should raise."""
        with self.assertRaises(Exception):
            trendline_breakout_navigator([1, 2, 3])

    def test_small_dataframe(self):
        """Should not crash on small data."""
        small = self.df.head(20)
        result = trendline_breakout_navigator(small, swing_short=3)
        self.assertEqual(len(result), 20)


class TestTrendlineBreakoutNavigatorSignal(unittest.TestCase):
    """Test cases for trendline_breakout_navigator_signal function."""

    def setUp(self):
        self.df = trendline_breakout_navigator(_make_ohlcv())

    def test_signal_column_added(self):
        result = trendline_breakout_navigator_signal(self.df)
        self.assertIn("tbn_signal", result.columns)

    def test_signal_values(self):
        """Signal must be -1, 0, or 1."""
        result = trendline_breakout_navigator_signal(self.df)
        unique = set(result["tbn_signal"].unique())
        self.assertTrue(unique.issubset({-1, 0, 1}))

    def test_signal_polars(self):
        pl_df = pl.from_pandas(self.df)
        result = trendline_breakout_navigator_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_custom_signal_column(self):
        result = trendline_breakout_navigator_signal(
            self.df, signal_column="my_sig")
        self.assertIn("my_sig", result.columns)

    def test_signal_consistency_with_composite(self):
        """Signal should match sign of composite trend."""
        result = trendline_breakout_navigator_signal(self.df)
        comp = result["tbn_composite_trend"]
        sig = result["tbn_signal"]
        for i in range(len(result)):
            if comp.iloc[i] > 0:
                self.assertEqual(sig.iloc[i], 1)
            elif comp.iloc[i] < 0:
                self.assertEqual(sig.iloc[i], -1)
            else:
                self.assertEqual(sig.iloc[i], 0)


class TestTrendlineBreakoutNavigatorStats(unittest.TestCase):
    """Test cases for get_trendline_breakout_navigator_stats function."""

    def setUp(self):
        self.df = trendline_breakout_navigator(_make_ohlcv())

    def test_returns_dict(self):
        stats = get_trendline_breakout_navigator_stats(self.df)
        self.assertIsInstance(stats, dict)

    def test_expected_keys(self):
        stats = get_trendline_breakout_navigator_stats(self.df)
        expected_keys = [
            "bullish_bars_long", "bearish_bars_long",
            "bullish_bars_medium", "bearish_bars_medium",
            "bullish_bars_short", "bearish_bars_short",
            "composite_bullish", "composite_bearish",
            "composite_bullish_pct", "composite_bearish_pct",
            "hh_count", "ll_count",
            "wick_bull_count", "wick_bear_count",
            "trend_changes", "active_trendline_bars",
        ]
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing key: {key}")

    def test_percentages_in_range(self):
        stats = get_trendline_breakout_navigator_stats(self.df)
        self.assertGreaterEqual(stats["composite_bullish_pct"], 0)
        self.assertLessEqual(stats["composite_bullish_pct"], 100)
        self.assertGreaterEqual(stats["composite_bearish_pct"], 0)
        self.assertLessEqual(stats["composite_bearish_pct"], 100)

    def test_counts_non_negative(self):
        stats = get_trendline_breakout_navigator_stats(self.df)
        for key in ["hh_count", "ll_count", "wick_bull_count",
                     "wick_bear_count", "trend_changes",
                     "active_trendline_bars"]:
            self.assertGreaterEqual(stats[key], 0, f"{key} is negative")

    def test_polars_input(self):
        pl_df = pl.from_pandas(self.df)
        stats = get_trendline_breakout_navigator_stats(pl_df)
        self.assertIsInstance(stats, dict)

    def test_invalid_input_raises(self):
        with self.assertRaises(Exception):
            get_trendline_breakout_navigator_stats([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
