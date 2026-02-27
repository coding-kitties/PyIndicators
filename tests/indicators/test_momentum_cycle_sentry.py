"""Tests for the Momentum Cycle Sentry indicator."""
import unittest
import pandas as pd
import polars as pl
import numpy as np
from pyindicators import (
    momentum_cycle_sentry,
    momentum_cycle_sentry_signal,
    get_momentum_cycle_sentry_stats,
)


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate realistic random OHLCV data."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_prices = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n)
    return pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


class TestMomentumCycleSentry(unittest.TestCase):
    """Test cases for momentum_cycle_sentry function."""

    def setUp(self):
        self.df = _make_ohlcv()

    def test_returns_pandas_dataframe(self):
        result = momentum_cycle_sentry(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars_dataframe(self):
        pl_df = pl.from_pandas(self.df)
        result = momentum_cycle_sentry(pl_df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_all_expected_columns(self):
        result = momentum_cycle_sentry(self.df)
        expected = [
            "mcs_p1", "mcs_p2", "mcs_p3", "mcs_p4", "mcs_p5",
            "mcs_ob_inner", "mcs_ob_outer",
            "mcs_os_inner", "mcs_os_outer",
            "mcs_trend", "mcs_retracing",
        ]
        for col in expected:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_row_count_unchanged(self):
        result = momentum_cycle_sentry(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_layers_ordering(self):
        """Faster layers should have smaller EMA lag — verify p5 is
        smoother (lower variance) than p1."""
        result = momentum_cycle_sentry(self.df)
        var_p1 = result["mcs_p1"].dropna().var()
        var_p5 = result["mcs_p5"].dropna().var()
        self.assertGreater(var_p1, var_p5)

    def test_ob_os_symmetry(self):
        """OB and OS bands should be symmetric mirrors of each other."""
        result = momentum_cycle_sentry(self.df)
        mask = result["mcs_ob_inner"].notna()
        np.testing.assert_array_almost_equal(
            result.loc[mask, "mcs_ob_inner"].values,
            -result.loc[mask, "mcs_os_inner"].values,
        )
        np.testing.assert_array_almost_equal(
            result.loc[mask, "mcs_ob_outer"].values,
            -result.loc[mask, "mcs_os_outer"].values,
        )

    def test_ob_outer_greater_than_inner(self):
        """Outer OB should be wider than inner OB."""
        result = momentum_cycle_sentry(self.df)
        mask = result["mcs_ob_inner"].notna()
        self.assertTrue(
            (result.loc[mask, "mcs_ob_outer"]
             >= result.loc[mask, "mcs_ob_inner"]).all()
        )

    def test_trend_values(self):
        """Trend must only contain -1, 0, or 1."""
        result = momentum_cycle_sentry(self.df)
        unique = set(result["mcs_trend"].unique())
        self.assertTrue(unique.issubset({-1, 0, 1}))

    def test_retracing_values(self):
        """Retracing must be binary (0 or 1)."""
        result = momentum_cycle_sentry(self.df)
        unique = set(result["mcs_retracing"].unique())
        self.assertTrue(unique.issubset({0, 1}))

    def test_custom_column_names(self):
        result = momentum_cycle_sentry(
            self.df,
            mcs_p1_column="fast",
            mcs_trend_column="dir",
        )
        self.assertIn("fast", result.columns)
        self.assertIn("dir", result.columns)

    def test_custom_parameters(self):
        """Different parameters should produce different output."""
        r1 = momentum_cycle_sentry(self.df, length=10)
        r2 = momentum_cycle_sentry(self.df, length=30)
        self.assertFalse(r1["mcs_p1"].equals(r2["mcs_p1"]))

    def test_magnitude_scales_layers(self):
        """Doubling magnitude should double all layer values."""
        r1 = momentum_cycle_sentry(self.df, magnitude=1.0)
        r2 = momentum_cycle_sentry(self.df, magnitude=2.0)
        mask = r1["mcs_p1"].notna()
        np.testing.assert_array_almost_equal(
            r2.loc[mask, "mcs_p1"].values,
            r1.loc[mask, "mcs_p1"].values * 2.0,
        )

    def test_polars_pandas_parity(self):
        """Polars and pandas paths must give the same numeric output."""
        pd_result = momentum_cycle_sentry(self.df)
        pl_result = momentum_cycle_sentry(
            pl.from_pandas(self.df)
        ).to_pandas()
        for col in ["mcs_p1", "mcs_p5", "mcs_trend", "mcs_retracing"]:
            np.testing.assert_array_almost_equal(
                pd_result[col].values, pl_result[col].values,
            )

    def test_invalid_input_raises(self):
        with self.assertRaises(Exception):
            momentum_cycle_sentry([1, 2, 3])


class TestMomentumCycleSentrySignal(unittest.TestCase):
    """Test cases for momentum_cycle_sentry_signal function."""

    def setUp(self):
        self.df = momentum_cycle_sentry(_make_ohlcv())

    def test_signal_column_added(self):
        result = momentum_cycle_sentry_signal(self.df)
        self.assertIn("mcs_signal", result.columns)

    def test_signal_values(self):
        """Signal must be -1, 0, or 1."""
        result = momentum_cycle_sentry_signal(self.df)
        unique = set(result["mcs_signal"].unique())
        self.assertTrue(unique.issubset({-1, 0, 1}))

    def test_retracing_forces_zero_signal(self):
        """When retracing == 1 the signal must be 0."""
        result = momentum_cycle_sentry_signal(self.df)
        retrace_mask = result["mcs_retracing"] == 1
        if retrace_mask.any():
            self.assertTrue(
                (result.loc[retrace_mask, "mcs_signal"] == 0).all()
            )

    def test_bullish_signal_requires_positive_trend(self):
        result = momentum_cycle_sentry_signal(self.df)
        bull_mask = result["mcs_signal"] == 1
        if bull_mask.any():
            self.assertTrue(
                (result.loc[bull_mask, "mcs_trend"] == 1).all()
            )

    def test_bearish_signal_requires_negative_trend(self):
        result = momentum_cycle_sentry_signal(self.df)
        bear_mask = result["mcs_signal"] == -1
        if bear_mask.any():
            self.assertTrue(
                (result.loc[bear_mask, "mcs_trend"] == -1).all()
            )

    def test_custom_signal_column_name(self):
        result = momentum_cycle_sentry_signal(
            self.df, signal_column="my_signal"
        )
        self.assertIn("my_signal", result.columns)

    def test_polars_signal(self):
        pl_df = pl.from_pandas(self.df)
        result = momentum_cycle_sentry_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn("mcs_signal", result.columns)


class TestGetMomentumCycleSentryStats(unittest.TestCase):
    """Test cases for get_momentum_cycle_sentry_stats function."""

    def setUp(self):
        self.df = momentum_cycle_sentry(_make_ohlcv())

    def test_stats_returns_dict(self):
        stats = get_momentum_cycle_sentry_stats(self.df)
        self.assertIsInstance(stats, dict)

    def test_stats_has_expected_keys(self):
        stats = get_momentum_cycle_sentry_stats(self.df)
        expected_keys = [
            "bullish_bars", "bearish_bars",
            "bullish_pct", "bearish_pct",
            "retracing_bars", "retracing_pct",
            "overbought_bars", "oversold_bars",
            "max_momentum", "min_momentum",
            "avg_momentum", "zero_crossings",
        ]
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing stats key: {key}")

    def test_bar_counts_sum(self):
        """Bullish + bearish bars should equal total non-neutral bars."""
        stats = get_momentum_cycle_sentry_stats(self.df)
        total = len(self.df)
        self.assertLessEqual(
            stats["bullish_bars"] + stats["bearish_bars"], total
        )

    def test_percentages_reasonable(self):
        stats = get_momentum_cycle_sentry_stats(self.df)
        self.assertGreaterEqual(stats["bullish_pct"], 0)
        self.assertLessEqual(stats["bullish_pct"], 100)
        self.assertGreaterEqual(stats["bearish_pct"], 0)
        self.assertLessEqual(stats["bearish_pct"], 100)

    def test_max_gte_min_momentum(self):
        stats = get_momentum_cycle_sentry_stats(self.df)
        self.assertGreaterEqual(stats["max_momentum"], stats["min_momentum"])

    def test_avg_momentum_non_negative(self):
        stats = get_momentum_cycle_sentry_stats(self.df)
        self.assertGreaterEqual(stats["avg_momentum"], 0)

    def test_zero_crossings_non_negative(self):
        stats = get_momentum_cycle_sentry_stats(self.df)
        self.assertGreaterEqual(stats["zero_crossings"], 0)

    def test_polars_stats(self):
        pl_df = pl.from_pandas(self.df)
        stats = get_momentum_cycle_sentry_stats(pl_df)
        self.assertIsInstance(stats, dict)
        self.assertIn("bullish_bars", stats)


if __name__ == "__main__":
    unittest.main()
