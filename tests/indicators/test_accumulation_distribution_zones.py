import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    accumulation_distribution_zones,
    accumulation_distribution_zones_signal,
    get_accumulation_distribution_zones_stats,
)
from pyindicators.exceptions import PyIndicatorException


def _make_converging_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLC data with clear converging swing
    structure to trigger accumulation and distribution patterns.
    """
    np.random.seed(seed)
    t = np.arange(n, dtype=float)

    # Create a pattern with converging highs and lows
    # First half: damped oscillation around an uptrend (accumulation)
    # Second half: damped oscillation around a downtrend (distribution)
    base = np.where(
        t < n // 2,
        100 + t * 0.05,
        100 + (n // 2) * 0.05 - (t - n // 2) * 0.05,
    )
    # Damped oscillation creates converging swings
    damping = np.exp(-t * 0.003)
    wave = 8 * damping * np.sin(t * 2 * np.pi / 30)
    close = base + wave + np.random.randn(n) * 0.3
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    opn = close - np.random.randn(n) * 0.3
    return pd.DataFrame({
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(1000, 10000, n),
    })


def _make_trending_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLC data with clear swing structure."""
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    base = np.where(
        t < n // 2,
        100 + t * 0.4,
        100 + (n // 2) * 0.4 - (t - n // 2) * 0.4,
    )
    wave = 5 * np.sin(t * 2 * np.pi / 25)
    close = base + wave + np.random.randn(n) * 0.5
    high = close + np.abs(np.random.randn(n)) * 2.0
    low = close - np.abs(np.random.randn(n)) * 2.0
    opn = close - np.random.randn(n) * 0.5
    return pd.DataFrame({
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(1000, 10000, n),
    })


class TestAccumulationDistributionZones(unittest.TestCase):
    """Tests for the accumulation_distribution_zones() function."""

    def setUp(self):
        self.df = _make_converging_data()

    def test_returns_pandas_dataframe(self):
        result = accumulation_distribution_zones(self.df, pivot_length=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars_dataframe(self):
        pl_df = pl.from_pandas(self.df)
        result = accumulation_distribution_zones(pl_df, pivot_length=5)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_expected_columns(self):
        result = accumulation_distribution_zones(self.df, pivot_length=5)
        expected = [
            "adz_accumulation", "adz_distribution",
            "adz_zone_top", "adz_zone_bottom",
            "adz_zone_left", "adz_zone_right",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_preserves_row_count(self):
        result = accumulation_distribution_zones(self.df, pivot_length=5)
        self.assertEqual(len(result), len(self.df))

    def test_formation_flags_are_binary(self):
        result = accumulation_distribution_zones(self.df, pivot_length=5)
        self.assertTrue(result["adz_accumulation"].isin([0, 1]).all())
        self.assertTrue(result["adz_distribution"].isin([0, 1]).all())

    def test_custom_column_names(self):
        result = accumulation_distribution_zones(
            self.df,
            pivot_length=5,
            accumulation_column="acc",
            distribution_column="dist",
            zone_top_column="zt",
            zone_bottom_column="zb",
        )
        self.assertIn("acc", result.columns)
        self.assertIn("dist", result.columns)
        self.assertIn("zt", result.columns)
        self.assertIn("zb", result.columns)

    def test_different_pivot_lengths(self):
        for length in [3, 5, 8]:
            result = accumulation_distribution_zones(
                self.df, pivot_length=length
            )
            self.assertIsInstance(result, pd.DataFrame)

    def test_fast_mode(self):
        result = accumulation_distribution_zones(
            self.df, pivot_length=5, mode="fast"
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_slow_mode(self):
        result = accumulation_distribution_zones(
            self.df, pivot_length=5, mode="slow"
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_invalid_mode_raises(self):
        with self.assertRaises(PyIndicatorException):
            accumulation_distribution_zones(
                self.df, pivot_length=5, mode="invalid"
            )

    def test_invalid_input_raises(self):
        with self.assertRaises(PyIndicatorException):
            accumulation_distribution_zones([1, 2, 3])

    def test_does_not_modify_original(self):
        original_cols = list(self.df.columns)
        accumulation_distribution_zones(self.df, pivot_length=5)
        self.assertEqual(list(self.df.columns), original_cols)

    def test_zone_top_above_bottom(self):
        """Zone top should always be >= zone bottom."""
        result = accumulation_distribution_zones(self.df, pivot_length=5)
        active = result.dropna(
            subset=["adz_zone_top", "adz_zone_bottom"]
        )
        if len(active) > 0:
            self.assertTrue(
                (active["adz_zone_top"] >= active["adz_zone_bottom"]).all()
            )

    def test_zone_left_before_right(self):
        """Zone left bar index should always be < zone right."""
        result = accumulation_distribution_zones(self.df, pivot_length=5)
        active = result.dropna(
            subset=["adz_zone_left", "adz_zone_right"]
        )
        if len(active) > 0:
            self.assertTrue(
                (active["adz_zone_left"] <= active["adz_zone_right"]).all()
            )


class TestAccumulationDistributionZonesSignal(unittest.TestCase):
    """Tests for accumulation_distribution_zones_signal()."""

    def setUp(self):
        self.df = accumulation_distribution_zones(
            _make_converging_data(), pivot_length=5
        )

    def test_signal_values(self):
        result = accumulation_distribution_zones_signal(self.df)
        self.assertTrue(result["adz_signal"].isin([0, 1, -1]).all())

    def test_adds_signal_column(self):
        result = accumulation_distribution_zones_signal(self.df)
        self.assertIn("adz_signal", result.columns)

    def test_accumulation_maps_to_positive(self):
        result = accumulation_distribution_zones_signal(self.df)
        acc_mask = self.df["adz_accumulation"] == 1
        if acc_mask.sum() > 0:
            self.assertTrue(
                (result.loc[acc_mask, "adz_signal"] == 1).all()
            )

    def test_distribution_maps_to_negative(self):
        result = accumulation_distribution_zones_signal(self.df)
        dist_mask = self.df["adz_distribution"] == 1
        if dist_mask.sum() > 0:
            self.assertTrue(
                (result.loc[dist_mask, "adz_signal"] == -1).all()
            )

    def test_polars_signal(self):
        pl_df = pl.from_pandas(self.df)
        result = accumulation_distribution_zones_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn("adz_signal", result.columns)

    def test_custom_signal_column(self):
        result = accumulation_distribution_zones_signal(
            self.df, signal_column="my_sig"
        )
        self.assertIn("my_sig", result.columns)


class TestAccumulationDistributionZonesStats(unittest.TestCase):
    """Tests for get_accumulation_distribution_zones_stats()."""

    def setUp(self):
        self.df = accumulation_distribution_zones(
            _make_converging_data(), pivot_length=5
        )

    def test_stats_keys(self):
        stats = get_accumulation_distribution_zones_stats(self.df)
        expected_keys = [
            "total_accumulation", "total_distribution",
            "total", "accumulation_ratio",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_total_is_sum(self):
        stats = get_accumulation_distribution_zones_stats(self.df)
        self.assertEqual(
            stats["total"],
            stats["total_accumulation"] + stats["total_distribution"],
        )

    def test_ratio_range(self):
        stats = get_accumulation_distribution_zones_stats(self.df)
        self.assertGreaterEqual(stats["accumulation_ratio"], 0.0)
        self.assertLessEqual(stats["accumulation_ratio"], 1.0)

    def test_polars_stats(self):
        pl_df = pl.from_pandas(self.df)
        stats = get_accumulation_distribution_zones_stats(pl_df)
        self.assertIn("total", stats)


class TestAccumulationDistributionZonesEdgeCases(unittest.TestCase):
    """Edge cases and special scenarios."""

    def test_short_data(self):
        """Data shorter than pivot_length should not crash."""
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.5, 101.5, 102.5],
        })
        result = accumulation_distribution_zones(df, pivot_length=5)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["adz_accumulation"].sum(), 0)
        self.assertEqual(result["adz_distribution"].sum(), 0)

    def test_flat_market(self):
        """Flat market should produce no patterns."""
        n = 200
        flat = np.full(n, 100.0)
        df = pd.DataFrame({
            "Open": flat,
            "High": flat + 0.01,
            "Low": flat - 0.01,
            "Close": flat,
        })
        result = accumulation_distribution_zones(df, pivot_length=5)
        total = (
            result["adz_accumulation"].sum()
            + result["adz_distribution"].sum()
        )
        self.assertEqual(total, 0)

    def test_strong_uptrend(self):
        """Monotonic uptrend should produce no converging patterns."""
        n = 200
        t = np.arange(n, dtype=float)
        close = 100 + t * 0.5
        df = pd.DataFrame({
            "Open": close - 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
        })
        result = accumulation_distribution_zones(df, pivot_length=5)
        total = (
            result["adz_accumulation"].sum()
            + result["adz_distribution"].sum()
        )
        self.assertEqual(total, 0)


if __name__ == "__main__":
    unittest.main()
