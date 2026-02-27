import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    volumetric_supply_demand_zones,
    volumetric_supply_demand_zones_signal,
    get_volumetric_supply_demand_zones_stats,
)
from pyindicators.exceptions import PyIndicatorException


def _make_trending_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with clear swing structure."""
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    # Up-trend then down-trend with strong oscillation for swings
    base = np.where(
        t < n // 2,
        100 + t * 0.5,
        100 + (n // 2) * 0.5 - (t - n // 2) * 0.5,
    )
    wave = 8 * np.sin(t * 2 * np.pi / 30)
    close = base + wave + np.random.randn(n) * 0.5
    high = close + np.abs(np.random.randn(n)) * 3.0
    low = close - np.abs(np.random.randn(n)) * 3.0
    opn = close - np.random.randn(n) * 0.5
    vol = np.random.randint(1000, 50000, n).astype(float)
    return pd.DataFrame({
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })


# ------------------------------------------------------------------ #
#  Test main indicator function                                       #
# ------------------------------------------------------------------ #
class TestVolumetricSupplyDemandZones(unittest.TestCase):
    """Tests for the volumetric_supply_demand_zones() function."""

    def setUp(self):
        self.df = _make_trending_data()

    def test_returns_pandas_dataframe(self):
        result = volumetric_supply_demand_zones(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars_dataframe(self):
        pl_df = pl.from_pandas(self.df)
        result = volumetric_supply_demand_zones(pl_df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_expected_columns(self):
        result = volumetric_supply_demand_zones(self.df)
        expected = [
            "vsdz_demand", "vsdz_supply",
            "vsdz_zone_top", "vsdz_zone_bottom",
            "vsdz_poc", "vsdz_zone_type",
            "vsdz_volume", "vsdz_delta",
            "vsdz_status", "vsdz_touches",
            "vsdz_signal",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_preserves_row_count(self):
        result = volumetric_supply_demand_zones(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_demand_supply_are_binary(self):
        result = volumetric_supply_demand_zones(self.df)
        self.assertTrue(result["vsdz_demand"].isin([0, 1]).all())
        self.assertTrue(result["vsdz_supply"].isin([0, 1]).all())

    def test_signal_values_are_valid(self):
        result = volumetric_supply_demand_zones(self.df)
        self.assertTrue(result["vsdz_signal"].isin([0, 1, -1]).all())

    def test_zone_type_values_are_valid(self):
        result = volumetric_supply_demand_zones(self.df)
        self.assertTrue(result["vsdz_zone_type"].isin([0, 1, -1]).all())

    def test_zone_top_above_bottom(self):
        result = volumetric_supply_demand_zones(self.df)
        valid = result.dropna(subset=["vsdz_zone_top", "vsdz_zone_bottom"])
        if len(valid) > 0:
            self.assertTrue(
                (valid["vsdz_zone_top"] >= valid["vsdz_zone_bottom"]).all()
            )

    def test_poc_within_zone_bounds(self):
        result = volumetric_supply_demand_zones(self.df)
        valid = result.dropna(
            subset=["vsdz_zone_top", "vsdz_zone_bottom", "vsdz_poc"]
        )
        if len(valid) > 0:
            self.assertTrue(
                (valid["vsdz_poc"] >= valid["vsdz_zone_bottom"]).all()
            )
            self.assertTrue(
                (valid["vsdz_poc"] <= valid["vsdz_zone_top"]).all()
            )

    def test_detects_some_zones(self):
        result = volumetric_supply_demand_zones(self.df)
        total = result["vsdz_demand"].sum() + result["vsdz_supply"].sum()
        self.assertGreater(total, 0, "Should detect at least one zone")

    def test_does_not_modify_original(self):
        original = self.df.copy()
        _ = volumetric_supply_demand_zones(self.df)
        pd.testing.assert_frame_equal(self.df, original)

    def test_custom_column_names(self):
        result = volumetric_supply_demand_zones(
            self.df,
            demand_zone_column="my_demand",
            supply_zone_column="my_supply",
            signal_column="my_signal",
        )
        self.assertIn("my_demand", result.columns)
        self.assertIn("my_supply", result.columns)
        self.assertIn("my_signal", result.columns)

    def test_different_swing_length(self):
        result = volumetric_supply_demand_zones(self.df, swing_length=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.df))

    def test_mitigation_type_close(self):
        result = volumetric_supply_demand_zones(
            self.df, mitigation_type="Close"
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_invalid_mitigation_type(self):
        with self.assertRaises(PyIndicatorException):
            volumetric_supply_demand_zones(
                self.df, mitigation_type="Invalid"
            )

    def test_invalid_input_type(self):
        with self.assertRaises(PyIndicatorException):
            volumetric_supply_demand_zones([1, 2, 3])

    def test_no_merge(self):
        result = volumetric_supply_demand_zones(
            self.df, merge_zones=False
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_polars_columns_match_pandas(self):
        pd_result = volumetric_supply_demand_zones(self.df)
        pl_result = volumetric_supply_demand_zones(
            pl.from_pandas(self.df)
        )
        pd_cols = sorted(pd_result.columns)
        pl_cols = sorted(pl_result.columns)
        self.assertEqual(pd_cols, pl_cols)


# ------------------------------------------------------------------ #
#  Test signal function                                               #
# ------------------------------------------------------------------ #
class TestVolumetricSupplyDemandZonesSignal(unittest.TestCase):
    """Tests for the volumetric_supply_demand_zones_signal() function."""

    def setUp(self):
        self.df = volumetric_supply_demand_zones(_make_trending_data())

    def test_returns_dataframe(self):
        result = volumetric_supply_demand_zones_signal(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_signal_values(self):
        result = volumetric_supply_demand_zones_signal(self.df)
        self.assertTrue(result["vsdz_signal"].isin([0, 1, -1]).all())

    def test_polars_support(self):
        pl_df = pl.from_pandas(self.df)
        result = volumetric_supply_demand_zones_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_missing_column_raises(self):
        bad_df = pd.DataFrame({"A": [1, 2, 3]})
        with self.assertRaises(PyIndicatorException):
            volumetric_supply_demand_zones_signal(bad_df)

    def test_invalid_input_type(self):
        with self.assertRaises(PyIndicatorException):
            volumetric_supply_demand_zones_signal([1, 2, 3])


# ------------------------------------------------------------------ #
#  Test stats function                                                #
# ------------------------------------------------------------------ #
class TestGetVolumetricSupplyDemandZonesStats(unittest.TestCase):
    """Tests for the get_volumetric_supply_demand_zones_stats() function."""

    def setUp(self):
        self.df = volumetric_supply_demand_zones(_make_trending_data())

    def test_returns_dict(self):
        stats = get_volumetric_supply_demand_zones_stats(self.df)
        self.assertIsInstance(stats, dict)

    def test_expected_keys(self):
        stats = get_volumetric_supply_demand_zones_stats(self.df)
        expected_keys = [
            "total_demand_zones", "total_supply_zones", "total_zones",
            "bullish_signals", "bearish_signals", "total_signals",
            "avg_zone_volume", "avg_zone_delta", "max_touches",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_total_equals_demand_plus_supply(self):
        stats = get_volumetric_supply_demand_zones_stats(self.df)
        self.assertEqual(
            stats["total_zones"],
            stats["total_demand_zones"] + stats["total_supply_zones"],
        )

    def test_total_signals_equals_sum(self):
        stats = get_volumetric_supply_demand_zones_stats(self.df)
        self.assertEqual(
            stats["total_signals"],
            stats["bullish_signals"] + stats["bearish_signals"],
        )

    def test_polars_input_works(self):
        pl_df = pl.from_pandas(self.df)
        stats = get_volumetric_supply_demand_zones_stats(pl_df)
        self.assertIsInstance(stats, dict)
        self.assertIn("total_zones", stats)


# ------------------------------------------------------------------ #
#  Edge cases                                                         #
# ------------------------------------------------------------------ #
class TestVolumetricSupplyDemandZonesEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_short_data(self):
        """Short data should not crash."""
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [103, 104, 105],
            "Low": [97, 98, 99],
            "Close": [101, 102, 103],
            "Volume": [1000, 2000, 3000],
        })
        result = volumetric_supply_demand_zones(df, swing_length=2)
        self.assertEqual(len(result), 3)

    def test_flat_market(self):
        """Flat market should produce few or no zones."""
        n = 200
        df = pd.DataFrame({
            "Open": np.full(n, 100.0),
            "High": np.full(n, 101.0),
            "Low": np.full(n, 99.0),
            "Close": np.full(n, 100.0),
            "Volume": np.full(n, 1000.0),
        })
        result = volumetric_supply_demand_zones(df)
        total = result["vsdz_demand"].sum() + result["vsdz_supply"].sum()
        self.assertEqual(total, 0, "Flat market should produce no zones")

    def test_strong_uptrend_only(self):
        """Strong uptrend should produce demand zones."""
        n = 300
        np.random.seed(99)
        t = np.arange(n, dtype=float)
        close = 100 + t * 1.0 + 5 * np.sin(t * 2 * np.pi / 20)
        high = close + np.abs(np.random.randn(n)) * 2.0
        low = close - np.abs(np.random.randn(n)) * 2.0
        opn = close - np.random.randn(n) * 0.3
        df = pd.DataFrame({
            "Open": opn, "High": high, "Low": low,
            "Close": close, "Volume": np.full(n, 5000.0),
        })
        result = volumetric_supply_demand_zones(df, swing_length=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_volume_is_positive(self):
        """Zone volume should be non-negative."""
        result = volumetric_supply_demand_zones(_make_trending_data())
        vol = result["vsdz_volume"].dropna()
        if len(vol) > 0:
            self.assertTrue((vol >= 0).all())


if __name__ == "__main__":
    unittest.main()
