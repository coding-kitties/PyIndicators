import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    volume_imbalance,
    volume_imbalance_signal,
    get_volume_imbalance_stats,
)
from pyindicators.exceptions import PyIndicatorException


def _make_vi_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate OHLC data with some volume imbalances."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    high = close + np.abs(np.random.randn(n)) * 2.0
    low = close - np.abs(np.random.randn(n)) * 2.0
    opn = close - np.random.randn(n) * 0.5
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close,
    })


def _make_clear_bullish_vi() -> pd.DataFrame:
    """Create data with a guaranteed bullish Volume Imbalance at bar 2."""
    # Bar 0: O=100, C=101 → body [100, 101], H=102
    # Bar 1: O=100, C=101 → body [100, 101], H=102
    # Bar 2: O=105, C=108 → body [105, 108]
    #   prev_body_hi=101, cur_body_lo=105: 101 < 105 ✓
    #   prev_high=102 < 105 ✓
    #   close[2]=108 > close[1]=101 ✓
    #   open[2]=105 > open[1]=100 ✓  → bullish VI
    return pd.DataFrame({
        "Open":  [100, 100, 105, 106, 104],
        "High":  [102, 102, 112, 110, 107],
        "Low":   [ 98,  98, 104, 104, 102],
        "Close": [101, 101, 108, 107, 105],
    })


def _make_clear_bearish_vi() -> pd.DataFrame:
    """Create data with a guaranteed bearish Volume Imbalance at bar 2."""
    # Bar 1: O=108, C=105 → body [105, 108], L=103
    # Bar 2: O=100, C=98 → body [98, 100]
    #   cur_body_hi=100, prev_body_lo=105: 100 < 105 ✓
    #   prev_low=103 > 100  ✓
    #   close[2]=98 < close[1]=105 ✓
    #   open[2]=100 < open[1]=108 ✓ → bearish VI
    return pd.DataFrame({
        "Open":  [110, 108, 100,  99, 101],
        "High":  [112, 110, 102, 101, 103],
        "Low":   [108, 103,  96,  97,  99],
        "Close": [109, 105,  98,  98, 100],
    })


class TestVolumeImbalance(unittest.TestCase):
    """Tests for volume_imbalance()."""

    def test_returns_pandas(self):
        df = _make_vi_data()
        result = volume_imbalance(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars(self):
        df = pl.from_pandas(_make_vi_data())
        result = volume_imbalance(df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_columns(self):
        result = volume_imbalance(_make_vi_data())
        for col in ["bullish_vi", "bearish_vi",
                     "bullish_vi_top", "bullish_vi_bottom",
                     "bearish_vi_top", "bearish_vi_bottom"]:
            self.assertIn(col, result.columns)

    def test_preserves_row_count(self):
        df = _make_vi_data()
        result = volume_imbalance(df)
        self.assertEqual(len(result), len(df))

    def test_flags_are_binary(self):
        result = volume_imbalance(_make_vi_data())
        self.assertTrue(result["bullish_vi"].isin([0, 1]).all())
        self.assertTrue(result["bearish_vi"].isin([0, 1]).all())

    def test_bullish_vi_detected(self):
        df = _make_clear_bullish_vi()
        result = volume_imbalance(df)
        self.assertEqual(result["bullish_vi"].iloc[2], 1)
        self.assertFalse(np.isnan(result["bullish_vi_top"].iloc[2]))
        self.assertFalse(np.isnan(result["bullish_vi_bottom"].iloc[2]))

    def test_bearish_vi_detected(self):
        df = _make_clear_bearish_vi()
        result = volume_imbalance(df)
        self.assertEqual(result["bearish_vi"].iloc[2], 1)
        self.assertFalse(np.isnan(result["bearish_vi_top"].iloc[2]))
        self.assertFalse(np.isnan(result["bearish_vi_bottom"].iloc[2]))

    def test_no_vi_at_bar_zero(self):
        result = volume_imbalance(_make_vi_data())
        self.assertEqual(result["bullish_vi"].iloc[0], 0)
        self.assertEqual(result["bearish_vi"].iloc[0], 0)

    def test_zone_boundaries_bullish(self):
        df = _make_clear_bullish_vi()
        result = volume_imbalance(df)
        idx = 2
        top = result["bullish_vi_top"].iloc[idx]
        bot = result["bullish_vi_bottom"].iloc[idx]
        self.assertGreater(top, bot)

    def test_zone_boundaries_bearish(self):
        df = _make_clear_bearish_vi()
        result = volume_imbalance(df)
        idx = 2
        top = result["bearish_vi_top"].iloc[idx]
        bot = result["bearish_vi_bottom"].iloc[idx]
        self.assertGreater(top, bot)

    def test_polars_matches_pandas(self):
        pdf = _make_clear_bullish_vi()
        pr = volume_imbalance(pdf.copy())
        plr = volume_imbalance(pl.from_pandas(pdf))
        plr_pd = plr.to_pandas()
        np.testing.assert_array_equal(
            pr["bullish_vi"].values, plr_pd["bullish_vi"].values
        )

    def test_custom_columns(self):
        df = _make_vi_data()
        result = volume_imbalance(
            df, bullish_vi_column="my_bull", bearish_vi_column="my_bear"
        )
        self.assertIn("my_bull", result.columns)
        self.assertIn("my_bear", result.columns)

    def test_invalid_input(self):
        with self.assertRaises(PyIndicatorException):
            volume_imbalance([1, 2, 3])


class TestVolumeImbalanceSignal(unittest.TestCase):
    """Tests for volume_imbalance_signal()."""

    def test_signal_column_added(self):
        df = volume_imbalance(_make_vi_data())
        result = volume_imbalance_signal(df)
        self.assertIn("vi_signal", result.columns)

    def test_signal_values(self):
        df = volume_imbalance(_make_vi_data())
        result = volume_imbalance_signal(df)
        self.assertTrue(result["vi_signal"].isin([1, 0, -1]).all())

    def test_bullish_signal(self):
        df = volume_imbalance(_make_clear_bullish_vi())
        result = volume_imbalance_signal(df)
        self.assertEqual(result["vi_signal"].iloc[2], 1)

    def test_bearish_signal(self):
        df = volume_imbalance(_make_clear_bearish_vi())
        result = volume_imbalance_signal(df)
        self.assertEqual(result["vi_signal"].iloc[2], -1)

    def test_polars_signal(self):
        df = volume_imbalance(_make_vi_data())
        pl_df = pl.from_pandas(df)
        result = volume_imbalance_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)


class TestVolumeImbalanceStats(unittest.TestCase):
    """Tests for get_volume_imbalance_stats()."""

    def test_stats_keys(self):
        df = volume_imbalance(_make_vi_data())
        stats = get_volume_imbalance_stats(df)
        for key in ["total_bullish", "total_bearish", "total",
                     "bullish_pct", "bearish_pct"]:
            self.assertIn(key, stats)

    def test_stats_types(self):
        df = volume_imbalance(_make_vi_data())
        stats = get_volume_imbalance_stats(df)
        self.assertIsInstance(stats["total"], int)
        self.assertIsInstance(stats["bullish_pct"], float)

    def test_stats_consistency(self):
        df = volume_imbalance(_make_vi_data())
        stats = get_volume_imbalance_stats(df)
        self.assertEqual(
            stats["total"],
            stats["total_bullish"] + stats["total_bearish"],
        )

    def test_polars_stats(self):
        df = volume_imbalance(pl.from_pandas(_make_vi_data()))
        stats = get_volume_imbalance_stats(df)
        self.assertIsInstance(stats["total"], int)


if __name__ == "__main__":
    unittest.main()
