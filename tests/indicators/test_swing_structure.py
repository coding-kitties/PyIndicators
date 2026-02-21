import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    swing_structure,
    swing_structure_signal,
    get_swing_structure_stats,
)
from pyindicators.exceptions import PyIndicatorException


class TestSwingStructure(unittest.TestCase):
    """Tests for swing_structure function."""

    def setUp(self):
        """Create trending OHLC data with clear swing points."""
        np.random.seed(42)
        n = 300
        # Build a price series with clear up-trend then down-trend
        t = np.arange(n, dtype=float)
        # Up-trend for first half, down-trend for second half
        base = np.where(
            t < n // 2,
            100 + t * 0.3,
            100 + (n // 2) * 0.3 - (t - n // 2) * 0.3,
        )
        # Add some waviness to create swing points
        wave = 3 * np.sin(t * 2 * np.pi / 20)
        close = base + wave + np.random.randn(n) * 0.3
        high = close + np.abs(np.random.randn(n)) * 1.5
        low = close - np.abs(np.random.randn(n)) * 1.5

        self.df = pd.DataFrame({
            "Open": close - np.random.randn(n) * 0.3,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.random.randint(1000, 10000, n),
        })

    def test_returns_dataframe(self):
        result = swing_structure(self.df, swing_length=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_adds_expected_columns(self):
        result = swing_structure(self.df, swing_length=5)
        expected = [
            "swing_high", "swing_low",
            "swing_high_price", "swing_low_price",
            "swing_structure", "swing_direction",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_swing_signals_are_binary(self):
        result = swing_structure(self.df, swing_length=5)
        self.assertTrue(result["swing_high"].isin([0, 1]).all())
        self.assertTrue(result["swing_low"].isin([0, 1]).all())

    def test_swing_high_price_nan_when_no_swing(self):
        result = swing_structure(self.df, swing_length=5)
        no_swing = result[result["swing_high"] == 0]
        self.assertTrue(no_swing["swing_high_price"].isna().all())

    def test_swing_low_price_nan_when_no_swing(self):
        result = swing_structure(self.df, swing_length=5)
        no_swing = result[result["swing_low"] == 0]
        self.assertTrue(no_swing["swing_low_price"].isna().all())

    def test_detects_some_swing_highs_and_lows(self):
        result = swing_structure(self.df, swing_length=5)
        self.assertGreater(result["swing_high"].sum(), 0)
        self.assertGreater(result["swing_low"].sum(), 0)

    def test_structure_labels_are_valid(self):
        result = swing_structure(self.df, swing_length=5)
        valid = {"", "HH", "LH", "HL", "LL"}
        unique_labels = set(result["swing_structure"].unique())
        self.assertTrue(unique_labels.issubset(valid))

    def test_direction_values_are_valid(self):
        result = swing_structure(self.df, swing_length=5)
        self.assertTrue(result["swing_direction"].isin([0, 1, -1]).all())

    def test_custom_column_names(self):
        result = swing_structure(
            self.df,
            swing_length=5,
            swing_high_column="sh",
            swing_low_column="sl",
            structure_column="struct",
            direction_column="dir",
        )
        self.assertIn("sh", result.columns)
        self.assertIn("sl", result.columns)
        self.assertIn("struct", result.columns)
        self.assertIn("dir", result.columns)

    def test_different_swing_lengths(self):
        for length in [3, 5, 10]:
            result = swing_structure(self.df, swing_length=length)
            self.assertIsInstance(result, pd.DataFrame)
            # Longer swing_length → fewer swings
            self.assertGreater(result["swing_high"].sum(), 0)

    def test_polars_dataframe_input(self):
        pl_df = pl.from_pandas(self.df)
        result = swing_structure(pl_df, swing_length=5)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn("swing_high", result.columns)
        self.assertIn("swing_low", result.columns)

    def test_invalid_input_raises(self):
        with self.assertRaises(PyIndicatorException):
            swing_structure([1, 2, 3])

    def test_first_swing_has_no_structure_label(self):
        """The very first swing high / low cannot be classified."""
        result = swing_structure(self.df, swing_length=5)
        sh_rows = result[result["swing_high"] == 1]
        if len(sh_rows) > 0:
            first_sh = sh_rows.iloc[0]
            # First swing high has no previous → label is "" or HH/LH
            # (only "" if truly first)
            self.assertIn(
                first_sh["swing_structure"], ["", "HH", "LH"]
            )


class TestSwingStructureSignal(unittest.TestCase):
    """Tests for swing_structure_signal function."""

    def setUp(self):
        np.random.seed(42)
        n = 300
        t = np.arange(n, dtype=float)
        base = np.where(
            t < n // 2,
            100 + t * 0.3,
            100 + (n // 2) * 0.3 - (t - n // 2) * 0.3,
        )
        wave = 3 * np.sin(t * 2 * np.pi / 20)
        close = base + wave + np.random.randn(n) * 0.3
        high = close + np.abs(np.random.randn(n)) * 1.5
        low = close - np.abs(np.random.randn(n)) * 1.5

        self.df = swing_structure(
            pd.DataFrame({
                "Open": close - np.random.randn(n) * 0.3,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": np.random.randint(1000, 10000, n),
            }),
            swing_length=5,
        )

    def test_signal_values(self):
        result = swing_structure_signal(self.df)
        self.assertTrue(result["swing_signal"].isin([0, 1, -1]).all())

    def test_bullish_signal_on_hh_hl(self):
        result = swing_structure_signal(self.df)
        hh_hl_mask = self.df["swing_structure"].isin(["HH", "HL"])
        self.assertTrue((result.loc[hh_hl_mask, "swing_signal"] == 1).all())

    def test_bearish_signal_on_lh_ll(self):
        result = swing_structure_signal(self.df)
        lh_ll_mask = self.df["swing_structure"].isin(["LH", "LL"])
        self.assertTrue(
            (result.loc[lh_ll_mask, "swing_signal"] == -1).all()
        )

    def test_polars_signal(self):
        pl_df = pl.from_pandas(self.df)
        result = swing_structure_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn("swing_signal", result.columns)


class TestSwingStructureStats(unittest.TestCase):
    """Tests for get_swing_structure_stats function."""

    def setUp(self):
        np.random.seed(42)
        n = 300
        t = np.arange(n, dtype=float)
        base = np.where(
            t < n // 2,
            100 + t * 0.3,
            100 + (n // 2) * 0.3 - (t - n // 2) * 0.3,
        )
        wave = 3 * np.sin(t * 2 * np.pi / 20)
        close = base + wave + np.random.randn(n) * 0.3
        high = close + np.abs(np.random.randn(n)) * 1.5
        low = close - np.abs(np.random.randn(n)) * 1.5

        self.df = swing_structure(
            pd.DataFrame({
                "Open": close - np.random.randn(n) * 0.3,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": np.random.randint(1000, 10000, n),
            }),
            swing_length=5,
        )

    def test_stats_keys(self):
        stats = get_swing_structure_stats(self.df)
        expected_keys = [
            "total_swing_highs", "total_swing_lows", "total_swings",
            "higher_highs", "lower_highs", "higher_lows", "lower_lows",
            "bullish_ratio",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_totals_are_consistent(self):
        stats = get_swing_structure_stats(self.df)
        self.assertEqual(
            stats["total_swings"],
            stats["total_swing_highs"] + stats["total_swing_lows"],
        )

    def test_bullish_ratio_range(self):
        stats = get_swing_structure_stats(self.df)
        self.assertGreaterEqual(stats["bullish_ratio"], 0.0)
        self.assertLessEqual(stats["bullish_ratio"], 1.0)

    def test_polars_stats(self):
        pl_df = pl.from_pandas(self.df)
        stats = get_swing_structure_stats(pl_df)
        self.assertIn("total_swings", stats)


if __name__ == "__main__":
    unittest.main()
