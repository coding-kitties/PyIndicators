import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    opening_gap,
    opening_gap_signal,
    get_opening_gap_stats,
)
from pyindicators.exceptions import PyIndicatorException


def _make_og_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate OHLC data with some opening gaps."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 2.0)
    high = close + np.abs(np.random.randn(n)) * 3.0
    low = close - np.abs(np.random.randn(n)) * 3.0
    opn = close - np.random.randn(n) * 0.5
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close,
    })


def _make_clear_bullish_og() -> pd.DataFrame:
    """Create data with a guaranteed bullish OG at bar 2."""
    # Bullish OG: low[2] > high[0]
    # Bar 0: H=102
    # Bar 2: L=105 → 105 > 102 ✓
    return pd.DataFrame({
        "Open":  [100, 103, 107, 108, 106],
        "High":  [102, 105, 110, 111, 109],
        "Low":   [ 98, 101, 105, 106, 104],
        "Close": [101, 104, 108, 109, 107],
    })


def _make_clear_bearish_og() -> pd.DataFrame:
    """Create data with a guaranteed bearish OG at bar 2."""
    # Bearish OG: high[2] < low[1]
    # Bar 1: L=103
    # Bar 2: H=100 → 100 < 103 ✓
    return pd.DataFrame({
        "Open":  [110, 108, 98,  97, 99],
        "High":  [112, 110, 100, 99, 101],
        "Low":   [108, 103, 95,  94, 97],
        "Close": [109, 105, 97,  96, 98],
    })


class TestOpeningGap(unittest.TestCase):
    """Tests for opening_gap()."""

    def test_returns_pandas(self):
        result = opening_gap(_make_og_data())
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars(self):
        df = pl.from_pandas(_make_og_data())
        result = opening_gap(df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_columns(self):
        result = opening_gap(_make_og_data())
        for col in ["bullish_og", "bearish_og",
                     "bullish_og_top", "bullish_og_bottom",
                     "bearish_og_top", "bearish_og_bottom"]:
            self.assertIn(col, result.columns)

    def test_preserves_row_count(self):
        df = _make_og_data()
        result = opening_gap(df)
        self.assertEqual(len(result), len(df))

    def test_flags_are_binary(self):
        result = opening_gap(_make_og_data())
        self.assertTrue(result["bullish_og"].isin([0, 1]).all())
        self.assertTrue(result["bearish_og"].isin([0, 1]).all())

    def test_bullish_og_detected(self):
        result = opening_gap(_make_clear_bullish_og())
        self.assertEqual(result["bullish_og"].iloc[2], 1)
        self.assertFalse(np.isnan(result["bullish_og_top"].iloc[2]))

    def test_bearish_og_detected(self):
        result = opening_gap(_make_clear_bearish_og())
        self.assertEqual(result["bearish_og"].iloc[2], 1)
        self.assertFalse(np.isnan(result["bearish_og_top"].iloc[2]))

    def test_no_og_at_first_two_bars(self):
        result = opening_gap(_make_og_data())
        self.assertEqual(result["bullish_og"].iloc[0], 0)
        self.assertEqual(result["bullish_og"].iloc[1], 0)
        self.assertEqual(result["bearish_og"].iloc[0], 0)
        self.assertEqual(result["bearish_og"].iloc[1], 0)

    def test_zone_boundaries_bullish(self):
        result = opening_gap(_make_clear_bullish_og())
        top = result["bullish_og_top"].iloc[2]
        bot = result["bullish_og_bottom"].iloc[2]
        self.assertGreater(top, bot)

    def test_zone_boundaries_bearish(self):
        result = opening_gap(_make_clear_bearish_og())
        top = result["bearish_og_top"].iloc[2]
        bot = result["bearish_og_bottom"].iloc[2]
        self.assertGreater(top, bot)

    def test_polars_matches_pandas(self):
        pdf = _make_clear_bullish_og()
        pr = opening_gap(pdf.copy())
        plr = opening_gap(pl.from_pandas(pdf)).to_pandas()
        np.testing.assert_array_equal(
            pr["bullish_og"].values, plr["bullish_og"].values
        )

    def test_custom_columns(self):
        result = opening_gap(
            _make_og_data(),
            bullish_og_column="my_bull",
            bearish_og_column="my_bear",
        )
        self.assertIn("my_bull", result.columns)
        self.assertIn("my_bear", result.columns)

    def test_no_overlap_no_gap(self):
        """Overlapping candles should not trigger OG."""
        df = pd.DataFrame({
            "Open":  [100, 101, 102, 103, 104],
            "High":  [102, 103, 104, 105, 106],
            "Low":   [ 98,  99, 100, 101, 102],
            "Close": [101, 102, 103, 104, 105],
        })
        result = opening_gap(df)
        self.assertEqual(result["bullish_og"].sum(), 0)

    def test_invalid_input(self):
        with self.assertRaises(PyIndicatorException):
            opening_gap("not a dataframe")


class TestOpeningGapSignal(unittest.TestCase):
    """Tests for opening_gap_signal()."""

    def test_signal_column(self):
        df = opening_gap(_make_og_data())
        result = opening_gap_signal(df)
        self.assertIn("og_signal", result.columns)

    def test_signal_values(self):
        df = opening_gap(_make_og_data())
        result = opening_gap_signal(df)
        self.assertTrue(result["og_signal"].isin([1, 0, -1]).all())

    def test_bullish_signal(self):
        df = opening_gap(_make_clear_bullish_og())
        result = opening_gap_signal(df)
        self.assertEqual(result["og_signal"].iloc[2], 1)

    def test_bearish_signal(self):
        df = opening_gap(_make_clear_bearish_og())
        result = opening_gap_signal(df)
        self.assertEqual(result["og_signal"].iloc[2], -1)

    def test_polars_signal(self):
        df = opening_gap(_make_og_data())
        result = opening_gap_signal(pl.from_pandas(df))
        self.assertIsInstance(result, pl.DataFrame)


class TestOpeningGapStats(unittest.TestCase):
    """Tests for get_opening_gap_stats()."""

    def test_stats_keys(self):
        df = opening_gap(_make_og_data())
        stats = get_opening_gap_stats(df)
        for key in ["total_bullish", "total_bearish", "total",
                     "bullish_pct", "bearish_pct"]:
            self.assertIn(key, stats)

    def test_stats_consistency(self):
        df = opening_gap(_make_og_data())
        stats = get_opening_gap_stats(df)
        self.assertEqual(
            stats["total"],
            stats["total_bullish"] + stats["total_bearish"],
        )

    def test_polars_stats(self):
        df = opening_gap(pl.from_pandas(_make_og_data()))
        stats = get_opening_gap_stats(df)
        self.assertIsInstance(stats["total"], int)


if __name__ == "__main__":
    unittest.main()
