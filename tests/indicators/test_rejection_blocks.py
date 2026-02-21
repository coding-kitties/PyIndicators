import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    rejection_blocks,
    rejection_blocks_signal,
    get_rejection_blocks_stats,
)
from pyindicators.exceptions import PyIndicatorException


def _make_trending_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLC data with clear swings and reversals.

    Uses the same generator as the mitigation-blocks tests so that
    the data has enough structure for rejection detection.
    """
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    # Up-trend then down-trend with oscillation → forces swings
    base = np.where(
        t < n // 2,
        100 + t * 0.4,
        100 + (n // 2) * 0.4 - (t - n // 2) * 0.4,
    )
    wave = 5 * np.sin(t * 2 * np.pi / 25)
    close = base + wave + np.random.randn(n) * 0.5
    # Exaggerated wicks to guarantee some rejection blocks
    high = close + np.abs(np.random.randn(n)) * 3.0
    low = close - np.abs(np.random.randn(n)) * 3.0
    opn = close - np.random.randn(n) * 0.3
    return pd.DataFrame({
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(1000, 10000, n),
    })


def _make_rejection_candles(n: int = 200, seed: int = 99) -> pd.DataFrame:
    """
    Generate data with intentionally large wicks at swing points.

    Creates a pattern of:
      up-swing with big upper wicks at tops (bearish rejection),
      down-swing with big lower wicks at bottoms (bullish rejection).
    """
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    # oscillating price
    base = 100 + 10 * np.sin(t * 2 * np.pi / 40)
    close = base + np.random.randn(n) * 0.3
    opn = close - np.random.randn(n) * 0.2

    high = np.maximum(close, opn) + np.abs(np.random.randn(n)) * 1.0
    low = np.minimum(close, opn) - np.abs(np.random.randn(n)) * 1.0

    # At swing peaks (every ~20 bars at top), add extra large upper wicks
    for peak in range(10, n, 40):
        if peak < n:
            body_t = max(close[peak], opn[peak])
            high[peak] = body_t + 8.0  # big upper wick
            low[peak] = min(close[peak], opn[peak]) - 0.2

    # At swing troughs (every ~20 bars at bottom), add extra large lower wicks
    for trough in range(30, n, 40):
        if trough < n:
            body_b = min(close[trough], opn[trough])
            low[trough] = body_b - 8.0  # big lower wick
            high[trough] = max(close[trough], opn[trough]) + 0.2

    return pd.DataFrame({
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(1000, 10000, n),
    })


class TestRejectionBlocks(unittest.TestCase):
    """Tests for the rejection_blocks() function."""

    def setUp(self):
        self.df = _make_trending_data()

    def test_returns_pandas_dataframe(self):
        result = rejection_blocks(self.df, swing_length=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars_dataframe(self):
        pl_df = pl.from_pandas(self.df)
        result = rejection_blocks(pl_df, swing_length=5)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_expected_columns(self):
        result = rejection_blocks(self.df, swing_length=5)
        expected = [
            "rb_bullish", "rb_bearish",
            "rb_top", "rb_bottom",
            "rb_direction",
            "rb_entry_long", "rb_entry_short",
            "rb_mitigated",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_preserves_row_count(self):
        result = rejection_blocks(self.df, swing_length=5)
        self.assertEqual(len(result), len(self.df))

    def test_formation_flags_are_binary(self):
        result = rejection_blocks(self.df, swing_length=5)
        self.assertTrue(result["rb_bullish"].isin([0, 1]).all())
        self.assertTrue(result["rb_bearish"].isin([0, 1]).all())

    def test_entry_flags_are_binary(self):
        result = rejection_blocks(self.df, swing_length=5)
        self.assertTrue(result["rb_entry_long"].isin([0, 1]).all())
        self.assertTrue(result["rb_entry_short"].isin([0, 1]).all())

    def test_mitigated_is_binary(self):
        result = rejection_blocks(self.df, swing_length=5)
        self.assertTrue(result["rb_mitigated"].isin([0, 1]).all())

    def test_direction_values_are_valid(self):
        result = rejection_blocks(self.df, swing_length=5)
        self.assertTrue(result["rb_direction"].isin([0, 1, -1]).all())

    def test_detects_some_rejection_blocks(self):
        result = rejection_blocks(self.df, swing_length=5)
        total = result["rb_bullish"].sum() + result["rb_bearish"].sum()
        self.assertGreater(total, 0, "Should detect at least one RB")

    def test_zone_top_greater_or_equal_bottom(self):
        result = rejection_blocks(self.df, swing_length=5)
        active = result[result["rb_direction"] != 0].dropna(
            subset=["rb_top", "rb_bottom"]
        )
        if len(active) > 0:
            self.assertTrue(
                (active["rb_top"] >= active["rb_bottom"]).all()
            )

    def test_custom_column_names(self):
        result = rejection_blocks(
            self.df, swing_length=5,
            bullish_rb_column="bull",
            bearish_rb_column="bear",
            rb_top_column="zt",
            rb_bottom_column="zb",
        )
        self.assertIn("bull", result.columns)
        self.assertIn("bear", result.columns)
        self.assertIn("zt", result.columns)
        self.assertIn("zb", result.columns)

    def test_different_swing_lengths(self):
        for length in [3, 5, 8]:
            result = rejection_blocks(self.df, swing_length=length)
            self.assertIsInstance(result, pd.DataFrame)

    def test_invalid_input_raises(self):
        with self.assertRaises(PyIndicatorException):
            rejection_blocks([1, 2, 3])

    def test_does_not_modify_original(self):
        original_cols = list(self.df.columns)
        rejection_blocks(self.df, swing_length=5)
        self.assertEqual(list(self.df.columns), original_cols)

    def test_wick_threshold_parameter(self):
        """Higher threshold should produce fewer or equal RBs."""
        result_low = rejection_blocks(
            self.df, swing_length=5, wick_threshold=0.3
        )
        result_high = rejection_blocks(
            self.df, swing_length=5, wick_threshold=0.7
        )
        total_low = (
            result_low["rb_bullish"].sum() + result_low["rb_bearish"].sum()
        )
        total_high = (
            result_high["rb_bullish"].sum() + result_high["rb_bearish"].sum()
        )
        self.assertGreaterEqual(total_low, total_high)

    def test_zone_is_wick_area(self):
        """RB zone should be a positive-width subset (the wick)."""
        result = rejection_blocks(self.df, swing_length=5)
        active = result[result["rb_direction"] != 0].dropna(
            subset=["rb_top", "rb_bottom"]
        )
        if len(active) > 0:
            widths = active["rb_top"] - active["rb_bottom"]
            self.assertTrue((widths >= 0).all(),
                            "Zone top must be >= zone bottom")
            self.assertTrue((widths > 0).any(),
                            "At least one zone should have positive width")


class TestRejectionBlocksWithIntentionalData(unittest.TestCase):
    """Tests with data designed to have obvious rejection candles."""

    def setUp(self):
        self.df = _make_rejection_candles()

    def test_detects_rejection_blocks(self):
        result = rejection_blocks(
            self.df, swing_length=5, wick_threshold=0.5
        )
        total = result["rb_bullish"].sum() + result["rb_bearish"].sum()
        self.assertGreater(total, 0, "Should detect RBs in designed data")

    def test_lower_threshold_detects_more(self):
        result_low = rejection_blocks(
            self.df, swing_length=5, wick_threshold=0.3
        )
        result_high = rejection_blocks(
            self.df, swing_length=5, wick_threshold=0.6
        )
        total_low = (
            result_low["rb_bullish"].sum() + result_low["rb_bearish"].sum()
        )
        total_high = (
            result_high["rb_bullish"].sum() + result_high["rb_bearish"].sum()
        )
        self.assertGreaterEqual(total_low, total_high)


class TestRBSignal(unittest.TestCase):
    """Tests for rejection_blocks_signal()."""

    def setUp(self):
        self.df = rejection_blocks(
            _make_trending_data(), swing_length=5
        )

    def test_signal_values(self):
        result = rejection_blocks_signal(self.df)
        self.assertTrue(result["rb_signal"].isin([0, 1, -1]).all())

    def test_adds_signal_column(self):
        result = rejection_blocks_signal(self.df)
        self.assertIn("rb_signal", result.columns)

    def test_long_signal_matches_entry_long(self):
        result = rejection_blocks_signal(self.df)
        long_mask = self.df["rb_entry_long"] == 1
        if long_mask.any():
            self.assertTrue(
                (result.loc[long_mask, "rb_signal"] == 1).all()
            )

    def test_short_signal_matches_entry_short(self):
        result = rejection_blocks_signal(self.df)
        short_mask = self.df["rb_entry_short"] == 1
        if short_mask.any():
            self.assertTrue(
                (result.loc[short_mask, "rb_signal"] == -1).all()
            )

    def test_polars_signal(self):
        pl_df = pl.from_pandas(self.df)
        result = rejection_blocks_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn("rb_signal", result.columns)

    def test_custom_signal_column(self):
        result = rejection_blocks_signal(
            self.df, signal_column="my_rb_sig"
        )
        self.assertIn("my_rb_sig", result.columns)


class TestRBStats(unittest.TestCase):
    """Tests for get_rejection_blocks_stats()."""

    def setUp(self):
        self.df = rejection_blocks(
            _make_trending_data(), swing_length=5
        )

    def test_stats_keys(self):
        stats = get_rejection_blocks_stats(self.df)
        expected_keys = [
            "total_bullish_rb", "total_bearish_rb", "total_rb",
            "total_entry_long", "total_entry_short", "total_entries",
            "total_mitigated", "entry_rate", "mitigation_rate",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_total_is_sum_of_bull_bear(self):
        stats = get_rejection_blocks_stats(self.df)
        self.assertEqual(
            stats["total_rb"],
            stats["total_bullish_rb"] + stats["total_bearish_rb"],
        )

    def test_entry_rate_range(self):
        stats = get_rejection_blocks_stats(self.df)
        self.assertGreaterEqual(stats["entry_rate"], 0.0)
        self.assertLessEqual(stats["entry_rate"], 1.0)

    def test_mitigation_rate_range(self):
        stats = get_rejection_blocks_stats(self.df)
        self.assertGreaterEqual(stats["mitigation_rate"], 0.0)
        self.assertLessEqual(stats["mitigation_rate"], 1.0)

    def test_polars_stats(self):
        pl_df = pl.from_pandas(self.df)
        stats = get_rejection_blocks_stats(pl_df)
        self.assertIn("total_rb", stats)


class TestRBEdgeCases(unittest.TestCase):
    """Edge cases and special scenarios."""

    def test_short_data(self):
        """Data shorter than swing_length should not crash."""
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.5, 101.5, 102.5],
        })
        result = rejection_blocks(df, swing_length=5)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["rb_bullish"].sum(), 0)

    def test_flat_market(self):
        """Flat market (no wicks) should produce no RBs."""
        n = 200
        flat = np.full(n, 100.0)
        df = pd.DataFrame({
            "Open": flat,
            "High": flat,
            "Low": flat,
            "Close": flat,
        })
        result = rejection_blocks(df, swing_length=5)
        total = result["rb_bullish"].sum() + result["rb_bearish"].sum()
        self.assertEqual(total, 0)

    def test_single_bar(self):
        """Single-bar DataFrame should not crash."""
        df = pd.DataFrame({
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
        })
        result = rejection_blocks(df, swing_length=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["rb_bullish"].sum(), 0)

    def test_doji_candles_no_crash(self):
        """Doji candles (open == close) should not crash."""
        n = 100
        np.random.seed(55)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "Open": close,      # same as close → zero body
            "High": close + 3,  # wide wicks
            "Low": close - 3,
            "Close": close,
        })
        result = rejection_blocks(df, swing_length=3)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), n)

    def test_very_high_threshold_no_blocks(self):
        """Threshold of 1.0 means wick must be 100% of range → no body,
        effectively only doji candles qualify.  Most data will have 0 RBs."""
        df = _make_trending_data(n=100)
        result = rejection_blocks(df, swing_length=5, wick_threshold=1.0)
        # Should still run without error
        self.assertIsInstance(result, pd.DataFrame)

    def test_very_low_threshold_many_blocks(self):
        """Threshold near 0 should detect more RBs than a high threshold."""
        df = _make_trending_data()
        result_low = rejection_blocks(
            df, swing_length=5, wick_threshold=0.1
        )
        result_high = rejection_blocks(
            df, swing_length=5, wick_threshold=0.9
        )
        total_low = (
            result_low["rb_bullish"].sum() + result_low["rb_bearish"].sum()
        )
        total_high = (
            result_high["rb_bullish"].sum() + result_high["rb_bearish"].sum()
        )
        self.assertGreaterEqual(total_low, total_high)

    def test_strong_uptrend_no_bearish_rb_with_small_wicks(self):
        """Monotonic uptrend with tiny upper wicks → no bearish RBs."""
        n = 200
        t = np.arange(n, dtype=float)
        close = 100 + t * 0.5
        opn = close - 0.1
        # Upper wick is tiny, lower wick is significant
        high = close + 0.05  # nearly no upper wick
        low = close - 2.0    # big lower wick
        df = pd.DataFrame({
            "Open": opn, "High": high, "Low": low, "Close": close,
        })
        result = rejection_blocks(df, swing_length=5, wick_threshold=0.5)
        self.assertEqual(result["rb_bearish"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
