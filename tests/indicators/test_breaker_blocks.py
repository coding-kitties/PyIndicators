import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    breaker_blocks,
    breaker_blocks_signal,
    get_breaker_blocks_stats,
)
from pyindicators.exceptions import PyIndicatorException


def _make_trending_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLC data with clear swing structure."""
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    # Up-trend then down-trend with oscillation → forces MSS events
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


class TestBreakerBlocks(unittest.TestCase):
    """Tests for the breaker_blocks() function."""

    def setUp(self):
        self.df = _make_trending_data()

    def test_returns_pandas_dataframe(self):
        result = breaker_blocks(self.df, swing_length=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars_dataframe(self):
        pl_df = pl.from_pandas(self.df)
        result = breaker_blocks(pl_df, swing_length=5)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_expected_columns(self):
        result = breaker_blocks(self.df, swing_length=5)
        expected = [
            "bb_bullish", "bb_bearish",
            "bb_top", "bb_bottom", "bb_center",
            "bb_direction",
            "bb_entry_long", "bb_entry_short",
            "bb_cancel", "bb_mitigated",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_preserves_row_count(self):
        result = breaker_blocks(self.df, swing_length=5)
        self.assertEqual(len(result), len(self.df))

    def test_formation_flags_are_binary(self):
        result = breaker_blocks(self.df, swing_length=5)
        self.assertTrue(result["bb_bullish"].isin([0, 1]).all())
        self.assertTrue(result["bb_bearish"].isin([0, 1]).all())

    def test_entry_flags_are_binary(self):
        result = breaker_blocks(self.df, swing_length=5)
        self.assertTrue(result["bb_entry_long"].isin([0, 1]).all())
        self.assertTrue(result["bb_entry_short"].isin([0, 1]).all())

    def test_cancel_and_mitigated_are_binary(self):
        result = breaker_blocks(self.df, swing_length=5)
        self.assertTrue(result["bb_cancel"].isin([0, 1]).all())
        self.assertTrue(result["bb_mitigated"].isin([0, 1]).all())

    def test_direction_values_are_valid(self):
        result = breaker_blocks(self.df, swing_length=5)
        self.assertTrue(result["bb_direction"].isin([0, 1, -1]).all())

    def test_detects_some_breaker_blocks(self):
        result = breaker_blocks(self.df, swing_length=5)
        total = result["bb_bullish"].sum() + result["bb_bearish"].sum()
        self.assertGreater(total, 0, "Should detect at least one BB")

    def test_center_is_midpoint_of_zone(self):
        result = breaker_blocks(self.df, swing_length=5)
        active = result[result["bb_direction"] != 0].dropna(
            subset=["bb_top", "bb_bottom", "bb_center"]
        )
        if len(active) > 0:
            expected_center = (active["bb_top"] + active["bb_bottom"]) / 2
            np.testing.assert_allclose(
                active["bb_center"].values,
                expected_center.values,
                atol=1e-10,
            )

    def test_custom_column_names(self):
        result = breaker_blocks(
            self.df,
            swing_length=5,
            bullish_bb_column="bull",
            bearish_bb_column="bear",
            bb_top_column="t",
            bb_bottom_column="b",
        )
        self.assertIn("bull", result.columns)
        self.assertIn("bear", result.columns)
        self.assertIn("t", result.columns)
        self.assertIn("b", result.columns)

    def test_different_swing_lengths(self):
        for length in [3, 5, 8]:
            result = breaker_blocks(self.df, swing_length=length)
            self.assertIsInstance(result, pd.DataFrame)

    def test_use_body_option(self):
        result = breaker_blocks(self.df, swing_length=5, use_body=True)
        self.assertIsInstance(result, pd.DataFrame)

    def test_use_2_candles_option(self):
        result = breaker_blocks(
            self.df, swing_length=5, use_2_candles=True
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_stop_at_first_center_break_false(self):
        result = breaker_blocks(
            self.df, swing_length=5,
            stop_at_first_center_break=False,
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_invalid_input_raises(self):
        with self.assertRaises(PyIndicatorException):
            breaker_blocks([1, 2, 3])

    def test_does_not_modify_original(self):
        original_cols = list(self.df.columns)
        breaker_blocks(self.df, swing_length=5)
        self.assertEqual(list(self.df.columns), original_cols)


class TestBreakerBlocksSignal(unittest.TestCase):
    """Tests for breaker_blocks_signal()."""

    def setUp(self):
        self.df = breaker_blocks(
            _make_trending_data(), swing_length=5
        )

    def test_signal_values(self):
        result = breaker_blocks_signal(self.df)
        self.assertTrue(result["bb_signal"].isin([0, 1, -1]).all())

    def test_adds_signal_column(self):
        result = breaker_blocks_signal(self.df)
        self.assertIn("bb_signal", result.columns)

    def test_long_signal_matches_entry_long(self):
        result = breaker_blocks_signal(self.df)
        long_mask = self.df["bb_entry_long"] == 1
        self.assertTrue((result.loc[long_mask, "bb_signal"] == 1).all())

    def test_short_signal_matches_entry_short(self):
        result = breaker_blocks_signal(self.df)
        short_mask = self.df["bb_entry_short"] == 1
        self.assertTrue((result.loc[short_mask, "bb_signal"] == -1).all())

    def test_polars_signal(self):
        pl_df = pl.from_pandas(self.df)
        result = breaker_blocks_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn("bb_signal", result.columns)

    def test_custom_signal_column(self):
        result = breaker_blocks_signal(
            self.df, signal_column="my_sig"
        )
        self.assertIn("my_sig", result.columns)


class TestBreakerBlocksStats(unittest.TestCase):
    """Tests for get_breaker_blocks_stats()."""

    def setUp(self):
        self.df = breaker_blocks(
            _make_trending_data(), swing_length=5
        )

    def test_stats_keys(self):
        stats = get_breaker_blocks_stats(self.df)
        expected_keys = [
            "total_bullish_bb", "total_bearish_bb", "total_bb",
            "total_entry_long", "total_entry_short",
            "total_cancels", "total_mitigated", "entry_rate",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_total_is_sum_of_bull_bear(self):
        stats = get_breaker_blocks_stats(self.df)
        self.assertEqual(
            stats["total_bb"],
            stats["total_bullish_bb"] + stats["total_bearish_bb"],
        )

    def test_entry_rate_range(self):
        stats = get_breaker_blocks_stats(self.df)
        self.assertGreaterEqual(stats["entry_rate"], 0.0)
        self.assertLessEqual(stats["entry_rate"], 1.0)

    def test_polars_stats(self):
        pl_df = pl.from_pandas(self.df)
        stats = get_breaker_blocks_stats(pl_df)
        self.assertIn("total_bb", stats)


class TestBreakerBlocksEdgeCases(unittest.TestCase):
    """Edge cases and special scenarios."""

    def test_short_data(self):
        """Data shorter than swing_length should not crash."""
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.5, 101.5, 102.5],
        })
        result = breaker_blocks(df, swing_length=5)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["bb_bullish"].sum(), 0)

    def test_flat_market(self):
        """Flat/sideways market should produce few or no BBs."""
        n = 200
        flat = np.full(n, 100.0)
        df = pd.DataFrame({
            "Open": flat,
            "High": flat + 0.01,
            "Low": flat - 0.01,
            "Close": flat,
        })
        result = breaker_blocks(df, swing_length=5)
        total = result["bb_bullish"].sum() + result["bb_bearish"].sum()
        self.assertEqual(total, 0)

    def test_strong_uptrend(self):
        """Monotonic uptrend with no reversals → no bearish BBs."""
        n = 200
        t = np.arange(n, dtype=float)
        close = 100 + t * 0.5
        df = pd.DataFrame({
            "Open": close - 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
        })
        result = breaker_blocks(df, swing_length=5)
        # Monotonic uptrend shouldn't produce bearish BBs
        self.assertEqual(result["bb_bearish"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
