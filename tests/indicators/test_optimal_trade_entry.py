import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    optimal_trade_entry,
    optimal_trade_entry_signal,
    get_optimal_trade_entry_stats,
)
from pyindicators.indicators.optimal_trade_entry import OTE_FIB_LEVELS
from pyindicators.exceptions import PyIndicatorException


def _make_trending_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLC data with clear swings and reversals."""
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


class TestOptimalTradeEntry(unittest.TestCase):
    """Tests for the optimal_trade_entry() function."""

    def setUp(self):
        self.df = _make_trending_data()

    def test_returns_pandas_dataframe(self):
        result = optimal_trade_entry(self.df, swing_length=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars_dataframe(self):
        pl_df = pl.from_pandas(self.df)
        result = optimal_trade_entry(pl_df, swing_length=5)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_expected_columns(self):
        result = optimal_trade_entry(self.df, swing_length=5)
        expected = [
            "ote_bullish", "ote_bearish",
            "ote_zone_top", "ote_zone_bottom",
            "ote_entry_long", "ote_entry_short",
            "ote_invalidated", "ote_direction",
            "ote_impulse_high", "ote_impulse_low",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_adds_fibonacci_level_columns(self):
        result = optimal_trade_entry(self.df, swing_length=5)
        for key in OTE_FIB_LEVELS:
            col = f"ote_{key}"
            self.assertIn(col, result.columns)

    def test_preserves_row_count(self):
        result = optimal_trade_entry(self.df, swing_length=5)
        self.assertEqual(len(result), len(self.df))

    def test_formation_flags_are_binary(self):
        result = optimal_trade_entry(self.df, swing_length=5)
        self.assertTrue(result["ote_bullish"].isin([0, 1]).all())
        self.assertTrue(result["ote_bearish"].isin([0, 1]).all())

    def test_entry_flags_are_binary(self):
        result = optimal_trade_entry(self.df, swing_length=5)
        self.assertTrue(result["ote_entry_long"].isin([0, 1]).all())
        self.assertTrue(result["ote_entry_short"].isin([0, 1]).all())

    def test_invalidated_is_binary(self):
        result = optimal_trade_entry(self.df, swing_length=5)
        self.assertTrue(result["ote_invalidated"].isin([0, 1]).all())

    def test_direction_values_are_valid(self):
        result = optimal_trade_entry(self.df, swing_length=5)
        self.assertTrue(result["ote_direction"].isin([0, 1, -1]).all())

    def test_detects_some_ote_zones(self):
        result = optimal_trade_entry(self.df, swing_length=5)
        total = result["ote_bullish"].sum() + result["ote_bearish"].sum()
        self.assertGreater(total, 0, "Should detect at least one OTE")

    def test_zone_top_greater_than_bottom(self):
        result = optimal_trade_entry(self.df, swing_length=5)
        active = result[result["ote_direction"] != 0].dropna(
            subset=["ote_zone_top", "ote_zone_bottom"]
        )
        if len(active) > 0:
            self.assertTrue(
                (active["ote_zone_top"] >= active["ote_zone_bottom"]).all()
            )

    def test_impulse_high_greater_than_low(self):
        result = optimal_trade_entry(self.df, swing_length=5)
        active = result.dropna(
            subset=["ote_impulse_high", "ote_impulse_low"]
        )
        if len(active) > 0:
            self.assertTrue(
                (active["ote_impulse_high"] >
                 active["ote_impulse_low"]).all()
            )

    def test_ote_zone_within_impulse_leg(self):
        """OTE zone should sit within the impulse leg boundaries."""
        result = optimal_trade_entry(self.df, swing_length=5)
        active = result.dropna(
            subset=["ote_zone_top", "ote_zone_bottom",
                     "ote_impulse_high", "ote_impulse_low"]
        )
        if len(active) > 0:
            self.assertTrue(
                (active["ote_zone_top"] <= active["ote_impulse_high"]).all()
            )
            self.assertTrue(
                (active["ote_zone_bottom"] >= active["ote_impulse_low"]).all()
            )

    def test_custom_fib_levels(self):
        """Custom OTE zone boundaries (e.g. 50%–70.5%)."""
        result = optimal_trade_entry(
            self.df, swing_length=5,
            ote_fib_start=0.5, ote_fib_end=0.705,
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_custom_column_names(self):
        result = optimal_trade_entry(
            self.df, swing_length=5,
            ote_bullish_column="bull",
            ote_bearish_column="bear",
            ote_zone_top_column="zt",
            ote_zone_bottom_column="zb",
        )
        self.assertIn("bull", result.columns)
        self.assertIn("bear", result.columns)
        self.assertIn("zt", result.columns)
        self.assertIn("zb", result.columns)

    def test_different_swing_lengths(self):
        for length in [3, 5, 8]:
            result = optimal_trade_entry(self.df, swing_length=length)
            self.assertIsInstance(result, pd.DataFrame)

    def test_premium_discount_filter(self):
        result = optimal_trade_entry(
            self.df, swing_length=5,
            premium_discount_filter=True,
        )
        self.assertIsInstance(result, pd.DataFrame)
        # With filter, should get equal or fewer OTEs
        result_no_filter = optimal_trade_entry(
            self.df, swing_length=5,
            premium_discount_filter=False,
        )
        total_filtered = (
            result["ote_bullish"].sum() + result["ote_bearish"].sum()
        )
        total_unfiltered = (
            result_no_filter["ote_bullish"].sum()
            + result_no_filter["ote_bearish"].sum()
        )
        self.assertLessEqual(total_filtered, total_unfiltered)

    def test_invalid_input_raises(self):
        with self.assertRaises(PyIndicatorException):
            optimal_trade_entry([1, 2, 3])

    def test_does_not_modify_original(self):
        original_cols = list(self.df.columns)
        optimal_trade_entry(self.df, swing_length=5)
        self.assertEqual(list(self.df.columns), original_cols)


class TestOTESignal(unittest.TestCase):
    """Tests for optimal_trade_entry_signal()."""

    def setUp(self):
        self.df = optimal_trade_entry(
            _make_trending_data(), swing_length=5
        )

    def test_signal_values(self):
        result = optimal_trade_entry_signal(self.df)
        self.assertTrue(result["ote_signal"].isin([0, 1, -1]).all())

    def test_adds_signal_column(self):
        result = optimal_trade_entry_signal(self.df)
        self.assertIn("ote_signal", result.columns)

    def test_long_signal_matches_entry_long(self):
        result = optimal_trade_entry_signal(self.df)
        long_mask = self.df["ote_entry_long"] == 1
        self.assertTrue((result.loc[long_mask, "ote_signal"] == 1).all())

    def test_short_signal_matches_entry_short(self):
        result = optimal_trade_entry_signal(self.df)
        short_mask = self.df["ote_entry_short"] == 1
        self.assertTrue(
            (result.loc[short_mask, "ote_signal"] == -1).all()
        )

    def test_polars_signal(self):
        pl_df = pl.from_pandas(self.df)
        result = optimal_trade_entry_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn("ote_signal", result.columns)

    def test_custom_signal_column(self):
        result = optimal_trade_entry_signal(
            self.df, signal_column="my_ote_sig"
        )
        self.assertIn("my_ote_sig", result.columns)


class TestOTEStats(unittest.TestCase):
    """Tests for get_optimal_trade_entry_stats()."""

    def setUp(self):
        self.df = optimal_trade_entry(
            _make_trending_data(), swing_length=5
        )

    def test_stats_keys(self):
        stats = get_optimal_trade_entry_stats(self.df)
        expected_keys = [
            "total_bullish_ote", "total_bearish_ote", "total_ote",
            "total_entry_long", "total_entry_short", "total_entries",
            "total_invalidated", "entry_rate", "invalidation_rate",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_total_is_sum_of_bull_bear(self):
        stats = get_optimal_trade_entry_stats(self.df)
        self.assertEqual(
            stats["total_ote"],
            stats["total_bullish_ote"] + stats["total_bearish_ote"],
        )

    def test_entry_rate_range(self):
        stats = get_optimal_trade_entry_stats(self.df)
        self.assertGreaterEqual(stats["entry_rate"], 0.0)
        self.assertLessEqual(stats["entry_rate"], 1.0)

    def test_invalidation_rate_range(self):
        stats = get_optimal_trade_entry_stats(self.df)
        self.assertGreaterEqual(stats["invalidation_rate"], 0.0)
        self.assertLessEqual(stats["invalidation_rate"], 1.0)

    def test_polars_stats(self):
        pl_df = pl.from_pandas(self.df)
        stats = get_optimal_trade_entry_stats(pl_df)
        self.assertIn("total_ote", stats)


class TestOTEEdgeCases(unittest.TestCase):
    """Edge cases and special scenarios."""

    def test_short_data(self):
        """Data shorter than swing_length should not crash."""
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.5, 101.5, 102.5],
        })
        result = optimal_trade_entry(df, swing_length=5)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["ote_bullish"].sum(), 0)

    def test_flat_market(self):
        """Flat/sideways market should produce no OTEs."""
        n = 200
        flat = np.full(n, 100.0)
        df = pd.DataFrame({
            "Open": flat,
            "High": flat + 0.01,
            "Low": flat - 0.01,
            "Close": flat,
        })
        result = optimal_trade_entry(df, swing_length=5)
        total = result["ote_bullish"].sum() + result["ote_bearish"].sum()
        self.assertEqual(total, 0)

    def test_strong_uptrend(self):
        """Monotonic uptrend with no reversals → no bearish OTEs."""
        n = 200
        t = np.arange(n, dtype=float)
        close = 100 + t * 0.5
        df = pd.DataFrame({
            "Open": close - 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
        })
        result = optimal_trade_entry(df, swing_length=5)
        self.assertEqual(result["ote_bearish"].sum(), 0)

    def test_custom_fib_prefix(self):
        """Custom fib prefix should rename all fib columns."""
        result = optimal_trade_entry(
            _make_trending_data(), swing_length=5,
            fib_prefix="my_ote",
        )
        for key in OTE_FIB_LEVELS:
            self.assertIn(f"my_ote_{key}", result.columns)


if __name__ == "__main__":
    unittest.main()
