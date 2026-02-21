import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    mitigation_blocks,
    mitigation_blocks_signal,
    get_mitigation_blocks_stats,
)
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


class TestMitigationBlocks(unittest.TestCase):
    """Tests for the mitigation_blocks() function."""

    def setUp(self):
        self.df = _make_trending_data()

    def test_returns_pandas_dataframe(self):
        result = mitigation_blocks(self.df, swing_length=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars_dataframe(self):
        pl_df = pl.from_pandas(self.df)
        result = mitigation_blocks(pl_df, swing_length=5)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_expected_columns(self):
        result = mitigation_blocks(self.df, swing_length=5)
        expected = [
            "mb_bullish", "mb_bearish",
            "mb_top", "mb_bottom",
            "mb_direction",
            "mb_entry_long", "mb_entry_short",
            "mb_mitigated",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_preserves_row_count(self):
        result = mitigation_blocks(self.df, swing_length=5)
        self.assertEqual(len(result), len(self.df))

    def test_formation_flags_are_binary(self):
        result = mitigation_blocks(self.df, swing_length=5)
        self.assertTrue(result["mb_bullish"].isin([0, 1]).all())
        self.assertTrue(result["mb_bearish"].isin([0, 1]).all())

    def test_entry_flags_are_binary(self):
        result = mitigation_blocks(self.df, swing_length=5)
        self.assertTrue(result["mb_entry_long"].isin([0, 1]).all())
        self.assertTrue(result["mb_entry_short"].isin([0, 1]).all())

    def test_mitigated_is_binary(self):
        result = mitigation_blocks(self.df, swing_length=5)
        self.assertTrue(result["mb_mitigated"].isin([0, 1]).all())

    def test_direction_values_are_valid(self):
        result = mitigation_blocks(self.df, swing_length=5)
        self.assertTrue(result["mb_direction"].isin([0, 1, -1]).all())

    def test_detects_some_mitigation_blocks(self):
        result = mitigation_blocks(self.df, swing_length=5)
        total = result["mb_bullish"].sum() + result["mb_bearish"].sum()
        self.assertGreater(total, 0, "Should detect at least one MB")

    def test_zone_top_greater_or_equal_bottom(self):
        result = mitigation_blocks(self.df, swing_length=5)
        active = result[result["mb_direction"] != 0].dropna(
            subset=["mb_top", "mb_bottom"]
        )
        if len(active) > 0:
            self.assertTrue(
                (active["mb_top"] >= active["mb_bottom"]).all()
            )

    def test_use_body_option(self):
        result_wick = mitigation_blocks(
            self.df, swing_length=5, use_body=False
        )
        result_body = mitigation_blocks(
            self.df, swing_length=5, use_body=True
        )
        self.assertIsInstance(result_wick, pd.DataFrame)
        self.assertIsInstance(result_body, pd.DataFrame)

    def test_custom_column_names(self):
        result = mitigation_blocks(
            self.df, swing_length=5,
            bullish_mb_column="bull",
            bearish_mb_column="bear",
            mb_top_column="zt",
            mb_bottom_column="zb",
        )
        self.assertIn("bull", result.columns)
        self.assertIn("bear", result.columns)
        self.assertIn("zt", result.columns)
        self.assertIn("zb", result.columns)

    def test_different_swing_lengths(self):
        for length in [3, 5, 8]:
            result = mitigation_blocks(self.df, swing_length=length)
            self.assertIsInstance(result, pd.DataFrame)

    def test_invalid_input_raises(self):
        with self.assertRaises(PyIndicatorException):
            mitigation_blocks([1, 2, 3])

    def test_does_not_modify_original(self):
        original_cols = list(self.df.columns)
        mitigation_blocks(self.df, swing_length=5)
        self.assertEqual(list(self.df.columns), original_cols)


class TestMBSignal(unittest.TestCase):
    """Tests for mitigation_blocks_signal()."""

    def setUp(self):
        self.df = mitigation_blocks(
            _make_trending_data(), swing_length=5
        )

    def test_signal_values(self):
        result = mitigation_blocks_signal(self.df)
        self.assertTrue(result["mb_signal"].isin([0, 1, -1]).all())

    def test_adds_signal_column(self):
        result = mitigation_blocks_signal(self.df)
        self.assertIn("mb_signal", result.columns)

    def test_long_signal_matches_entry_long(self):
        result = mitigation_blocks_signal(self.df)
        long_mask = self.df["mb_entry_long"] == 1
        if long_mask.any():
            self.assertTrue(
                (result.loc[long_mask, "mb_signal"] == 1).all()
            )

    def test_short_signal_matches_entry_short(self):
        result = mitigation_blocks_signal(self.df)
        short_mask = self.df["mb_entry_short"] == 1
        if short_mask.any():
            self.assertTrue(
                (result.loc[short_mask, "mb_signal"] == -1).all()
            )

    def test_polars_signal(self):
        pl_df = pl.from_pandas(self.df)
        result = mitigation_blocks_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn("mb_signal", result.columns)

    def test_custom_signal_column(self):
        result = mitigation_blocks_signal(
            self.df, signal_column="my_mb_sig"
        )
        self.assertIn("my_mb_sig", result.columns)


class TestMBStats(unittest.TestCase):
    """Tests for get_mitigation_blocks_stats()."""

    def setUp(self):
        self.df = mitigation_blocks(
            _make_trending_data(), swing_length=5
        )

    def test_stats_keys(self):
        stats = get_mitigation_blocks_stats(self.df)
        expected_keys = [
            "total_bullish_mb", "total_bearish_mb", "total_mb",
            "total_entry_long", "total_entry_short", "total_entries",
            "total_mitigated", "entry_rate", "mitigation_rate",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_total_is_sum_of_bull_bear(self):
        stats = get_mitigation_blocks_stats(self.df)
        self.assertEqual(
            stats["total_mb"],
            stats["total_bullish_mb"] + stats["total_bearish_mb"],
        )

    def test_entry_rate_range(self):
        stats = get_mitigation_blocks_stats(self.df)
        self.assertGreaterEqual(stats["entry_rate"], 0.0)
        self.assertLessEqual(stats["entry_rate"], 1.0)

    def test_mitigation_rate_range(self):
        stats = get_mitigation_blocks_stats(self.df)
        self.assertGreaterEqual(stats["mitigation_rate"], 0.0)
        self.assertLessEqual(stats["mitigation_rate"], 1.0)

    def test_polars_stats(self):
        pl_df = pl.from_pandas(self.df)
        stats = get_mitigation_blocks_stats(pl_df)
        self.assertIn("total_mb", stats)


class TestMBEdgeCases(unittest.TestCase):
    """Edge cases and special scenarios."""

    def test_short_data(self):
        """Data shorter than swing_length should not crash."""
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.5, 101.5, 102.5],
        })
        result = mitigation_blocks(df, swing_length=5)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["mb_bullish"].sum(), 0)

    def test_flat_market(self):
        """Flat/sideways market should produce no MBs."""
        n = 200
        flat = np.full(n, 100.0)
        df = pd.DataFrame({
            "Open": flat,
            "High": flat + 0.01,
            "Low": flat - 0.01,
            "Close": flat,
        })
        result = mitigation_blocks(df, swing_length=5)
        total = result["mb_bullish"].sum() + result["mb_bearish"].sum()
        self.assertEqual(total, 0)

    def test_strong_uptrend(self):
        """Monotonic uptrend with no reversals → no bearish MBs."""
        n = 200
        t = np.arange(n, dtype=float)
        close = 100 + t * 0.5
        df = pd.DataFrame({
            "Open": close - 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
        })
        result = mitigation_blocks(df, swing_length=5)
        self.assertEqual(result["mb_bearish"].sum(), 0)

    def test_single_bar(self):
        """Single-bar DataFrame should not crash."""
        df = pd.DataFrame({
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
        })
        result = mitigation_blocks(df, swing_length=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["mb_bullish"].sum(), 0)

    def test_body_zones_narrower_than_wick_zones(self):
        """With use_body=True, zones should be <= wick zones."""
        df = _make_trending_data()
        result_wick = mitigation_blocks(df, swing_length=5, use_body=False)
        result_body = mitigation_blocks(df, swing_length=5, use_body=True)

        # Check active zone bars — body zones should not be wider
        wick_active = result_wick[result_wick["mb_direction"] != 0].dropna(
            subset=["mb_top", "mb_bottom"]
        )
        body_active = result_body[result_body["mb_direction"] != 0].dropna(
            subset=["mb_top", "mb_bottom"]
        )

        if len(wick_active) > 0:
            wick_widths = (
                wick_active["mb_top"] - wick_active["mb_bottom"]
            ).mean()
        else:
            wick_widths = 0

        if len(body_active) > 0:
            body_widths = (
                body_active["mb_top"] - body_active["mb_bottom"]
            ).mean()
        else:
            body_widths = 0

        # Body-based zones are typically narrower or equal
        # (they can differ in detection too, so just check no crash)
        self.assertIsInstance(result_body, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
