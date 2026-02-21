import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    internal_external_liquidity_zones,
    internal_external_liquidity_zones_signal,
    get_internal_external_liquidity_zones_stats,
)
from pyindicators.exceptions import PyIndicatorException


def _make_ohlc(n=300, seed=42):
    """Synthetic OHLC with clear swing points."""
    np.random.seed(seed)
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
    return pd.DataFrame({
        "Open": close - np.random.randn(n) * 0.3,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(1000, 10000, n),
    })


class TestInternalExternalLiquidityZones(unittest.TestCase):
    """Tests for internal_external_liquidity_zones function."""

    def setUp(self):
        self.df = _make_ohlc()

    def test_returns_dataframe(self):
        result = internal_external_liquidity_zones(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_adds_expected_columns(self):
        result = internal_external_liquidity_zones(self.df)
        expected = [
            "ielz_ext_high", "ielz_ext_low",
            "ielz_ext_high_price", "ielz_ext_low_price",
            "ielz_int_high", "ielz_int_low",
            "ielz_int_high_price", "ielz_int_low_price",
            "ielz_range_high", "ielz_range_low",
            "ielz_ext_sweep_bull", "ielz_ext_sweep_bear",
            "ielz_int_sweep_bull", "ielz_int_sweep_bear",
            "ielz_ext_structure", "ielz_int_structure",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_ext_high_low_are_binary(self):
        result = internal_external_liquidity_zones(self.df)
        self.assertTrue(result["ielz_ext_high"].isin([0, 1]).all())
        self.assertTrue(result["ielz_ext_low"].isin([0, 1]).all())

    def test_int_high_low_are_binary(self):
        result = internal_external_liquidity_zones(self.df)
        self.assertTrue(result["ielz_int_high"].isin([0, 1]).all())
        self.assertTrue(result["ielz_int_low"].isin([0, 1]).all())

    def test_sweep_columns_are_binary(self):
        result = internal_external_liquidity_zones(self.df)
        for col in [
            "ielz_ext_sweep_bull", "ielz_ext_sweep_bear",
            "ielz_int_sweep_bull", "ielz_int_sweep_bear",
        ]:
            self.assertTrue(result[col].isin([0, 1]).all())

    def test_detects_some_external_pivots(self):
        result = internal_external_liquidity_zones(self.df)
        self.assertGreater(result["ielz_ext_high"].sum(), 0)
        self.assertGreater(result["ielz_ext_low"].sum(), 0)

    def test_ext_price_nan_when_no_detection(self):
        result = internal_external_liquidity_zones(self.df)
        no_ext_h = result[result["ielz_ext_high"] == 0]
        self.assertTrue(no_ext_h["ielz_ext_high_price"].isna().all())

    def test_structure_labels_are_valid(self):
        result = internal_external_liquidity_zones(self.df)
        valid_ext = {"", "eBOS", "eCHoCH"}
        valid_int = {"", "iBOS", "iCHoCH"}
        self.assertTrue(
            set(result["ielz_ext_structure"].unique()).issubset(valid_ext)
        )
        self.assertTrue(
            set(result["ielz_int_structure"].unique()).issubset(valid_int)
        )

    def test_every_pivot_mode(self):
        result = internal_external_liquidity_zones(
            self.df,
            internal_mode="every_pivot",
            require_internal_inside=False,
        )
        self.assertIsInstance(result, pd.DataFrame)
        # every_pivot mode should detect at least as many as equal_hl
        self.assertGreater(result["ielz_int_high"].sum(), 0)

    def test_equal_hl_mode(self):
        result = internal_external_liquidity_zones(
            self.df, internal_mode="equal_hl"
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_custom_column_names(self):
        result = internal_external_liquidity_zones(
            self.df,
            ext_high_column="eh",
            ext_low_column="el",
            int_high_column="ih",
            int_low_column="il",
        )
        self.assertIn("eh", result.columns)
        self.assertIn("el", result.columns)
        self.assertIn("ih", result.columns)
        self.assertIn("il", result.columns)

    def test_different_sweep_modes(self):
        for mode in ("wick", "close", "wick_close"):
            result = internal_external_liquidity_zones(
                self.df, sweep_mode=mode,
            )
            self.assertIsInstance(result, pd.DataFrame)

    def test_invalid_internal_mode_raises(self):
        with self.assertRaises(PyIndicatorException):
            internal_external_liquidity_zones(
                self.df, internal_mode="bad"
            )

    def test_invalid_sweep_mode_raises(self):
        with self.assertRaises(PyIndicatorException):
            internal_external_liquidity_zones(
                self.df, sweep_mode="bad"
            )

    def test_invalid_input_raises(self):
        with self.assertRaises(PyIndicatorException):
            internal_external_liquidity_zones([1, 2, 3])

    def test_does_not_modify_original_dataframe(self):
        original_cols = list(self.df.columns)
        internal_external_liquidity_zones(self.df)
        self.assertEqual(list(self.df.columns), original_cols)

    def test_polars_dataframe_input(self):
        pl_df = pl.from_pandas(self.df)
        result = internal_external_liquidity_zones(pl_df)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn("ielz_ext_high", result.columns)
        self.assertIn("ielz_int_high", result.columns)

    def test_polars_matches_pandas(self):
        pd_result = internal_external_liquidity_zones(self.df)
        pl_result = internal_external_liquidity_zones(
            pl.from_pandas(self.df)
        ).to_pandas()
        for col in [
            "ielz_ext_high", "ielz_ext_low",
            "ielz_int_high", "ielz_int_low",
        ]:
            np.testing.assert_array_equal(
                pd_result[col].values, pl_result[col].values
            )

    def test_range_columns_populated(self):
        result = internal_external_liquidity_zones(self.df)
        # After enough bars, the range columns should have values
        non_nan = result["ielz_range_high"].dropna()
        self.assertGreater(len(non_nan), 0)


class TestInternalExternalLiquidityZonesSignal(unittest.TestCase):
    """Tests for internal_external_liquidity_zones_signal function."""

    def setUp(self):
        self.df = internal_external_liquidity_zones(
            _make_ohlc(),
            internal_mode="every_pivot",
            require_internal_inside=False,
        )

    def test_signal_values(self):
        result = internal_external_liquidity_zones_signal(self.df)
        self.assertTrue(
            result["ielz_signal"].isin([0, 1, -1]).all()
        )

    def test_signal_custom_column(self):
        result = internal_external_liquidity_zones_signal(
            self.df, signal_column="my_signal"
        )
        self.assertIn("my_signal", result.columns)

    def test_polars_signal(self):
        pl_df = pl.from_pandas(self.df)
        result = internal_external_liquidity_zones_signal(pl_df)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn("ielz_signal", result.columns)

    def test_invalid_input_raises(self):
        with self.assertRaises(PyIndicatorException):
            internal_external_liquidity_zones_signal([1, 2, 3])


class TestInternalExternalLiquidityZonesStats(unittest.TestCase):
    """Tests for get_internal_external_liquidity_zones_stats function."""

    def setUp(self):
        self.df = internal_external_liquidity_zones(
            _make_ohlc(),
            internal_mode="every_pivot",
            require_internal_inside=False,
        )

    def test_stats_returns_dict(self):
        stats = get_internal_external_liquidity_zones_stats(self.df)
        self.assertIsInstance(stats, dict)

    def test_stats_keys_present(self):
        stats = get_internal_external_liquidity_zones_stats(self.df)
        expected_keys = [
            "total_ext_highs", "total_ext_lows",
            "total_int_highs", "total_int_lows",
            "total_ext_sweeps", "total_int_sweeps",
            "ext_bos_count", "ext_choch_count",
            "int_bos_count", "int_choch_count",
            "bullish_sweep_ratio",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_bullish_sweep_ratio_range(self):
        stats = get_internal_external_liquidity_zones_stats(self.df)
        self.assertGreaterEqual(stats["bullish_sweep_ratio"], 0.0)
        self.assertLessEqual(stats["bullish_sweep_ratio"], 1.0)

    def test_polars_stats(self):
        pl_df = pl.from_pandas(self.df)
        stats = get_internal_external_liquidity_zones_stats(pl_df)
        self.assertIn("total_ext_highs", stats)

    def test_totals_are_non_negative(self):
        stats = get_internal_external_liquidity_zones_stats(self.df)
        for key in [
            "total_ext_highs", "total_ext_lows",
            "total_int_highs", "total_int_lows",
            "total_ext_sweeps", "total_int_sweeps",
        ]:
            self.assertGreaterEqual(stats[key], 0)


if __name__ == "__main__":
    unittest.main()
