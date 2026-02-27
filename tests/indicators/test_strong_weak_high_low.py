import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    strong_weak_high_low,
    strong_weak_high_low_signal,
    get_strong_weak_high_low_stats,
)
from pyindicators.exceptions import PyIndicatorException


def _make_swing_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate OHLCV data with clear swing structure."""
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    base = np.where(
        t < n // 2,
        100 + t * 0.3,
        100 + (n // 2) * 0.3 - (t - n // 2) * 0.3,
    )
    wave = 6 * np.sin(t * 2 * np.pi / 30)
    close = base + wave + np.random.randn(n) * 0.5
    high = close + np.abs(np.random.randn(n)) * 2.0
    low = close - np.abs(np.random.randn(n)) * 2.0
    opn = close - np.random.randn(n) * 0.5
    vol = np.random.randint(1000, 10000, n).astype(float)
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low,
        "Close": close, "Volume": vol,
    })


def _make_short_data() -> pd.DataFrame:
    """Very short data — edge case."""
    return pd.DataFrame({
        "Open": [100, 101, 102],
        "High": [102, 103, 104],
        "Low": [98, 99, 100],
        "Close": [101, 102, 103],
        "Volume": [1000, 2000, 3000],
    })


class TestStrongWeakHighLow(unittest.TestCase):
    """Tests for strong_weak_high_low()."""

    def setUp(self):
        self.df = _make_swing_data()

    def test_returns_pandas(self):
        result = strong_weak_high_low(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars(self):
        pl_df = pl.from_pandas(self.df)
        result = strong_weak_high_low(pl_df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_expected_columns(self):
        result = strong_weak_high_low(self.df)
        expected = [
            "sw_high", "sw_low",
            "sw_high_price", "sw_low_price",
            "sw_high_type", "sw_low_type",
            "sw_high_volume", "sw_low_volume",
            "sw_high_vol_pct", "sw_low_vol_pct",
            "sw_equilibrium",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_preserves_row_count(self):
        result = strong_weak_high_low(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_flags_are_binary(self):
        result = strong_weak_high_low(self.df)
        self.assertTrue(result["sw_high"].isin([0, 1]).all())
        self.assertTrue(result["sw_low"].isin([0, 1]).all())

    def test_type_values(self):
        result = strong_weak_high_low(self.df)
        high_mask = result["sw_high"] == 1
        if high_mask.any():
            types = result.loc[high_mask, "sw_high_type"].dropna().unique()
            for t in types:
                self.assertIn(t, ["Strong", "Weak"])

    def test_swings_detected(self):
        result = strong_weak_high_low(self.df)
        self.assertGreater(result["sw_high"].sum(), 0)
        self.assertGreater(result["sw_low"].sum(), 0)

    def test_prices_at_swings(self):
        result = strong_weak_high_low(self.df)
        high_mask = result["sw_high"] == 1
        if high_mask.any():
            prices = result.loc[high_mask, "sw_high_price"]
            self.assertTrue(prices.notna().all())

    def test_volume_at_swings(self):
        result = strong_weak_high_low(self.df)
        high_mask = result["sw_high"] == 1
        if high_mask.any():
            vols = result.loc[high_mask, "sw_high_volume"]
            self.assertTrue(vols.notna().all())
            self.assertTrue((vols > 0).all())

    def test_equilibrium_computed(self):
        result = strong_weak_high_low(self.df)
        # After first swing high and low are found, equilibrium
        # should be non-NaN for remaining rows
        eq = result["sw_equilibrium"]
        self.assertTrue(eq.notna().any())

    def test_short_data_no_crash(self):
        df = _make_short_data()
        result = strong_weak_high_low(df, swing_lookback=2)
        self.assertEqual(len(result), 3)

    def test_custom_lookback(self):
        result = strong_weak_high_low(self.df, swing_lookback=20)
        self.assertGreater(result["sw_high"].sum(), 0)

    def test_custom_columns(self):
        result = strong_weak_high_low(
            self.df,
            sw_high_column="my_h",
            sw_low_column="my_l",
        )
        self.assertIn("my_h", result.columns)
        self.assertIn("my_l", result.columns)

    def test_polars_matches_pandas(self):
        pr = strong_weak_high_low(self.df.copy())
        plr = strong_weak_high_low(pl.from_pandas(self.df)).to_pandas()
        np.testing.assert_array_equal(
            pr["sw_high"].values, plr["sw_high"].values
        )

    def test_invalid_input(self):
        with self.assertRaises(PyIndicatorException):
            strong_weak_high_low(42)


class TestStrongWeakHighLowSignal(unittest.TestCase):
    """Tests for strong_weak_high_low_signal()."""

    def test_signal_column(self):
        df = strong_weak_high_low(_make_swing_data())
        result = strong_weak_high_low_signal(df)
        self.assertIn("sw_signal", result.columns)

    def test_signal_values(self):
        df = strong_weak_high_low(_make_swing_data())
        result = strong_weak_high_low_signal(df)
        self.assertTrue(result["sw_signal"].isin([1, 0, -1]).all())

    def test_strong_low_gives_bullish(self):
        df = strong_weak_high_low(_make_swing_data())
        result = strong_weak_high_low_signal(df)
        strong_low_mask = (
            (result["sw_low"] == 1) &
            (result["sw_low_type"] == "Strong")
        )
        if strong_low_mask.any():
            sigs = result.loc[strong_low_mask, "sw_signal"]
            self.assertTrue((sigs == 1).all())

    def test_strong_high_gives_bearish(self):
        df = strong_weak_high_low(_make_swing_data())
        result = strong_weak_high_low_signal(df)
        strong_high_mask = (
            (result["sw_high"] == 1) &
            (result["sw_high_type"] == "Strong")
        )
        if strong_high_mask.any():
            sigs = result.loc[strong_high_mask, "sw_signal"]
            self.assertTrue((sigs == -1).all())

    def test_polars_signal(self):
        df = strong_weak_high_low(_make_swing_data())
        result = strong_weak_high_low_signal(pl.from_pandas(df))
        self.assertIsInstance(result, pl.DataFrame)


class TestStrongWeakHighLowStats(unittest.TestCase):
    """Tests for get_strong_weak_high_low_stats()."""

    def test_stats_keys(self):
        df = strong_weak_high_low(_make_swing_data())
        stats = get_strong_weak_high_low_stats(df)
        for key in ["total_swing_highs", "total_swing_lows",
                     "strong_highs", "weak_highs",
                     "strong_lows", "weak_lows", "total"]:
            self.assertIn(key, stats)

    def test_stats_consistency(self):
        df = strong_weak_high_low(_make_swing_data())
        stats = get_strong_weak_high_low_stats(df)
        self.assertEqual(
            stats["total"],
            stats["total_swing_highs"] + stats["total_swing_lows"],
        )
        self.assertEqual(
            stats["total_swing_highs"],
            stats["strong_highs"] + stats["weak_highs"],
        )
        self.assertEqual(
            stats["total_swing_lows"],
            stats["strong_lows"] + stats["weak_lows"],
        )

    def test_polars_stats(self):
        df = strong_weak_high_low(pl.from_pandas(_make_swing_data()))
        stats = get_strong_weak_high_low_stats(df)
        self.assertIsInstance(stats["total"], int)


if __name__ == "__main__":
    unittest.main()
