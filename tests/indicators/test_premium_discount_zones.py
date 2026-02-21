import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    premium_discount_zones,
    premium_discount_zones_signal,
    get_premium_discount_zones_stats,
)
from pyindicators.exceptions import PyIndicatorException


class TestPremiumDiscountZones(unittest.TestCase):
    """Tests for premium_discount_zones function."""

    def setUp(self):
        """Create OHLC data with clear swing points."""
        np.random.seed(42)
        n = 300
        t = np.arange(n, dtype=float)
        # Up-trend then down-trend to create clear range
        base = np.where(
            t < n // 2,
            100 + t * 0.3,
            100 + (n // 2) * 0.3 - (t - n // 2) * 0.3,
        )
        wave = 3 * np.sin(t * 2 * np.pi / 25)
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

    def test_returns_pandas_dataframe(self):
        result = premium_discount_zones(self.df, swing_length=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars_dataframe(self):
        pl_df = pl.from_pandas(self.df)
        result = premium_discount_zones(pl_df, swing_length=5)
        self.assertIsInstance(result, pl.DataFrame)

    def test_raises_on_invalid_input(self):
        with self.assertRaises(PyIndicatorException):
            premium_discount_zones({"not": "a dataframe"})

    def test_adds_expected_columns(self):
        result = premium_discount_zones(self.df, swing_length=5)
        expected = [
            "pdz_range_high",
            "pdz_range_low",
            "pdz_equilibrium",
            "pdz_zone",
            "pdz_zone_pct",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_custom_column_names(self):
        result = premium_discount_zones(
            self.df,
            swing_length=5,
            range_high_column="rh",
            range_low_column="rl",
            equilibrium_column="eq",
            zone_column="zn",
            zone_pct_column="zp",
        )
        for col in ["rh", "rl", "eq", "zn", "zp"]:
            self.assertIn(col, result.columns)

    def test_zone_values_are_valid(self):
        result = premium_discount_zones(self.df, swing_length=5)
        valid = {"", "premium", "discount", "equilibrium"}
        unique = set(result["pdz_zone"].unique())
        self.assertTrue(unique.issubset(valid))

    def test_equilibrium_is_midpoint(self):
        """Equilibrium should be exactly (range_high + range_low) / 2."""
        result = premium_discount_zones(self.df, swing_length=5)
        mask = result["pdz_equilibrium"].notna()
        eq = result.loc[mask, "pdz_equilibrium"]
        rh = result.loc[mask, "pdz_range_high"]
        rl = result.loc[mask, "pdz_range_low"]
        expected = (rh + rl) / 2.0
        np.testing.assert_allclose(eq.values, expected.values)

    def test_range_high_above_range_low(self):
        """Range high should be >= range low when both are set."""
        result = premium_discount_zones(self.df, swing_length=5)
        mask = result["pdz_range_high"].notna() & result["pdz_range_low"].notna()
        self.assertTrue(
            (result.loc[mask, "pdz_range_high"]
             >= result.loc[mask, "pdz_range_low"]).all()
        )

    def test_zone_pct_in_range(self):
        """Zone percentage should be between 0 and 100."""
        result = premium_discount_zones(self.df, swing_length=5)
        mask = result["pdz_zone_pct"].notna()
        pcts = result.loc[mask, "pdz_zone_pct"]
        self.assertTrue((pcts >= 0).all())
        self.assertTrue((pcts <= 100).all())

    def test_premium_zone_when_price_above_equilibrium(self):
        """When zone is 'premium', Close should be > equilibrium."""
        result = premium_discount_zones(self.df, swing_length=5)
        premium = result[result["pdz_zone"] == "premium"]
        if len(premium) > 0:
            self.assertTrue(
                (premium["Close"] > premium["pdz_equilibrium"]).all()
            )

    def test_discount_zone_when_price_below_equilibrium(self):
        """When zone is 'discount', Close should be < equilibrium."""
        result = premium_discount_zones(self.df, swing_length=5)
        discount = result[result["pdz_zone"] == "discount"]
        if len(discount) > 0:
            self.assertTrue(
                (discount["Close"] < discount["pdz_equilibrium"]).all()
            )

    def test_some_premium_and_discount_bars_detected(self):
        """With trending data there should be both zone types."""
        result = premium_discount_zones(self.df, swing_length=5)
        self.assertGreater(
            (result["pdz_zone"] == "premium").sum(), 0,
            "Expected at least some premium bars"
        )
        self.assertGreater(
            (result["pdz_zone"] == "discount").sum(), 0,
            "Expected at least some discount bars"
        )

    def test_early_bars_have_no_zone(self):
        """Before first swing is confirmed, zone should be empty."""
        result = premium_discount_zones(self.df, swing_length=10)
        # At minimum the first swing_length bars cannot have pivots
        first_bars = result.head(10)
        self.assertTrue(
            (first_bars["pdz_zone"] == "").all()
            or first_bars["pdz_equilibrium"].isna().all()
        )

    def test_different_swing_lengths(self):
        """Indicator should work with different swing lengths."""
        for length in [3, 5, 10, 15]:
            result = premium_discount_zones(self.df, swing_length=length)
            self.assertIn("pdz_zone", result.columns)

    def test_does_not_modify_original_dataframe(self):
        """Original DataFrame should not be modified."""
        original_cols = list(self.df.columns)
        premium_discount_zones(self.df, swing_length=5)
        self.assertEqual(list(self.df.columns), original_cols)

    def test_polars_matches_pandas(self):
        """Polars and pandas output should be equivalent."""
        pd_result = premium_discount_zones(self.df, swing_length=5)
        pl_df = pl.from_pandas(self.df)
        pl_result = premium_discount_zones(pl_df, swing_length=5).to_pandas()

        np.testing.assert_array_equal(
            pd_result["pdz_zone"].values, pl_result["pdz_zone"].values
        )
        mask = pd_result["pdz_zone_pct"].notna()
        np.testing.assert_allclose(
            pd_result.loc[mask, "pdz_zone_pct"].values,
            pl_result.loc[mask, "pdz_zone_pct"].values,
        )


class TestPremiumDiscountZonesSignal(unittest.TestCase):
    """Tests for premium_discount_zones_signal function."""

    def setUp(self):
        np.random.seed(42)
        n = 300
        t = np.arange(n, dtype=float)
        base = np.where(
            t < n // 2,
            100 + t * 0.3,
            100 + (n // 2) * 0.3 - (t - n // 2) * 0.3,
        )
        wave = 3 * np.sin(t * 2 * np.pi / 25)
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
        self.result = premium_discount_zones(self.df, swing_length=5)

    def test_signal_values(self):
        sig = premium_discount_zones_signal(self.result)
        self.assertTrue(sig["pdz_signal"].isin([0, 1, -1]).all())

    def test_discount_gives_buy_signal(self):
        sig = premium_discount_zones_signal(self.result)
        discount = sig[sig["pdz_zone"] == "discount"]
        if len(discount) > 0:
            self.assertTrue((discount["pdz_signal"] == 1).all())

    def test_premium_gives_sell_signal(self):
        sig = premium_discount_zones_signal(self.result)
        premium = sig[sig["pdz_zone"] == "premium"]
        if len(premium) > 0:
            self.assertTrue((premium["pdz_signal"] == -1).all())

    def test_polars_signal(self):
        pl_result = pl.from_pandas(self.result)
        sig = premium_discount_zones_signal(pl_result)
        self.assertIsInstance(sig, pl.DataFrame)
        self.assertIn("pdz_signal", sig.columns)

    def test_raises_on_invalid_input(self):
        with self.assertRaises(PyIndicatorException):
            premium_discount_zones_signal({"not": "a dataframe"})


class TestGetPremiumDiscountZonesStats(unittest.TestCase):
    """Tests for get_premium_discount_zones_stats function."""

    def setUp(self):
        np.random.seed(42)
        n = 300
        t = np.arange(n, dtype=float)
        base = np.where(
            t < n // 2,
            100 + t * 0.3,
            100 + (n // 2) * 0.3 - (t - n // 2) * 0.3,
        )
        wave = 3 * np.sin(t * 2 * np.pi / 25)
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
        self.result = premium_discount_zones(self.df, swing_length=5)

    def test_stats_returns_dict(self):
        stats = get_premium_discount_zones_stats(self.result)
        self.assertIsInstance(stats, dict)

    def test_stats_keys_present(self):
        stats = get_premium_discount_zones_stats(self.result)
        expected_keys = [
            "total_bars",
            "premium_bars",
            "discount_bars",
            "equilibrium_bars",
            "no_zone_bars",
            "premium_ratio",
            "discount_ratio",
            "avg_premium_pct",
            "avg_discount_pct",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_total_bars_matches(self):
        stats = get_premium_discount_zones_stats(self.result)
        self.assertEqual(stats["total_bars"], len(self.result))

    def test_bars_sum_to_total(self):
        stats = get_premium_discount_zones_stats(self.result)
        total = (
            stats["premium_bars"]
            + stats["discount_bars"]
            + stats["equilibrium_bars"]
            + stats["no_zone_bars"]
        )
        self.assertEqual(total, stats["total_bars"])

    def test_ratios_between_zero_and_one(self):
        stats = get_premium_discount_zones_stats(self.result)
        self.assertGreaterEqual(stats["premium_ratio"], 0)
        self.assertLessEqual(stats["premium_ratio"], 1)
        self.assertGreaterEqual(stats["discount_ratio"], 0)
        self.assertLessEqual(stats["discount_ratio"], 1)

    def test_polars_stats(self):
        pl_result = pl.from_pandas(self.result)
        stats = get_premium_discount_zones_stats(pl_result)
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["total_bars"], len(self.result))


if __name__ == "__main__":
    unittest.main()
