import unittest
import pandas as pd
import numpy as np
import polars as pl
from pyindicators import (
    z_score_predictive_zones,
    z_score_predictive_zones_signal,
    get_z_score_predictive_zones_stats,
)


def _make_trending_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with clear swing structure."""
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    # Up-trend then down-trend with oscillation
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
    vol = np.random.randint(1000, 10000, n).astype(float)
    return pd.DataFrame({
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })


class TestZScorePredictiveZones(unittest.TestCase):
    """Tests for the z_score_predictive_zones() function."""

    def setUp(self):
        self.df = _make_trending_data()

    def test_returns_pandas_dataframe(self):
        result = z_score_predictive_zones(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_returns_polars_dataframe(self):
        pl_df = pl.from_pandas(self.df)
        result = z_score_predictive_zones(pl_df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_adds_expected_columns(self):
        result = z_score_predictive_zones(self.df)
        expected = [
            "zspz_z_score", "zspz_mean", "zspz_std",
            "zspz_avg_top_level", "zspz_avg_bot_level",
            "zspz_res_band_high", "zspz_res_band_low",
            "zspz_sup_band_high", "zspz_sup_band_low",
            "zspz_rsi_ma",
            "zspz_long_signal", "zspz_short_signal", "zspz_signal",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_preserves_row_count(self):
        result = z_score_predictive_zones(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_signal_values_are_valid(self):
        result = z_score_predictive_zones(self.df)
        self.assertTrue(result["zspz_signal"].isin([0, 1, -1]).all())

    def test_long_short_signals_are_binary(self):
        result = z_score_predictive_zones(self.df)
        self.assertTrue(result["zspz_long_signal"].isin([0, 1]).all())
        self.assertTrue(result["zspz_short_signal"].isin([0, 1]).all())

    def test_does_not_modify_original(self):
        original = self.df.copy()
        _ = z_score_predictive_zones(self.df)
        pd.testing.assert_frame_equal(self.df, original)

    def test_custom_parameters(self):
        result = z_score_predictive_zones(
            self.df, z_length=50, smooth=10,
            history_depth=15, threshold=1.0,
        )
        self.assertEqual(len(result), len(self.df))
        self.assertIn("zspz_z_score", result.columns)

    def test_z_score_within_reasonable_range(self):
        result = z_score_predictive_zones(self.df)
        z = result["zspz_z_score"].dropna()
        if len(z) > 0:
            self.assertTrue(z.abs().max() < 10,
                            "Z-score should be within a reasonable range")

    def test_res_band_above_sup_band(self):
        result = z_score_predictive_zones(self.df)
        valid = result.dropna(
            subset=["zspz_res_band_low", "zspz_sup_band_high"]
        )
        if len(valid) > 0:
            self.assertTrue(
                (valid["zspz_res_band_low"] >= valid["zspz_sup_band_high"])
                .all(),
                "Resistance band should be above support band",
            )


class TestZScorePredictiveZonesSignal(unittest.TestCase):
    """Tests for the z_score_predictive_zones_signal() function."""

    def setUp(self):
        self.df = _make_trending_data()
        self.result = z_score_predictive_zones(self.df)

    def test_returns_dataframe(self):
        sig_df = z_score_predictive_zones_signal(self.result)
        self.assertIsInstance(sig_df, pd.DataFrame)

    def test_signal_column_exists(self):
        sig_df = z_score_predictive_zones_signal(self.result)
        self.assertIn("zspz_signal", sig_df.columns)

    def test_polars_passthrough(self):
        pl_df = pl.from_pandas(self.result)
        sig_df = z_score_predictive_zones_signal(pl_df)
        self.assertIsInstance(sig_df, pl.DataFrame)


class TestZScorePredictiveZonesStats(unittest.TestCase):
    """Tests for the get_z_score_predictive_zones_stats() function."""

    def setUp(self):
        self.df = _make_trending_data()
        self.result = z_score_predictive_zones(self.df)

    def test_expected_keys(self):
        stats = get_z_score_predictive_zones_stats(self.result)
        expected_keys = [
            "total_bars", "long_signals", "short_signals",
            "total_signals", "long_pct", "short_pct",
            "avg_z_score", "max_z_score", "min_z_score",
            "avg_top_level", "avg_bot_level",
            "avg_res_band_width", "avg_sup_band_width",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_total_signals_is_sum(self):
        stats = get_z_score_predictive_zones_stats(self.result)
        self.assertEqual(
            stats["total_signals"],
            stats["long_signals"] + stats["short_signals"],
        )

    def test_total_bars_matches(self):
        stats = get_z_score_predictive_zones_stats(self.result)
        self.assertEqual(stats["total_bars"], len(self.result))

    def test_pct_in_range(self):
        stats = get_z_score_predictive_zones_stats(self.result)
        self.assertGreaterEqual(stats["long_pct"], 0)
        self.assertLessEqual(stats["long_pct"], 1)
        self.assertGreaterEqual(stats["short_pct"], 0)
        self.assertLessEqual(stats["short_pct"], 1)

    def test_polars_compatibility(self):
        pl_df = pl.from_pandas(self.result)
        stats = get_z_score_predictive_zones_stats(pl_df)
        self.assertEqual(stats["total_bars"], len(self.result))


class TestZScorePredictiveZonesEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_short_data(self):
        df = _make_trending_data(n=50)
        result = z_score_predictive_zones(df, z_length=20, smooth=5)
        self.assertEqual(len(result), 50)

    def test_flat_market(self):
        n = 200
        df = pd.DataFrame({
            "Open": [100.0] * n,
            "High": [101.0] * n,
            "Low": [99.0] * n,
            "Close": [100.0] * n,
            "Volume": [5000.0] * n,
        })
        result = z_score_predictive_zones(df, z_length=20, smooth=5)
        self.assertEqual(len(result), n)

    def test_monotonic_uptrend(self):
        n = 300
        t = np.arange(n, dtype=float)
        close = 100 + t * 0.5
        df = pd.DataFrame({
            "Open": close - 0.1,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 5000.0),
        })
        result = z_score_predictive_zones(df, z_length=50, smooth=10)
        self.assertEqual(len(result), n)


if __name__ == "__main__":
    unittest.main()
