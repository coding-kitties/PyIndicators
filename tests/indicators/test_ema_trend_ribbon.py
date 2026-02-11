import numpy as np
from unittest import TestCase
import pandas as pd
import polars as pl
from pyindicators import ema_trend_ribbon
from pyindicators.exceptions import PyIndicatorException


class TestEmaTrendRibbon(TestCase):

    def _make_pandas_df(self, n=100):
        """Create a pandas DataFrame with trending close prices."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({"Close": close})

    def _make_polars_df(self, n=100):
        """Create a polars DataFrame with trending close prices."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"Close": close.tolist()})

    # ----------------------------------------------------------------
    # Basic output tests
    # ----------------------------------------------------------------

    def test_output_columns_pandas(self):
        """Test that default output columns are created with pandas."""
        df = self._make_pandas_df()
        result = ema_trend_ribbon(df.copy())
        default_lengths = [8, 14, 20, 26, 32, 38, 44, 50, 60]

        for length in default_lengths:
            self.assertIn(f'ema_ribbon_{length}', result.columns)

        self.assertIn('ema_ribbon_trend', result.columns)
        self.assertIn('ema_ribbon_bullish_count', result.columns)
        self.assertIn('ema_ribbon_bearish_count', result.columns)

    def test_output_columns_polars(self):
        """Test that default output columns are created with polars."""
        df = self._make_polars_df()
        result = ema_trend_ribbon(df)
        default_lengths = [8, 14, 20, 26, 32, 38, 44, 50, 60]

        for length in default_lengths:
            self.assertIn(f'ema_ribbon_{length}', result.columns)

        self.assertIn('ema_ribbon_trend', result.columns)
        self.assertIn('ema_ribbon_bullish_count', result.columns)
        self.assertIn('ema_ribbon_bearish_count', result.columns)

    def test_custom_column_names(self):
        """Test that custom column names and prefix are used."""
        df = self._make_pandas_df()
        result = ema_trend_ribbon(
            df.copy(),
            ema_column_prefix='ribbon',
            trend_column='tr',
            bullish_count_column='bulls',
            bearish_count_column='bears',
        )
        self.assertIn('ribbon_8', result.columns)
        self.assertIn('tr', result.columns)
        self.assertIn('bulls', result.columns)
        self.assertIn('bears', result.columns)

    def test_custom_ema_lengths(self):
        """Test that custom EMA lengths are used."""
        df = self._make_pandas_df()
        result = ema_trend_ribbon(
            df.copy(),
            ema_lengths=[5, 10, 20],
            threshold=2,
        )
        self.assertIn('ema_ribbon_5', result.columns)
        self.assertIn('ema_ribbon_10', result.columns)
        self.assertIn('ema_ribbon_20', result.columns)
        self.assertNotIn('ema_ribbon_8', result.columns)

    def test_output_length_matches_input(self):
        """Test that output has the same number of rows as input."""
        df = self._make_pandas_df(80)
        result = ema_trend_ribbon(df.copy())
        self.assertEqual(len(result), 80)

    # ----------------------------------------------------------------
    # Trend and count tests
    # ----------------------------------------------------------------

    def test_trend_values_valid(self):
        """Test that trend column only contains -1, 0, or 1."""
        df = self._make_pandas_df(200)
        result = ema_trend_ribbon(df.copy())

        for v in result['ema_ribbon_trend'].values:
            self.assertIn(v, [-1, 0, 1])

    def test_counts_sum_to_num_emas(self):
        """Test that bullish + bearish count <= total EMAs."""
        df = self._make_pandas_df(100)
        result = ema_trend_ribbon(df.copy())
        num_emas = 9

        for i in range(len(result)):
            bull = result['ema_ribbon_bullish_count'].iloc[i]
            bear = result['ema_ribbon_bearish_count'].iloc[i]
            self.assertLessEqual(bull + bear, num_emas)
            self.assertGreaterEqual(bull, 0)
            self.assertGreaterEqual(bear, 0)

    def test_bullish_trend_requires_threshold(self):
        """Test that bullish trend only when bullish_count >= threshold."""
        df = self._make_pandas_df(200)
        result = ema_trend_ribbon(df.copy(), threshold=7)

        for i in range(len(result)):
            if result['ema_ribbon_trend'].iloc[i] == 1:
                self.assertGreaterEqual(
                    result['ema_ribbon_bullish_count'].iloc[i], 7
                )

    def test_bearish_trend_requires_threshold(self):
        """Test that bearish trend only when bearish_count >= threshold."""
        df = self._make_pandas_df(200)
        result = ema_trend_ribbon(df.copy(), threshold=7)

        for i in range(len(result)):
            if result['ema_ribbon_trend'].iloc[i] == -1:
                self.assertGreaterEqual(
                    result['ema_ribbon_bearish_count'].iloc[i], 7
                )

    def test_strong_uptrend_detected(self):
        """Test that a strong uptrend is classified as bullish."""
        # Monotonically rising prices
        close = np.arange(1, 201, dtype=float)
        df = pd.DataFrame({"Close": close})
        result = ema_trend_ribbon(df, smoothing_period=1, threshold=7)

        # Near the end, all EMAs should be rising → bullish
        self.assertEqual(result['ema_ribbon_trend'].iloc[-1], 1)

    def test_strong_downtrend_detected(self):
        """Test that a strong downtrend is classified as bearish."""
        # Monotonically falling prices
        close = np.arange(200, 0, -1, dtype=float)
        df = pd.DataFrame({"Close": close})
        result = ema_trend_ribbon(df, smoothing_period=1, threshold=7)

        # Near the end, all EMAs should be falling → bearish
        self.assertEqual(result['ema_ribbon_trend'].iloc[-1], -1)

    # ----------------------------------------------------------------
    # EMA ordering
    # ----------------------------------------------------------------

    def test_shorter_ema_more_responsive(self):
        """Test that shorter EMAs react faster to price changes."""
        # Step function: flat then jump
        close = np.concatenate([np.full(50, 100.0), np.full(50, 120.0)])
        df = pd.DataFrame({"Close": close})
        result = ema_trend_ribbon(df)

        # Right after the step, shorter EMA should be closer to new price
        idx = 55
        ema_8 = result['ema_ribbon_8'].iloc[idx]
        ema_60 = result['ema_ribbon_60'].iloc[idx]
        self.assertGreater(ema_8, ema_60)

    # ----------------------------------------------------------------
    # Polars / Pandas parity
    # ----------------------------------------------------------------

    def test_pandas_polars_close_results(self):
        """Test that pandas and polars produce similar results."""
        pd_df = self._make_pandas_df(100)
        pl_df = self._make_polars_df(100)

        pd_result = ema_trend_ribbon(pd_df.copy())
        pl_result = ema_trend_ribbon(pl_df)

        pd_trend = pd_result['ema_ribbon_trend'].values
        pl_trend = pl_result['ema_ribbon_trend'].to_numpy()
        np.testing.assert_array_equal(pd_trend, pl_trend)

        pd_ema = pd_result['ema_ribbon_8'].values
        pl_ema = pl_result['ema_ribbon_8'].to_numpy()
        np.testing.assert_allclose(pd_ema, pl_ema, rtol=1e-10)

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------

    def test_invalid_smoothing_period(self):
        """Test that smoothing_period < 1 raises an exception."""
        df = self._make_pandas_df()
        with self.assertRaises(PyIndicatorException):
            ema_trend_ribbon(df.copy(), smoothing_period=0)

    def test_invalid_threshold(self):
        """Test that threshold < 1 raises an exception."""
        df = self._make_pandas_df()
        with self.assertRaises(PyIndicatorException):
            ema_trend_ribbon(df.copy(), threshold=0)

    def test_threshold_exceeds_num_emas(self):
        """Test that threshold > number of EMAs raises an exception."""
        df = self._make_pandas_df()
        with self.assertRaises(PyIndicatorException):
            ema_trend_ribbon(df.copy(), threshold=10)

    def test_too_few_ema_lengths(self):
        """Test that fewer than 2 EMA lengths raises an exception."""
        df = self._make_pandas_df()
        with self.assertRaises(PyIndicatorException):
            ema_trend_ribbon(df.copy(), ema_lengths=[10], threshold=1)

    def test_invalid_ema_length(self):
        """Test that an EMA length < 1 raises an exception."""
        df = self._make_pandas_df()
        with self.assertRaises(PyIndicatorException):
            ema_trend_ribbon(
                df.copy(), ema_lengths=[0, 10, 20], threshold=2
            )

    def test_missing_source_column(self):
        """Test that a missing source column raises an exception."""
        df = self._make_pandas_df()
        with self.assertRaises(PyIndicatorException):
            ema_trend_ribbon(df.copy(), source_column='Missing')

    # ----------------------------------------------------------------
    # Edge cases
    # ----------------------------------------------------------------

    def test_first_bars_neutral(self):
        """Test that the first bars are neutral (not enough history)."""
        df = self._make_pandas_df(10)
        result = ema_trend_ribbon(df.copy(), smoothing_period=2)

        # Bar 0 and 1 have no smoothing_period lookback → counts are 0
        self.assertEqual(result['ema_ribbon_bullish_count'].iloc[0], 0)
        self.assertEqual(result['ema_ribbon_bearish_count'].iloc[0], 0)
        self.assertEqual(result['ema_ribbon_trend'].iloc[0], 0)
