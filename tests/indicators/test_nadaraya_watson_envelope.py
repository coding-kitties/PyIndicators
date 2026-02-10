import numpy as np
from unittest import TestCase
import pandas as pd
import polars as pl
from pyindicators import nadaraya_watson_envelope
from pyindicators.exceptions import PyIndicatorException


class TestNadarayaWatsonEnvelope(TestCase):

    def _make_pandas_df(self, n=50):
        """Create a simple pandas DataFrame with trending close prices."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({"Close": close})

    def _make_polars_df(self, n=50):
        """Create a simple polars DataFrame with trending close prices."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({"Close": close.tolist()})

    # ----------------------------------------------------------------
    # Basic output tests
    # ----------------------------------------------------------------

    def test_output_columns_pandas(self):
        """Test that default output columns are created with pandas."""
        df = self._make_pandas_df()
        result = nadaraya_watson_envelope(df.copy(), bandwidth=5, lookback=20)

        self.assertIn('nwe_upper', result.columns)
        self.assertIn('nwe_lower', result.columns)
        self.assertIn('nwe_middle', result.columns)

    def test_output_columns_polars(self):
        """Test that default output columns are created with polars."""
        df = self._make_polars_df()
        result = nadaraya_watson_envelope(df, bandwidth=5, lookback=20)

        self.assertIn('nwe_upper', result.columns)
        self.assertIn('nwe_lower', result.columns)
        self.assertIn('nwe_middle', result.columns)

    def test_custom_column_names(self):
        """Test that custom output column names are used."""
        df = self._make_pandas_df()
        result = nadaraya_watson_envelope(
            df.copy(),
            bandwidth=5,
            lookback=20,
            upper_column='env_upper',
            lower_column='env_lower',
            middle_column='env_mid',
        )

        self.assertIn('env_upper', result.columns)
        self.assertIn('env_lower', result.columns)
        self.assertIn('env_mid', result.columns)

    def test_output_length_matches_input(self):
        """Test that output DataFrame has same number of rows as input."""
        df = self._make_pandas_df(30)
        result = nadaraya_watson_envelope(df.copy(), bandwidth=5, lookback=20)
        self.assertEqual(len(result), 30)

    # ----------------------------------------------------------------
    # Envelope ordering
    # ----------------------------------------------------------------

    def test_upper_above_lower(self):
        """Test that upper envelope is always >= lower envelope."""
        df = self._make_pandas_df(100)
        result = nadaraya_watson_envelope(
            df.copy(), bandwidth=8, mult=3.0, lookback=50
        )

        upper = result['nwe_upper'].values
        lower = result['nwe_lower'].values

        for i in range(len(result)):
            if not np.isnan(upper[i]) and not np.isnan(lower[i]):
                self.assertGreaterEqual(upper[i], lower[i])

    def test_middle_between_upper_and_lower(self):
        """Test that the middle line is between upper and lower."""
        df = self._make_pandas_df(100)
        result = nadaraya_watson_envelope(
            df.copy(), bandwidth=8, mult=3.0, lookback=50
        )

        for i in range(len(result)):
            u = result['nwe_upper'].iloc[i]
            m = result['nwe_middle'].iloc[i]
            low = result['nwe_lower'].iloc[i]
            if not np.isnan(u) and not np.isnan(m) and not np.isnan(low):
                self.assertGreaterEqual(u, m)
                self.assertGreaterEqual(m, low)

    # ----------------------------------------------------------------
    # Smoothing behavior
    # ----------------------------------------------------------------

    def test_constant_price_gives_constant_envelope(self):
        """With constant prices, middle should equal source and envelope
        should collapse (upper == lower == middle)."""
        df = pd.DataFrame({"Close": [100.0] * 20})
        result = nadaraya_watson_envelope(
            df.copy(), bandwidth=5, mult=3.0, lookback=10
        )

        for i in range(len(result)):
            self.assertAlmostEqual(
                result['nwe_middle'].iloc[i], 100.0, places=5
            )
            self.assertAlmostEqual(
                result['nwe_upper'].iloc[i], 100.0, places=5
            )
            self.assertAlmostEqual(
                result['nwe_lower'].iloc[i], 100.0, places=5
            )

    def test_higher_bandwidth_smoother(self):
        """Higher bandwidth should produce a smoother (less variable) middle."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 2)
        df = pd.DataFrame({"Close": close})

        result_low_bw = nadaraya_watson_envelope(
            df.copy(), bandwidth=3, lookback=50
        )
        result_high_bw = nadaraya_watson_envelope(
            df.copy(), bandwidth=20, lookback=50
        )

        # Variance of middle should be lower with higher bandwidth
        var_low = np.nanvar(np.diff(result_low_bw['nwe_middle'].values))
        var_high = np.nanvar(np.diff(result_high_bw['nwe_middle'].values))
        self.assertLess(var_high, var_low)

    def test_higher_mult_wider_envelope(self):
        """Higher multiplier should produce a wider envelope."""
        df = self._make_pandas_df(50)

        result_narrow = nadaraya_watson_envelope(
            df.copy(), bandwidth=5, mult=1.0, lookback=20
        )
        result_wide = nadaraya_watson_envelope(
            df.copy(), bandwidth=5, mult=5.0, lookback=20
        )

        width_narrow = (
            result_narrow['nwe_upper'].iloc[-1]
            - result_narrow['nwe_lower'].iloc[-1]
        )
        width_wide = (
            result_wide['nwe_upper'].iloc[-1]
            - result_wide['nwe_lower'].iloc[-1]
        )
        self.assertGreater(width_wide, width_narrow)

    def test_zero_mult_collapses_envelope(self):
        """With mult=0, upper and lower should equal middle."""
        df = self._make_pandas_df(30)
        result = nadaraya_watson_envelope(
            df.copy(), bandwidth=5, mult=0.0, lookback=20
        )

        for i in range(len(result)):
            m = result['nwe_middle'].iloc[i]
            u = result['nwe_upper'].iloc[i]
            low = result['nwe_lower'].iloc[i]
            if not np.isnan(m):
                self.assertAlmostEqual(u, m, places=10)
                self.assertAlmostEqual(low, m, places=10)

    # ----------------------------------------------------------------
    # Polars consistency
    # ----------------------------------------------------------------

    def test_pandas_polars_consistency(self):
        """Test that pandas and polars produce the same results."""
        np.random.seed(42)
        close = (100 + np.cumsum(np.random.randn(50) * 0.5)).tolist()

        pd_df = pd.DataFrame({"Close": close})
        pl_df = pl.DataFrame({"Close": close})

        pd_result = nadaraya_watson_envelope(
            pd_df.copy(), bandwidth=5, lookback=20
        )
        pl_result = nadaraya_watson_envelope(
            pl_df, bandwidth=5, lookback=20
        )

        np.testing.assert_array_almost_equal(
            pd_result['nwe_middle'].values,
            pl_result['nwe_middle'].to_numpy(),
            decimal=8
        )
        np.testing.assert_array_almost_equal(
            pd_result['nwe_upper'].values,
            pl_result['nwe_upper'].to_numpy(),
            decimal=8
        )
        np.testing.assert_array_almost_equal(
            pd_result['nwe_lower'].values,
            pl_result['nwe_lower'].to_numpy(),
            decimal=8
        )

    # ----------------------------------------------------------------
    # Custom source column
    # ----------------------------------------------------------------

    def test_custom_source_column(self):
        """Test using a non-default source column."""
        df = pd.DataFrame({
            "Price": [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        })
        result = nadaraya_watson_envelope(
            df.copy(), source_column='Price', bandwidth=3, lookback=5
        )

        self.assertIn('nwe_middle', result.columns)
        self.assertFalse(result['nwe_middle'].isna().all())

    # ----------------------------------------------------------------
    # Edge cases
    # ----------------------------------------------------------------

    def test_single_row(self):
        """Test with a single row DataFrame."""
        df = pd.DataFrame({"Close": [100.0]})
        result = nadaraya_watson_envelope(
            df.copy(), bandwidth=5, lookback=10
        )

        self.assertAlmostEqual(
            result['nwe_middle'].iloc[0], 100.0, places=5
        )

    def test_lookback_larger_than_data(self):
        """Test when lookback exceeds the number of data points."""
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
        result = nadaraya_watson_envelope(
            df.copy(), bandwidth=5, lookback=500
        )

        self.assertEqual(len(result), 3)
        self.assertFalse(result['nwe_middle'].isna().all())

    # ----------------------------------------------------------------
    # Validation / error handling
    # ----------------------------------------------------------------

    def test_invalid_bandwidth_zero(self):
        """Test that bandwidth=0 raises an exception."""
        df = self._make_pandas_df()
        with self.assertRaises(PyIndicatorException):
            nadaraya_watson_envelope(df.copy(), bandwidth=0)

    def test_invalid_bandwidth_negative(self):
        """Test that negative bandwidth raises an exception."""
        df = self._make_pandas_df()
        with self.assertRaises(PyIndicatorException):
            nadaraya_watson_envelope(df.copy(), bandwidth=-1)

    def test_invalid_mult_negative(self):
        """Test that negative multiplier raises an exception."""
        df = self._make_pandas_df()
        with self.assertRaises(PyIndicatorException):
            nadaraya_watson_envelope(df.copy(), mult=-1)

    def test_invalid_lookback(self):
        """Test that lookback < 1 raises an exception."""
        df = self._make_pandas_df()
        with self.assertRaises(PyIndicatorException):
            nadaraya_watson_envelope(df.copy(), lookback=0)

    def test_missing_source_column(self):
        """Test that a missing source column raises an exception."""
        df = pd.DataFrame({"Price": [100, 101, 102]})
        with self.assertRaises(PyIndicatorException):
            nadaraya_watson_envelope(df.copy(), source_column='Close')
