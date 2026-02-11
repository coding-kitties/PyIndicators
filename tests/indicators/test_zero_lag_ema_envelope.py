import numpy as np
from unittest import TestCase
import pandas as pd
import polars as pl
from pyindicators import zero_lag_ema_envelope
from pyindicators.exceptions import PyIndicatorException


class TestZeroLagEmaEnvelope(TestCase):

    def _make_pandas_df(self, n=250):
        """Create a pandas DataFrame with OHLC data."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return pd.DataFrame({
            "High": high,
            "Low": low,
            "Close": close,
        })

    def _make_polars_df(self, n=250):
        """Create a polars DataFrame with OHLC data."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return pl.DataFrame({
            "High": high.tolist(),
            "Low": low.tolist(),
            "Close": close.tolist(),
        })

    # ----------------------------------------------------------------
    # Basic output tests
    # ----------------------------------------------------------------

    def test_output_columns_pandas(self):
        """Test that default output columns are created with pandas."""
        df = self._make_pandas_df()
        result = zero_lag_ema_envelope(
            df.copy(), length=20, atr_length=5
        )
        for col in [
            'zlema_upper', 'zlema_lower', 'zlema_middle',
            'zlema_trend', 'zlema_signal',
        ]:
            self.assertIn(col, result.columns)

    def test_output_columns_polars(self):
        """Test that default output columns are created with polars."""
        df = self._make_polars_df()
        result = zero_lag_ema_envelope(df, length=20, atr_length=5)
        for col in [
            'zlema_upper', 'zlema_lower', 'zlema_middle',
            'zlema_trend', 'zlema_signal',
        ]:
            self.assertIn(col, result.columns)

    def test_custom_column_names(self):
        """Test that custom output column names are used."""
        df = self._make_pandas_df()
        result = zero_lag_ema_envelope(
            df.copy(),
            length=20,
            atr_length=5,
            upper_column='up',
            lower_column='dn',
            middle_column='mid',
            trend_column='tr',
            signal_column='sig',
        )
        for col in ['up', 'dn', 'mid', 'tr', 'sig']:
            self.assertIn(col, result.columns)

    def test_output_length_matches_input(self):
        """Test that output has the same number of rows as input."""
        df = self._make_pandas_df(100)
        result = zero_lag_ema_envelope(
            df.copy(), length=20, atr_length=5
        )
        self.assertEqual(len(result), 100)

    # ----------------------------------------------------------------
    # Envelope ordering
    # ----------------------------------------------------------------

    def test_upper_above_lower(self):
        """Test that upper band is always >= lower band."""
        df = self._make_pandas_df(200)
        result = zero_lag_ema_envelope(
            df.copy(), length=20, mult=2.0, atr_length=10
        )
        upper = result['zlema_upper'].values
        lower = result['zlema_lower'].values

        for i in range(len(result)):
            if not np.isnan(upper[i]) and not np.isnan(lower[i]):
                self.assertGreaterEqual(upper[i], lower[i])

    def test_middle_between_upper_and_lower(self):
        """Test that the ZLEMA line is between upper and lower."""
        df = self._make_pandas_df(200)
        result = zero_lag_ema_envelope(
            df.copy(), length=20, mult=2.0, atr_length=10
        )
        for i in range(len(result)):
            u = result['zlema_upper'].iloc[i]
            m = result['zlema_middle'].iloc[i]
            lo = result['zlema_lower'].iloc[i]
            if not np.isnan(u) and not np.isnan(m) and not np.isnan(lo):
                self.assertGreaterEqual(u, m)
                self.assertGreaterEqual(m, lo)

    # ----------------------------------------------------------------
    # Trend and signal tests
    # ----------------------------------------------------------------

    def test_trend_values_valid(self):
        """Test that trend column only contains -1, 0, or 1."""
        df = self._make_pandas_df(250)
        result = zero_lag_ema_envelope(
            df.copy(), length=20, atr_length=5
        )
        for v in result['zlema_trend'].values:
            self.assertIn(v, [-1, 0, 1])

    def test_signal_values_valid(self):
        """Test that signal column only contains -1, 0, or 1."""
        df = self._make_pandas_df(250)
        result = zero_lag_ema_envelope(
            df.copy(), length=20, atr_length=5
        )
        for v in result['zlema_signal'].values:
            self.assertIn(v, [-1, 0, 1])

    def test_signal_only_on_trend_flip(self):
        """Test that signals only fire when trend changes."""
        df = self._make_pandas_df(250)
        result = zero_lag_ema_envelope(
            df.copy(), length=20, atr_length=5
        )
        trend = result['zlema_trend'].values
        signal = result['zlema_signal'].values

        for i in range(1, len(result)):
            if signal[i] != 0:
                self.assertNotEqual(trend[i], trend[i - 1])

    # ----------------------------------------------------------------
    # Polars / Pandas parity
    # ----------------------------------------------------------------

    def test_pandas_polars_close_results(self):
        """Test that pandas and polars produce similar results."""
        pd_df = self._make_pandas_df(200)
        pl_df = self._make_polars_df(200)

        pd_result = zero_lag_ema_envelope(
            pd_df.copy(), length=20, atr_length=5
        )
        pl_result = zero_lag_ema_envelope(
            pl_df, length=20, atr_length=5
        )

        pd_mid = pd_result['zlema_middle'].values
        pl_mid = pl_result['zlema_middle'].to_numpy()

        np.testing.assert_allclose(pd_mid, pl_mid, rtol=1e-10)

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------

    def test_invalid_length(self):
        """Test that length < 1 raises an exception."""
        df = self._make_pandas_df(50)
        with self.assertRaises(PyIndicatorException):
            zero_lag_ema_envelope(df.copy(), length=0, atr_length=5)

    def test_invalid_mult(self):
        """Test that negative multiplier raises an exception."""
        df = self._make_pandas_df(50)
        with self.assertRaises(PyIndicatorException):
            zero_lag_ema_envelope(df.copy(), length=20, mult=-1.0)

    def test_invalid_atr_length(self):
        """Test that atr_length < 1 raises an exception."""
        df = self._make_pandas_df(50)
        with self.assertRaises(PyIndicatorException):
            zero_lag_ema_envelope(df.copy(), length=20, atr_length=0)

    def test_invalid_confirm_bars_low(self):
        """Test that confirm_bars < 1 raises an exception."""
        df = self._make_pandas_df(50)
        with self.assertRaises(PyIndicatorException):
            zero_lag_ema_envelope(
                df.copy(), length=20, atr_length=5, confirm_bars=0
            )

    def test_invalid_confirm_bars_high(self):
        """Test that confirm_bars > 3 raises an exception."""
        df = self._make_pandas_df(50)
        with self.assertRaises(PyIndicatorException):
            zero_lag_ema_envelope(
                df.copy(), length=20, atr_length=5, confirm_bars=4
            )

    def test_missing_source_column(self):
        """Test that a missing source column raises an exception."""
        df = self._make_pandas_df(50)
        with self.assertRaises(PyIndicatorException):
            zero_lag_ema_envelope(
                df.copy(), source_column='Missing', length=20, atr_length=5
            )

    def test_missing_high_column(self):
        """Test that a missing High column raises an exception."""
        df = self._make_pandas_df(50).drop(columns=['High'])
        with self.assertRaises(PyIndicatorException):
            zero_lag_ema_envelope(df.copy(), length=20, atr_length=5)

    def test_missing_low_column(self):
        """Test that a missing Low column raises an exception."""
        df = self._make_pandas_df(50).drop(columns=['Low'])
        with self.assertRaises(PyIndicatorException):
            zero_lag_ema_envelope(df.copy(), length=20, atr_length=5)

    # ----------------------------------------------------------------
    # ZLEMA correctness
    # ----------------------------------------------------------------

    def test_zlema_reduces_lag(self):
        """
        Test that ZLEMA has less lag than a plain EMA by comparing
        the responsiveness to a step change in price.
        """
        n = 100
        length = 20
        # Create a step function: flat at 100, then jumps to 110
        close = np.concatenate([
            np.full(50, 100.0), np.full(50, 110.0)
        ])
        high = close + 1.0
        low = close - 1.0

        df = pd.DataFrame({"High": high, "Low": low, "Close": close})
        result = zero_lag_ema_envelope(
            df.copy(), length=length, atr_length=5
        )

        zlema_vals = result['zlema_middle'].values

        # Compute a plain EMA for comparison
        alpha = 2.0 / (length + 1)
        ema = np.empty(n)
        ema[0] = close[0]

        for i in range(1, n):
            ema[i] = close[i] * alpha + ema[i - 1] * (1 - alpha)

        # After the step, ZLEMA should reach the new level faster
        idx = 55  # 5 bars after the step
        self.assertGreater(zlema_vals[idx], ema[idx])

    def test_mult_zero_bands_equal_middle(self):
        """Test that with mult=0 the bands equal the ZLEMA line."""
        df = self._make_pandas_df(50)
        result = zero_lag_ema_envelope(
            df.copy(), length=10, mult=0.0, atr_length=5
        )
        np.testing.assert_allclose(
            result['zlema_upper'].values,
            result['zlema_middle'].values,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            result['zlema_lower'].values,
            result['zlema_middle'].values,
            rtol=1e-10,
        )
