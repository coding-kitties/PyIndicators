import numpy as np
from unittest import TestCase
import pandas as pd
import polars as pl
from pyindicators import moving_average_envelope, sma_envelope, ema_envelope


class TestMovingAverageEnvelope(TestCase):

    def test_sma_envelope_pandas(self):
        """Test SMA envelope calculation with pandas."""
        df = pd.DataFrame({
            "Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })
        result = moving_average_envelope(df.copy(), period=10, percentage=2.5)

        self.assertIn('ma_envelope_middle', result.columns)
        self.assertIn('ma_envelope_upper', result.columns)
        self.assertIn('ma_envelope_lower', result.columns)

        # Upper should be > middle > lower
        valid_idx = 15
        self.assertGreater(
            result['ma_envelope_upper'].iloc[valid_idx],
            result['ma_envelope_middle'].iloc[valid_idx]
        )
        self.assertGreater(
            result['ma_envelope_middle'].iloc[valid_idx],
            result['ma_envelope_lower'].iloc[valid_idx]
        )

    def test_sma_envelope_polars(self):
        """Test SMA envelope calculation with polars."""
        df = pl.DataFrame({
            "Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })
        result = moving_average_envelope(df, period=10, percentage=2.5)

        self.assertIn('ma_envelope_middle', result.columns)
        self.assertIn('ma_envelope_upper', result.columns)
        self.assertIn('ma_envelope_lower', result.columns)

    def test_ema_envelope_pandas(self):
        """Test EMA envelope calculation with pandas."""
        df = pd.DataFrame({
            "Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })
        result = moving_average_envelope(
            df.copy(), period=10, percentage=2.5, ma_type='ema'
        )

        self.assertIn('ma_envelope_middle', result.columns)
        self.assertIn('ma_envelope_upper', result.columns)
        self.assertIn('ma_envelope_lower', result.columns)

    def test_envelope_percentage_calculation(self):
        """Test that envelope percentage is calculated correctly."""
        df = pd.DataFrame({
            "Close": [100.0] * 20  # Constant price
        })
        result = moving_average_envelope(df.copy(), period=10, percentage=5.0)

        # Middle should be 100
        middle = result['ma_envelope_middle'].iloc[-1]
        upper = result['ma_envelope_upper'].iloc[-1]
        lower = result['ma_envelope_lower'].iloc[-1]

        self.assertAlmostEqual(middle, 100.0, places=5)
        # Upper should be 100 * 1.05 = 105
        self.assertAlmostEqual(upper, 105.0, places=5)
        # Lower should be 100 * 0.95 = 95
        self.assertAlmostEqual(lower, 95.0, places=5)

    def test_envelope_symmetry(self):
        """Test that envelope bands are symmetric around the middle."""
        df = pd.DataFrame({
            "Close": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118,
                      120, 122, 124, 126, 128, 130, 132, 134, 136, 138]
        })
        result = moving_average_envelope(df.copy(), period=10, percentage=3.0)

        # For each row, upper - middle should equal middle - lower
        for idx in range(10, len(result)):
            middle = result['ma_envelope_middle'].iloc[idx]
            upper = result['ma_envelope_upper'].iloc[idx]
            lower = result['ma_envelope_lower'].iloc[idx]

            upper_diff = upper - middle
            lower_diff = middle - lower

            self.assertAlmostEqual(upper_diff, lower_diff, places=5)

    def test_sma_envelope_convenience_function(self):
        """Test the sma_envelope convenience function."""
        df = pd.DataFrame({
            "Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })
        result = sma_envelope(df.copy(), period=10, percentage=2.5)

        self.assertIn('sma_envelope_middle', result.columns)
        self.assertIn('sma_envelope_upper', result.columns)
        self.assertIn('sma_envelope_lower', result.columns)

    def test_ema_envelope_convenience_function(self):
        """Test the ema_envelope convenience function."""
        df = pd.DataFrame({
            "Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })
        result = ema_envelope(df.copy(), period=10, percentage=2.5)

        self.assertIn('ema_envelope_middle', result.columns)
        self.assertIn('ema_envelope_upper', result.columns)
        self.assertIn('ema_envelope_lower', result.columns)

    def test_envelope_custom_columns(self):
        """Test envelope with custom column names."""
        df = pd.DataFrame({
            "Price": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })
        result = moving_average_envelope(
            df.copy(),
            source_column='Price',
            period=10,
            percentage=2.5,
            middle_column='center',
            upper_column='top',
            lower_column='bottom'
        )

        self.assertIn('center', result.columns)
        self.assertIn('top', result.columns)
        self.assertIn('bottom', result.columns)

    def test_envelope_datetime_index(self):
        """Test envelope works with DatetimeIndex."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            "Close": np.linspace(100, 130, 30)
        }, index=dates)

        result = moving_average_envelope(df.copy(), period=10, percentage=2.5)

        self.assertIn('ma_envelope_middle', result.columns)
        self.assertEqual(len(result), 30)

    def test_envelope_different_percentages(self):
        """Test that different percentages produce different band widths."""
        df = pd.DataFrame({
            "Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })

        result_narrow = moving_average_envelope(df.copy(), period=10, percentage=1.0)
        result_wide = moving_average_envelope(df.copy(), period=10, percentage=5.0)

        # Wide envelope should have larger band width
        narrow_width = (result_narrow['ma_envelope_upper'].iloc[-1] -
                        result_narrow['ma_envelope_lower'].iloc[-1])
        wide_width = (result_wide['ma_envelope_upper'].iloc[-1] -
                      result_wide['ma_envelope_lower'].iloc[-1])

        self.assertGreater(wide_width, narrow_width)

    def test_invalid_ma_type(self):
        """Test that invalid ma_type raises exception."""
        df = pd.DataFrame({
            "Close": [100, 102, 104, 103, 105]
        })

        from pyindicators import PyIndicatorException
        with self.assertRaises(PyIndicatorException):
            moving_average_envelope(df.copy(), period=3, ma_type='invalid')

    def test_envelope_real_world_trend(self):
        """Test envelope with realistic trending data (like the gold/silver chart)."""
        # Simulate an uptrend with some noise
        np.random.seed(42)
        base_trend = np.linspace(3600, 5700, 100)  # Like gold price movement
        noise = np.random.normal(0, 30, 100)
        prices = base_trend + noise

        df = pd.DataFrame({"Close": prices})
        result = moving_average_envelope(df.copy(), period=20, percentage=2.0)

        # In an uptrend, recent prices should be near or above the upper envelope
        # and the envelope should follow the trend upward
        early_middle = result['ma_envelope_middle'].iloc[30]
        late_middle = result['ma_envelope_middle'].iloc[-1]

        self.assertGreater(late_middle, early_middle)

    def test_envelope_polars_ema(self):
        """Test EMA envelope with polars DataFrame."""
        df = pl.DataFrame({
            "Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })
        result = moving_average_envelope(
            df, period=10, percentage=2.5, ma_type='ema'
        )

        self.assertIn('ma_envelope_middle', result.columns)
        self.assertIn('ma_envelope_upper', result.columns)
        self.assertIn('ma_envelope_lower', result.columns)

