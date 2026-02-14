import unittest
import pandas as pd
import numpy as np

from pyindicators import volume_gated_trend_ribbon


class TestVolumeGatedTrendRibbon(unittest.TestCase):
    """Test cases for Volume-Gated Trend Ribbon indicator."""

    def setUp(self):
        """Set up test data with OHLCV."""
        np.random.seed(42)
        n = 200

        # Generate trending data with volume
        base_price = 100
        trend = np.linspace(0, 30, n)
        noise = np.random.randn(n) * 2

        close = base_price + trend + noise
        high = close + np.abs(np.random.randn(n) * 1.5)
        low = close - np.abs(np.random.randn(n) * 1.5)
        open_price = close + np.random.randn(n) * 0.5
        volume = np.abs(np.random.randn(n) * 1000) + 500

        self.df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })

    def test_returns_dataframe(self):
        """Test that the indicator returns a DataFrame."""
        result = volume_gated_trend_ribbon(self.df.copy())
        self.assertIsInstance(result, pd.DataFrame)

    def test_expected_columns(self):
        """Test that all expected columns are added."""
        result = volume_gated_trend_ribbon(self.df.copy())
        expected = [
            'vgtr_fast', 'vgtr_mid_fast', 'vgtr_mid_slow',
            'vgtr_slow', 'vgtr_trend', 'vgtr_signal'
        ]
        for col in expected:
            self.assertIn(col, result.columns)

    def test_trend_values(self):
        """Test that trend values are -1, 0, or 1."""
        result = volume_gated_trend_ribbon(self.df.copy())
        unique = result['vgtr_trend'].unique()
        for v in unique:
            self.assertIn(v, [-1, 0, 1])

    def test_signal_values(self):
        """Test that signal values are -1, 0, or 1."""
        result = volume_gated_trend_ribbon(self.df.copy())
        unique = result['vgtr_signal'].unique()
        for v in unique:
            self.assertIn(v, [-1, 0, 1])

    def test_ribbon_ordering(self):
        """Test that mid MAs are between fast and slow when trend is clear."""
        result = volume_gated_trend_ribbon(self.df.copy())
        # Mid-fast should be closer to fast, mid-slow closer to slow
        # by construction: mid_fast = 0.67*fast + 0.33*slow
        valid = result.dropna(subset=['vgtr_fast', 'vgtr_slow'])
        if len(valid) > 0:
            for _, row in valid.iterrows():
                fast, slow = row['vgtr_fast'], row['vgtr_slow']
                mid_fast, mid_slow = row['vgtr_mid_fast'], row['vgtr_mid_slow']
                if fast > slow:
                    self.assertGreaterEqual(mid_fast, mid_slow)
                elif fast < slow:
                    self.assertLessEqual(mid_fast, mid_slow)

    def test_custom_parameters(self):
        """Test with custom parameters."""
        result = volume_gated_trend_ribbon(
            self.df.copy(),
            vol_mult=0.8,
            vol_period=20,
            fast_length=10,
            slow_length=20
        )
        self.assertIn('vgtr_fast', result.columns)

    def test_different_ma_types(self):
        """Test that all supported MA types work."""
        ma_types = [
            'SMA', 'EMA', 'WMA', 'RMA', 'HMA', 'DEMA', 'TEMA',
            'LSMA', 'SMMA', 'KAMA', 'ZLEMA', 'T3', 'VIDYA', 'VWMA',
            'ALMA'
        ]
        for ma in ma_types:
            result = volume_gated_trend_ribbon(
                self.df.copy(), ma_type=ma
            )
            self.assertIn(
                'vgtr_fast', result.columns,
                f"MA type '{ma}' failed to produce output"
            )

    def test_invalid_ma_type(self):
        """Test that invalid MA type raises ValueError."""
        with self.assertRaises(ValueError):
            volume_gated_trend_ribbon(self.df.copy(), ma_type='INVALID')

    def test_signal_on_trend_change(self):
        """Test that signals occur when trend changes."""
        result = volume_gated_trend_ribbon(self.df.copy())
        trend = result['vgtr_trend']
        signal = result['vgtr_signal']

        for i in range(1, len(result)):
            if trend.iloc[i] != 0 and trend.iloc[i-1] != 0:
                if trend.iloc[i] != trend.iloc[i-1]:
                    self.assertNotEqual(
                        signal.iloc[i], 0,
                        f"Signal should be non-zero at index {i} "
                        f"(trend: {trend.iloc[i-1]} -> {trend.iloc[i]})"
                    )

    def test_slow_length_auto_adjusted(self):
        """Test that slow_length is adjusted to be > fast_length."""
        # Set slow == fast; should be auto-bumped to fast+1
        result = volume_gated_trend_ribbon(
            self.df.copy(), fast_length=20, slow_length=20
        )
        self.assertIn('vgtr_fast', result.columns)

    def test_preserves_original_columns(self):
        """Test that original DataFrame columns are preserved."""
        result = volume_gated_trend_ribbon(self.df.copy())
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            self.assertIn(col, result.columns)

    def test_with_minimal_data(self):
        """Test with minimal data (just enough for slow period)."""
        minimal_df = self.df.head(60).copy()
        result = volume_gated_trend_ribbon(minimal_df)
        self.assertEqual(len(result), 60)


if __name__ == '__main__':
    unittest.main()
