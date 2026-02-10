import unittest
import pandas as pd
import numpy as np

from pyindicators import supertrend_ai, supertrend, get_supertrend_stats


class TestSuperTrend(unittest.TestCase):
    """Test cases for SuperTrend indicator."""

    def setUp(self):
        """Set up test data."""
        # Create sample OHLC data with a clear trend
        np.random.seed(42)
        n = 200

        # Generate trending data
        base_price = 100
        trend = np.linspace(0, 30, n)  # Upward trend
        noise = np.random.randn(n) * 2

        close = base_price + trend + noise
        high = close + np.abs(np.random.randn(n) * 1.5)
        low = close - np.abs(np.random.randn(n) * 1.5)
        open_price = close + np.random.randn(n) * 0.5

        self.df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close
        })

    def test_supertrend_returns_dataframe(self):
        """Test that supertrend returns a DataFrame."""
        result = supertrend(self.df.copy())
        self.assertIsInstance(result, pd.DataFrame)

    def test_supertrend_columns(self):
        """Test that supertrend adds expected columns."""
        result = supertrend(self.df.copy())

        expected_columns = [
            'supertrend', 'supertrend_trend',
            'supertrend_upper', 'supertrend_lower', 'supertrend_signal'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_supertrend_trend_values(self):
        """Test that trend values are 0 or 1."""
        result = supertrend(self.df.copy())
        unique_trends = result['supertrend_trend'].unique()
        for trend in unique_trends:
            self.assertIn(trend, [0, 1])

    def test_supertrend_signal_values(self):
        """Test that signal values are -1, 0, or 1."""
        result = supertrend(self.df.copy())
        unique_signals = result['supertrend_signal'].unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])

    def test_supertrend_custom_parameters(self):
        """Test supertrend with custom parameters."""
        result = supertrend(self.df.copy(), atr_length=14, factor=2.0)
        self.assertIn('supertrend', result.columns)

    def test_supertrend_ai_returns_dataframe(self):
        """Test that supertrend_ai returns a DataFrame."""
        result = supertrend_ai(self.df.copy(), max_data=100)
        self.assertIsInstance(result, pd.DataFrame)

    def test_supertrend_ai_columns(self):
        """Test that supertrend_ai adds expected columns."""
        result = supertrend_ai(self.df.copy(), max_data=100)

        expected_columns = [
            'supertrend', 'supertrend_trend', 'supertrend_ama',
            'supertrend_perf_idx', 'supertrend_factor', 'supertrend_signal'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_supertrend_ai_factor_range(self):
        """Test that supertrend_ai uses factors within specified range."""
        min_mult, max_mult = 1.0, 3.0
        result = supertrend_ai(
            self.df.copy(),
            min_mult=min_mult,
            max_mult=max_mult,
            max_data=100
        )

        # Factor should be within range (or NaN for initial bars)
        valid_factors = result['supertrend_factor'].dropna()
        for factor in valid_factors:
            self.assertGreaterEqual(factor, min_mult - 0.1)
            self.assertLessEqual(factor, max_mult + 0.1)

    def test_supertrend_ai_cluster_options(self):
        """Test supertrend_ai with different cluster options."""
        for cluster in ['best', 'average', 'worst']:
            result = supertrend_ai(
                self.df.copy(),
                from_cluster=cluster,
                max_data=100
            )
            self.assertIn('supertrend', result.columns)

    def test_supertrend_ai_invalid_factor_range(self):
        """Test that invalid factor range raises error."""
        with self.assertRaises(ValueError):
            supertrend_ai(self.df.copy(), min_mult=5.0, max_mult=1.0)

    def test_get_supertrend_stats(self):
        """Test statistics function."""
        result = supertrend(self.df.copy())
        stats = get_supertrend_stats(result)

        self.assertIn('buy_signals', stats)
        self.assertIn('sell_signals', stats)
        self.assertIn('current_trend', stats)
        self.assertIn(stats['current_trend'], ['bullish', 'bearish'])

    def test_get_supertrend_stats_ai(self):
        """Test statistics for AI version includes additional fields."""
        result = supertrend_ai(self.df.copy(), max_data=100)
        stats = get_supertrend_stats(result)

        self.assertIn('avg_factor', stats)
        self.assertIn('avg_perf_idx', stats)

    def test_supertrend_upper_lower_relationship(self):
        """Test that upper band is always above lower band."""
        result = supertrend(self.df.copy())

        # After initialization period
        for i in range(10, len(result)):
            upper = result['supertrend_upper'].iloc[i]
            lower = result['supertrend_lower'].iloc[i]
            if not np.isnan(upper) and not np.isnan(lower):
                self.assertGreater(upper, lower)

    def test_supertrend_with_minimal_data(self):
        """Test SuperTrend with minimal data."""
        minimal_df = self.df.head(30).copy()
        result = supertrend(minimal_df)
        self.assertEqual(len(result), 30)

    def test_supertrend_signal_on_trend_change(self):
        """Test that signals occur when trend changes."""
        result = supertrend(self.df.copy())

        # Find where trend changes
        trend_changes = result['supertrend_trend'].diff().abs()

        # Where trend changes, there should be a non-zero signal
        for i in range(1, len(result)):
            if trend_changes.iloc[i] == 1:
                self.assertNotEqual(
                    result['supertrend_signal'].iloc[i], 0,
                    f"Signal should be non-zero at index {i}"
                )


if __name__ == '__main__':
    unittest.main()
