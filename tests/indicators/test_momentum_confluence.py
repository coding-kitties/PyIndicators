"""dTests for the Momentum Confluence indicator.
"""
import unittest
import pandas as pd
import numpy as np
from pyindicators import (
    momentum_confluence,
    momentum_confluence_signal,
    get_momentum_confluence_stats
)


class TestMomentumConfluence(unittest.TestCase):
    """Test cases for momentum_confluence function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n = 100

        # Generate realistic OHLCV data
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_prices = close + np.random.randn(n) * 0.5
        volume = np.random.randint(1000, 10000, n)

        self.df = pd.DataFrame({
            'Open': open_prices,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })

    def test_momentum_confluence_returns_dataframe(self):
        """Test that momentum_confluence returns a DataFrame."""
        result = momentum_confluence(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_momentum_confluence_adds_all_columns(self):
        """Test that all expected columns are added."""
        result = momentum_confluence(self.df)

        expected_columns = [
            'money_flow',
            'mf_upper_threshold',
            'mf_lower_threshold',
            'overflow_bullish',
            'overflow_bearish',
            'trend_wave',
            'trend_wave_signal',
            'divergence_bullish',
            'divergence_bearish',
            'reversal_bullish',
            'reversal_bearish',
            'reversal_strong_bullish',
            'reversal_strong_bearish',
            'confluence',
            'mc_trend'
        ]

        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_money_flow_range(self):
        """Test that money flow is within expected range."""
        result = momentum_confluence(self.df)
        money_flow = result['money_flow'].dropna()

        self.assertTrue((money_flow >= -100).all())
        self.assertTrue((money_flow <= 100).all())

    def test_trend_wave_range(self):
        """Test that trend wave is within 0-100 range."""
        result = momentum_confluence(self.df)
        trend_wave = result['trend_wave'].dropna()

        self.assertTrue((trend_wave >= 0).all())
        self.assertTrue((trend_wave <= 100).all())

    def test_confluence_range(self):
        """Test that confluence is within -100 to 100 range."""
        result = momentum_confluence(self.df)
        confluence = result['confluence'].dropna()

        self.assertTrue((confluence >= -100).all())
        self.assertTrue((confluence <= 100).all())

    def test_trend_values(self):
        """Test that trend values are -1, 0, or 1."""
        result = momentum_confluence(self.df)
        trend = result['mc_trend'].unique()

        for val in trend:
            self.assertIn(val, [-1, 0, 1])

    def test_trend_wave_signal_values(self):
        """Test that trend wave signal values are -1, 0, or 1."""
        result = momentum_confluence(self.df)
        signal = result['trend_wave_signal'].unique()

        for val in signal:
            self.assertIn(val, [-1, 0, 1])

    def test_binary_columns_are_0_or_1(self):
        """Test that binary columns contain only 0 or 1."""
        result = momentum_confluence(self.df)

        binary_columns = [
            'overflow_bullish',
            'overflow_bearish',
            'divergence_bullish',
            'divergence_bearish',
            'reversal_bullish',
            'reversal_bearish',
            'reversal_strong_bullish',
            'reversal_strong_bearish'
        ]

        for col in binary_columns:
            unique_vals = result[col].unique()
            for val in unique_vals:
                self.assertIn(val, [0, 1])

    def test_custom_column_names(self):
        """Test that custom column names work."""
        result = momentum_confluence(
            self.df,
            money_flow_column='mf_custom',
            trend_wave_column='tw_custom',
            confluence_column='conf_custom'
        )

        self.assertIn('mf_custom', result.columns)
        self.assertIn('tw_custom', result.columns)
        self.assertIn('conf_custom', result.columns)

    def test_custom_parameters(self):
        """Test that custom parameters are applied."""
        result1 = momentum_confluence(self.df, money_flow_length=7)
        result2 = momentum_confluence(self.df, money_flow_length=21)

        # Results should be different with different parameters
        self.assertFalse(
            result1['money_flow'].equals(result2['money_flow'])
        )

    def test_without_volume(self):
        """Test that indicator works without volume column."""
        df_no_volume = self.df.drop(columns=['Volume'])
        result = momentum_confluence(df_no_volume)

        self.assertIn('money_flow', result.columns)
        self.assertIn('trend_wave', result.columns)


class TestMomentumConfluenceSignal(unittest.TestCase):
    """Test cases for momentum_confluence_signal function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n = 100

        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_prices = close + np.random.randn(n) * 0.5
        volume = np.random.randint(1000, 10000, n)

        self.df = pd.DataFrame({
            'Open': open_prices,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
        self.df = momentum_confluence(self.df)

    def test_signal_column_added(self):
        """Test that signal column is added."""
        result = momentum_confluence_signal(self.df)
        self.assertIn('mc_signal', result.columns)

    def test_signal_values(self):
        """Test that signal values are in expected range."""
        result = momentum_confluence_signal(self.df)
        unique_vals = result['mc_signal'].unique()

        for val in unique_vals:
            self.assertIn(val, [-2, -1, 0, 1, 2])

    def test_custom_signal_column_name(self):
        """Test that custom signal column name works."""
        result = momentum_confluence_signal(
            self.df,
            signal_column='custom_signal'
        )
        self.assertIn('custom_signal', result.columns)


class TestGetMomentumConfluenceStats(unittest.TestCase):
    """Test cases for get_momentum_confluence_stats function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n = 100

        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_prices = close + np.random.randn(n) * 0.5
        volume = np.random.randint(1000, 10000, n)

        self.df = pd.DataFrame({
            'Open': open_prices,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
        self.df = momentum_confluence(self.df)

    def test_stats_returns_dict(self):
        """Test that stats returns a dictionary."""
        stats = get_momentum_confluence_stats(self.df)
        self.assertIsInstance(stats, dict)

    def test_stats_has_expected_keys(self):
        """Test that stats has all expected keys."""
        stats = get_momentum_confluence_stats(self.df)

        expected_keys = [
            'reversal_bullish_count',
            'reversal_bearish_count',
            'strong_reversal_bullish_count',
            'strong_reversal_bearish_count',
            'divergence_bullish_count',
            'divergence_bearish_count',
            'overflow_bullish_count',
            'overflow_bearish_count'
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

    def test_stats_values_are_non_negative(self):
        """Test that all stat values are non-negative integers."""
        stats = get_momentum_confluence_stats(self.df)

        for key, value in stats.items():
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)


if __name__ == '__main__':
    unittest.main()
