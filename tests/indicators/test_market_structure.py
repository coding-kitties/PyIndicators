import unittest
import pandas as pd
import numpy as np
from pyindicators import (
    market_structure_break,
    market_structure_ob,
    msb_signal,
    ob_quality_signal,
    get_market_structure_stats,
    market_structure_choch_bos,
    choch_bos_signal,
    get_choch_bos_stats
)
from pyindicators.exceptions import PyIndicatorException


class TestMarketStructureBreak(unittest.TestCase):
    """Tests for market_structure_break function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n = 200
        # Create trending data with clear pivots
        trend = np.linspace(0, 20, n)
        noise = np.cumsum(np.random.randn(n) * 0.5)
        close = 100 + trend + noise
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2

        self.df = pd.DataFrame({
            'Open': close - np.random.randn(n) * 0.5,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': np.random.randint(1000, 10000, n)
        })

    def test_market_structure_break_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        result = market_structure_break(self.df, pivot_length=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_market_structure_break_adds_columns(self):
        """Test that required columns are added."""
        result = market_structure_break(self.df, pivot_length=5)
        expected_columns = [
            'msb_bullish', 'msb_bearish',
            'last_pivot_high', 'last_pivot_low',
            'momentum_z'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_market_structure_break_signal_values(self):
        """Test that MSB signals are 0 or 1."""
        result = market_structure_break(self.df, pivot_length=5)
        self.assertTrue(result['msb_bullish'].isin([0, 1]).all())
        self.assertTrue(result['msb_bearish'].isin([0, 1]).all())

    def test_market_structure_break_custom_columns(self):
        """Test custom column names."""
        result = market_structure_break(
            self.df,
            pivot_length=5,
            msb_bullish_column='custom_bull',
            msb_bearish_column='custom_bear'
        )
        self.assertIn('custom_bull', result.columns)
        self.assertIn('custom_bear', result.columns)


class TestMarketStructureOB(unittest.TestCase):
    """Tests for market_structure_ob function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n = 200
        trend = np.linspace(0, 20, n)
        noise = np.cumsum(np.random.randn(n) * 0.5)
        close = 100 + trend + noise
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2

        self.df = pd.DataFrame({
            'Open': close - np.random.randn(n) * 0.5,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': np.random.randint(1000, 10000, n)
        })

    def test_market_structure_ob_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        result = market_structure_ob(self.df, pivot_length=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_market_structure_ob_adds_columns(self):
        """Test that required columns are added."""
        result = market_structure_ob(self.df, pivot_length=5)
        expected_columns = [
            'msb_bullish', 'msb_bearish',
            'ob_bullish', 'ob_bearish',
            'ob_top', 'ob_bottom',
            'ob_quality', 'ob_is_hpz', 'ob_mitigated'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_market_structure_ob_quality_range(self):
        """Test that quality scores are in valid range."""
        result = market_structure_ob(self.df, pivot_length=5)
        valid_quality = result['ob_quality'].dropna()
        if len(valid_quality) > 0:
            self.assertTrue((valid_quality >= 0).all())
            self.assertTrue((valid_quality <= 100).all())

    def test_market_structure_ob_hpz_flag(self):
        """Test HPZ flag is boolean."""
        result = market_structure_ob(self.df, pivot_length=5)
        self.assertTrue(result['ob_is_hpz'].dtype == bool)


class TestMSBSignal(unittest.TestCase):
    """Tests for msb_signal function."""

    def setUp(self):
        """Set up test data with MSB columns."""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            'Close': np.random.randn(n) * 10 + 100,
            'msb_bullish': np.random.choice([0, 1], n, p=[0.9, 0.1]),
            'msb_bearish': np.random.choice([0, 1], n, p=[0.9, 0.1])
        })

    def test_msb_signal_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        result = msb_signal(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_msb_signal_values(self):
        """Test signal values are -1, 0, or 1."""
        result = msb_signal(self.df)
        self.assertTrue(result['msb_signal'].isin([-1, 0, 1]).all())

    def test_msb_signal_bullish_mapping(self):
        """Test that bullish MSB maps to 1."""
        df = pd.DataFrame({
            'Close': [100, 101, 102],
            'msb_bullish': [0, 1, 0],
            'msb_bearish': [0, 0, 0]
        })
        result = msb_signal(df)
        self.assertEqual(result['msb_signal'].iloc[1], 1)

    def test_msb_signal_bearish_mapping(self):
        """Test that bearish MSB maps to -1."""
        df = pd.DataFrame({
            'Close': [100, 99, 98],
            'msb_bullish': [0, 0, 0],
            'msb_bearish': [0, 1, 0]
        })
        result = msb_signal(df)
        self.assertEqual(result['msb_signal'].iloc[1], -1)


class TestGetMarketStructureStats(unittest.TestCase):
    """Tests for get_market_structure_stats function."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'msb_bullish': [0, 1, 0, 1, 0],
            'msb_bearish': [1, 0, 1, 0, 0],
            'ob_bullish': [0, 1, 0, 0, 0],
            'ob_bearish': [0, 0, 1, 0, 0],
            'ob_mitigated': [0, 0, 0, 1, 0],
            'ob_is_hpz': [False, True, False, False, False]
        })

    def test_get_market_structure_stats_returns_dict(self):
        """Test that function returns a dictionary."""
        result = get_market_structure_stats(self.df)
        self.assertIsInstance(result, dict)

    def test_get_market_structure_stats_keys(self):
        """Test that all expected keys are present."""
        result = get_market_structure_stats(self.df)
        expected_keys = [
            'total_obs', 'total_mitigated', 'reliability',
            'hpz_count', 'bullish_msb_count', 'bearish_msb_count'
        ]
        for key in expected_keys:
            self.assertIn(key, result)

    def test_get_market_structure_stats_counts(self):
        """Test that counts are correct."""
        result = get_market_structure_stats(self.df)
        self.assertEqual(result['bullish_msb_count'], 2)
        self.assertEqual(result['bearish_msb_count'], 2)
        self.assertEqual(result['total_obs'], 2)
        self.assertEqual(result['total_mitigated'], 1)


class TestMarketStructureChochBos(unittest.TestCase):
    """Tests for market_structure_choch_bos function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n = 200
        # Create data with clear swings
        t = np.linspace(0, 4 * np.pi, n)
        base = 100 + 10 * np.sin(t)
        noise = np.random.randn(n) * 0.5
        close = base + noise
        high = close + np.abs(np.random.randn(n)) * 1.5
        low = close - np.abs(np.random.randn(n)) * 1.5

        self.df = pd.DataFrame({
            'High': high,
            'Low': low,
            'Close': close
        })

    def test_choch_bos_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        result = market_structure_choch_bos(self.df, length=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_choch_bos_adds_columns(self):
        """Test that required columns are added."""
        result = market_structure_choch_bos(self.df, length=5)
        expected_columns = [
            'choch_bullish', 'choch_bearish',
            'bos_bullish', 'bos_bearish',
            'support_level', 'resistance_level',
            'support_broken', 'resistance_broken',
            'market_trend'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_choch_bos_signal_values(self):
        """Test that CHoCH/BOS signals are 0 or 1."""
        result = market_structure_choch_bos(self.df, length=5)
        self.assertTrue(result['choch_bullish'].isin([0, 1]).all())
        self.assertTrue(result['choch_bearish'].isin([0, 1]).all())
        self.assertTrue(result['bos_bullish'].isin([0, 1]).all())
        self.assertTrue(result['bos_bearish'].isin([0, 1]).all())

    def test_choch_bos_trend_values(self):
        """Test that trend values are -1, 0, or 1."""
        result = market_structure_choch_bos(self.df, length=5)
        self.assertTrue(result['market_trend'].isin([-1, 0, 1]).all())

    def test_choch_bos_minimum_length(self):
        """Test that length must be at least 3."""
        with self.assertRaises(PyIndicatorException):
            market_structure_choch_bos(self.df, length=2)

    def test_choch_bos_custom_columns(self):
        """Test custom column names."""
        result = market_structure_choch_bos(
            self.df,
            length=5,
            choch_bullish_column='custom_choch_bull',
            bos_bullish_column='custom_bos_bull'
        )
        self.assertIn('custom_choch_bull', result.columns)
        self.assertIn('custom_bos_bull', result.columns)

    def test_choch_bos_support_resistance_levels(self):
        """Test that S/R levels are numeric or NaN."""
        result = market_structure_choch_bos(self.df, length=5)
        support = result['support_level'].dropna()
        resistance = result['resistance_level'].dropna()
        if len(support) > 0:
            self.assertTrue(np.issubdtype(support.dtype, np.floating))
        if len(resistance) > 0:
            self.assertTrue(np.issubdtype(resistance.dtype, np.floating))


class TestChochBosSignal(unittest.TestCase):
    """Tests for choch_bos_signal function."""

    def setUp(self):
        """Set up test data with CHoCH/BOS columns."""
        self.df = pd.DataFrame({
            'choch_bullish': [0, 1, 0, 0, 0],
            'choch_bearish': [0, 0, 0, 1, 0],
            'bos_bullish': [0, 0, 1, 0, 0],
            'bos_bearish': [1, 0, 0, 0, 0]
        })

    def test_choch_bos_signal_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        result = choch_bos_signal(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_choch_bos_signal_values(self):
        """Test signal values are -2, -1, 0, 1, or 2."""
        result = choch_bos_signal(self.df)
        self.assertTrue(result['structure_signal'].isin([-2, -1, 0, 1, 2]).all())

    def test_choch_bos_signal_bullish_choch_mapping(self):
        """Test that bullish CHoCH maps to 2."""
        result = choch_bos_signal(self.df)
        self.assertEqual(result['structure_signal'].iloc[1], 2)

    def test_choch_bos_signal_bearish_choch_mapping(self):
        """Test that bearish CHoCH maps to -2."""
        result = choch_bos_signal(self.df)
        self.assertEqual(result['structure_signal'].iloc[3], -2)

    def test_choch_bos_signal_bullish_bos_mapping(self):
        """Test that bullish BOS maps to 1."""
        result = choch_bos_signal(self.df)
        self.assertEqual(result['structure_signal'].iloc[2], 1)

    def test_choch_bos_signal_bearish_bos_mapping(self):
        """Test that bearish BOS maps to -1."""
        result = choch_bos_signal(self.df)
        self.assertEqual(result['structure_signal'].iloc[0], -1)


class TestGetChochBosStats(unittest.TestCase):
    """Tests for get_choch_bos_stats function."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'choch_bullish': [0, 1, 0, 1, 0],
            'choch_bearish': [1, 0, 1, 0, 0],
            'choch_plus_bullish': [0, 0, 0, 1, 0],
            'choch_plus_bearish': [0, 1, 0, 0, 0],
            'bos_bullish': [0, 0, 1, 0, 1],
            'bos_bearish': [1, 0, 0, 1, 0]
        })

    def test_get_choch_bos_stats_returns_dict(self):
        """Test that function returns a dictionary."""
        result = get_choch_bos_stats(self.df)
        self.assertIsInstance(result, dict)

    def test_get_choch_bos_stats_keys(self):
        """Test that all expected keys are present."""
        result = get_choch_bos_stats(self.df)
        expected_keys = [
            'choch_bullish_count', 'choch_bearish_count',
            'choch_plus_bullish_count',
            'choch_plus_bearish_count',
            'bos_bullish_count', 'bos_bearish_count',
            'total_choch', 'total_choch_plus', 'total_bos'
        ]
        for key in expected_keys:
            self.assertIn(key, result)

    def test_get_choch_bos_stats_counts(self):
        """Test that counts are correct."""
        result = get_choch_bos_stats(self.df)
        self.assertEqual(result['choch_bullish_count'], 2)
        self.assertEqual(result['choch_bearish_count'], 2)
        self.assertEqual(result['choch_plus_bullish_count'], 1)
        self.assertEqual(result['choch_plus_bearish_count'], 1)
        self.assertEqual(result['bos_bullish_count'], 2)
        self.assertEqual(result['bos_bearish_count'], 2)
        self.assertEqual(result['total_choch'], 4)
        self.assertEqual(result['total_choch_plus'], 2)
        self.assertEqual(result['total_bos'], 4)


class TestPolarsSupport(unittest.TestCase):
    """Tests for Polars DataFrame support."""

    def setUp(self):
        """Set up test data."""
        try:
            import polars as pl
            self.polars_available = True
            np.random.seed(42)
            n = 100
            close = 100 + np.cumsum(np.random.randn(n) * 0.5)
            high = close + np.abs(np.random.randn(n))
            low = close - np.abs(np.random.randn(n))

            self.df_polars = pl.DataFrame({
                'Open': close - np.random.randn(n) * 0.5,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': np.random.randint(1000, 10000, n)
            })
        except ImportError:
            self.polars_available = False

    def test_market_structure_break_polars(self):
        """Test market_structure_break with Polars DataFrame."""
        if not self.polars_available:
            self.skipTest("Polars not available")
        import polars as pl
        result = market_structure_break(self.df_polars, pivot_length=5)
        self.assertIsInstance(result, pl.DataFrame)

    def test_market_structure_choch_bos_polars(self):
        """Test market_structure_choch_bos with Polars DataFrame."""
        if not self.polars_available:
            self.skipTest("Polars not available")
        import polars as pl
        result = market_structure_choch_bos(self.df_polars, length=5)
        self.assertIsInstance(result, pl.DataFrame)


if __name__ == '__main__':
    unittest.main()
