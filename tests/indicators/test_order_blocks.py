from unittest import TestCase
import pandas as pd
import polars as pl
import numpy as np
from pyindicators import order_blocks, ob_signal, get_active_order_blocks
from pyindicators.exceptions import PyIndicatorException


class TestOrderBlocks(TestCase):

    def test_order_blocks_pandas(self):
        """Test Order Blocks detection with pandas."""
        # Create sample data with enough bars
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_price = close + np.random.randn(n) * 0.5

        df = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close
        })
        result = order_blocks(df.copy(), swing_length=5)

        self.assertIn('bullish_ob', result.columns)
        self.assertIn('bearish_ob', result.columns)
        self.assertIn('bullish_ob_top', result.columns)
        self.assertIn('bullish_ob_bottom', result.columns)
        self.assertIn('bearish_ob_top', result.columns)
        self.assertIn('bearish_ob_bottom', result.columns)
        self.assertIn('bullish_breaker', result.columns)
        self.assertIn('bearish_breaker', result.columns)

    def test_order_blocks_polars(self):
        """Test Order Blocks detection with polars."""
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_price = close + np.random.randn(n) * 0.5

        df = pl.DataFrame({
            "Open": open_price.tolist(),
            "High": high.tolist(),
            "Low": low.tolist(),
            "Close": close.tolist()
        })
        result = order_blocks(df, swing_length=5)

        self.assertIn('bullish_ob', result.columns)
        self.assertIn('bearish_ob', result.columns)

    def test_order_blocks_custom_columns(self):
        """Test Order Blocks with custom column names."""
        np.random.seed(42)
        n = 30
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_price = close + np.random.randn(n) * 0.5

        df = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close
        })
        result = order_blocks(
            df.copy(),
            swing_length=5,
            bullish_ob_column='bull_ob',
            bearish_ob_column='bear_ob'
        )

        self.assertIn('bull_ob', result.columns)
        self.assertIn('bear_ob', result.columns)

    def test_order_blocks_invalid_dataframe_type(self):
        """Test that invalid input type raises exception."""
        with self.assertRaises(PyIndicatorException):
            order_blocks([1, 2, 3])

    def test_order_blocks_use_body(self):
        """Test Order Blocks with use_body=True."""
        np.random.seed(42)
        n = 30
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2
        open_price = close + np.random.randn(n) * 0.5

        df = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close
        })
        result = order_blocks(df.copy(), swing_length=5, use_body=True)

        self.assertIn('bullish_ob', result.columns)
        self.assertIn('bearish_ob', result.columns)

    def test_order_blocks_minimum_data(self):
        """Test Order Blocks with minimum data."""
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [102, 103, 104],
            "Low": [99, 100, 101],
            "Close": [101, 102, 103]
        })
        # Should work with small data (but likely no OBs detected)
        result = order_blocks(df.copy(), swing_length=2)

        self.assertEqual(len(result), 3)
        self.assertIn('bullish_ob', result.columns)


class TestOBSignal(TestCase):

    def test_ob_signal_pandas(self):
        """Test OB signal with pandas."""
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_price = close + np.random.randn(n) * 0.5

        df = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close
        })
        df = order_blocks(df, swing_length=5)
        result = ob_signal(df)

        self.assertIn('ob_signal', result.columns)

    def test_ob_signal_polars(self):
        """Test OB signal with polars."""
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_price = close + np.random.randn(n) * 0.5

        df = pl.DataFrame({
            "Open": open_price.tolist(),
            "High": high.tolist(),
            "Low": low.tolist(),
            "Close": close.tolist()
        })
        df = order_blocks(df, swing_length=5)
        result = ob_signal(df)

        self.assertIn('ob_signal', result.columns)

    def test_ob_signal_custom_column(self):
        """Test OB signal with custom signal column name."""
        np.random.seed(42)
        n = 30
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_price = close + np.random.randn(n) * 0.5

        df = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close
        })
        df = order_blocks(df, swing_length=5)
        result = ob_signal(df, signal_column='my_signal')

        self.assertIn('my_signal', result.columns)

    def test_ob_signal_invalid_dataframe_type(self):
        """Test that invalid input type raises exception."""
        with self.assertRaises(PyIndicatorException):
            ob_signal([1, 2, 3])


class TestGetActiveOrderBlocks(TestCase):

    def test_get_active_order_blocks(self):
        """Test getting active order blocks."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_price = close + np.random.randn(n) * 0.5

        df = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close
        })
        df = order_blocks(df, swing_length=5)
        result = get_active_order_blocks(df)

        self.assertIn('bullish', result)
        self.assertIn('bearish', result)
        self.assertIsInstance(result['bullish'], list)
        self.assertIsInstance(result['bearish'], list)

    def test_get_active_order_blocks_max_count(self):
        """Test max bullish/bearish count parameter."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_price = close + np.random.randn(n) * 0.5

        df = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close
        })
        df = order_blocks(df, swing_length=5)
        result = get_active_order_blocks(df, max_bullish=1, max_bearish=1)

        self.assertLessEqual(len(result['bullish']), 1)
        self.assertLessEqual(len(result['bearish']), 1)

    def test_get_active_order_blocks_polars(self):
        """Test getting active order blocks with polars."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_price = close + np.random.randn(n) * 0.5

        df = pl.DataFrame({
            "Open": open_price.tolist(),
            "High": high.tolist(),
            "Low": low.tolist(),
            "Close": close.tolist()
        })
        df = order_blocks(df, swing_length=5)
        result = get_active_order_blocks(df)

        self.assertIn('bullish', result)
        self.assertIn('bearish', result)


class TestOrderBlocksEdgeCases(TestCase):

    def test_order_blocks_trending_up(self):
        """Test Order Blocks in strong uptrend."""
        n = 50
        close = np.linspace(100, 150, n)
        high = close + 1
        low = close - 1
        open_price = close - 0.5

        df = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close
        })
        result = order_blocks(df.copy(), swing_length=5)

        # Should complete without errors
        self.assertEqual(len(result), n)

    def test_order_blocks_trending_down(self):
        """Test Order Blocks in strong downtrend."""
        n = 50
        close = np.linspace(150, 100, n)
        high = close + 1
        low = close - 1
        open_price = close + 0.5

        df = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close
        })
        result = order_blocks(df.copy(), swing_length=5)

        # Should complete without errors
        self.assertEqual(len(result), n)

    def test_order_blocks_flat_market(self):
        """Test Order Blocks in flat market."""
        n = 50
        close = np.full(n, 100.0) + np.random.randn(n) * 0.1
        high = close + 0.5
        low = close - 0.5
        open_price = close

        df = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close
        })
        result = order_blocks(df.copy(), swing_length=5)

        # Should complete without errors
        self.assertEqual(len(result), n)

    def test_order_blocks_different_swing_lengths(self):
        """Test Order Blocks with different swing lengths."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_price = close + np.random.randn(n) * 0.5

        df = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close
        })

        for swing_length in [3, 5, 10, 20]:
            result = order_blocks(df.copy(), swing_length=swing_length)
            self.assertIn('bullish_ob', result.columns)
            self.assertIn('bearish_ob', result.columns)
