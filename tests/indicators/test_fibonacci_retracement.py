import numpy as np
from unittest import TestCase
import pandas as pd
import polars as pl
from pyindicators import fibonacci_retracement, fibonacci_retracement_levels, \
    fibonacci_extension


class TestFibonacciRetracement(TestCase):

    def test_fibonacci_retracement_pandas(self):
        """Test Fibonacci retracement calculation with pandas."""
        df = pd.DataFrame({
            "High": [100, 105, 110, 115, 120, 118, 115, 112, 110, 108],
            "Low": [95, 100, 105, 110, 115, 113, 110, 107, 105, 103],
            "Close": [98, 103, 108, 113, 118, 116, 113, 110, 108, 105]
        })
        result = fibonacci_retracement(df.copy())

        # Check all standard Fibonacci levels are present
        self.assertIn('fib_0.0', result.columns)
        self.assertIn('fib_0.236', result.columns)
        self.assertIn('fib_0.382', result.columns)
        self.assertIn('fib_0.5', result.columns)
        self.assertIn('fib_0.618', result.columns)
        self.assertIn('fib_0.786', result.columns)
        self.assertIn('fib_1.0', result.columns)
        self.assertIn('fib_swing_high', result.columns)
        self.assertIn('fib_swing_low', result.columns)

        # fib_0.0 should equal swing high
        self.assertEqual(result['fib_0.0'].iloc[0], result['fib_swing_high'].iloc[0])
        # fib_1.0 should equal swing low
        self.assertEqual(result['fib_1.0'].iloc[0], result['fib_swing_low'].iloc[0])

    def test_fibonacci_retracement_polars(self):
        """Test Fibonacci retracement calculation with polars."""
        df = pl.DataFrame({
            "High": [100, 105, 110, 115, 120, 118, 115, 112, 110, 108],
            "Low": [95, 100, 105, 110, 115, 113, 110, 107, 105, 103],
            "Close": [98, 103, 108, 113, 118, 116, 113, 110, 108, 105]
        })
        result = fibonacci_retracement(df)

        self.assertIn('fib_0.0', result.columns)
        self.assertIn('fib_0.618', result.columns)
        self.assertIn('fib_1.0', result.columns)

    def test_fibonacci_retracement_manual_swing(self):
        """Test Fibonacci retracement with manual swing high/low."""
        df = pd.DataFrame({
            "High": [100, 105, 110],
            "Low": [95, 100, 105],
            "Close": [98, 103, 108]
        })

        # Manual swing values like in the chart (Gold example)
        result = fibonacci_retracement(
            df.copy(),
            swing_high=5698.75,
            swing_low=3613.155
        )

        # Verify calculations match expected values from chart
        self.assertAlmostEqual(result['fib_0.0'].iloc[0], 5698.75, places=2)
        self.assertAlmostEqual(result['fib_1.0'].iloc[0], 3613.155, places=2)

        # Check 0.236 level (should be around 5130.15)
        expected_236 = 5698.75 - (5698.75 - 3613.155) * 0.236
        self.assertAlmostEqual(result['fib_0.236'].iloc[0], expected_236, places=2)

    def test_fibonacci_retracement_levels_utility(self):
        """Test the utility function for calculating levels."""
        levels = fibonacci_retracement_levels(
            swing_high=5698.75,
            swing_low=3613.155
        )

        self.assertEqual(levels[0.0], 5698.75)
        self.assertEqual(levels[1.0], 3613.155)

        # Check 0.5 level (midpoint)
        expected_50 = 5698.75 - (5698.75 - 3613.155) * 0.5
        self.assertAlmostEqual(levels[0.5], expected_50, places=2)

    def test_fibonacci_retracement_custom_levels(self):
        """Test Fibonacci retracement with custom levels."""
        df = pd.DataFrame({
            "High": [100, 110, 120],
            "Low": [90, 100, 110]
        })

        custom_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = fibonacci_retracement(df.copy(), levels=custom_levels)

        self.assertIn('fib_0.25', result.columns)
        self.assertIn('fib_0.75', result.columns)
        # Standard levels should not be present
        self.assertNotIn('fib_0.236', result.columns)
        self.assertNotIn('fib_0.618', result.columns)

    def test_fibonacci_retracement_lookback_period(self):
        """Test Fibonacci retracement with lookback period."""
        # Create data with different highs/lows in different periods
        df = pd.DataFrame({
            "High": [100, 105, 110, 150, 145, 140, 135, 130, 125, 120],
            "Low": [95, 100, 105, 145, 140, 135, 130, 125, 120, 115]
        })

        # Without lookback - should use full range
        result_full = fibonacci_retracement(df.copy())

        # With lookback of 5 - should only look at last 5 rows
        result_lookback = fibonacci_retracement(df.copy(), lookback_period=5)

        # Different swing highs expected
        self.assertNotEqual(
            result_full['fib_swing_high'].iloc[0],
            result_lookback['fib_swing_high'].iloc[0]
        )

    def test_fibonacci_retracement_custom_prefix(self):
        """Test Fibonacci retracement with custom result prefix."""
        df = pd.DataFrame({
            "High": [100, 110, 120],
            "Low": [90, 100, 110]
        })

        result = fibonacci_retracement(df.copy(), result_prefix='gold_fib')

        self.assertIn('gold_fib_0.0', result.columns)
        self.assertIn('gold_fib_0.618', result.columns)
        self.assertNotIn('fib_0.0', result.columns)

    def test_fibonacci_retracement_datetime_index(self):
        """Test that Fibonacci retracement works with DatetimeIndex."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            "High": [100, 105, 110, 115, 120, 118, 115, 112, 110, 108],
            "Low": [95, 100, 105, 110, 115, 113, 110, 107, 105, 103]
        }, index=dates)

        result = fibonacci_retracement(df.copy())

        self.assertIn('fib_0.618', result.columns)
        self.assertEqual(len(result), 10)


class TestFibonacciExtension(TestCase):

    def test_fibonacci_extension_pandas(self):
        """Test Fibonacci extension calculation with pandas."""
        df = pd.DataFrame({
            "High": [100, 105, 110, 115, 120],
            "Low": [95, 100, 105, 110, 115]
        })
        result = fibonacci_extension(df.copy())

        # Check extension levels are present
        self.assertIn('fib_ext_1.0', result.columns)
        self.assertIn('fib_ext_1.618', result.columns)
        self.assertIn('fib_ext_2.0', result.columns)

    def test_fibonacci_extension_polars(self):
        """Test Fibonacci extension calculation with polars."""
        df = pl.DataFrame({
            "High": [100, 105, 110, 115, 120],
            "Low": [95, 100, 105, 110, 115]
        })
        result = fibonacci_extension(df)

        self.assertIn('fib_ext_1.618', result.columns)

    def test_fibonacci_extension_calculation(self):
        """Test that extension levels are calculated correctly."""
        df = pd.DataFrame({
            "High": [100, 110, 120],  # swing high = 120
            "Low": [90, 100, 110]     # swing low = 90
        })

        result = fibonacci_extension(df.copy())

        # Range is 30 (120 - 90)
        # 1.618 extension = 120 + (30 * 0.618) = 138.54
        expected_1618 = 120 + (30 * 0.618)
        self.assertAlmostEqual(result['fib_ext_1.618'].iloc[0], expected_1618, places=2)

        # 2.0 extension = 120 + (30 * 1.0) = 150
        self.assertAlmostEqual(result['fib_ext_2.0'].iloc[0], 150.0, places=2)


class TestFibonacciLevelsRealWorld(TestCase):

    def test_gold_chart_levels(self):
        """Test Fibonacci levels matching the Gold chart example."""
        # From the chart: Gold swing high ~5698.75, swing low ~3613.155
        levels = fibonacci_retracement_levels(
            swing_high=5698.75,
            swing_low=3613.155
        )

        # Verify levels match chart annotations
        # 0.236 level should be around 5130.15
        self.assertAlmostEqual(levels[0.236], 5206.58, delta=1)

        # 0.382 level should be around 4840.25
        self.assertAlmostEqual(levels[0.382], 4901.89, delta=1)

        # 0.5 level should be around 4605.95
        self.assertAlmostEqual(levels[0.5], 4655.95, delta=1)

        # 0.618 level should be around 4371.65
        self.assertAlmostEqual(levels[0.618], 4410.02, delta=1)

        # 0.786 level should be around 4038.07
        self.assertAlmostEqual(levels[0.786], 4060.16, delta=1)

    def test_silver_chart_levels(self):
        """Test Fibonacci levels for a Silver-like scenario."""
        # Silver example with higher volatility
        levels = fibonacci_retracement_levels(
            swing_high=121.65,
            swing_low=40.73
        )

        # 0.236 level
        expected_236 = 121.65 - (121.65 - 40.73) * 0.236
        self.assertAlmostEqual(levels[0.236], expected_236, places=2)

        # 0.618 (Golden ratio)
        expected_618 = 121.65 - (121.65 - 40.73) * 0.618
        self.assertAlmostEqual(levels[0.618], expected_618, places=2)

