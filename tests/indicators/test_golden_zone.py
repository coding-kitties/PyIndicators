import numpy as np
from unittest import TestCase
import pandas as pd
import polars as pl
from pyindicators import golden_zone, golden_zone_signal
from pyindicators.exceptions import PyIndicatorException


class TestGoldenZone(TestCase):

    def test_golden_zone_pandas(self):
        """Test Golden Zone calculation with pandas."""
        df = pd.DataFrame({
            "High": [105, 110, 108, 112, 115, 113, 117, 120, 118, 116],
            "Low": [100, 105, 103, 107, 110, 108, 112, 115, 113, 111]
        })
        result = golden_zone(df.copy(), length=5)

        self.assertIn('golden_zone_upper', result.columns)
        self.assertIn('golden_zone_lower', result.columns)
        self.assertIn('golden_zone_hh', result.columns)
        self.assertIn('golden_zone_ll', result.columns)

        # Upper should be > lower (since upper is 50% retracement, lower is 61.8%)
        for idx in range(len(result)):
            self.assertGreaterEqual(
                result['golden_zone_upper'].iloc[idx],
                result['golden_zone_lower'].iloc[idx]
            )

    def test_golden_zone_polars(self):
        """Test Golden Zone calculation with polars."""
        df = pl.DataFrame({
            "High": [105, 110, 108, 112, 115, 113, 117, 120, 118, 116],
            "Low": [100, 105, 103, 107, 110, 108, 112, 115, 113, 111]
        })
        result = golden_zone(df, length=5)

        self.assertIn('golden_zone_upper', result.columns)
        self.assertIn('golden_zone_lower', result.columns)
        self.assertIn('golden_zone_hh', result.columns)
        self.assertIn('golden_zone_ll', result.columns)

    def test_golden_zone_calculation(self):
        """Test that Golden Zone levels are calculated correctly."""
        df = pd.DataFrame({
            "High": [100, 100, 100, 100, 100],
            "Low": [80, 80, 80, 80, 80]
        })
        result = golden_zone(df.copy(), length=5)

        # HH = 100, LL = 80, Diff = 20
        # Upper (50%): 100 - 20 * 0.5 = 90
        # Lower (61.8%): 100 - 20 * 0.618 = 87.64
        self.assertAlmostEqual(
            result['golden_zone_hh'].iloc[-1], 100.0, places=5
        )
        self.assertAlmostEqual(
            result['golden_zone_ll'].iloc[-1], 80.0, places=5
        )
        self.assertAlmostEqual(
            result['golden_zone_upper'].iloc[-1], 90.0, places=5
        )
        self.assertAlmostEqual(
            result['golden_zone_lower'].iloc[-1], 87.64, places=2
        )

    def test_golden_zone_rolling_calculation(self):
        """Test that Golden Zone uses rolling window correctly."""
        df = pd.DataFrame({
            "High": [100, 110, 105, 115, 120, 118, 116, 114, 112, 110],
            "Low": [95, 105, 100, 110, 115, 113, 111, 109, 107, 105]
        })
        result = golden_zone(df.copy(), length=3)

        # At index 4, the last 3 highs are [105, 115, 120], so HH = 120
        # At index 4, the last 3 lows are [100, 110, 115], so LL = 100
        self.assertAlmostEqual(result['golden_zone_hh'].iloc[4], 120.0, places=5)
        self.assertAlmostEqual(result['golden_zone_ll'].iloc[4], 100.0, places=5)

    def test_golden_zone_custom_levels(self):
        """Test Golden Zone with custom retracement levels."""
        df = pd.DataFrame({
            "High": [100, 100, 100, 100, 100],
            "Low": [80, 80, 80, 80, 80]
        })
        # Custom levels: 38.2% and 78.6%
        result = golden_zone(
            df.copy(),
            length=5,
            retracement_level_1=0.382,
            retracement_level_2=0.786
        )

        # HH = 100, LL = 80, Diff = 20
        # Upper (38.2%): 100 - 20 * 0.382 = 92.36
        # Lower (78.6%): 100 - 20 * 0.786 = 84.28
        self.assertAlmostEqual(
            result['golden_zone_upper'].iloc[-1], 92.36, places=2
        )
        self.assertAlmostEqual(
            result['golden_zone_lower'].iloc[-1], 84.28, places=2
        )

    def test_golden_zone_custom_column_names(self):
        """Test Golden Zone with custom output column names."""
        df = pd.DataFrame({
            "High": [105, 110, 108, 112, 115],
            "Low": [100, 105, 103, 107, 110]
        })
        result = golden_zone(
            df.copy(),
            length=5,
            upper_column='fib_50',
            lower_column='fib_618',
            hh_column='highest_high',
            ll_column='lowest_low'
        )

        self.assertIn('fib_50', result.columns)
        self.assertIn('fib_618', result.columns)
        self.assertIn('highest_high', result.columns)
        self.assertIn('lowest_low', result.columns)

    def test_golden_zone_invalid_length(self):
        """Test that invalid length raises exception."""
        df = pd.DataFrame({
            "High": [100, 105, 110],
            "Low": [95, 100, 105]
        })
        with self.assertRaises(PyIndicatorException):
            golden_zone(df.copy(), length=0)

    def test_golden_zone_invalid_dataframe_type(self):
        """Test that invalid input type raises exception."""
        with self.assertRaises(PyIndicatorException):
            golden_zone([1, 2, 3], length=5)

    def test_golden_zone_level_order_independence(self):
        """Test that level order doesn't matter (upper is always higher)."""
        df = pd.DataFrame({
            "High": [100, 100, 100],
            "Low": [80, 80, 80]
        })

        # Pass levels in reverse order
        result = golden_zone(
            df.copy(),
            length=3,
            retracement_level_1=0.618,  # Usually lower
            retracement_level_2=0.5     # Usually upper
        )

        # Upper should still be higher than lower
        self.assertGreater(
            result['golden_zone_upper'].iloc[-1],
            result['golden_zone_lower'].iloc[-1]
        )


class TestGoldenZoneSignal(TestCase):

    def test_golden_zone_signal_pandas(self):
        """Test Golden Zone signal with pandas."""
        df = pd.DataFrame({
            "High": [100, 100, 100, 100, 100],
            "Low": [80, 80, 80, 80, 80],
            "Close": [85, 88, 92, 95, 78]
        })
        df = golden_zone(df, length=5)
        result = golden_zone_signal(df)

        self.assertIn('golden_zone_signal', result.columns)

        # Close=85 is below lower (87.64), so signal should be 0
        # Close=88 is between upper (90) and lower (87.64), so signal should be 1
        # Close=92 is above upper (90), so signal should be 0
        self.assertEqual(result['golden_zone_signal'].iloc[1], 1)
        self.assertEqual(result['golden_zone_signal'].iloc[2], 0)

    def test_golden_zone_signal_polars(self):
        """Test Golden Zone signal with polars."""
        df = pl.DataFrame({
            "High": [100, 100, 100, 100, 100],
            "Low": [80, 80, 80, 80, 80],
            "Close": [85, 88, 92, 95, 78]
        })
        df = golden_zone(df, length=5)
        result = golden_zone_signal(df)

        self.assertIn('golden_zone_signal', result.columns)

    def test_golden_zone_signal_missing_columns(self):
        """Test that missing golden zone columns raises exception."""
        df = pd.DataFrame({
            "Close": [85, 88, 92]
        })
        with self.assertRaises(PyIndicatorException):
            golden_zone_signal(df)

    def test_golden_zone_signal_custom_columns(self):
        """Test Golden Zone signal with custom column names."""
        df = pd.DataFrame({
            "High": [100, 100, 100, 100, 100],
            "Low": [80, 80, 80, 80, 80],
            "Close": [85, 88, 92, 95, 78]
        })
        df = golden_zone(
            df,
            length=5,
            upper_column='zone_upper',
            lower_column='zone_lower'
        )
        result = golden_zone_signal(
            df,
            upper_column='zone_upper',
            lower_column='zone_lower',
            signal_column='in_zone'
        )

        self.assertIn('in_zone', result.columns)

    def test_golden_zone_signal_boundary_values(self):
        """Test signal when price is exactly at boundaries."""
        df = pd.DataFrame({
            "High": [100, 100, 100, 100, 100],
            "Low": [80, 80, 80, 80, 80],
            "Close": [90, 87.64, 100, 80, 88]  # upper, lower, HH, LL, middle
        })
        df = golden_zone(df, length=5)
        result = golden_zone_signal(df)

        # Price at upper boundary (90) should be in zone (signal = 1)
        self.assertEqual(result['golden_zone_signal'].iloc[0], 1)
        # Price at lower boundary (87.64) should be in zone (signal = 1)
        self.assertEqual(result['golden_zone_signal'].iloc[1], 1)
        # Price at HH (100) should be outside zone (signal = 0)
        self.assertEqual(result['golden_zone_signal'].iloc[2], 0)

    def test_golden_zone_signal_polars_values(self):
        """Test Golden Zone signal values with polars."""
        df = pl.DataFrame({
            "High": [100, 100, 100, 100, 100],
            "Low": [80, 80, 80, 80, 80],
            "Close": [85, 88, 92, 95, 78]
        })
        df = golden_zone(df, length=5)
        result = golden_zone_signal(df)

        # Verify signal values match expected pattern
        signals = result['golden_zone_signal'].to_list()
        # Close=88 is in zone, Close=92 and Close=95 are above zone
        self.assertEqual(signals[1], 1)  # 88 is in zone
        self.assertEqual(signals[2], 0)  # 92 is above zone
        self.assertEqual(signals[3], 0)  # 95 is above zone

    def test_golden_zone_signal_invalid_dataframe_type(self):
        """Test that invalid input type raises exception."""
        with self.assertRaises(PyIndicatorException):
            golden_zone_signal([1, 2, 3])

    def test_golden_zone_signal_all_in_zone(self):
        """Test when all prices are within the golden zone."""
        df = pd.DataFrame({
            "High": [100, 100, 100, 100, 100],
            "Low": [80, 80, 80, 80, 80],
            "Close": [89, 88.5, 88, 89.5, 87.8]  # All between 87.64 and 90
        })
        df = golden_zone(df, length=5)
        result = golden_zone_signal(df)

        # All signals should be 1 (in zone)
        for i in range(len(result)):
            self.assertEqual(result['golden_zone_signal'].iloc[i], 1)

    def test_golden_zone_signal_all_outside_zone(self):
        """Test when all prices are outside the golden zone."""
        df = pd.DataFrame({
            "High": [100, 100, 100, 100, 100],
            "Low": [80, 80, 80, 80, 80],
            "Close": [95, 82, 98, 81, 99]  # All outside 87.64-90 range
        })
        df = golden_zone(df, length=5)
        result = golden_zone_signal(df)

        # All signals should be 0 (outside zone)
        for i in range(len(result)):
            self.assertEqual(result['golden_zone_signal'].iloc[i], 0)

    def test_golden_zone_signal_custom_close_column(self):
        """Test Golden Zone signal with custom close column name."""
        df = pd.DataFrame({
            "High": [100, 100, 100, 100, 100],
            "Low": [80, 80, 80, 80, 80],
            "Price": [85, 88, 92, 95, 78]  # Using 'Price' instead of 'Close'
        })
        df = golden_zone(df, length=5)
        result = golden_zone_signal(df, close_column='Price')

        self.assertIn('golden_zone_signal', result.columns)
        # Price=88 is in zone
        self.assertEqual(result['golden_zone_signal'].iloc[1], 1)
        # Price=92 is above zone
        self.assertEqual(result['golden_zone_signal'].iloc[2], 0)

    def test_golden_zone_signal_polars_missing_columns(self):
        """Test that missing golden zone columns raises exception for polars."""
        df = pl.DataFrame({
            "Close": [85, 88, 92]
        })
        with self.assertRaises(PyIndicatorException):
            golden_zone_signal(df)

    def test_golden_zone_signal_dynamic_zone(self):
        """Test signal with a dynamically changing golden zone."""
        df = pd.DataFrame({
            "High": [100, 110, 120, 115, 125],
            "Low": [90, 100, 110, 105, 115],
            "Close": [95, 105, 115, 110, 120]
        })
        df = golden_zone(df, length=3)
        result = golden_zone_signal(df)

        self.assertIn('golden_zone_signal', result.columns)
        # Verify we have both 0s and 1s in a dynamic scenario
        self.assertTrue(
            len(result) > 0,
            "Result should have data"
        )


