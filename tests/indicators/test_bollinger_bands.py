import numpy as np
from unittest import TestCase
import pandas as pd
import polars as pl
from pyindicators import bollinger_bands, bollinger_width, bollinger_overshoot


class TestBollingerBands(TestCase):

    def test_bollinger_bands_pandas(self):
        """Test basic Bollinger Bands calculation with pandas."""
        df = pd.DataFrame({
            "Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })
        result = bollinger_bands(df.copy(), period=10, std_dev=2)

        self.assertIn('bollinger_middle', result.columns)
        self.assertIn('bollinger_upper', result.columns)
        self.assertIn('bollinger_lower', result.columns)

        # First 9 values should be NaN (period=10)
        self.assertTrue(pd.isna(result['bollinger_middle'].iloc[8]))
        self.assertFalse(pd.isna(result['bollinger_middle'].iloc[9]))

        # Upper band should be greater than middle, middle greater than lower
        valid_idx = 10
        self.assertGreater(
            result['bollinger_upper'].iloc[valid_idx],
            result['bollinger_middle'].iloc[valid_idx]
        )
        self.assertGreater(
            result['bollinger_middle'].iloc[valid_idx],
            result['bollinger_lower'].iloc[valid_idx]
        )

    def test_bollinger_bands_polars(self):
        """Test basic Bollinger Bands calculation with polars."""
        df = pl.DataFrame({
            "Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })
        result = bollinger_bands(df, period=10, std_dev=2)

        self.assertIn('bollinger_middle', result.columns)
        self.assertIn('bollinger_upper', result.columns)
        self.assertIn('bollinger_lower', result.columns)


class TestBollingerWidth(TestCase):

    def test_bollinger_width_pandas(self):
        """Test Bollinger Width calculation with pandas."""
        df = pd.DataFrame({
            "Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })
        result = bollinger_width(df.copy(), period=10, std_dev=2)

        self.assertIn('Bollinger_Width', result.columns)
        # Width should be positive
        valid_width = result['Bollinger_Width'].dropna()
        self.assertTrue(all(valid_width > 0))

    def test_bollinger_width_polars(self):
        """Test Bollinger Width calculation with polars."""
        df = pl.DataFrame({
            "Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                      111, 113, 112, 114, 116, 115, 117, 119, 118, 120]
        })
        result = bollinger_width(df, period=10, std_dev=2)

        self.assertIn('Bollinger_Width', result.columns)


class TestBollingerOvershoot(TestCase):

    def test_overshoot_within_bands_pandas(self):
        """Test that overshoot is 0 when price is within bands."""
        # Stable prices with minimal volatility - price should stay within bands
        df = pd.DataFrame({
            "Close": [100.0] * 30
        })
        result = bollinger_overshoot(df.copy(), period=10, std_dev=2)

        self.assertIn('bollinger_overshoot', result.columns)
        # All valid values should be 0 (price exactly at middle band)
        valid_overshoot = result['bollinger_overshoot'].iloc[10:]
        self.assertTrue(all(valid_overshoot == 0))

    def test_overshoot_above_upper_band_pandas(self):
        """Test positive overshoot when price exceeds upper band."""
        # Start with stable prices, then spike up
        prices = [100.0] * 20 + [100.0, 100.0, 100.0, 100.0, 100.0, 150.0]
        df = pd.DataFrame({"Close": prices})

        result = bollinger_overshoot(df.copy(), period=10, std_dev=2)

        # The last value should have a positive overshoot (price spiked above band)
        last_overshoot = result['bollinger_overshoot'].iloc[-1]
        self.assertGreater(last_overshoot, 0)

    def test_overshoot_below_lower_band_pandas(self):
        """Test negative overshoot when price drops below lower band."""
        # Start with stable prices, then drop sharply
        prices = [100.0] * 20 + [100.0, 100.0, 100.0, 100.0, 100.0, 50.0]
        df = pd.DataFrame({"Close": prices})

        result = bollinger_overshoot(df.copy(), period=10, std_dev=2)

        # The last value should have a negative overshoot (price dropped below band)
        last_overshoot = result['bollinger_overshoot'].iloc[-1]
        self.assertLess(last_overshoot, 0)

    def test_overshoot_calculation_accuracy_pandas(self):
        """Test that overshoot percentage is calculated correctly."""
        # Create a scenario with stable prices followed by a significant spike
        # The spike should create a measurable overshoot
        prices = [100.0] * 25 + [130.0]  # 30% spike after stable period
        df = pd.DataFrame({"Close": prices})

        result = bollinger_overshoot(df.copy(), period=10, std_dev=2)

        # Also calculate bands to verify manually
        bands = bollinger_bands(df.copy(), period=10, std_dev=2)

        last_price = bands['Close'].iloc[-1]
        last_upper = bands['bollinger_upper'].iloc[-1]
        last_middle = bands['bollinger_middle'].iloc[-1]
        half_band_width = last_upper - last_middle

        # Manual calculation
        expected_overshoot = ((last_price - last_upper) / half_band_width) * 100

        actual_overshoot = result['bollinger_overshoot'].iloc[-1]

        # Verify the calculation matches
        self.assertAlmostEqual(actual_overshoot, expected_overshoot, places=5)
        # Should be positive (above upper band)
        self.assertGreater(actual_overshoot, 0)

    def test_overshoot_with_datetime_index_pandas(self):
        """Test that overshoot works correctly with DatetimeIndex."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        prices = [100.0] * 20 + [100.0, 100.0, 100.0, 100.0, 100.0, 150.0, 100.0, 100.0, 50.0, 100.0]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = bollinger_overshoot(df.copy(), period=10, std_dev=2)

        self.assertIn('bollinger_overshoot', result.columns)
        # Should have both positive and negative overshoots
        self.assertTrue(any(result['bollinger_overshoot'] > 0))
        self.assertTrue(any(result['bollinger_overshoot'] < 0))

    def test_overshoot_polars(self):
        """Test overshoot calculation with polars DataFrame."""
        # Start with stable prices, then spike up, then drop
        prices = [100.0] * 20 + [100.0, 100.0, 150.0, 100.0, 50.0]
        df = pl.DataFrame({"Close": prices})

        result = bollinger_overshoot(df, period=10, std_dev=2)

        self.assertIn('bollinger_overshoot', result.columns)

        overshoot_values = result['bollinger_overshoot'].to_list()
        # Should have both positive and negative overshoots
        self.assertTrue(any(v > 0 for v in overshoot_values if v is not None))
        self.assertTrue(any(v < 0 for v in overshoot_values if v is not None))

    def test_overshoot_custom_column_names(self):
        """Test overshoot with custom source and result column names."""
        df = pd.DataFrame({
            "Price": [100.0] * 20 + [150.0]
        })

        result = bollinger_overshoot(
            df.copy(),
            source_column='Price',
            period=10,
            std_dev=2,
            result_column='my_overshoot'
        )

        self.assertIn('my_overshoot', result.columns)
        self.assertNotIn('bollinger_overshoot', result.columns)

    def test_overshoot_different_std_dev(self):
        """Test that different std_dev values affect overshoot."""
        prices = [100.0] * 20 + [120.0]
        df = pd.DataFrame({"Close": prices})

        # With std_dev=1, bands are tighter, so overshoot should be larger
        result_std1 = bollinger_overshoot(df.copy(), period=10, std_dev=1)
        # With std_dev=3, bands are wider, so overshoot should be smaller
        result_std3 = bollinger_overshoot(df.copy(), period=10, std_dev=3)

        overshoot_std1 = result_std1['bollinger_overshoot'].iloc[-1]
        overshoot_std3 = result_std3['bollinger_overshoot'].iloc[-1]

        # Overshoot with tighter bands should be larger (or equal if within bands)
        self.assertGreaterEqual(overshoot_std1, overshoot_std3)

    def test_overshoot_no_temp_columns_remain(self):
        """Test that temporary columns are removed from result."""
        df = pd.DataFrame({
            "Close": [100.0] * 25
        })

        result = bollinger_overshoot(df.copy(), period=10, std_dev=2)

        # Temporary columns should not be in result
        self.assertNotIn('BB_middle_temp', result.columns)
        self.assertNotIn('BB_upper_temp', result.columns)
        self.assertNotIn('BB_lower_temp', result.columns)

    def test_overshoot_preserves_original_columns(self):
        """Test that original DataFrame columns are preserved."""
        df = pd.DataFrame({
            "Close": [100.0] * 25,
            "Volume": [1000] * 25,
            "Open": [99.0] * 25
        })

        result = bollinger_overshoot(df.copy(), period=10, std_dev=2)

        self.assertIn('Close', result.columns)
        self.assertIn('Volume', result.columns)
        self.assertIn('Open', result.columns)
        self.assertIn('bollinger_overshoot', result.columns)

    def test_overshoot_real_world_scenario(self):
        """Test overshoot with a realistic price movement scenario."""
        # Simulate: trending up, then sharp spike (like silver 40% overshoot)
        np.random.seed(42)
        base_trend = np.linspace(100, 120, 50)
        noise = np.random.normal(0, 1, 50)
        prices = base_trend + noise

        # Add a sharp spike at the end
        prices[-1] = prices[-2] + 15  # Sharp spike

        df = pd.DataFrame({"Close": prices})
        result = bollinger_overshoot(df.copy(), period=20, std_dev=2)

        # Should detect significant positive overshoot at the spike
        last_overshoot = result['bollinger_overshoot'].iloc[-1]
        self.assertGreater(last_overshoot, 0)
        print(f"Real-world scenario overshoot: {last_overshoot:.2f}%")

