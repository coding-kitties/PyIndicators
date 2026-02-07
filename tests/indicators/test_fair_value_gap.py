from unittest import TestCase
import pandas as pd
import polars as pl
from pyindicators import fair_value_gap, fvg_signal, fvg_filled
from pyindicators.exceptions import PyIndicatorException


class TestFairValueGap(TestCase):

    def test_fvg_pandas(self):
        """Test Fair Value Gap detection with pandas."""
        df = pd.DataFrame({
            "High": [100, 102, 106, 108, 105],
            "Low": [98, 100, 104, 106, 103],
            "Close": [99, 101, 105, 107, 104]
        })
        result = fair_value_gap(df.copy())

        self.assertIn('bullish_fvg', result.columns)
        self.assertIn('bearish_fvg', result.columns)
        self.assertIn('bullish_fvg_top', result.columns)
        self.assertIn('bullish_fvg_bottom', result.columns)
        self.assertIn('bearish_fvg_top', result.columns)
        self.assertIn('bearish_fvg_bottom', result.columns)

    def test_fvg_polars(self):
        """Test Fair Value Gap detection with polars."""
        df = pl.DataFrame({
            "High": [100, 102, 106, 108, 105],
            "Low": [98, 100, 104, 106, 103],
            "Close": [99, 101, 105, 107, 104]
        })
        result = fair_value_gap(df)

        self.assertIn('bullish_fvg', result.columns)
        self.assertIn('bearish_fvg', result.columns)
        self.assertIn('bullish_fvg_top', result.columns)
        self.assertIn('bullish_fvg_bottom', result.columns)

    def test_bullish_fvg_detection(self):
        """Test bullish FVG is detected correctly."""
        # Create clear bullish FVG: bar 2's low > bar 0's high
        # AND close[1] > high[0] (LuxAlgo confirmation)
        df = pd.DataFrame({
            "High": [100, 105, 115, 120, 118],  # Bar 0 high = 100
            "Low": [95, 100, 102, 115, 113],    # Bar 2 low = 102 > 100 = bullish FVG
            "Close": [98, 101, 110, 117, 115]   # Close[1] = 101 > High[0] = 100
        })
        result = fair_value_gap(df.copy())

        # First two bars should have no FVG (not enough history)
        self.assertEqual(result['bullish_fvg'].iloc[0], 0)
        self.assertEqual(result['bullish_fvg'].iloc[1], 0)
        # Bar 2: Low(102) > High[0](100) = bullish FVG
        self.assertEqual(result['bullish_fvg'].iloc[2], 1)

    def test_bearish_fvg_detection(self):
        """Test bearish FVG is detected correctly."""
        # Create clear bearish FVG: bar 2's high < bar 0's low
        # AND close[1] < low[0] (LuxAlgo confirmation)
        df = pd.DataFrame({
            "High": [110, 105, 95, 92, 94],   # Bar 2 high = 95
            "Low": [100, 98, 90, 88, 89],     # Bar 0 low = 100, so 95 < 100 = bearish FVG
            "Close": [105, 99, 92, 90, 91]    # Close[1] = 99 < Low[0] = 100
        })
        result = fair_value_gap(df.copy())

        # Bar 2: High(95) < Low[0](100) = bearish FVG
        self.assertEqual(result['bearish_fvg'].iloc[2], 1)
        self.assertEqual(result['bullish_fvg'].iloc[2], 0)

    def test_no_fvg_when_overlapping(self):
        """Test no FVG when candles overlap."""
        # Overlapping candles - no gap
        df = pd.DataFrame({
            "High": [100, 102, 103, 104, 105],
            "Low": [98, 99, 100, 101, 102],
            "Close": [99, 101, 102, 103, 104]
        })
        result = fair_value_gap(df.copy())

        # No FVG should be detected when candles overlap
        for i in range(2, len(result)):
            self.assertEqual(result['bullish_fvg'].iloc[i], 0)
            self.assertEqual(result['bearish_fvg'].iloc[i], 0)

    def test_fvg_zone_boundaries_bullish(self):
        """Test bullish FVG zone boundaries are calculated correctly."""
        df = pd.DataFrame({
            "High": [100, 105, 115, 120, 118],
            "Low": [95, 100, 102, 115, 113],
            "Close": [98, 101, 110, 117, 115]  # Close[1] = 101 > High[0] = 100
        })
        result = fair_value_gap(df.copy())

        # At bar 2: bullish FVG between High[0]=100 and Low[2]=102
        self.assertAlmostEqual(result['bullish_fvg_top'].iloc[2], 102.0, places=5)
        self.assertAlmostEqual(result['bullish_fvg_bottom'].iloc[2], 100.0, places=5)

    def test_fvg_zone_boundaries_bearish(self):
        """Test bearish FVG zone boundaries are calculated correctly."""
        df = pd.DataFrame({
            "High": [110, 105, 95, 92, 94],
            "Low": [100, 98, 90, 88, 89],
            "Close": [105, 99, 92, 90, 91]  # Close[1] = 99 < Low[0] = 100
        })
        result = fair_value_gap(df.copy())

        # At bar 2: bearish FVG between Low[0]=100 and High[2]=95
        self.assertAlmostEqual(result['bearish_fvg_top'].iloc[2], 100.0, places=5)
        self.assertAlmostEqual(result['bearish_fvg_bottom'].iloc[2], 95.0, places=5)

    def test_fvg_custom_column_names(self):
        """Test FVG with custom output column names."""
        df = pd.DataFrame({
            "High": [100, 105, 115],
            "Low": [95, 100, 102],
            "Close": [98, 101, 110]  # Close[1] = 101 > High[0] = 100
        })
        result = fair_value_gap(
            df.copy(),
            bullish_fvg_column='bull_gap',
            bearish_fvg_column='bear_gap',
            bullish_fvg_top_column='bull_top',
            bullish_fvg_bottom_column='bull_bottom',
            bearish_fvg_top_column='bear_top',
            bearish_fvg_bottom_column='bear_bottom'
        )

        self.assertIn('bull_gap', result.columns)
        self.assertIn('bear_gap', result.columns)
        self.assertIn('bull_top', result.columns)
        self.assertIn('bull_bottom', result.columns)

    def test_fvg_invalid_dataframe_type(self):
        """Test that invalid input type raises exception."""
        with self.assertRaises(PyIndicatorException):
            fair_value_gap([1, 2, 3])

    def test_fvg_polars_values(self):
        """Test FVG values with polars DataFrame."""
        df = pl.DataFrame({
            "High": [100, 105, 115, 120, 118],
            "Low": [95, 100, 102, 115, 113],
            "Close": [98, 101, 110, 117, 115]  # Close[1] = 101 > High[0] = 100
        })
        result = fair_value_gap(df)

        bullish_fvg = result['bullish_fvg'].to_list()
        self.assertEqual(bullish_fvg[2], 1)  # Bullish FVG at bar 2

    def test_multiple_fvgs(self):
        """Test detection of multiple FVGs in data."""
        df = pd.DataFrame({
            # Two bullish FVGs
            "High": [100, 105, 115, 116, 130, 131],
            "Low": [95, 100, 102, 110, 118, 125],
            # Close[1]=101 > High[0]=100 for first FVG
            # Close[3]=116 > High[2]=115 for second FVG
            "Close": [98, 101, 110, 116, 125, 128]
        })
        result = fair_value_gap(df.copy())

        # Bar 2: Low(102) > High[0](100) AND Close[1](101) > High[0](100) = bullish
        self.assertEqual(result['bullish_fvg'].iloc[2], 1)
        # Bar 4: Low(118) > High[2](115) AND Close[3](116) > High[2](115) = bullish
        self.assertEqual(result['bullish_fvg'].iloc[4], 1)


class TestFVGSignal(TestCase):

    def test_fvg_signal_pandas(self):
        """Test FVG signal with pandas."""
        df = pd.DataFrame({
            "High": [100, 105, 115, 120, 118],
            "Low": [95, 100, 102, 115, 113],
            "Close": [98, 103, 110, 117, 101]  # Close at 101 is in bullish FVG zone
        })
        df = fair_value_gap(df)
        result = fvg_signal(df)

        self.assertIn('fvg_signal', result.columns)

    def test_fvg_signal_polars(self):
        """Test FVG signal with polars."""
        df = pl.DataFrame({
            "High": [100, 105, 115, 120, 118],
            "Low": [95, 100, 102, 115, 113],
            "Close": [98, 103, 110, 117, 101]
        })
        df = fair_value_gap(df)
        result = fvg_signal(df)

        self.assertIn('fvg_signal', result.columns)

    def test_fvg_signal_in_bullish_zone(self):
        """Test signal when price is in bullish FVG zone."""
        # Create a single bullish FVG at bar 2: Low(102) > High[0](100)
        # Bars 3 and 4 must not create new FVGs
        df = pd.DataFrame({
            "High": [100, 105, 115, 118, 103],  # Bar 3 Low(104) < High[1](105), no FVG
            "Low": [95, 100, 102, 104, 98],     # Bar 4 Low(98) < High[2](115), no FVG
            "Close": [98, 103, 110, 116, 101]   # 101 is between 100 and 102
        })
        df = fair_value_gap(df)
        result = fvg_signal(df)

        # Close at 101 is within bullish FVG zone (100-102)
        self.assertEqual(result['fvg_signal'].iloc[4], 1)

    def test_fvg_signal_in_bearish_zone(self):
        """Test signal when price is in bearish FVG zone."""
        df = pd.DataFrame({
            "High": [110, 105, 95, 92, 94],
            "Low": [100, 98, 90, 88, 89],
            "Close": [105, 100, 92, 90, 97]  # 97 is between 95 and 100
        })
        df = fair_value_gap(df)
        result = fvg_signal(df)

        # Close at 97 is within bearish FVG zone (95-100)
        self.assertEqual(result['fvg_signal'].iloc[4], -1)

    def test_fvg_signal_outside_zone(self):
        """Test signal when price is outside FVG zones."""
        # Create a single bullish FVG at bar 2: Low(102) > High[0](100)
        df = pd.DataFrame({
            "High": [100, 105, 115, 118, 103],
            "Low": [95, 100, 102, 104, 98],
            "Close": [98, 103, 110, 116, 115]  # 115 is above bullish zone (100-102)
        })
        df = fair_value_gap(df)
        result = fvg_signal(df)

        # Close at 115 is above the bullish FVG zone (100-102)
        self.assertEqual(result['fvg_signal'].iloc[4], 0)

    def test_fvg_signal_missing_columns(self):
        """Test that missing FVG columns raises exception."""
        df = pd.DataFrame({
            "Close": [100, 105, 110]
        })
        with self.assertRaises(PyIndicatorException):
            fvg_signal(df)

    def test_fvg_signal_custom_columns(self):
        """Test FVG signal with custom column names."""
        df = pd.DataFrame({
            "High": [100, 105, 115],
            "Low": [95, 100, 102],
            "Close": [98, 103, 101]
        })
        df = fair_value_gap(
            df,
            bullish_fvg_top_column='bull_top',
            bullish_fvg_bottom_column='bull_bottom',
            bearish_fvg_top_column='bear_top',
            bearish_fvg_bottom_column='bear_bottom'
        )
        result = fvg_signal(
            df,
            bullish_fvg_top_column='bull_top',
            bullish_fvg_bottom_column='bull_bottom',
            bearish_fvg_top_column='bear_top',
            bearish_fvg_bottom_column='bear_bottom',
            signal_column='my_signal'
        )

        self.assertIn('my_signal', result.columns)

    def test_fvg_signal_invalid_dataframe_type(self):
        """Test that invalid input type raises exception."""
        with self.assertRaises(PyIndicatorException):
            fvg_signal([1, 2, 3])

    def test_fvg_signal_polars_missing_columns(self):
        """Test that missing FVG columns raises exception for polars."""
        df = pl.DataFrame({
            "Close": [100, 105, 110]
        })
        with self.assertRaises(PyIndicatorException):
            fvg_signal(df)

    def test_fvg_signal_custom_close_column(self):
        """Test FVG signal with custom close column name."""
        df = pd.DataFrame({
            "High": [100, 105, 115, 120, 118],
            "Low": [95, 100, 102, 115, 113],
            "Price": [98, 103, 110, 117, 101]
        })
        df = fair_value_gap(df, close_column='Price')
        result = fvg_signal(df, close_column='Price')

        self.assertIn('fvg_signal', result.columns)


class TestFVGFilled(TestCase):

    def test_fvg_filled_pandas(self):
        """Test FVG filled detection with pandas."""
        df = pd.DataFrame({
            "High": [100, 105, 115, 120, 118, 115],
            "Low": [95, 100, 102, 115, 113, 99],  # Last low fills bullish FVG
            "Close": [98, 101, 110, 117, 115, 100]  # Close[1] = 101 > High[0] = 100
        })
        df = fair_value_gap(df)
        result = fvg_filled(df)

        self.assertIn('bullish_fvg_filled', result.columns)
        self.assertIn('bearish_fvg_filled', result.columns)

    def test_fvg_filled_polars(self):
        """Test FVG filled detection with polars."""
        df = pl.DataFrame({
            "High": [100, 105, 115, 120, 118, 115],
            "Low": [95, 100, 102, 115, 113, 99],
            "Close": [98, 101, 110, 117, 115, 100]  # Close[1] = 101 > High[0] = 100
        })
        df = fair_value_gap(df)
        result = fvg_filled(df)

        self.assertIn('bullish_fvg_filled', result.columns)
        self.assertIn('bearish_fvg_filled', result.columns)

    def test_bullish_fvg_filled_detection(self):
        """Test bullish FVG filled is detected correctly."""
        df = pd.DataFrame({
            "High": [100, 105, 115, 120, 118, 115],
            "Low": [95, 100, 102, 115, 113, 99],  # Low at 99 < 100 (bottom of FVG)
            "Close": [98, 101, 110, 117, 115, 100]  # Close[1] = 101 > High[0] = 100
        })
        df = fair_value_gap(df)
        result = fvg_filled(df)

        # Bullish FVG (100-102) is filled when low reaches 100 or below
        # At bar 5, low=99 which fills the bullish FVG
        self.assertEqual(result['bullish_fvg_filled'].iloc[5], 1)

    def test_bearish_fvg_filled_detection(self):
        """Test bearish FVG filled is detected correctly."""
        df = pd.DataFrame({
            "High": [110, 105, 95, 92, 94, 101],  # High at 101 > 100 (top of FVG)
            "Low": [100, 98, 90, 88, 89, 96],
            "Close": [105, 99, 92, 90, 91, 98]  # Close[1] = 99 < Low[0] = 100
        })
        df = fair_value_gap(df)
        result = fvg_filled(df)

        # Bearish FVG (95-100) is filled when high reaches 100 or above
        # At bar 5, high=101 which fills the bearish FVG
        self.assertEqual(result['bearish_fvg_filled'].iloc[5], 1)

    def test_fvg_not_filled(self):
        """Test FVG not filled when price doesn't reach zone."""
        df = pd.DataFrame({
            "High": [100, 105, 115, 120, 118, 116],
            "Low": [95, 100, 102, 115, 113, 110],  # Low doesn't reach 100
            "Close": [98, 101, 110, 117, 115, 112]  # Close[1] = 101 > High[0] = 100
        })
        df = fair_value_gap(df)
        result = fvg_filled(df)

        # Bullish FVG at 100-102 not filled since low stays above 100
        self.assertEqual(result['bullish_fvg_filled'].iloc[5], 0)

    def test_fvg_filled_custom_columns(self):
        """Test FVG filled with custom column names."""
        df = pd.DataFrame({
            "High": [100, 105, 115, 120, 118, 115],
            "Low": [95, 100, 102, 115, 113, 99],
            "Close": [98, 101, 110, 117, 115, 100]  # Close[1] = 101 > High[0] = 100
        })
        df = fair_value_gap(df)
        result = fvg_filled(
            df,
            bullish_filled_column='bull_filled',
            bearish_filled_column='bear_filled'
        )

        self.assertIn('bull_filled', result.columns)
        self.assertIn('bear_filled', result.columns)

    def test_fvg_filled_invalid_dataframe_type(self):
        """Test that invalid input type raises exception."""
        with self.assertRaises(PyIndicatorException):
            fvg_filled([1, 2, 3])


class TestFVGEdgeCases(TestCase):

    def test_fvg_minimum_data(self):
        """Test FVG with minimum required data (3 bars)."""
        df = pd.DataFrame({
            "High": [100, 105, 115],
            "Low": [95, 100, 102],
            "Close": [98, 101, 110]  # Close[1] = 101 > High[0] = 100
        })
        result = fair_value_gap(df.copy())

        # Should work with just 3 bars
        self.assertEqual(len(result), 3)
        self.assertEqual(result['bullish_fvg'].iloc[2], 1)

    def test_fvg_first_two_bars_nan(self):
        """Test that first two bars have no FVG (not enough history)."""
        df = pd.DataFrame({
            "High": [100, 102, 106, 108, 105],
            "Low": [98, 100, 104, 106, 103],
            "Close": [99, 101, 105, 107, 104]
        })
        result = fair_value_gap(df.copy())

        # First two bars should have no FVG
        self.assertEqual(result['bullish_fvg'].iloc[0], 0)
        self.assertEqual(result['bullish_fvg'].iloc[1], 0)
        self.assertEqual(result['bearish_fvg'].iloc[0], 0)
        self.assertEqual(result['bearish_fvg'].iloc[1], 0)

    def test_fvg_exact_touch_no_gap(self):
        """Test when price exactly touches (no actual gap)."""
        df = pd.DataFrame({
            "High": [100, 105, 110],
            "Low": [95, 100, 100],  # Low exactly equals previous High[0] - no gap
            "Close": [98, 101, 105]
        })
        result = fair_value_gap(df.copy())

        # No bullish FVG when low equals high (no gap)
        self.assertEqual(result['bullish_fvg'].iloc[2], 0)

    def test_fvg_large_gap(self):
        """Test FVG with a large price gap."""
        df = pd.DataFrame({
            "High": [100, 105, 200, 210, 205],
            "Low": [95, 100, 150, 195, 190],  # Large gap
            "Close": [98, 110, 180, 200, 195]  # Close[1] = 110 > High[0] = 100
        })
        result = fair_value_gap(df.copy())

        # Large bullish gap: Low(150) > High[0](100)
        self.assertEqual(result['bullish_fvg'].iloc[2], 1)
        self.assertAlmostEqual(result['bullish_fvg_top'].iloc[2], 150.0)
        self.assertAlmostEqual(result['bullish_fvg_bottom'].iloc[2], 100.0)

    def test_fvg_signal_boundary_values(self):
        """Test FVG signal at exact boundary values."""
        # Create a single bullish FVG at bar 2: Low(102) > High[0](100)
        df = pd.DataFrame({
            "High": [100, 105, 115, 118, 103],
            "Low": [95, 100, 102, 104, 98],
            "Close": [98, 103, 110, 116, 100]  # Close exactly at bottom
        })
        df = fair_value_gap(df)
        result = fvg_signal(df)

        # Close at exactly 100 (bottom boundary) should be in zone
        self.assertEqual(result['fvg_signal'].iloc[4], 1)

    def test_fvg_signal_boundary_top(self):
        """Test FVG signal at exact top boundary."""
        # Create a single bullish FVG at bar 2: Low(102) > High[0](100)
        df = pd.DataFrame({
            "High": [100, 105, 115, 118, 103],
            "Low": [95, 100, 102, 104, 98],
            "Close": [98, 103, 110, 116, 102]  # Close exactly at top
        })
        df = fair_value_gap(df)
        result = fvg_signal(df)

        # Close at exactly 102 (top boundary) should be in zone
        self.assertEqual(result['fvg_signal'].iloc[4], 1)

