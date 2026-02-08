import unittest
import pandas as pd
import numpy as np

from pyindicators import (
    market_structure_break,
    market_structure_ob,
    msb_signal,
    ob_quality_signal,
    get_market_structure_stats
)


def create_sample_ohlcv_data():
    """Create sample OHLCV data with clear structure breaks."""
    np.random.seed(42)
    n = 200

    # Create data with clear trends and reversals
    close = np.zeros(n)
    close[0] = 100

    # Uptrend
    for i in range(1, 50):
        close[i] = close[i-1] + np.random.uniform(0.5, 2.0)

    # Downtrend
    for i in range(50, 100):
        close[i] = close[i-1] - np.random.uniform(0.5, 2.0)

    # Uptrend
    for i in range(100, 150):
        close[i] = close[i-1] + np.random.uniform(0.5, 2.0)

    # Downtrend
    for i in range(150, 200):
        close[i] = close[i-1] - np.random.uniform(0.3, 1.5)

    high = close + np.random.uniform(0.5, 2.0, n)
    low = close - np.random.uniform(0.5, 2.0, n)
    open_price = close + np.random.uniform(-1.0, 1.0, n)
    volume = np.random.uniform(1000, 10000, n)

    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })


class TestMarketStructureBreak(unittest.TestCase):
    """Tests for market_structure_break function."""

    def setUp(self):
        """Set up test data."""
        self.sample_ohlcv_data = create_sample_ohlcv_data()

    def test_basic_msb_detection(self):
        """Test basic MSB detection."""
        result = market_structure_break(self.sample_ohlcv_data)

        self.assertIn('msb_bullish', result.columns)
        self.assertIn('msb_bearish', result.columns)
        self.assertIn('last_pivot_high', result.columns)
        self.assertIn('last_pivot_low', result.columns)
        self.assertIn('momentum_z', result.columns)

    def test_msb_values_are_binary(self):
        """Test that MSB signals are binary (0 or 1)."""
        result = market_structure_break(self.sample_ohlcv_data)

        self.assertTrue(set(result['msb_bullish'].unique()).issubset({0, 1}))
        self.assertTrue(set(result['msb_bearish'].unique()).issubset({0, 1}))

    def test_msb_with_custom_columns(self):
        """Test MSB with custom column names."""
        result = market_structure_break(
            self.sample_ohlcv_data,
            msb_bullish_column='bull_msb',
            msb_bearish_column='bear_msb'
        )

        self.assertIn('bull_msb', result.columns)
        self.assertIn('bear_msb', result.columns)

    def test_msb_pivot_length_parameter(self):
        """Test MSB with different pivot lengths."""
        result_short = market_structure_break(
            self.sample_ohlcv_data.copy(), pivot_length=3
        )
        result_long = market_structure_break(
            self.sample_ohlcv_data.copy(), pivot_length=14
        )

        # Different pivot lengths should give different results
        short_msb_count = result_short['msb_bullish'].sum()
        long_msb_count = result_long['msb_bullish'].sum()

        # With shorter pivot length, we expect more signals
        self.assertGreaterEqual(short_msb_count, 0)
        self.assertGreaterEqual(long_msb_count, 0)

    def test_msb_zscore_threshold(self):
        """Test MSB with different z-score thresholds."""
        result_low = market_structure_break(
            self.sample_ohlcv_data.copy(), momentum_zscore_threshold=0.1
        )
        result_high = market_structure_break(
            self.sample_ohlcv_data.copy(), momentum_zscore_threshold=2.0
        )

        # Higher threshold should result in fewer signals
        low_count = result_low['msb_bullish'].sum() + \
            result_low['msb_bearish'].sum()
        high_count = result_high['msb_bullish'].sum() + \
            result_high['msb_bearish'].sum()

        self.assertGreaterEqual(low_count, high_count)


class TestMarketStructureOB(unittest.TestCase):
    """Tests for market_structure_ob function."""

    def setUp(self):
        """Set up test data."""
        self.sample_ohlcv_data = create_sample_ohlcv_data()

    def test_basic_ob_detection(self):
        """Test basic order block detection with quality scoring."""
        result = market_structure_ob(self.sample_ohlcv_data)

        self.assertIn('msb_bullish', result.columns)
        self.assertIn('msb_bearish', result.columns)
        self.assertIn('ob_bullish', result.columns)
        self.assertIn('ob_bearish', result.columns)
        self.assertIn('ob_top', result.columns)
        self.assertIn('ob_bottom', result.columns)
        self.assertIn('ob_quality', result.columns)
        self.assertIn('ob_is_hpz', result.columns)
        self.assertIn('ob_mitigated', result.columns)

    def test_ob_quality_range(self):
        """Test that quality scores are in valid range (0-100)."""
        result = market_structure_ob(self.sample_ohlcv_data)

        valid_quality = result['ob_quality'].dropna()
        if len(valid_quality) > 0:
            self.assertGreaterEqual(valid_quality.min(), 0)
            self.assertLessEqual(valid_quality.max(), 100)

    def test_hpz_flag_consistency(self):
        """Test that HPZ flag is consistent with quality > 80."""
        result = market_structure_ob(self.sample_ohlcv_data)

        # Where we have quality scores
        has_quality = ~result['ob_quality'].isna()
        if has_quality.any():
            high_quality = result.loc[has_quality, 'ob_quality'] > 80
            hpz_flag = result.loc[has_quality, 'ob_is_hpz']
            # HPZ should match high quality
            self.assertTrue((high_quality == hpz_flag).all())

    def test_ob_top_bottom_relationship(self):
        """Test that OB top is always >= OB bottom."""
        result = market_structure_ob(self.sample_ohlcv_data)

        # Where we have both top and bottom
        has_both = ~result['ob_top'].isna() & ~result['ob_bottom'].isna()
        if has_both.any():
            tops = result.loc[has_both, 'ob_top']
            bottoms = result.loc[has_both, 'ob_bottom']
            self.assertTrue((tops >= bottoms).all())

    def test_ob_with_custom_max_active(self):
        """Test order block with custom max active parameter."""
        result = market_structure_ob(
            self.sample_ohlcv_data, max_active_obs=5
        )

        self.assertIn('ob_bullish', result.columns)
        self.assertIn('ob_bearish', result.columns)


class TestMSBSignal(unittest.TestCase):
    """Tests for msb_signal function."""

    def setUp(self):
        """Set up test data."""
        self.sample_ohlcv_data = create_sample_ohlcv_data()

    def test_msb_signal_values(self):
        """Test that MSB signal values are correct."""
        df = market_structure_break(self.sample_ohlcv_data)
        result = msb_signal(df)

        self.assertIn('msb_signal', result.columns)
        self.assertTrue(set(result['msb_signal'].unique()).issubset({-1, 0, 1}))

    def test_msb_signal_matches_components(self):
        """Test that signal matches bullish/bearish columns."""
        df = market_structure_break(self.sample_ohlcv_data)
        result = msb_signal(df)

        # Where bullish MSB = 1, signal should be 1
        bullish_mask = result['msb_bullish'] == 1
        if bullish_mask.any():
            self.assertTrue(
                (result.loc[bullish_mask, 'msb_signal'] == 1).all()
            )

        # Where bearish MSB = 1, signal should be -1
        bearish_mask = result['msb_bearish'] == 1
        if bearish_mask.any():
            self.assertTrue(
                (result.loc[bearish_mask, 'msb_signal'] == -1).all()
            )

    def test_msb_signal_custom_column(self):
        """Test MSB signal with custom column name."""
        df = market_structure_break(self.sample_ohlcv_data)
        result = msb_signal(df, signal_column='my_signal')

        self.assertIn('my_signal', result.columns)


class TestOBQualitySignal(unittest.TestCase):
    """Tests for ob_quality_signal function."""

    def setUp(self):
        """Set up test data."""
        self.sample_ohlcv_data = create_sample_ohlcv_data()

    def test_ob_quality_signal_basic(self):
        """Test basic OB quality signal generation."""
        df = market_structure_ob(self.sample_ohlcv_data)
        result = ob_quality_signal(df)

        self.assertIn('ob_quality_signal', result.columns)
        self.assertTrue(
            set(result['ob_quality_signal'].unique()).issubset({-1, 0, 1})
        )

    def test_ob_quality_signal_min_quality(self):
        """Test OB quality signal with min_quality filter."""
        df = market_structure_ob(self.sample_ohlcv_data)

        result_low = ob_quality_signal(df.copy(), min_quality=30)
        result_high = ob_quality_signal(df.copy(), min_quality=90)

        # Higher min quality should result in fewer signals
        low_signals = (result_low['ob_quality_signal'] != 0).sum()
        high_signals = (result_high['ob_quality_signal'] != 0).sum()

        self.assertGreaterEqual(low_signals, high_signals)

    def test_ob_quality_signal_hpz_only(self):
        """Test OB quality signal with HPZ only filter."""
        df = market_structure_ob(self.sample_ohlcv_data)

        result_all = ob_quality_signal(df.copy(), hpz_only=False)
        result_hpz = ob_quality_signal(df.copy(), hpz_only=True)

        # HPZ only should have equal or fewer signals
        all_signals = (result_all['ob_quality_signal'] != 0).sum()
        hpz_signals = (result_hpz['ob_quality_signal'] != 0).sum()

        self.assertGreaterEqual(all_signals, hpz_signals)


class TestGetMarketStructureStats(unittest.TestCase):
    """Tests for get_market_structure_stats function."""

    def setUp(self):
        """Set up test data."""
        self.sample_ohlcv_data = create_sample_ohlcv_data()

    def test_stats_structure(self):
        """Test that stats return correct structure."""
        df = market_structure_ob(self.sample_ohlcv_data)
        stats = get_market_structure_stats(df)

        self.assertIn('total_obs', stats)
        self.assertIn('total_mitigated', stats)
        self.assertIn('reliability', stats)
        self.assertIn('hpz_count', stats)
        self.assertIn('bullish_msb_count', stats)
        self.assertIn('bearish_msb_count', stats)

    def test_stats_types(self):
        """Test that stats have correct types."""
        df = market_structure_ob(self.sample_ohlcv_data)
        stats = get_market_structure_stats(df)

        self.assertIsInstance(stats['total_obs'], int)
        self.assertIsInstance(stats['total_mitigated'], int)
        self.assertIsInstance(stats['reliability'], float)
        self.assertIsInstance(stats['hpz_count'], int)
        self.assertIsInstance(stats['bullish_msb_count'], int)
        self.assertIsInstance(stats['bearish_msb_count'], int)

    def test_reliability_range(self):
        """Test that reliability is in valid range."""
        df = market_structure_ob(self.sample_ohlcv_data)
        stats = get_market_structure_stats(df)

        self.assertGreaterEqual(stats['reliability'], 0)
        self.assertLessEqual(stats['reliability'], 100)

    def test_stats_non_negative(self):
        """Test that all counts are non-negative."""
        df = market_structure_ob(self.sample_ohlcv_data)
        stats = get_market_structure_stats(df)

        self.assertGreaterEqual(stats['total_obs'], 0)
        self.assertGreaterEqual(stats['total_mitigated'], 0)
        self.assertGreaterEqual(stats['hpz_count'], 0)
        self.assertGreaterEqual(stats['bullish_msb_count'], 0)
        self.assertGreaterEqual(stats['bearish_msb_count'], 0)


class TestPolarsSupport(unittest.TestCase):
    """Tests for Polars DataFrame support."""

    def setUp(self):
        """Set up test data."""
        self.sample_ohlcv_data = create_sample_ohlcv_data()

    def test_msb_with_polars(self):
        """Test MSB with Polars DataFrame."""
        import polars as pl
        pl_df = pl.from_pandas(self.sample_ohlcv_data)

        result = market_structure_break(pl_df)

        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn('msb_bullish', result.columns)
        self.assertIn('msb_bearish', result.columns)

    def test_market_structure_ob_with_polars(self):
        """Test market_structure_ob with Polars DataFrame."""
        import polars as pl
        pl_df = pl.from_pandas(self.sample_ohlcv_data)

        result = market_structure_ob(pl_df)

        self.assertIsInstance(result, pl.DataFrame)
        self.assertIn('ob_quality', result.columns)
        self.assertIn('ob_is_hpz', result.columns)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_short_data(self):
        """Test with very short data."""
        df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000, 1000, 1000]
        })

        result = market_structure_break(df, pivot_length=1)
        self.assertEqual(len(result), 3)

    def test_no_volume_column(self):
        """Test without volume column."""
        df = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Close': np.random.randn(100).cumsum() + 100
        })
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

        # Should work without volume
        result = market_structure_ob(df)
        self.assertIn('ob_quality', result.columns)

    def test_constant_prices(self):
        """Test with constant prices (no movement)."""
        df = pd.DataFrame({
            'Open': [100] * 50,
            'High': [101] * 50,
            'Low': [99] * 50,
            'Close': [100] * 50,
            'Volume': [1000] * 50
        })

        result = market_structure_break(df)
        # Should not crash, MSB should be 0
        self.assertEqual(result['msb_bullish'].sum(), 0)
        self.assertEqual(result['msb_bearish'].sum(), 0)


if __name__ == '__main__':
    unittest.main()

