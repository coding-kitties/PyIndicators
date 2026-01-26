from .indicators import sma, rsi, ema, wilders_rsi, adx, roc, cci, \
    crossover, is_crossover, wma, macd, willr, is_crossunder, crossunder, \
    is_lower_low_detected, is_divergence, bollinger_width, \
    is_below, is_above, get_slope, has_any_higher_then_threshold, \
    has_slope_above_threshold, has_any_lower_then_threshold, \
    has_values_above_threshold, has_values_below_threshold, is_down_trend, \
    is_up_trend, up_and_downtrends, detect_peaks, atr, bollinger_bands, \
    bearish_divergence, bullish_divergence, stochastic_oscillator, \
    bearish_divergence_multi_dataframe, bullish_divergence_multi_dataframe
from .exceptions import PyIndicatorException
from .date_range import DateRange

# Version information
__version__ = "0.15.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))


def get_version():
    """Return the version string of pyindicators."""
    return __version__


__all__ = [
    '__version__',
    '__version_info__',
    'get_version',
    'sma',
    'wma',
    'is_crossover',
    'crossunder',
    'is_crossunder',
    'crossover',
    'is_crossover',
    'ema',
    'rsi',
    "wilders_rsi",
    'macd',
    'willr',
    'adx',
    'is_lower_low_detected',
    'is_below',
    'is_above',
    'get_slope',
    'has_any_higher_then_threshold',
    'has_slope_above_threshold',
    'has_any_lower_then_threshold',
    'has_values_above_threshold',
    'has_values_below_threshold',
    'PyIndicatorException',
    'is_down_trend',
    'is_up_trend',
    'up_and_downtrends',
    'DateRange',
    'detect_peaks',
    'bearish_divergence',
    'bullish_divergence',
    'is_divergence',
    'stochastic_oscillator',
    'bearish_divergence_multi_dataframe',
    'bullish_divergence_multi_dataframe',
    'bollinger_bands',
    'bollinger_width',
    'atr',
    'cci',
    'roc'
]
