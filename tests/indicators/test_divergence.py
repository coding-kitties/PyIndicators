import numpy as np
from unittest import TestCase
from pyindicators import is_divergence, bearish_divergence_multi_dataframe, \
    PyIndicatorException, bullish_divergence_multi_dataframe, \
    bearish_divergence, bullish_divergence

import pandas as pd

class TestBearishDivergence(TestCase):

    def test_detect_bearish_divergence_pandas(self):
        df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            "Close_highs": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)
        df = bearish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=1
        )

        # Check that at least one true value in the df
        self.assertTrue(any(df["bearish_divergence"]))

        df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            "Close_highs": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)
        df = bearish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=1
        )

        # Check that at least one true value in the df
        self.assertFalse(any(df["bearish_divergence"]))

        df = bearish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=2
        )
        self.assertTrue(any(df["bearish_divergence"]))

        df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            "Close_highs": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })
        df.set_index("DateTime", inplace=True)
        df = bearish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=1
        )
        self.assertFalse(any(df["bearish_divergence"]))
        df = bearish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=2
        )
        self.assertFalse(any(df["bearish_divergence"]))

        df = pd.DataFrame({
            "RSI_highs": [0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
            "Close_highs": [0, 0, -1, 0, 0, 0, 0, 0, 0, 1],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        df = bearish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=2
        )
        self.assertFalse(any(df["bearish_divergence"]))

        df = bearish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=7
        )
        self.assertTrue(any(df["bearish_divergence"]))


class TestBullishDivergence(TestCase):

    def test_detect_bullish_divergence_pandas(self):
        df = pd.DataFrame({
            "RSI_lows": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            "Close_lows": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)
        df = bullish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=1
        )

        # Check that at least one true value in the df
        self.assertTrue(any(df["bullish_divergence"]))

        df = pd.DataFrame({
            "RSI_lows": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            "Close_lows": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)
        df = bullish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=1
        )

        # Check that at least one true value in the df
        self.assertFalse(any(df["bullish_divergence"]))

        df = bullish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=2
        )
        self.assertTrue(any(df["bullish_divergence"]))

        df = pd.DataFrame({
            "RSI_lows": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            "Close_lows": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })
        df.set_index("DateTime", inplace=True)
        df = bullish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=1
        )
        self.assertFalse(any(df["bullish_divergence"]))
        df = bullish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=2
        )
        self.assertFalse(any(df["bullish_divergence"]))

        df = pd.DataFrame({
            "RSI_lows": [0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
            "Close_lows": [0, 0, -1, 0, 0, 0, 0, 0, 0, 1],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        df = bullish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=2
        )
        self.assertFalse(any(df["bullish_divergence"]))

        df = bullish_divergence(
            data=df,
            first_column="RSI",
            second_column="Close",
            window_size=7
        )
        self.assertTrue(any(df["bullish_divergence"]))

class TestDetectDivergence(TestCase):

    def test_detect_bearish_divergence_pandas(self):
        df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            "Close_highs": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)

        self.assertFalse(
            is_divergence(df, window_size=2, number_of_data_points=2, column_one="RSI_highs", column_two="Close_highs")
        )
        self.assertTrue(is_divergence(df, window_size=2, number_of_data_points=10, column_one="RSI_highs", column_two="Close_highs"))

        df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            "Close_highs": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)
        self.assertFalse(is_divergence(df, window_size=10, number_of_data_points=1, column_one="RSI_highs", column_two="Close_highs"))

        df = pd.DataFrame({
            "RSI_highs": [0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
            "Close_highs": [0, 0, -1, 0, 0, 0, 0, 0, 0, 1],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)

        self.assertFalse(is_divergence(df, window_size=6, number_of_data_points=10, column_one="RSI_highs", column_two="Close_highs"))
        self.assertTrue(is_divergence(df, window_size=7, number_of_data_points=10, column_one="RSI_highs", column_two="Close_highs"))

        df = pd.DataFrame({
            "RSI_highs": [-1.0],
            "Close_highs": [1.0],
            "DateTime": pd.date_range("2021-01-01", periods=1, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)

        self.assertTrue(is_divergence(df, window_size=1, number_of_data_points=1, column_one="RSI_highs", column_two="Close_highs"))

    def test_detect_bearish_sequence_with_wrong_params_pandas(self):

        df = pd.DataFrame({
            "RSI_highs": [0, 1, 0, -1, 0, 0, 0, 0, -1, 0],
            "Close_highs": [0, 0, -1, 0, 0, 0, 0, 1, 0, 1],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)
        self.assertTrue(is_divergence(df, window_size=8, number_of_data_points=10, column_one="RSI_highs", column_two="Close_highs"))
        self.assertFalse(is_divergence(df, window_size=8, number_of_data_points=1, column_one="RSI_highs", column_two="Close_highs"))

        df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            "Close_highs": [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })
        self.assertFalse(is_divergence(df, window_size=6, number_of_data_points=10, column_one="RSI_highs", column_two="Close_highs"))

        df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            "Close_highs": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })
        self.assertFalse(is_divergence(df, window_size=6, number_of_data_points=10, column_one="RSI_highs", column_two="Close_highs"))

        df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            "Close_highs": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })
        self.assertTrue(is_divergence(df, window_size=6, number_of_data_points=10, column_one="RSI_highs", column_two="Close_highs"))

        df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            "Close_highs": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })
        self.assertTrue(is_divergence(df, window_size=1, number_of_data_points=10, column_one="RSI_highs", column_two="Close_highs"))

        df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            "Close_highs": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })
        self.assertFalse(is_divergence(df, window_size=1, number_of_data_points=10, column_one="RSI_highs", column_two="Close_highs"))


class TestBearishDivergenceMultiDataFrame(TestCase):

    def test_bearish_divergence_detected(self):
        # Setup indicator (e.g., RSI) and price (e.g., Close) with divergence
        indicator = pd.DataFrame({
            "RSI": [50, 60, 70, 65, 60, 58, 55],
        }, index=pd.date_range("2022-01-01", periods=7))
        price = pd.DataFrame({
            "Close": [100, 102, 105, 108, 110, 112, 115],  # Higher highs
        }, index=pd.date_range("2022-01-01", periods=7))

        result = pd.DataFrame(index=indicator.index)

        # Force peaks manually for deterministic test
        indicator["RSI_highs"] = [0, 0, 0, 0, -1, 0, 0]  # Two indicator highs
        price["Close_highs"] = [0, 0, 0, 0, 1, 0, 0]    # Two price highs

        out = bearish_divergence_multi_dataframe(
            first_df=indicator,
            second_df=price,
            result_df=result,
            first_column="RSI",
            second_column="Close",
            window_size=2,
            result_column="bearish_divergence"
        )

        self.assertIn("bearish_divergence", out.columns)
        self.assertTrue(any(out["bearish_divergence"]))

        indicator = pd.DataFrame({
            "RSI": [50, 60, 70, 65, 60, 58, 55],
        }, index=pd.date_range("2022-01-01", periods=7))
        price = pd.DataFrame({
            "Close": [100, 102, 105, 108, 110, 112, 115],  # Higher highs
        }, index=pd.date_range("2022-01-01", periods=7))

        result = pd.DataFrame(index=indicator.index)

        # Force peaks manually for deterministic test
        indicator["RSI_highs"] = [0, 0, 0, 0, 1, 0, 0]  # Two indicator highs
        price["Close_highs"] = [0, 0, 0, 0, 1, 0, 0]  # Two price highs

        out = bearish_divergence_multi_dataframe(
            first_df=indicator,
            second_df=price,
            result_df=result,
            first_column="RSI",
            second_column="Close",
            window_size=2,
            result_column="bearish_divergence"
        )

        self.assertIn("bearish_divergence", out.columns)
        self.assertFalse(any(out["bearish_divergence"]))

        indicator = pd.DataFrame({
            "RSI": [50, 60, 70, 65, 60, 58, 55],
        }, index=pd.date_range("2022-01-01", periods=7))
        price = pd.DataFrame({
            "Close": [100, 102, 105, 108, 110, 112, 115],  # Higher highs
        }, index=pd.date_range("2022-01-01", periods=7))

        result = pd.DataFrame(index=indicator.index)

        # Force peaks manually for deterministic test
        indicator["RSI_highs"] = [0, 0, 0, -1, 0, 0, 0]  # Two indicator highs
        price["Close_highs"] = [0, 0, 0, 0, 0, 1, 0]  # Two price highs

        out = bearish_divergence_multi_dataframe(
            first_df=indicator,
            second_df=price,
            result_df=result,
            first_column="RSI",
            second_column="Close",
            window_size=2,
            result_column="bearish_divergence"
        )

        self.assertIn("bearish_divergence", out.columns)
        self.assertFalse(any(out["bearish_divergence"]))

        out = bearish_divergence_multi_dataframe(
            first_df=indicator,
            second_df=price,
            result_df=result,
            first_column="RSI",
            second_column="Close",
            window_size=3,
            result_column="bearish_divergence"
        )

        self.assertIn("bearish_divergence", out.columns)
        self.assertTrue(any(out["bearish_divergence"]))

    def test_missing_column_exception(self):
        df1 = pd.DataFrame({"RSI": [50, 60]}, index=pd.date_range("2022-01-01", periods=2))
        df2 = pd.DataFrame({"Close": [100, 110]}, index=pd.date_range("2022-01-01", periods=2))
        result = pd.DataFrame(index=df1.index)

        with self.assertRaises(PyIndicatorException):
            bearish_divergence_multi_dataframe(
                first_df=df1.drop("RSI", axis=1),
                second_df=df2,
                result_df=result,
                first_column="RSI",
                second_column="Close"
            )

    def test_not_enough_data_exception(self):
        df1 = pd.DataFrame({"RSI": [50]}, index=pd.date_range("2022-01-01", periods=1))
        df2 = pd.DataFrame({"Close": [100]}, index=pd.date_range("2022-01-01", periods=1))
        result = pd.DataFrame(index=df1.index)

        # Assume detect_peaks adds _highs column
        df1["RSI_highs"] = [1]
        df2["Close_highs"] = [1]

        with self.assertRaises(PyIndicatorException):
            bearish_divergence_multi_dataframe(
                first_df=df1,
                second_df=df2,
                result_df=result,
                first_column="RSI",
                second_column="Close",
                window_size=3
            )

    def test_different_timeframes_align_correctly(self):
        daily_index = pd.date_range("2022-01-01", periods=2, freq="D")
        indicator_df = pd.DataFrame({
            "RSI": [65, 60],
        }, index=daily_index)

        # 2-hour close prices — only some times will match the daily timestamps
        two_hour_index = pd.date_range("2022-01-01", periods=12, freq="2h")
        price_df = pd.DataFrame({
            "Close": [100, 102, 105, 108, 110, 112, 115, 117, 120, 122, 125, 130]
        }, index=two_hour_index)

        result_df = pd.DataFrame(index=price_df.index)

        # Inject fake peaks
        indicator_df["RSI_highs"] = [-1, 0]
        price_df["Close_highs"] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        result = bearish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=result_df,
            first_column="RSI",
            second_column="Close",
            window_size=2,
            result_column="bearish_divergence"
        )
        self.assertIn("bearish_divergence", result.columns)
        self.assertTrue(any(result["bearish_divergence"]))

    def test_standard_situations(self):
        two_hour_index = pd.date_range("2022-01-01", periods=10, freq="2h")
        indicator_df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        }, index=two_hour_index)

        # 2-hour close prices — only some times will match the daily timestamps
        price_df = pd.DataFrame({
            "Close_highs": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        }, index=two_hour_index)

        df = bearish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=1
        )

        # Check that at least one true value in the df
        self.assertTrue(any(df["bearish_divergence"]))

        two_hour_index = pd.date_range("2022-01-01", periods=10, freq="2h")
        indicator_df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        }, index=two_hour_index)

        # 2-hour close prices — only some times will match the daily timestamps
        price_df = pd.DataFrame({
            "Close_highs": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        }, index=two_hour_index)

        df = bearish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=1
        )

        self.assertFalse(any(df["bearish_divergence"]))

        df = bearish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=2
        )

        self.assertTrue(any(df["bearish_divergence"]))

        two_hour_index = pd.date_range("2022-01-01", periods=10, freq="2h")

        indicator_df = pd.DataFrame({
            "RSI_highs": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        }, index=two_hour_index)

        # 2-hour close prices — only some times will match the daily timestamps
        price_df = pd.DataFrame({
            "Close_highs": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        }, index=two_hour_index)

        df = bearish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=1
        )
        self.assertFalse(any(df["bearish_divergence"]))
        df = bearish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=2
        )
        self.assertFalse(any(df["bearish_divergence"]))

        indicator_df = pd.DataFrame({
            "RSI_highs": [0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
        }, index=two_hour_index)

        # 2-hour close prices — only some times will match the daily timestamps
        price_df = pd.DataFrame({
            "Close_highs": [0, 0, -1, 0, 0, 0, 0, 0, 0, 1],
        }, index=two_hour_index)

        df = bearish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=2
        )
        self.assertFalse(any(df["bearish_divergence"]))

        df = bearish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=7
        )
        self.assertTrue(any(df["bearish_divergence"]))


class TestBullishDivergenceMultiDataFrame(TestCase):

    def test_bullish_divergence_detected(self):
        # Setup indicator (e.g., RSI) and price (e.g., Close) with divergence
        indicator = pd.DataFrame({
            "RSI": [50, 60, 70, 65, 60, 58, 55],
        }, index=pd.date_range("2022-01-01", periods=7))
        price = pd.DataFrame({
            "Close": [100, 102, 105, 108, 110, 112, 115],  # Higher highs
        }, index=pd.date_range("2022-01-01", periods=7))

        result = pd.DataFrame(index=indicator.index)

        # Force peaks manually for deterministic test
        indicator["RSI_lows"] = [0, 0, 0, 0, -1, 0, 0]  # Two indicator highs
        price["Close_lows"] = [0, 0, 0, 0, 1, 0, 0]    # Two price highs

        out = bullish_divergence_multi_dataframe(
            first_df=indicator,
            second_df=price,
            result_df=result,
            first_column="RSI",
            second_column="Close",
            window_size=2,
            result_column="bullish_divergence"
        )

        self.assertIn("bullish_divergence", out.columns)
        self.assertTrue(any(out["bullish_divergence"]))

        indicator = pd.DataFrame({
            "RSI": [50, 60, 70, 65, 60, 58, 55],
        }, index=pd.date_range("2022-01-01", periods=7))
        price = pd.DataFrame({
            "Close": [100, 102, 105, 108, 110, 112, 115],  # Higher highs
        }, index=pd.date_range("2022-01-01", periods=7))

        result = pd.DataFrame(index=indicator.index)

        # Force peaks manually for deterministic test
        indicator["RSI_lows"] = [0, 0, 0, 0, 1, 0, 0]  # Two indicator highs
        price["Close_lows"] = [0, 0, 0, 0, 1, 0, 0]  # Two price highs

        out = bullish_divergence_multi_dataframe(
            first_df=indicator,
            second_df=price,
            result_df=result,
            first_column="RSI",
            second_column="Close",
            window_size=2,
            result_column="bullish_divergence"
        )

        self.assertIn("bullish_divergence", out.columns)
        self.assertFalse(any(out["bullish_divergence"]))

        indicator = pd.DataFrame({
            "RSI": [50, 60, 70, 65, 60, 58, 55],
        }, index=pd.date_range("2022-01-01", periods=7))
        price = pd.DataFrame({
            "Close": [100, 102, 105, 108, 110, 112, 115],  # Higher highs
        }, index=pd.date_range("2022-01-01", periods=7))

        result = pd.DataFrame(index=indicator.index)

        # Force peaks manually for deterministic test
        indicator["RSI_lows"] = [0, 0, 0, -1, 0, 0, 0]  # Two indicator highs
        price["Close_lows"] = [0, 0, 0, 0, 0, 1, 0]  # Two price highs

        out = bullish_divergence_multi_dataframe(
            first_df=indicator,
            second_df=price,
            result_df=result,
            first_column="RSI",
            second_column="Close",
            window_size=2,
            result_column="bullish_divergence"
        )

        self.assertIn("bullish_divergence", out.columns)
        self.assertFalse(any(out["bullish_divergence"]))

        out = bullish_divergence_multi_dataframe(
            first_df=indicator,
            second_df=price,
            result_df=result,
            first_column="RSI",
            second_column="Close",
            window_size=3,
            result_column="bullish_divergence"
        )

        self.assertIn("bullish_divergence", out.columns)
        self.assertTrue(any(out["bullish_divergence"]))

    def test_missing_column_exception(self):
        df1 = pd.DataFrame({"RSI": [50, 60]}, index=pd.date_range("2022-01-01", periods=2))
        df2 = pd.DataFrame({"Close": [100, 110]}, index=pd.date_range("2022-01-01", periods=2))
        result = pd.DataFrame(index=df1.index)

        with self.assertRaises(PyIndicatorException):
            bullish_divergence_multi_dataframe(
                first_df=df1.drop("RSI", axis=1),
                second_df=df2,
                result_df=result,
                first_column="RSI",
                second_column="Close"
            )

    def test_not_enough_data_exception(self):
        df1 = pd.DataFrame({"RSI": [50]}, index=pd.date_range("2022-01-01", periods=1))
        df2 = pd.DataFrame({"Close": [100]}, index=pd.date_range("2022-01-01", periods=1))
        result = pd.DataFrame(index=df1.index)

        # Assume detect_peaks adds _highs column
        df1["RSI_lows"] = [1]
        df2["Close_lows"] = [1]

        with self.assertRaises(PyIndicatorException):
            bullish_divergence_multi_dataframe(
                first_df=df1,
                second_df=df2,
                result_df=result,
                first_column="RSI",
                second_column="Close",
                window_size=3
            )

    def test_different_timeframes_align_correctly(self):
        daily_index = pd.date_range("2022-01-01", periods=2, freq="D")
        indicator_df = pd.DataFrame({
            "RSI": [65, 60],
        }, index=daily_index)

        # 2-hour close prices — only some times will match the daily timestamps
        two_hour_index = pd.date_range("2022-01-01", periods=12, freq="2h")
        price_df = pd.DataFrame({
            "Close": [100, 102, 105, 108, 110, 112, 115, 117, 120, 122, 125, 130]
        }, index=two_hour_index)

        result_df = pd.DataFrame(index=price_df.index)

        # Inject fake peaks
        indicator_df["RSI_lows"] = [-1, 0]
        price_df["Close_lows"] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        result = bullish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=result_df,
            first_column="RSI",
            second_column="Close",
            window_size=2,
            result_column="bullish_divergence"
        )
        self.assertIn("bullish_divergence", result.columns)
        self.assertTrue(any(result["bullish_divergence"]))

    def test_standard_situations(self):
        two_hour_index = pd.date_range("2022-01-01", periods=10, freq="2h")
        indicator_df = pd.DataFrame({
            "RSI_lows": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        }, index=two_hour_index)

        # 2-hour close prices — only some times will match the daily timestamps
        price_df = pd.DataFrame({
            "Close_lows": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        }, index=two_hour_index)

        df = bullish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=1
        )

        # Check that at least one true value in the df
        self.assertTrue(any(df["bearish_divergence"]))

        two_hour_index = pd.date_range("2022-01-01", periods=10, freq="2h")
        indicator_df = pd.DataFrame({
            "RSI_lows": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        }, index=two_hour_index)

        # 2-hour close prices — only some times will match the daily timestamps
        price_df = pd.DataFrame({
            "Close_lows": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        }, index=two_hour_index)

        df = bullish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=1
        )

        self.assertFalse(any(df["bearish_divergence"]))

        df = bullish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=2
        )

        self.assertTrue(any(df["bearish_divergence"]))

        two_hour_index = pd.date_range("2022-01-01", periods=10, freq="2h")

        indicator_df = pd.DataFrame({
            "RSI_lows": [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        }, index=two_hour_index)

        # 2-hour close prices — only some times will match the daily timestamps
        price_df = pd.DataFrame({
            "Close_lows": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        }, index=two_hour_index)

        df = bullish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=1
        )
        self.assertFalse(any(df["bearish_divergence"]))
        df = bullish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=2
        )
        self.assertFalse(any(df["bearish_divergence"]))

        indicator_df = pd.DataFrame({
            "RSI_lows": [0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
        }, index=two_hour_index)

        # 2-hour close prices — only some times will match the daily timestamps
        price_df = pd.DataFrame({
            "Close_lows": [0, 0, -1, 0, 0, 0, 0, 0, 0, 1],
        }, index=two_hour_index)

        df = bullish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=2
        )
        self.assertFalse(any(df["bearish_divergence"]))

        df = bullish_divergence_multi_dataframe(
            first_df=indicator_df,
            second_df=price_df,
            result_df=pd.DataFrame(index=indicator_df.index),
            first_column="RSI",
            second_column="Close",
            window_size=7
        )
        self.assertTrue(any(df["bearish_divergence"]))


class TestCheckDivergencePatternWithDatetimeIndex(TestCase):
    """Tests to verify divergence functions work correctly with DatetimeIndex."""

    def test_check_divergence_pattern_with_datetime_index(self):
        """Test that check_divergence_pattern works with DatetimeIndex Series."""
        from pyindicators.indicators.divergence import check_divergence_pattern

        series_a = pd.Series(
            [np.nan, -1, np.nan],
            index=pd.date_range('2024-01-01', periods=3, freq='1h')
        )
        series_b = pd.Series(
            [np.nan, np.nan, 1],
            index=pd.date_range('2024-01-01', periods=3, freq='1h')
        )

        result = check_divergence_pattern(series_a, series_b, target_a=-1, target_b=1)
        self.assertTrue(result)

    def test_check_divergence_pattern_with_integer_index(self):
        """Test that check_divergence_pattern still works with integer index."""
        from pyindicators.indicators.divergence import check_divergence_pattern

        series_a = pd.Series([np.nan, -1, np.nan])
        series_b = pd.Series([np.nan, np.nan, 1])

        result = check_divergence_pattern(series_a, series_b, target_a=-1, target_b=1)
        self.assertTrue(result)

    def test_check_divergence_pattern_with_numpy_arrays(self):
        """Test that check_divergence_pattern works with numpy arrays."""
        from pyindicators.indicators.divergence import check_divergence_pattern

        series_a = np.array([np.nan, -1, np.nan])
        series_b = np.array([np.nan, np.nan, 1])

        result = check_divergence_pattern(series_a, series_b, target_a=-1, target_b=1)
        self.assertTrue(result)

    def test_bearish_divergence_multi_dataframe_with_datetime_index(self):
        """Test bearish_divergence_multi_dataframe with DatetimeIndex DataFrames."""
        from pyindicators import macd, detect_peaks

        # Data with DatetimeIndex (common when loading from CSV or APIs)
        df = pd.DataFrame({
            'Close': [100, 101, 102, 101, 100, 99, 100, 101, 102, 103, 102, 101]
        }, index=pd.date_range('2024-01-01', periods=12, freq='1h'))

        # Apply indicators
        macd_data = macd(
            df.copy(),
            source_column="Close",
            short_period=3,
            long_period=6,
            signal_period=3
        )
        close_data = detect_peaks(
            df.copy(),
            source_column="Close",
            number_of_neighbors_to_compare=2,
            min_consecutive=1
        )
        macd_data = detect_peaks(
            macd_data,
            source_column="macd_histogram",
            number_of_neighbors_to_compare=2,
            min_consecutive=1
        )

        # This should not raise ValueError with DatetimeIndex
        divergence_data = bearish_divergence_multi_dataframe(
            first_df=macd_data,
            second_df=close_data,
            result_df=close_data.copy(),
            first_column="macd_histogram",
            second_column="Close",
            window_size=5,
            result_column="bearish_divergence"
        )

        # Verify the result has the expected column and index
        self.assertIn("bearish_divergence", divergence_data.columns)
        self.assertIsInstance(divergence_data.index, pd.DatetimeIndex)

    def test_bullish_divergence_multi_dataframe_with_datetime_index(self):
        """Test bullish_divergence_multi_dataframe with DatetimeIndex DataFrames."""
        from pyindicators import macd, detect_peaks

        # Data with DatetimeIndex
        df = pd.DataFrame({
            'Close': [103, 102, 101, 100, 99, 100, 101, 102, 101, 100, 99, 98]
        }, index=pd.date_range('2024-01-01', periods=12, freq='1h'))

        # Apply indicators
        macd_data = macd(
            df.copy(),
            source_column="Close",
            short_period=3,
            long_period=6,
            signal_period=3
        )
        close_data = detect_peaks(
            df.copy(),
            source_column="Close",
            number_of_neighbors_to_compare=2,
            min_consecutive=1
        )
        macd_data = detect_peaks(
            macd_data,
            source_column="macd_histogram",
            number_of_neighbors_to_compare=2,
            min_consecutive=1
        )

        # This should not raise ValueError with DatetimeIndex
        divergence_data = bullish_divergence_multi_dataframe(
            first_df=macd_data,
            second_df=close_data,
            result_df=close_data.copy(),
            first_column="macd_histogram",
            second_column="Close",
            window_size=5,
            result_column="bullish_divergence"
        )

        # Verify the result has the expected column and index
        self.assertIn("bullish_divergence", divergence_data.columns)
        self.assertIsInstance(divergence_data.index, pd.DatetimeIndex)

