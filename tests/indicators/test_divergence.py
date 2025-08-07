import numpy as np
from unittest import TestCase
from pyindicators import is_divergence, bearish_divergence_multi_dataframe, \
    PyIndicatorException

import pandas as pd

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
