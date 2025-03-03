import pandas as pd
import polars as pl
from unittest import TestCase

from pyindicators import is_crossunder


class TestCrossover(TestCase):

    def test_crossunder_pandas(self):
        df = pd.DataFrame({
            "EMA_50": [200, 201, 202, 203, 204, 205, 207, 205, 204, 205],
            "EMA_200": [200, 201, 202, 203, 204, 205, 206, 207, 209, 209],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)

        self.assertTrue(
            is_crossunder(df, first_column="EMA_50", second_column="EMA_200")
        )
        self.assertFalse(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=3
            )
        )
        self.assertTrue(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=5
            )
        )

        df = pd.DataFrame({
            "EMA_50": [200, 201, 202, 203, 204, 205, 207, 205, 209, 209],
            "EMA_200": [200, 201, 202, 203, 204, 205, 206, 209, 209, 209],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)

        self.assertTrue(
            is_crossunder(df, first_column="EMA_50", second_column="EMA_200")
        )
        self.assertFalse(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=3
            )
        )
        self.assertFalse(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=3
            )
        )

        # Check how strict works
        df = pd.DataFrame({
            "EMA_50": [200, 201, 202, 203, 204, 205, 207, 206, 205, 209],
            "EMA_200": [200, 201, 202, 203, 204, 205, 206, 206, 206, 209],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)

        self.assertFalse(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=4,
                strict=True
            )
        )
        self.assertTrue(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=4,
                strict=False
            )
        )


    def test_crossunder_polars(self):
        df = pl.DataFrame({
            "EMA_50": [200, 201, 202, 203, 204, 205, 207, 205, 204, 205],
            "EMA_200": [200, 201, 202, 203, 204, 205, 206, 207, 209, 209],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        self.assertTrue(
            is_crossunder(df, first_column="EMA_50", second_column="EMA_200")
        )
        self.assertFalse(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=3
            )
        )
        self.assertTrue(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=5
            )
        )

        df = pl.DataFrame({
            "EMA_50": [200, 201, 202, 203, 204, 205, 207, 205, 209, 209],
            "EMA_200": [200, 201, 202, 203, 204, 205, 206, 209, 209, 209],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        self.assertTrue(
            is_crossunder(df, first_column="EMA_50", second_column="EMA_200")
        )
        self.assertFalse(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=3
            )
        )
        self.assertFalse(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=3
            )
        )

        # Check how strict works
        df = pl.DataFrame({
            "EMA_50": [200, 201, 202, 203, 204, 205, 207, 206, 205, 209],
            "EMA_200": [200, 201, 202, 203, 204, 205, 206, 206, 206, 209],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        self.assertFalse(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=4,
                strict=True
            )
        )
        self.assertTrue(
            is_crossunder(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=4,
                strict=False
            )
        )
