import pandas as pd
import polars as pl
from unittest import TestCase

from pyindicators import is_crossover, PyIndicatorException


class TestCrossover(TestCase):

    def test_crossover_pandas(self):
        df = pd.DataFrame({
            "EMA_50": [200, 201, 202, 203, 204, 205, 206, 208, 208, 210],
            "EMA_200": [200, 201, 202, 203, 204, 205, 206, 207, 209, 209],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)

        self.assertTrue(is_crossover(df, first_column="EMA_50", second_column="EMA_200"))
        self.assertTrue(is_crossover(df, first_column="EMA_50", second_column="EMA_200", number_of_data_points=3))

        df = pd.DataFrame({
            "EMA_50": [200, 201, 202, 203, 204, 205, 206, 208, 210, 210],
            "EMA_200": [200, 201, 202, 203, 204, 205, 206, 209, 209, 209],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)

        self.assertTrue(
            is_crossover(df, first_column="EMA_50", second_column="EMA_200")
        )
        self.assertTrue(
            is_crossover(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=3
            )
        )

        df = pd.DataFrame({
            "EMA_50": [200, 201, 202, 200],
            "EMA_200": [200, 201, 202, 200],
            "DateTime": pd.date_range("2021-01-01", periods=4, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)
        self.assertFalse(
            is_crossover(df, first_column="EMA_50", second_column="EMA_200")
        )
        self.assertFalse(
            is_crossover(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=3
            )
        )

    def test_crossover_polars(self):
        df = pl.DataFrame({
            "EMA_50": [200, 201, 202, 203, 204, 205, 206, 208, 208, 210],
            "EMA_200": [200, 201, 202, 203, 204, 205, 206, 207, 209, 209],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        self.assertTrue(is_crossover(df, first_column="EMA_50", second_column="EMA_200"))
        self.assertTrue(is_crossover(df, first_column="EMA_50", second_column="EMA_200", number_of_data_points=3))

        df = pl.DataFrame({
            "EMA_50": [200, 201, 202, 203, 204, 205, 206, 208, 210, 210],
            "EMA_200": [200, 201, 202, 203, 204, 205, 206, 209, 209, 209],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        self.assertTrue(
            is_crossover(df, first_column="EMA_50", second_column="EMA_200")
        )
        self.assertTrue(
            is_crossover(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=3
            )
        )

        df = pl.DataFrame({
            "EMA_50": [200, 201, 202, 200],
            "EMA_200": [200, 201, 202, 200],
            "DateTime": pd.date_range("2021-01-01", periods=4, freq="D")
        })

        self.assertFalse(
            is_crossover(df, first_column="EMA_50", second_column="EMA_200")
        )
        self.assertFalse(
            is_crossover(
                df,
                first_column="EMA_50",
                second_column="EMA_200",
                number_of_data_points=3
            )
        )

    def test_throws_exception_when_number_of_data_points_is_less_than_2(self):
        df = pd.DataFrame({
            "EMA_50": [200, 201, 202, 203, 204, 205, 206, 208, 208, 210],
            "EMA_200": [200, 201, 202, 203, 204, 205, 206, 207, 209, 209],
            "DateTime": pd.date_range("2021-01-01", periods=10, freq="D")
        })

        # Set index to DateTime
        df.set_index("DateTime", inplace=True)

        with self.assertRaises(PyIndicatorException):
            is_crossover(
                df, first_column="EMA_50", second_column="EMA_200", number_of_data_points=1
            )