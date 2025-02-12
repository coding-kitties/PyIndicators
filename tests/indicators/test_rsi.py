import pandas as pd
import polars as pl
import pandas.testing as pdt
from polars.testing import assert_frame_equal

from tests.resources import TestBaseline
from pyindicators import rsi


class Test(TestBaseline):
    correct_output_csv_filename = \
        "RSI_14_BTC-EUR_BINANCE_15m_2023-12-01-00-00_2023-12-25-00-00.csv"

    def generate_pandas_df(self, polars_source_df):
        polars_source_df = rsi(
            data=polars_source_df,
            period=14,
            result_column="RSI_14",
            source_column="Close"
        )
        return polars_source_df

    def generate_polars_df(self, pandas_source_df):
        pandas_source_df = rsi(
            data=pandas_source_df,
            period=14,
            result_column="RSI_14",
            source_column="Close"
        )
        return pandas_source_df

    def test_comparison_pandas(self):

        # Load the correct output in a pandas dataframe
        correct_output_pd = pd.read_csv(self.get_correct_output_csv_path())

        # Load the source in a pandas dataframe
        source = pd.read_csv(self.get_source_csv_path())

        # Generate the pandas dataframe
        output = self.generate_pandas_df(source)
        output = output[correct_output_pd.columns]
        output["Datetime"] = \
            pd.to_datetime(output["Datetime"]).dt.tz_localize(None)
        correct_output_pd["Datetime"] = \
            pd.to_datetime(correct_output_pd["Datetime"]).dt.tz_localize(None)

        pdt.assert_frame_equal(correct_output_pd, output)

    def test_comparison_polars(self):

        # Load the correct output in a polars dataframe
        correct_output_pl = pl.read_csv(self.get_correct_output_csv_path())

        # Load the source in a polars dataframe
        source = pl.read_csv(self.get_source_csv_path())

        # Generate the polars dataframe
        output = self.generate_polars_df(source)

        # Convert the datetime columns to datetime
        # Convert the 'Datetime' column in both DataFrames to datetime
        output = output.with_columns(
            pl.col("Datetime").str.strptime(pl.Datetime).alias("Datetime")
        )

        correct_output_pl = correct_output_pl.with_columns(
            pl.col("Datetime").str.strptime(pl.Datetime).alias("Datetime")
        )
        output = output[correct_output_pl.columns]
        output = self.make_polars_column_datetime_naive(output, "Datetime")
        correct_output_pl = self.make_polars_column_datetime_naive(
            correct_output_pl, "Datetime"
        )

        assert_frame_equal(correct_output_pl, output)
