import os

import polars as pl
from abc import abstractmethod
from unittest import TestCase


class TestBaseline(TestCase):
    correct_output_csv_filename = None
    source_csv_filename = \
        "OHLCV_BTC-EUR_BINANCE_15m_2023-12-01:00:00_2023-12-25:00:00.csv"
    result_column = None

    @abstractmethod
    def generate_polars_df(self, pandas_source_df):
        pass

    @abstractmethod
    def generate_pandas_df(self, polars_source_df):
        pass

    def make_polars_column_datetime_naive(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        # Ensure the column is in the correct type, then convert to UTC and remove timezone
        return df.with_columns(
            pl.col(column).cast(pl.Datetime).dt.convert_time_zone("UTC").cast(pl.Datetime).alias(column)
        )

    def get_correct_output_csv_path(self):
        return os.path.abspath(
            os.path.join(
                os.path.join(
                    os.path.join(
                        os.path.join(
                            os.path.join(
                                os.path.realpath(__file__),
                                os.pardir
                            ),
                            os.pardir
                        ),
                        "test_data"
                    ),
                    "correct_test_data"
                ),
                self.correct_output_csv_filename
            )
        )

    def get_source_csv_path(self):
        return os.path.abspath(
            os.path.join(
                os.path.join(
                    os.path.join(
                        os.path.join(
                            os.path.join(
                                os.path.realpath(__file__),
                                os.pardir
                            ),
                            os.pardir
                        ),
                        "test_data"
                    ),
                    "source_test_data"
                ),
                self.source_csv_filename
            )
        )
