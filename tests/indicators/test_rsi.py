# from datetime import timedelta
# from unittest import TestCase

# import pandas as pd
# import numpy as np
# import tulipy as ti
# from investing_algorithm_framework import CSVOHLCVMarketDataSource

# import pyindicators as pyi


# class Test(TestCase):

#     def test(self):
#         data_source = CSVOHLCVMarketDataSource(
#             csv_file_path="../test_data/OHLCV_BTC-EUR_BINANCE_15m"
#                           "_2023-12-01:00:00_2023-12-25:00:00.csv",
#         )
#         data_source.end_date = data_source.start_date \
#             + timedelta(days=4, hours=4)

#         while not data_source.empty():
#             data = data_source.get_data(market_credential_service=None)
#             df = pd.DataFrame(
#                 data,
#                 columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
#             )
#             pyi_rsi = pyi.rsi(series=df["Close"], timeperiod=14)
#             ta_rsi = ta.RSI(df["Close"], timeperiod=14).astype('float64')
#             ti_rsi = pd.Series(ti.rsi(df["Close"].to_numpy(), period=14))
#             # # Define a tolerance for comparison
#             tolerance = 1e-9
#             #
#             # # Compare the two Series with tolerance
#             nan_mask = ~np.isnan(pyi_rsi) & ~np.isnan(ta_rsi)
#             comparison_result = np.abs(
#                 ta_rsi[nan_mask] - ti_rsi[nan_mask]) <= tolerance

#             print(ta_rsi.iloc[-1], ti_rsi.iloc[-1])
#             # data_source.start_date = \
#             #     data_source.start_date + timedelta(minutes=15)
#             # data_source.end_date = data_source.end_date + timedelta(minutes=15)
#             # self.assertTrue(all(comparison_result))
