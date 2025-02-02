import pandas as pd


def sma(series: pd.Series, timeperiod=14):
    return series.rolling(window=timeperiod).mean()
