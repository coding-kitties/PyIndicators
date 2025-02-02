import pandas as pd


def ema(series: pd.Series, period, adjust=False):
    return series.ewm(span=period, adjust=adjust).mean()


