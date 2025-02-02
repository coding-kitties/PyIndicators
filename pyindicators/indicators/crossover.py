import pandas as pd


def crossover(series_a: pd.Series, series_b: pd.Series):
    """
    Returns a boolean series indicating if series_a has crossed over series_b
    """
    return (series_a > series_b).astype(int).diff().astype('Int64') == 1
