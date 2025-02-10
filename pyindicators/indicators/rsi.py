from pandas import Series


def rsi(series: Series, timeperiod=14):
    """
    Calculate the Relative Strength Index (RSI) for a given Pandas series.

    Parameters:
    - series: Pandas series instance containing prices
    - period: Lookback period for RSI calculation (default is 14)

    Returns:
    - series representing the RSI values
    """
    # Calculate daily price changes
    delta = series.diff()

    # Calculate gains (positive changes) and losses (negative changes)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate initial smoothed averages
    avg_gain = gains.rolling(window=timeperiod, min_periods=1).mean()
    avg_loss = losses.rolling(window=timeperiod, min_periods=1).mean()

    # Calculate SMMA for gains and losses
    for i in range(1, len(series)):
        avg_gain[i] = (avg_gain[i - 1] * (timeperiod - 1) + gains[
            i]) / timeperiod
        avg_loss[i] = (avg_loss[i - 1] * (timeperiod - 1) + losses[
            i]) / timeperiod

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss.replace(
        0, 1
    )  # Replace 0s with 1 to avoid division by zero

    # Calculate RSI
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values
    # df_deep_copy = df.copy(deep=True)
    # # df_deep_copy['Delta'] = df_deep_copy['Close'].diff()
    #
    # # Separate gains and losses
    # gains = df_deep_copy['Delta'].where(df_deep_copy['Delta'] > 0, 0)
    # losses = -df_deep_copy['Delta'].where(df_deep_copy['Delta'] < 0, 0)
    #
    # # Calculate average gains and average losses over the specified period
    # avg_gains = gains.rolling(window=period, min_periods=1).mean()
    # avg_losses = losses.rolling(window=period, min_periods=1).mean()
    #
    # # Calculate relative strength (RS)
    # rs = avg_gains / avg_losses.replace(0, 1)  # Avoid division by zero
    #
    # # Calculate RSI
    # df_deep_copy[f'Rsi_{period}'] = 100 - (100 / (1 + rs))
    #
    # # Replace NaN values in 'Rsi' column with 0
    # df_deep_copy[f'Rsi_{period}'] = df_deep_copy[f'Rsi_{period}'].fillna(0)
    #
    # # Drop intermediate columns
    # df_deep_copy.drop(['Delta'], axis=1, inplace=True)
    # return df_deep_copy[[f'Rsi_{period}']]
