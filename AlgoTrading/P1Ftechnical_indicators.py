"""Module to store functions to calculate technical indicators"""
import numpy as np
import  pandas as pd


def calculate_sma(prices, window):
    """
    Calculate Simple Moving Average (SMA) from first principles.
    """
    sma = []
    for i in range(len(prices)):
        # Ensure enough data to calc SMA
        if i >= window - 1:
            # Compute SMA as the mean of the last `window` prices
            sma.append(sum(prices[i - window + 1:i + 1]) / window)
        else:
            # Not enough data for SMA, append None
            sma.append(None)

    return pd.Series(sma, index=prices.index)


def calculate_ema(prices, window):
    """
    Calculate Exponential Moving Average (EMA) from first principles.

    Parameters:
        prices (list or np.ndarray): List of prices.
        window (int): Moving average window size.

    Returns:
        list: EMA values.
    """
    # Convert prices to a Pandas Series
    prices = pd.Series(prices)

    # list to store EMA values
    ema = []
    # calc the smoothing factor
    alpha = 2 / (window + 1)

    # Calc the first EMA value as the SMA of the first `window` prices
    if len(prices) >= window:
        sma = sum(prices[:window]) / window  # Calculate SMA
        ema.append(sma)  # Use SMA as the first EMA value
    else:
        raise ValueError("Not enough data points to calc the initial SMA for EMA.")

    # Calc subsequent EMA values
    for i in range(window, len(prices)):
        ema.append(alpha * prices[i] + (1 - alpha) * ema[-1])

    # Add None for the first (window - 1) entries, as EMA is undefined for those
    ema = [np.nan] * (window - 1) + ema

    return pd.Series(ema, index=prices.index)


def calculate_rolling_std(prices, window):
    """
    Calculate rolling standard deviation from first principles.
    """
    # List to store standard deviation values
    std_values = []

    for i in range(len(prices)):
        if i >= window - 1:
            # Extract the rolling window of prices
            window_prices = prices[i - window + 1:i + 1]
            # Calculate the mean of the window
            mean = sum(window_prices) / window
            # Calculate the variance
            variance = sum((p - mean) ** 2 for p in window_prices) / window
            # Standard deviation is the square root of variance
            std_values.append(variance ** 0.5)
        else:
            std_values.append(None)
    return pd.Series(std_values, index=prices.index)


def calculate_rsi_smoothed(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) using the smoothed formula.

    Parameters:
        prices (pd.Series): Series of prices.
        window (int): Lookback period for RSI (default is 14).

    Returns:
        pd.Series: RSI values.
    """
    # ensure prices are pd.Series
    prices = pd.Series(prices)
    # Calculate daily price changes
    delta = prices.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)  # Positive price changes
    loss = -delta.where(delta < 0, 0)  # Absolute value of negative price changes

    # Initialise the first average gain and loss
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Smooth the averages
    for i in range(window, len(prices)):
        avg_gain.iloc[i] = ((avg_gain.iloc[i - 1] * (window - 1)) + gain.iloc[i]) / window
        avg_loss.iloc[i] = ((avg_loss.iloc[i - 1] * (window - 1)) + loss.iloc[i]) / window

    # Calculate RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Fill undefined values with NaN
    rsi.iloc[:window] = np.nan

    return rsi


def calculate_bollinger_bands(prices, window=20):
    """
    Calculate Bollinger Bands using SMA and rolling standard deviation.
    """
    prices = pd.Series(prices)
    # calc SMA
    sma = calculate_sma(prices, window)
    # calc rolling stand dev
    rolling_std = calculate_rolling_std(prices, window)
    # calc upper and lower bands
    upper_bb = sma + 2 * rolling_std
    lower_bb = sma - 2 * rolling_std
    return sma, upper_bb, lower_bb


def calculate_adx(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate the Average Directional Index (ADX) along with +DI and -DI.

    Parameters:
        data (pd.DataFrame): DataFrame with columns 'High', 'Low', 'Close'.
        window (int): Lookback period for ADX (default is 14).

    Returns:
        pd.DataFrame: Updated DataFrame with +DI, -DI, and ADX columns.
    """
    # Calculate True Range (TR) if not present
    if "TR" not in data.columns:
        data["TR"] = np.maximum(data["High"] - data["Low"],
                                np.maximum(abs(data["High"] - data["Close"].shift(1)),
                                           abs(data["Low"] - data["Close"].shift(1))))

    # Calculate +DM and -DM if not present
    if "+DM" not in data.columns:
        data["+DM"] = np.where((data["High"] - data["High"].shift(1)) > (data["Low"].shift(1) - data["Low"]),
                               np.maximum(data["High"] - data["High"].shift(1), 0), 0)
    if "-DM" not in data.columns:
        data["-DM"] = np.where((data["Low"].shift(1) - data["Low"]) > (data["High"] - data["High"].shift(1)),
                               np.maximum(data["Low"].shift(1) - data["Low"], 0), 0)

    # Smooth TR, +DM, and -DM using EMA if not already present
    if "TR_smooth" not in data.columns:
        data["TR_smooth"] = data["TR"].ewm(span=window, adjust=False).mean()
    if "+DM_smooth" not in data.columns:
        data["+DM_smooth"] = data["+DM"].ewm(span=window, adjust=False).mean()
    if "-DM_smooth" not in data.columns:
        data["-DM_smooth"] = data["-DM"].ewm(span=window, adjust=False).mean()

    # Calculate +DI and -DI if not present
    if "+DI" not in data.columns:
        data["+DI"] = 100 * (data["+DM_smooth"] / data["TR_smooth"])
    if "-DI" not in data.columns:
        data["-DI"] = 100 * (data["-DM_smooth"] / data["TR_smooth"])

    # Calculate DX if not present
    if "DX" not in data.columns:
        data["DX"] = 100 * abs(data["+DI"] - data["-DI"]) / (data["+DI"] + data["-DI"])

    # Calculate ADX if not present
    if "ADX" not in data.columns:
        data["ADX"] = data["DX"].ewm(span=window, adjust=False).mean()

    return data


def calculate_macd(data: pd.Series, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
    """
    Calculate MACD line, Signal line, and Histogram.
    """
    data = pd.Series(data)
    # calc short EMA
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    # calc longer EMA
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    # calc signal line
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histogram = macd - signal

    return macd, signal, histogram