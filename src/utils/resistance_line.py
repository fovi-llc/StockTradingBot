 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

#####################
## Resistance Line ##
#####################

def find_resistance_line(prices):
    n_minutes = len(prices)
    coefficients = np.polyfit(range(n_minutes), prices, 1)
    resistance_line = np.polyval(coefficients, range(n_minutes))
    return resistance_line

def draw_resistance_line(ticker, peroid, interval):
    data = yf.download(ticker, period=peroid, interval=interval, progress=False)
    minute_data = data.reset_index().to_dict('records')
    prices = np.array([point['Close'] for point in minute_data])
    timestamps = np.array([point['Datetime'] for point in minute_data])
    resistance_line = find_resistance_line(prices)

    plt.figure(figsize=(5, 3))
    plt.plot(timestamps, prices, label='Minute Data', marker='o')
    plt.plot(timestamps, resistance_line, label='Resistance Line', linestyle='--')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.title(f'Trending Resistance Line for {ticker}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

#def plot_metrics_with_resistance(ticker, peroid, interval):
def plot_metrics_with_resistance(ticker, data):
    #data = yf.download(ticker, period=peroid, interval=interval, progress=False)
    minute_data = data.reset_index().to_dict('records')
    prices = np.array([point['Close'] for point in minute_data])
    resistance_line = find_resistance_line(prices)

    volume = np.array([point['Volume'] for point in minute_data])
    volume_resistance_line = find_resistance_line(volume)
    
    prices_series = pd.Series(prices)
    
    # Calculate RSI, MACD, and Bollinger Bands
    rsi = calculate_rsi(prices_series)
    rsi_resistance_line = find_resistance_line(rsi)
    
    macd, signal = calculate_macd(prices_series)
    upper_band, lower_band = calculate_bollinger_bands(prices_series)

    # close plot
    plt.close()
    
    # Plot the data and metrics on separate subplots
    plt.figure(figsize=(8, 12))
    
    plt.subplot(5, 1, 1)
    plt.plot(data.index, prices, label='Price', color='b', marker='o')
    plt.plot(data.index, resistance_line, label='Resistance Line', linestyle='--', color='r')
    plt.title(f'Price and Resistance Line for {ticker}')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    
    plt.subplot(5, 1, 2)
    plt.plot(data.index, volume, label='Volume', color='purple', marker='o')
    plt.plot(data.index, volume_resistance_line, label='Resistance Line', linestyle='--', color='r')
    plt.title(f'Volume and Resistance Line for {ticker}')
    plt.xlabel('Timestamp')
    plt.ylabel('Volume')
    plt.legend()
    
    plt.subplot(5, 1, 3)
    plt.plot(data.index, rsi, label='RSI', color='g', marker='o')
    plt.plot(data.index, rsi_resistance_line, label='Resistance Line', linestyle='--', color='r')
    plt.title(f'Relative Strength Index (RSI) and Resistance Line for {ticker}')
    plt.xlabel('Timestamp')
    plt.ylabel('RSI Value')
    plt.legend()
    
    plt.subplot(5, 1, 4)
    plt.plot(data.index, macd, label='MACD', color='m', marker='o')
    plt.plot(data.index, signal, label='Signal Line', linestyle='--', color='orange')
    plt.title(f'Moving Average Convergence Divergence (MACD) for {ticker}')
    plt.xlabel('Timestamp')
    plt.ylabel('MACD Value')
    plt.legend()
    
    plt.subplot(5, 1, 5)
    plt.plot(data.index, prices, label='Price', color='b', marker='o')
    plt.plot(data.index, resistance_line, label='Resistance Line', linestyle='--', color='r')
    plt.fill_between(data.index, upper_band, lower_band, alpha=0.2, color='gray')
    plt.title(f'Bollinger Bands for {ticker}')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)

def calculate_rsi(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    exp_short = prices.ewm(span=short_window, adjust=False).mean()
    exp_long = prices.ewm(span=long_window, adjust=False).mean()
    macd = exp_short - exp_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(prices, window=20, k=2):
    rolling_mean = prices.rolling(window=window, min_periods=1).mean()
    rolling_std = prices.rolling(window=window, min_periods=1).std()
    upper_band = rolling_mean + k * rolling_std
    lower_band = rolling_mean - k * rolling_std
    return upper_band, lower_band

def above_or_below_resistance_line(ticker, data):
    #data = yf.download(ticker, period=peroid, interval=interval, progress=False)
    minute_data = data.reset_index().to_dict('records')
    prices = np.array([point['Close'] for point in minute_data])
    resistance_line = find_resistance_line(prices)
    
    volumes = np.array([point['Volume'] for point in minute_data])
    volume_resistance_line = find_resistance_line(volumes)
    
    rsi = calculate_rsi(prices)
    rsi_resistance_line = find_resistance_line(rsi)
    
    prices_series = pd.Series(prices)
    macd, signal = calculate_macd(prices_series)
    upper_band, lower_band = calculate_bollinger_bands(prices_series)
    
    # Compare each price data point with the resistance line
    PRICES = ['Above' if price > resistance_line[i] else 'Below' for i, price in enumerate(prices)]

    # Compare each volume with the resistance line
    VOL = ['Above' if volume > volume_resistance_line[i] else 'Below' for i, volume in enumerate(volumes)]

    # Compare each RSIs value with the resistance line
    RSI = ['Above' if value > rsi_resistance_line[i] else 'Below' for i, value in enumerate(rsi)]

    # Compare each MACD value with the signal line and the resistance line
    MACD = ['Above' if macd[i] > signal[i] and prices[i] > resistance_line[i] else 'Below' for i in range(len(macd))]

    # Compare each price with the resistance line, upper band, and lower band
    bollinger_bands = []
    for i, price in enumerate(prices):
        if price > resistance_line[i] and price > upper_band[i]:
            bollinger_bands.append('Above (Breakout)')
        elif price > resistance_line[i] and price <= upper_band[i]:
            bollinger_bands.append('Above')
        elif price < resistance_line[i] and price < lower_band[i]:
            bollinger_bands.append('Below (Breakout)')
        elif price < resistance_line[i] and price >= lower_band[i]:
            bollinger_bands.append('Below')
        else:
            bollinger_bands.append('Equal to Resistance Line')
    
    return PRICES, VOL, RSI, MACD, bollinger_bands
