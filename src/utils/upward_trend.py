import talib
import yfinance as yf


def SMA_check_upward_trend(symbol, window=50):
    # Get historical stock data
    stock_data = yf.download(symbol, period='1y')

    # Calculate the simple moving average (SMA)
    stock_data['SMA'] = stock_data['Close'].rolling(window=window).mean()

    # Check if the latest closing price is above the SMA
    latest_close = stock_data['Close'].iloc[-1]
    latest_sma = stock_data['SMA'].iloc[-1]

    if latest_close > latest_sma:
        return True
    else:
        return False

def RSI_check_upward_trend(symbol, window=14, rsi_threshold=50):
    # Get historical stock data
    stock_data = yf.download(symbol, period='1y')

    # Calculate the RSI
    stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=window)

    # Check if the latest RSI is above the threshold
    latest_rsi = stock_data['RSI'].iloc[-1]

    if latest_rsi > rsi_threshold:
        return True
    else:
        return False

def MACD_check_upward_trend(symbol):
    # Get historical stock data
    stock_data = yf.download(symbol, period='1y')

    # Calculate the MACD
    macd, signal, _ = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Check if the latest MACD is above the signal line
    latest_macd = macd.iloc[-1]
    latest_signal = signal.iloc[-1]

    if latest_macd > latest_signal:
        return True
    else:
        return False

def ADX_check_upward_trend(symbol, window=14, adx_threshold=25):
    # Get historical stock data
    stock_data = yf.download(symbol, period='1y')

    # Calculate the ADX
    adx = talib.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=window)

    # Check if the latest ADX is above the threshold
    latest_adx = adx.iloc[-1]

    if latest_adx > adx_threshold:
        return True
    else:
        return False

def MAE_check_upward_trend(symbol, window=20, deviation=0.05):
    # Get historical stock data
    stock_data = yf.download(symbol, period='1y')

    # Calculate the moving averages
    ma = stock_data['Close'].rolling(window=window).mean()

    # Calculate the upper and lower envelope lines
    upper_line = ma * (1 + deviation)
    lower_line = ma * (1 - deviation)

    # Check if the latest closing price is above the upper envelope line
    latest_close = stock_data['Close'].iloc[-1]
    latest_upper = upper_line.iloc[-1]

    if latest_close > latest_upper:
        return True
    else:
        return False

def OBV_check_upward_trend(symbol):
    # Get historical stock data
    stock_data = yf.download(symbol, period='1y')

    # Calculate the OBV
    obv = talib.OBV(stock_data['Close'], stock_data['Volume'])

    # Check if the latest OBV is increasing
    latest_obv = obv.iloc[-1]
    previous_obv = obv.iloc[-2]

    if latest_obv > previous_obv:
        return True
    else:
        return False

def SAR_check_upward_trend(symbol):
    # Get historical stock data
    stock_data = yf.download(symbol, period='1y')

    # Calculate the Parabolic SAR
    sar = talib.SAR(stock_data['High'], stock_data['Low'])

    # Check if the latest price is above the Parabolic SAR
    latest_close = stock_data['Close'].iloc[-1]
    latest_sar = sar.iloc[-1]

    if latest_close > latest_sar:
        return True
    else:
        return False

def MAR_check_upward_trend(symbol, periods=[20, 50, 100, 200]):
    # MOVING AVERAGE RIBBON
    # Get historical stock data
    stock_data = yf.download(symbol, period='1y')

    # Calculate the moving averages
    ma_data = {}
    for period in periods:
        ma_data[period] = stock_data['Close'].rolling(window=period).mean()

    # Check if the moving averages align in ascending order
    previous_ma = None
    for period in periods:
        current_ma = ma_data[period].iloc[-1]
        if previous_ma is not None and current_ma <= previous_ma:
            return False
        previous_ma = current_ma

    return True

def VWAPcheck_upward_trend(symbol, window=20):
    # Get historical stock data
    stock_data = yf.download(symbol, period='1y')

    # Calculate the VWAP
    vwap = talib.VWAP(stock_data['High'], stock_data['Low'], stock_data['Close'], stock_data['Volume'], timeperiod=window)

    # Check if the latest price is above the VWAP
    latest_close = stock_data['Close'].iloc[-1]
    latest_vwap = vwap.iloc[-1]

    if latest_close > latest_vwap:
        return True
    else:
        return False

def SO_check_upward_trend(symbol, window=14, k_threshold=80):
    # STOCHASTIC OSCILLATOR
    
    # Get historical stock data
    stock_data = yf.download(symbol, period='1y')

    # Calculate the Stochastic Oscillator
    slowk, slowd = talib.STOCH(stock_data['High'], stock_data['Low'], stock_data['Close'], fastk_period=window, slowk_period=3, slowd_period=3)

    # Check if the latest %K value is above the threshold
    latest_k = slowk.iloc[-1]

    if latest_k > k_threshold:
        return True
    else:
        return False
