import time

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from utils import utils, upward_trend, resistance_line




################
## Parameters ##
################

peroid = '120m' #1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
interval = '1m' # 1m, 2m, 5m, 15m, 30m, 60m 90m, 1h, 1d, 5d, 1wk, 1mo
ticker = "RTX"

#######################
## Find Upward Trend ##
#######################
#stocks = upward_trend.find_upward()
#ticker = next(iter(stocks))
#exit()

################
## simulation ##
################

start_price = yf.Ticker(ticker).history().tail(1)['Close'].values[0]

BUY_PRICES = []
TOTAL = 0.0
CASH = 10000

while True:
    ##########################
    ## Plot Resistance_line ##
    ##########################
    resistance_line.plot_metrics_with_resistance(ticker, peroid, interval)
    prices, volume, RSIs, macd, bollinger_bands = resistance_line.above_or_below_resistance_line(ticker, peroid, interval)

    current_price = yf.Ticker(ticker).history().tail(1)['Close'].values[0]
    print(f"{prices[-10:]=}")

    #################
    ## Trade Logic ##
    #################
    if prices[-1] == "Below":
        
        if (CASH - current_price) > 0:
        
            BUY_PRICES.append(current_price)
            CASH -= current_price

            print(f"{utils.light_green}BUY @ {current_price=:.4f} with {CASH=:.4f} left{utils.reset}")
            
        else: print(f"{prices[-1]=} cannot buy b/c {(CASH - current_price) > 0=:.4f}")

    elif prices[-1] == "Above":
        if len(BUY_PRICES) > 0:

            TOTAL += current_price - BUY_PRICES[0]
            CASH += current_price

            print(f"{utils.light_blue}SELL @ {current_price=:.4f} (+-) {current_price - BUY_PRICES[0]=:.4f} {CASH=:.4f}{utils.reset}")
            BUY_PRICES = BUY_PRICES[1:]
            
        else: print(f"{utils.gray}{prices[-1]=} cannot sell b/c {len(BUY_PRICES) > 0=}{utils.reset}")

    else: print("THIS SHOULD NEVER HAPPEN")

    ##############
    ## OVERVIEW ##
    ##############
    trade_pct = (((start_price + TOTAL)-start_price) / abs(start_price))* 100
    pct = (((current_price-start_price) / abs(start_price))* 100)
    if TOTAL < 0:
        print(f"{utils.light_red}{CASH=:.4f} {len(BUY_PRICES)=} {TOTAL=:.4f} {trade_pct=:.4f}%{utils.reset}")
    else:
        print(f"{utils.green}{CASH=:.4f} {len(BUY_PRICES)=} {TOTAL=:.4f} {trade_pct=:.4f}%{utils.reset}")

    if current_price - start_price < 0:
        print(f"Started trading @ {start_price=:.4f} Without trading bot {utils.light_red} {current_price - start_price=:.4f} {pct=:.4f}%{utils.reset}")
    else:
        print(f"Started trading @ {start_price=:.4f} Without trading bot {utils.light_yellow} {current_price - start_price=:.4f} {pct=:.4f}%{utils.reset}")
    print()
    
    start_time = time.time()
    while time.time() - start_time < 15:
        print(f"Run_Time {time.time() - start_time:.1f}", end="\r")
        time.sleep(1)
