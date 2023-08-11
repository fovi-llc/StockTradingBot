import sys
import time
import select
import threading
import itertools

import requests
import pandas as pd
import numpy as np
import robin_stocks.robinhood as robin_stocks

from utils import utils, NN, upward_trend, resistance_line

utils.login()

#while True:
#    quote = robin_stocks.get_latest_price("AAPL", priceType=None, includeExtendedHours=True)
#    print(quote)
#    time.sleep(1)

symbols = ["AAPL", "GOOGL", "NVDA", "PLTR", "TSLA", "NIO", "AMD", "AMZN", "MSFT"]
symbols = [
    'ABNB', 'ALGN', 'AMD', 'CEG', 'AMZN', 'AMGN', 'AEP', 'ADI', 'ANSS', 'AAPL',
    'AMAT', 'GEHC', 'ASML', 'TEAM', 'ADSK', 'ATVI', 'ADP', 'AZN', 'BKR', 'AVGO',
    'BIIB', 'BKNG', 'CDNS', 'ADBE', 'CHTR', 'CPRT', 'CSGP', 'CRWD', 'CTAS', 'CSCO',
    'CMCSA', 'COST', 'CSX', 'CTSH', 'DDOG', 'DXCM', 'FANG', 'DLTR', 'EA', 'EBAY',
    'ENPH', 'ON', 'EXC', 'FAST', 'GFS', 'META', 'FI', 'FTNT', 'GILD', 'GOOG',
    'GOOGL', 'HON', 'ILMN', 'INTC', 'INTU', 'ISRG', 'MRVL', 'IDXX', 'JD', 'KDP',
    'KLAC', 'KHC', 'LRCX', 'LCID', 'LULU', 'MELI', 'MAR', 'MCHP', 'MDLZ', 'MRNA',
    'MNST', 'MSFT', 'MU', 'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ORLY', 'PCAR', 'PANW',
    'PAYX', 'PDD', 'PLTR', 'PYPL', 'PEP', 'QCOM', 'REGN', 'ROST', 'SIRI', 'SGEN',
    'SBUX', 'SNPS', 'TSLA', 'TXN', 'TMUS', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY',
    'XEL', 'ZM', 'ZS', 'NIO'
]
spread_pcts = []
for symbol in symbols:
    quote = robin_stocks.stocks.get_quotes(symbol, info=None)[0]
    ask = float(quote["ask_price"])
    bid = float(quote["bid_price"])
    spread = ask - bid
    spread_pct = spread / bid
    spread_pcts.append(spread_pct)
    
    print(f"{symbol=} {spread=} {spread_pct=}")
argmin = np.argmin(spread_pcts)
print(f"MIN SPREAD {symbols[argmin]} {spread_pcts[argmin]}")
print()


    
def main():
    symbol = symbols[argmin]
    interval='5minute'
    span='day'

    utils.login()

    profile = robin_stocks.build_user_profile()
    print(profile)

    print("#############")
    print("## PROFILE ##")
    print("#############")
    profile = robin_stocks.profiles.load_account_profile()
    for key, value in profile.items():
        print(f"{key} = {value}")
            
    print("###########")
    print("## QUOTE ##")
    print("###########")
    quote = robin_stocks.stocks.get_quotes(symbol, info=None)[0]
    for key, value in quote.items():
        print(f"{key} = {value}")

    print("####################")
    print("## GATHERING DATA ##")
    print("####################")
    data = utils.collect_stock_data(symbol, min_units=15, max_units=60, downtime=10, df=None)
    
    plot_metrics = True
    BUY_PRICES = []
    BUY_ORDER_PRICE = []
    SELL_ORDER_PRICE = []
    ORDER_IDS = []
    TOTAL = 0.0
    CASH = 10000
    UNITS = 1
    SELL_ALL = False
    PURCHASE = True
    METRIC = "prices" # prices, volume, RSIs, macd, bollinger_bands
    
        
    print(f"Processing symbol: {symbol}")

    start_cash = CASH
    #start_price = np.array(utils.get_rh_stock_data(symbol, interval, span)['Close'])[-1]
    start_price = np.array(data['ask_price'])[-1]

    start_time = time.time()
    while True:

        #data = utils.get_rh_stock_data(symbol, interval, span)
        #current_price = np.array(data['Close'])[-1]
        data = utils.collect_stock_data(symbol, min_units=15, max_units=60, downtime=0, df=data)
        current_price = np.array(data['ask_price'])[-1]
        
        #############
        ## METRICS ##
        #############

        RESISTANCE_LINE = resistance_line.above_or_below_resistance_line(symbol, data, metric=METRIC, key="ask_price")
        print(f"Trading based on {METRIC}: {RESISTANCE_LINE[-10:]=}")
         
        if plot_metrics:
            #resistance_line.plot_metrics_with_resistance(symbol, data)
            resistance_line.plot_ask_bid(symbol, data)

        ######################################
        ## check if any orders went through ##
        ######################################
        for buy_order_price in BUY_ORDER_PRICE:

            # get latest price
            quote = robin_stocks.stocks.get_quotes(symbol)[0]
            if quote is not None:
                ask_price = float(quote["ask_price"])
                bid_price = float(quote["bid_price"])
                round_price = round((ask_price + bid_price) / 2, 6)
            else: print("503 Server Error: Service Unavailable for url"); continue

            if buy_order_price > ask_price:
                if CASH - buy_order_price > 0:
                    print(f"{utils.light_green}BUY ORDER FILLED{utils.reset}")

                    BUY_PRICES.append(buy_order_price)

                    # update cash
                    CASH -= buy_order_price
                    print(f"{utils.light_green}BUY @ {buy_order_price=:.6f} with {CASH=:.6f} left. current {ask_price=:.6f}{utils.reset}")

                    BUY_ORDER_PRICE.remove(buy_order_price)
                else:
                    print ("NOT ENOUGH CASH {CASH - buy_order_price=:.6f}")
                    BUY_ORDER_PRICE.remove(buy_order_price)
                
            else:
                print(f"{utils.gray}BUY ORDER CONFIRMED @ {buy_order_price=:.6f} current {ask_price=:.6f}{utils.reset}")

        for sell_order_price in SELL_ORDER_PRICE:

            # get latest price
            quote = robin_stocks.stocks.get_quotes(symbol)[0]
            if quote is not None:
                ask_price = float(quote["ask_price"])
                bid_price = float(quote["bid_price"])
                round_price = round((ask_price + bid_price) / 2, 6)
            else: print("503 Server Error: Service Unavailable for url"); continue

            if sell_order_price < bid_price:
                print(f"{utils.light_green}SELL ORDER FILLED{utils.reset}")

                if BUY_PRICES:
                    buy_price = BUY_PRICES.pop(0)
                    amount = sell_order_price - buy_price
                    TOTAL += amount

                    # update cash
                    CASH += sell_order_price
                    print(f"{utils.light_blue}SELL @ {sell_order_price=:.6f} (+-) {amount=:.6f} {CASH=:.6f} current {bid_price=:.6f}{utils.reset}")
                else:
                    print(f"{utils.light_blue}CANNOT SELL @ {sell_order_price=:.6f} (+-) {amount=:.6f} {CASH=:.6f} current {bid_price=:.6f}{utils.reset}")
                    print(f"{utils.light_blue}B/C {BUY_PRICES=} CANCELING ORDER{utils.reset}")

                # remove order_id from list
                SELL_ORDER_PRICE.remove(sell_order_price)
                
            else:
                print(f"{utils.gray}SELL ORDER CONFIRMED @ {sell_order_price=:.6f} current {bid_price=:.6f}{utils.reset}")
                
        
        #################
        ## Trade Logic ##
        #################
        # BUY the stock
        if RESISTANCE_LINE[-1] == "Below":

            # Cancel all SELL orders that did not go through.
            for sell_order_price in SELL_ORDER_PRICE:
                print(f"Canceling {sell_order_price=:.6f} b/c did not go through")
            SELL_ORDER_PRICE = []

            # BUY N UNITS of the stock
            for _ in range(0, UNITS):
                    
                # Have money to BUY stock
                if (CASH - current_price) > 0 and PURCHASE:

                    # get latest prices
                    quote = robin_stocks.stocks.get_quotes(symbol)[0]
                    
                    # Bid Price: The bid price is the highest price that a buyer is willing to pay
                    ask_price = float(quote["ask_price"])
                    bid_price = float(quote["bid_price"])
                    round_price = round((ask_price + bid_price) / 2, 6)

                    print(f"{utils.light_green}PLACE BUY ORDER @ {round_price=:.6f}{utils.reset}")
                    BUY_ORDER_PRICE.append(round_price)
                
                else: print(f"{RESISTANCE_LINE[-1]=} cannot buy b/c {(CASH - current_price) > 0=:.6f} or {PURCHASE=}")
            
        # SELL the stock
        elif RESISTANCE_LINE[-1] == "Above":

            # Cancel all BUY orders that did not go through.
            for buy_order_price in BUY_ORDER_PRICE:
                print(f"Canceling {buy_order_price=:.6f} b/c did not go through")
            BUY_ORDER_PRICE = []

            # SELL N UNITS of the stock
            for _ in range (0, UNITS):
                
                # Have stock to SELL
                if len(BUY_PRICES) > 0:

                    # get latest prices
                    quote = robin_stocks.stocks.get_quotes(symbol)[0]
                    
                    # Asking Price: The asking price is the lowest price at which a seller is willing to sell
                    ask_price = float(quote["ask_price"])
                    bid_price = float(quote["bid_price"])
                    round_price = round((ask_price + bid_price) / 2, 6)

                    print(f"{utils.light_blue}PLACE SELL ORDER @ {round_price=:.6f}{utils.reset}")
                    SELL_ORDER_PRICE.append(round_price)
                
                else: print(f"{utils.gray}{RESISTANCE_LINE[-1]=} cannot sell b/c {len(BUY_PRICES) > 0=}{utils.reset}")
    
        else: print("THIS SHOULD NEVER HAPPEN")
    
        ##############
        ## OVERVIEW ##
        ##############
        trade_pct = (TOTAL / abs(start_cash))* 100
        pct = (((current_price-start_price) / abs(start_price))* 100)
        if TOTAL < 0:
            print(f"{utils.light_red}{CASH=:.6f} {len(BUY_PRICES)=} {TOTAL=:.6f} {trade_pct=:.6f}%{utils.reset}")
        else:
            print(f"{utils.green}{CASH=:.6f} {len(BUY_PRICES)=} {TOTAL=:.6f} {trade_pct=:.6f}%{utils.reset}")
    
        if current_price - start_price < 0:
            print(f"Started trading @ {start_price=:.6f} Without trading bot {utils.light_red} {current_price - start_price=:.6f} {pct=:.6f}%{utils.reset}")
        else:
            print(f"Started trading @ {start_price=:.6f} Without trading bot {utils.light_yellow} {current_price - start_price=:.6f} {pct=:.6f}%{utils.reset}")
        print()

        if SELL_ALL and len(BUY_PRICES) == 0: exit()

        ##########
        ## Wait ##
        ##########
        symbol, interval, span, UNITS, SELL_ALL, METRIC, plot_metrics, PURCHASE = utils.get_user_input(start_time=start_time,
                                                                                                       symbol=symbol,
                                                                                                       interval=interval,
                                                                                                       span=span,
                                                                                                       UNITS=UNITS,
                                                                                                       SELL_ALL=SELL_ALL,
                                                                                                       METRIC=METRIC,
                                                                                                       plot_metrics=plot_metrics,
                                                                                                       PURCHASE=PURCHASE)
        # reset time
        start_time = time.time()
        
if __name__ == "__main__":
    main()
