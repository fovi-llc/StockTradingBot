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

#data = utils.collect_crypto_data("XLM", min_units=15, max_units=240, df=None)
#while True:
#    data = utils.collect_crypto_data("XLM", min_units=15, max_units=240, df=data)
#    resistance_line.plot_ask_bid("XLM", data)
#    time.sleep(10)
#    
#input()
#exit()

#symbols = ["AAPL", "GOOGL", "NVDA", "PLTR", "TSLA", "NIO", "AMD"]
#
#for symbol in symbols:
#    quote = robin_stocks.stocks.get_quotes(symbol, info=None)[0]
#    ask = float(quote["ask_price"])
#    bid = float(quote["bid_price"])
#    spread = ask - bid
#    spread_pct = spread / bid
#    
#    print(f"{symbol=} {spread=} {spread_pct=}")
#print()

symbols = ['BTC', 'ETH', 'DOGE', 'SHIB', 'AVAX', 'ETC', 'UNI', 'LTC', 'LINK', 'XLM', 'BCH', 'XTZ', 'AAVE', 'COMP']
spread_pcts = []
for symbol in symbols:
    quote = robin_stocks.crypto.get_crypto_quote(symbol)
    ask = float(quote["ask_price"])
    bid = float(quote["bid_price"])
    spread = ask - bid
    spread_pct = spread / bid
    spread_pcts.append(spread_pct)
    
    print(f"{symbol=} {spread=} {spread_pct=}")

argmin = np.argmin(spread_pcts)

print(f"MIN SPREAD {symbols[argmin]} {spread_pcts[argmin]}")
exit()
    
def main():
    symbol = 'XLM' # 'BTC', 'ETH', 'SOL', 'DOGE' "SHIB", 'AVAX', 'ETC', 'UNI', 'LTC', 'LINK', 'XLM', 'BCH', 'XTZ', 'AAVE', 'COMP',
    symbol = 'DOGE'
    interval='15second'
    span='hour'

    utils.login()

    profile = robin_stocks.build_user_profile()
    print(profile)

    print("#############")
    print("## PROFILE ##")
    print("#############")
    profile = robin_stocks.profiles.load_account_profile()
    for key, value in profile.items():
        print(f"{key} = {value}")

#    print("############")
#    print("## ORDERS ##")
#    print("############")
#    orders = robin_stocks.orders.get_all_crypto_orders(info=None)
#    for order in orders:
#        print(order, "\n")
#
#    print("##########################")
#    print("## CANCEL CRYPTO ORDERS ##")
#    print("##########################")
#    for order_id in ORDER_IDS:
#        order = robin_stocks.orders.cancel_all_crypto_orders(order_id)
#        print(order)
        
    print("#######################")
    print("## CANCEL ALL ORDERS ##")
    print("#######################")
    orders = robin_stocks.orders.cancel_all_crypto_orders()
    for order in orders:
        print(order, "\n")

    print("#################")
    print("## OPEN ORDERS ##")
    print("#################")
    orders = robin_stocks.orders.get_all_open_crypto_orders(info=None)
    for order in orders:
        print(order, "\n")

    print("###############")
    print("## POSITIONS ##")
    print("###############")
    positions = robin_stocks.crypto.get_crypto_positions()
    for position in positions:
        for key, value in position.items():
            print(f"{key} = {value}")
            
    print("###########")
    print("## QUOTE ##")
    print("###########")
    quote = robin_stocks.crypto.get_crypto_quote(symbol)
    for key, value in quote.items():
        print(f"{key} = {value}")
    
    plot_metrics = True
    BUY_PRICES = []
    BUY_ORDER_PRICE = []
    SELL_ORDER_PRICE = []
    ORDER_IDS = []
    TOTAL = 0.0
    CASH = np.array(profile["crypto_buying_power"], dtype=np.float64)
    UNITS = 1
    SELL_ALL = False
    PURCHASE = True
    METRIC = "prices" # prices, volume, RSIs, macd, bollinger_bands
    
        
    print(f"Processing symbol: {symbol}")

    start_cash = CASH
    start_price = np.array(utils.get_crypto_data(symbol, interval, span)['Close'])[-1]

    start_time = time.time()
    while True:

        data = utils.get_crypto_data(symbol, interval, span)
        current_price = np.array(data['Close'])[-1]
        
        #############
        ## METRICS ##
        #############

        RESISTANCE_LINE = resistance_line.above_or_below_resistance_line(symbol, data, metric=METRIC)
        print(f"Trading based on {METRIC}: {RESISTANCE_LINE[-10:]=}")
         
        if plot_metrics:
            resistance_line.plot_metrics_with_resistance(symbol, data)


        ######################################
        ## check if any orders went through ##
        ######################################
        for buy_order_price in BUY_ORDER_PRICE:

            # get latest price
            quote = robin_stocks.crypto.get_crypto_quote(symbol)
            if quote is not None:
                ask_price = float(quote["ask_price"])
                bid_price = float(quote["bid_price"])
                round_price = round((ask_price + bid_price) / 2, 6)
            else: print("503 Server Error: Service Unavailable for url"); continue

            if buy_order_price > ask_price:
                print(f"{utils.light_green}BUY ORDER FILLED{utils.reset}")

                BUY_PRICES.append(buy_order_price)

                # update cash
                CASH -= buy_order_price
                print(f"{utils.light_green}BUY @ {buy_order_price=:.6f} with {CASH=:.6f} left. current {ask_price=:.6f}{utils.reset}")

                BUY_ORDER_PRICE.remove(buy_order_price)
                
            else:
                print(f"{utils.gray}BUY ORDER CONFIRMED @ {buy_order_price=:.6f} current {ask_price=:.6f}{utils.reset}")

        for sell_order_price in SELL_ORDER_PRICE:

            # get latest price
            quote = robin_stocks.crypto.get_crypto_quote(symbol)
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
                    CASH += amount
                    print(f"{utils.light_blue}SELL @ {sell_order_price=:.6f} (+-) {amount=:.6f} {CASH=:.6f} current {bid_price=:.6f}{utils.reset}")
                else:
                    print(f"{utils.light_blue}CANNOT SELL @ {sell_order_price=:.6f} (+-) {amount=:.6f} {CASH=:.6f} current {bid_price=:.6f}{utils.reset}")
                    print(f"{utils.light_blue}B/C {BUY_PRICES=} CANCELING ORDER{utils.reset}")

                # remove order_id from list
                SELL_ORDER_PRICE.remove(sell_order_price)
                
            else:
                print(f"{utils.gray}SELL ORDER CONFIRMED @ {sell_order_price=:.6f} current {bid_price=:.6f}{utils.reset}")
                
#        utils.login()
#        profile = robin_stocks.build_user_profile()
#        for order_id in ORDER_IDS:
#            
#            profile = robin_stocks.profiles.load_account_profile()
#            
#            try:
#                order_info = robin_stocks.orders.get_crypto_order_info(order_id)
#                state = order_info["state"]
#                side = order_info["side"]
#                price = float(order_info['price'])
#
#                quote = robin_stocks.crypto.get_crypto_quote(symbol)
#                ask_price = float(quote["ask_price"])
#                bid_price = float(quote["bid_price"])
#                round_price = round((ask_price + bid_price) / 2, 6)
#
#                if state == "filled":
#
#                    print(f"{utils.light_green}{order_id} {side.upper()} ORDER {state.upper()}{utils.reset}")
#                
#                    if side == "buy":
#                    
#                        BUY_PRICES.append(float(order_info['price']))
#                            
#                        # update CASH
#                        profile = robin_stocks.profiles.load_account_profile()
#                        CASH = float(profile["crypto_buying_power"])
#                        print(f"{utils.light_green}BUY @ {price=:.6f} with {CASH=:.6f} left {round_price=:.6f} {ask_price=:.6f}{utils.reset}")
#
#                        # remove order_id from list
#                        ORDER_IDS.remove(order_id)
#                   
#                    elif side == "sell":
#                        buy_price = BUY_PRICES.pop(0)
#                        TOTAL += float(order_info["price"]) - buy_price
#
#                        # update CASH
#                        profile = robin_stocks.profiles.load_account_profile()
#                        CASH = float(profile["crypto_buying_power"])
#                        print(f"{utils.light_blue}SELL @ {float(order_info['price'])=:.6f} (+-) {float(order_info['price']) - buy_price=:.6f} {CASH=:.6f} {round_price=:.6f} {bid_price=:.6f}{utils.reset}")
#
#                        # remove order_id from list
#                        ORDER_IDS.remove(order_id)
#                    
#                elif state == "failed" or state == "canceled" or state == "rejected":
#                    print(f"{utils.light_red}{order_id} {side.upper()} ORDER {state.upper()} @ {price=:.6f} {round_price=:.6f}{utils.reset}")
#                
#                    # remove order_id from list
#                    ORDER_IDS.remove(order_id)
#                    break
#            
#                else:
#                    print(f"{utils.gray}{side.upper()} ORDER {state.upper()} @ {price=:.6f} {round_price=:.6f} {bid_price=:.6f} {ask_price=:.6f}{utils.reset}")
#                    
#            except Exception as e:
#                print(f"Error: {str(e)} removing {order_id}")
#                order_info = robin_stocks.orders.get_crypto_order_info(order_id)
#                print(order_info)
#                ORDER_IDS.remove(order_id)
#                continue
                

        ##############
        ## SELL ALL ##
        ##############
        if SELL_ALL:
            print("#######################")
            print("## load profile info ##")
            print("#######################")
            profile = robin_stocks.profiles.load_account_profile()
            for key, value in profile.items():
                print(f"{key} = {value}")

            print("##########################")
            print("## CANCEL CRYPTO ORDERS ##")
            print("##########################")
            for order_id in ORDER_IDS:
                order = robin_stocks.orders.cancel_all_crypto_orders(order_id)
                print(order)
                
            print("#######################")
            print("## CANCEL ALL ORDERS ##")
            print("#######################")
            orders = robin_stocks.orders.cancel_all_crypto_orders()
            for order in orders:
                print(order, "\n")

            print("###############")
            print("## POSITIONS ##")
            print("###############")
            positions = robin_stocks.crypto.get_crypto_positions()
            for position in positions:
                for key, value in position.items():
                    print(f"{key} = {value}")
            exit()
        
        #################
        ## Trade Logic ##
        #################
        # BUY the stock
        if RESISTANCE_LINE[-1] == "Below":

            # BUY N UNITS of the stock
            for _ in range(0, UNITS):
                    
                # Have money to BUY stock
                if (CASH - current_price) > 0 and PURCHASE:

                    # place order
                    #order_id = utils.buy_crypto(symbol)

                    # track order ID
                    #ORDER_IDS.append(order_id)

                    # get latest prices
                    quote = robin_stocks.crypto.get_crypto_quote(symbol)
                    
                    # Bid Price: The bid price is the highest price that a buyer is willing to pay
                    ask_price = float(quote["ask_price"])
                    bid_price = float(quote["bid_price"])
                    round_price = round((ask_price + bid_price) / 2, 6)

                    print(f"{utils.light_green}PLACE BUY ORDER @ {round_price=:.6f}{utils.reset}")
                    BUY_ORDER_PRICE.append(round_price)
                
                else: print(f"{RESISTANCE_LINE[-1]=} cannot buy b/c {(CASH - current_price) > 0=:.6f} or {PURCHASE=}")
            
        # SELL the stock
        elif RESISTANCE_LINE[-1] == "Above":

            # SELL N UNITS of the stock
            for _ in range (0, UNITS):
                
                # Have stock to SELL
                if len(BUY_PRICES) > 0:

                    # place order
                    #order_id = utils.sell_crypto(symbol)

                    # track order ID
                    #ORDER_IDS.append(order_id)

                    # get latest prices
                    quote = robin_stocks.crypto.get_crypto_quote(symbol)
                    
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
