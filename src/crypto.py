 
import requests
import pandas as pd

def get_crypto_history_data(crypto_symbol, currency_symbol, limit=1000):
    base_url = 'https://min-api.cryptocompare.com/data/v2'
    endpoint = '/histominute'
    url = f'{base_url}{endpoint}?fsym={crypto_symbol}&tsym={currency_symbol}&limit={limit}'

    try:
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None

def main():
    # Replace 'BTC' and 'USD' with the desired cryptocurrency and currency symbols
    crypto_symbol = 'ASM'
    currency_symbol = 'USD'
    limit = 1000  # Number of 1-minute data points to retrieve

    data = get_crypto_history_data(crypto_symbol, currency_symbol, limit)

    if data is not None:
        print(data)
    else:
        print("Failed to retrieve data.")

if __name__ == "__main__":
    main()
