# Standard Library
import os

# Third party dependencies
from twelvedata import TDClient

client = None
api_key = os.getenv("TWELVEDATA_API_KEY")


def create_client():
    global client
    # Initialize client - apikey parameter is required
    api_key = os.getenv("TWELVEDATA_API_KEY")
    client = TDClient(apikey=api_key)
    return client


def get_client():
    global client
    if not client:
        create_client()
    return client


def get_time_series(symbol="AAPL", interval="1min", outputsize=10, start_date=None, end_date=None):
    global client

    client = create_client()

    if start_date and end_date:
        ts = client.time_series(
            symbol=symbol, interval=interval, outputsize=outputsize, start_date=start_date, end_date=end_date
        )
    elif start_date:
        ts = client.time_series(symbol=symbol, interval=interval, outputsize=outputsize, start_date=start_date)
    elif end_date:
        ts = client.time_series(symbol=symbol, interval=interval, outputsize=outputsize, end_date=end_date)
    else:
        ts = client.time_series(symbol=symbol, interval=interval, outputsize=outputsize)

    # add macd
    ts = (
        ts.with_macd()
        .with_rsi()
        .with_ema(time_period=200)
        .with_ema(time_period=50)
        .with_sma(time_period=20)
        .with_vwap()
    )

    return ts.as_pandas()


def get_stocks_list(country="USA"):
    global client

    client = get_client()

    stocks_list = client.get_stocks_list(country=country, exchange="NASDAQ").as_json()
    # Standard Library
    import json

    json.dump(stocks_list, open("stocks.json", "w"))

    tickers = [stock["symbol"] for stock in stocks_list]

    return tickers
    # return ["TSLA", "AAPL", "MSFT", "GOOG", "AMZN"]


if __name__ == "__main__":
    # a = get_time_series()
    a = get_stocks_list()
    print(len(a))
    # print(a.head())
    # print(a.tail())
