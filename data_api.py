# Standard Library
import json
import os

# Third party dependencies
from twelvedata import TDClient

# Current project dependencies
import helper as h

client = None
api_key = os.getenv("TWELVEDATA_API_KEY")


def create_client():
    global client
    # Initialize client - apikey parameter is required
    api_key = os.getenv("TWELVEDATA_API_KEY")
    client = TDClient(apikey=api_key)
    return client


def get_time_series(symbol="AAPL", interval="1min", outputsize=10, start_date=None, end_date=None):
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


def get_stocks_list(country="USA", from_file=False):
    if not from_file:
        client = create_client()

        stocks_list = client.get_stocks_list(country=country).as_json()

        json.dump(stocks_list, open("assorted/stocks.json", "w"))

        tickers = [stock["symbol"] for stock in stocks_list]

        return tickers
    else:
        return [stock["symbol"] for stock in json.load(open("assorted/stocks.json", "r"))]


def get_actionable_stocks_list():
    return [symbol for symbol in get_stocks_list(from_file=True) if symbol in h.get_reasonable_tickers()]
    # ["QCOM"]
    #
    # ["SCHW", "PFE", "INTC"]
    # ["ACHR", "AIR", "BAM", "BRN", "CNM"]
    #


if __name__ == "__main__":
    # a = get_time_series()
    a = get_stocks_list()
    print(len(a))
    # print(a.head())
    # print(a.tail())
