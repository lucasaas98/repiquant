# Standard Library
import json
import os

TICKER = os.getenv("TICKER")
INTERVAL = os.getenv("INTERVAL")

MULTIPLE_FEED_TICKERS = json.loads(os.getenv("TICKER_LIST"))
MULTIPLE_FEED_EXCHANGE = os.getenv("EXCHANGE")

LONG_MODEL = os.getenv("LONG_MODEL")
SHORT_MODEL = os.getenv("SHORT_MODEL")
SCALER = os.getenv("SCALER")
