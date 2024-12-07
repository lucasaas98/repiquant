# Standard Library
import json
import os

TICKER = "TSLA"
INTERVAL = "5min"

MULTIPLE_FEED_TICKERS = json.loads(os.getenv("TICKER_LIST"))
MULTIPLE_FEED_EXCHANGE = os.getenv("EXCHANGE")
