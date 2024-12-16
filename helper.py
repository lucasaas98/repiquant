# Standard Library
import os
from time import time

# Third party dependencies
import pandas as pd

RAW_DATA_FOLDER = "raw_data"
PROCESSED_DATA_FOLDER = "processed_data"
MODELS_FOLDER = "models"
DEFAULT_TICKERS = ["TSLA", "AAPL", "MSFT", "GOOG", "AMZN"]
DEFAULT_INTERVALS = ["5min", "1day", "1min", "1h"]


class DataTranslator:
    def __init__(self, ticker, interval):
        self.hloc_data = None
        self.increments_data = None
        self.ticker = ticker
        self.interval = interval
        self.increments_data_file = get_scaled_previous_combined_parquet(ticker, interval)
        self.hloc_data_file = get_ticker_file_parquet(ticker, interval)
        self.load_data()

    def load_data(self):
        self.hloc_data = pd.read_parquet(self.hloc_data_file, engine="fastparquet")
        self.increments_data = pd.read_parquet(self.increments_data_file, engine="fastparquet")

    def get_increments_for_bar(self, timestamp):
        real_index = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        if real_index in self.increments_data.index.to_list():
            return self.increments_data.loc[real_index]

    def get_close_for_bar(self, timestamp):
        real_index = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        if real_index in self.hloc_data.index.to_list():
            return self.hloc_data.loc[real_index]


def get_tickers(from_folder=False):
    if from_folder:
        return os.listdir(RAW_DATA_FOLDER)
    else:
        return DEFAULT_TICKERS


def get_reasonable_tickers():
    # we are not able to work with all the tickers we currently have in TwelveData for the US. Too many tickers.
    # let's instead decide on a number of nice tickers - we are getting the tickers based on market cap
    default_tickers = DEFAULT_TICKERS
    valuable_tickers = [
        "NVDA",
        "META",
        "AVGO",
        "COST",
        "NFLX",
        "LLY",
        "WMT",
        "JPM",
        "V",
        "UNH",
        "XOM",
        "ORCL",
        "MA",
        "HD",
        "PG",
        "JNJ",
        "BAC",
        "CRM",
        "ABBV",
        "CVX",
        "TMUS",
        "KO",
        "MRK",
        "WFC",
        "CSCO",
        "AMD",
        "ADBE",
        "PEP",
        "MS",
        "DIS",
    ]

    return default_tickers + valuable_tickers


def get_intervals(from_folder=False):
    if from_folder:
        ticker = "TSLA"
        return os.listdir(f"{RAW_DATA_FOLDER}/{ticker}")
    else:
        return DEFAULT_INTERVALS


def get_latest_scaler():
    latest_scaler_path = max(os.listdir(f"{MODELS_FOLDER}/scalers"))
    return os.path.join(MODELS_FOLDER, "scalers", latest_scaler_path)


def get_specific_scaler(timestamp):
    return os.path.join(MODELS_FOLDER, "scalers", f"{timestamp}_scaler.gzip")


def ignore_files_list():
    return [
        "all_current.csv",
        "all_current.parquet",
        "trade_outcomes.csv",
        "trade_outcomes.parquet",
        "less_than_2_years",
        "fucked_in_the_api",
    ]


def get_ticker_file(ticker, interval):
    return os.path.join(RAW_DATA_FOLDER, ticker, interval, "all_current.csv")


def get_ticker_file_parquet(ticker, interval):
    return os.path.join(RAW_DATA_FOLDER, ticker, interval, "all_current.parquet")


def get_ticker_files(ticker, interval):
    return [f for f in os.listdir(os.path.join(RAW_DATA_FOLDER, ticker, interval)) if f not in ignore_files_list()]


def get_ticker_files_parquet(ticker, interval):
    return [
        f
        for f in os.listdir(os.path.join(RAW_DATA_FOLDER, ticker, interval))
        if f not in ignore_files_list() and f.split(".")[-1] == "parquet"
    ]


def get_trade_outcomes_file(ticker, interval, max_bar=50):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, f"{max_bar}_max_bars_trade_outcomes.csv")


def get_trade_outcomes_file_parquet(ticker, interval, max_bar=50):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, f"{max_bar}_max_bars_trade_outcomes.parquet")


def get_scaled_labeled(ticker, interval, max_bar=50, short=False):
    file_name = f"scaled_labeled_{max_bar}_bars.csv" if not short else f"scaled_labeled_{max_bar}_bars_short.csv"
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, file_name)


def get_scaled_labeled_parquet(ticker, interval, max_bar=50, short=False):
    file_name = (
        f"scaled_labeled_{max_bar}_bars.parquet" if not short else f"scaled_labeled_{max_bar}_bars_short.parquet"
    )
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, file_name)


def get_scaled_labeled_both_sides(ticker, interval, max_bar=50):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, f"scaled_labeled_{max_bar}_bars_both_sides.csv")


def get_scaled_labeled_both_sides_parquet(ticker, interval, max_bar=50):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, f"scaled_labeled_{max_bar}_bars_both_sides.parquet")


def get_labeled_outcomes(ticker, interval, max_bar=50):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, f"{max_bar}_max_bars_labeled_trade_outcomes.csv")


def get_labeled_outcomes_parquet(ticker, interval, max_bar=50):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, f"{max_bar}_max_bars_labeled_trade_outcomes.parquet")


def get_previous_combined(ticker, interval):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, "previous_combined_by_datetime.csv")


def get_previous_combined_parquet(ticker, interval):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, "previous_combined_by_datetime.parquet")


def get_scaled_previous_combined(ticker, interval):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, "scaled_previous_combined_by_datetime.csv")


def get_scaled_previous_combined_parquet(ticker, interval):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, "scaled_previous_combined_by_datetime.parquet")


def get_new_scaler():
    return os.path.join(MODELS_FOLDER, "scalers", f"{int(time())}_scaler.gzip")


def get_new_classifier():
    return os.path.join(MODELS_FOLDER, "classifiers", f"{int(time())}_classifier(SVC).gzip")
