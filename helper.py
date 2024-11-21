# Standard Library
import os
from time import time

# Current project dependencies
import file_combiner as fc

RAW_DATA_FOLDER = "raw_data"
PROCESSED_DATA_FOLDER = "processed_data"
MODELS_FOLDER = "models"


def get_tickers(from_folder=False):
    if from_folder:
        return os.listdir(RAW_DATA_FOLDER)
    else:
        return ["TSLA", "AAPL", "MSFT", "GOOG", "AMZN"]


def get_intervals(from_folder=False):
    if from_folder:
        ticker = "TSLA"
        return os.listdir(f"{RAW_DATA_FOLDER}/{ticker}")
    else:
        return ["5min", "1day", "1min", "1h"]


def get_latest_scaler():
    latest_scaler_path = max(os.listdir(f"{MODELS_FOLDER}/scalers"))
    return os.path.join(MODELS_FOLDER, "scalers", latest_scaler_path)


def get_specific_scaler(timestamp):
    return os.path.join(MODELS_FOLDER, "scalers", f"{timestamp}_scaler.gzip")


def ignore_files_list():
    return ["all_current.csv", "trade_outcomes.csv"]


def get_ticker_file(ticker, interval):
    return os.path.join(RAW_DATA_FOLDER, ticker, interval, "all_current.csv")


def get_ticker_files(ticker, interval):
    return [f for f in os.listdir(os.path.join(RAW_DATA_FOLDER, ticker, interval)) if f not in ignore_files_list()]


def get_trade_outcomes_file(ticker, interval, max_bar=50):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, f"{max_bar}_max_bars_trade_outcomes.csv")


def get_scaled_labeled(ticker, interval, max_bar=50):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, f"scaled_labeled_{max_bar}_bars.csv")


def get_labeled_outcomes(ticker, interval, max_bar=50):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, f"{max_bar}_max_bars_labeled_trade_outcomes.csv")


def get_previous_combined(ticker, interval):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, "previous_combined_by_datetime.csv")


def get_scaled_previous_combined(ticker, interval):
    return os.path.join(PROCESSED_DATA_FOLDER, ticker, interval, "scaled_previous_combined_by_datetime.csv")


def get_new_scaler():
    return os.path.join(MODELS_FOLDER, "scalers", f"{int(time())}_scaler.gzip")