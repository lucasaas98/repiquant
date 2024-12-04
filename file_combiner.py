# Standard Library
import os
import re

# Third party dependencies
import pandas as pd

# Current project dependencies
import data_api
import helper as h


def reset_raw_data_folder():
    pattern = (
        r"[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]\.csv"
    )

    for ticker in h.get_tickers(from_folder=True):
        for interval in h.get_intervals(from_folder=True):
            if os.path.isdir(f"{h.RAW_DATA_FOLDER}/{ticker}/{interval}"):
                files = os.listdir(f"{h.RAW_DATA_FOLDER}/{ticker}/{interval}")
                for file in files:
                    # check the pattern of the file name, if it's like 1639387800_1731426900 keep, else remove
                    if re.match(pattern, file):
                        pass
                    else:
                        os.remove(f"{h.RAW_DATA_FOLDER}/{ticker}/{interval}/{file}")


def combine_ticker_files(ticker, interval):
    dataframes = [
        pd.read_csv(f"{h.RAW_DATA_FOLDER}/{ticker}/{interval}/{file}", index_col=0)
        for file in sorted(h.get_ticker_files(ticker, interval), reverse=True)
    ]
    full_dataframe = pd.concat(dataframes).reset_index().set_index("datetime")[::-1]
    full_dataframe = full_dataframe[~full_dataframe.index.duplicated(keep="first")]
    full_dataframe.to_csv(h.get_ticker_file(ticker, interval))


def combine_all_tickers(from_data_api=False):
    tickers = None
    if from_data_api is True:
        tickers = data_api.get_actionable_stocks_list()
    else:
        tickers = h.get_tickers(from_folder=True)

    for ticker in tickers:
        print(f"Combining {ticker}")
        for interval in h.get_intervals(from_folder=True):
            try:
                combine_ticker_files(ticker, interval)
                print(f"{ticker} {interval} done")
            except Exception as e:
                print(e)
                print(f"{ticker} {interval} failed")


if __name__ == "__main__":
    combine_all_tickers()
