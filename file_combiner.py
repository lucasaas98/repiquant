# Standard Library
import os

# Third party dependencies
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

# Current project dependencies
import data_api
import helper as h


def combine_ticker_files_parquet(ticker, interval):
    print(f"{ticker} {interval} combining")

    first = True
    all_data = None

    for file in sorted(h.get_ticker_files_parquet(ticker, interval), reverse=True):
        if first:
            all_data = (
                pl.read_parquet(f"{h.RAW_DATA_FOLDER}/{ticker}/{interval}/{file}")
                .with_columns(datetime=pl.col("datetime").cast(str).str.split(".").list.first())
                .sort(by="datetime")
            )
            first = False
        else:
            data = (
                pl.read_parquet(f"{h.RAW_DATA_FOLDER}/{ticker}/{interval}/{file}")
                .with_columns(datetime=pl.col("datetime").cast(str).str.split(".").list.first())
                .sort(by="datetime")
            )
            all_data = all_data.merge_sorted(data, key="datetime")

    all_data = all_data.unique(subset=["datetime"], keep="last", maintain_order=True)
    all_data.write_parquet(h.get_ticker_file_parquet(ticker, interval))

    print(f"{ticker} {interval} done")


def convert_ticker_file_to_parquet(ticker, interval):
    full_dataframe = pl.read_csv(h.get_ticker_file(ticker, interval))
    full_dataframe.write_parquet(h.get_ticker_file_parquet(ticker, interval))


def convert_ticker_files_to_parquet(ticker, interval):
    for file in h.get_ticker_files(ticker, interval):
        df = pl.read_csv(f"{h.RAW_DATA_FOLDER}/{ticker}/{interval}/{file}")
        df.write_parquet(f"{h.RAW_DATA_FOLDER}/{ticker}/{interval}/{file.split('.')[0] + '.parquet'}")


def combine_all_tickers(from_data_api=False):
    tickers = None
    if from_data_api is True:
        tickers = data_api.get_actionable_stocks_list()
    else:
        tickers = h.get_tickers(from_folder=True)

    Parallel(n_jobs=16)(
        delayed(combine_ticker_files_parquet)(ticker, interval)
        for ticker in tickers
        for interval in h.get_intervals(from_folder=True)
    )


def convert_to_parquet(from_data_api=False):
    tickers = None
    if from_data_api is True:
        tickers = data_api.get_actionable_stocks_list()
    else:
        tickers = h.get_tickers(from_folder=True)

    for ticker, interval in tqdm(
        [(ticker, interval) for ticker in tickers for interval in h.get_intervals(from_folder=True)]
    ):
        try:
            convert_ticker_files_to_parquet(ticker, interval)
        except Exception as e:
            print(e)
            print(f"{ticker} {interval} failed")


def delete_csv(ticker, interval):
    for file in h.get_ticker_files(ticker, interval):
        if os.path.exists(f"{h.RAW_DATA_FOLDER}/{ticker}/{interval}/{file.split('.')[0] + '.parquet'}"):
            os.remove(f"{h.RAW_DATA_FOLDER}/{ticker}/{interval}/{file}")


def delete_csvs(from_data_api=False):
    tickers = None
    if from_data_api is True:
        tickers = data_api.get_actionable_stocks_list()
    else:
        tickers = h.get_tickers(from_folder=True)

    for ticker, interval in tqdm(
        [(ticker, interval) for ticker in tickers for interval in h.get_intervals(from_folder=True)]
    ):
        try:
            delete_csv(ticker, interval)
        except Exception as e:
            print(e)
            print(f"{ticker} {interval} failed")


if __name__ == "__main__":
    combine_all_tickers(from_data_api=True)
