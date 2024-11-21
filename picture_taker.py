# Standard Library
import datetime
import os
import time

# Current project dependencies
import data_api
import engine
import helper as h


def get_bar_data(ticker="TSLA", interval="5min", per_page=5000, start_date=None, end_date=None):
    dataframe = data_api.get_time_series(
        symbol=ticker, interval=interval, outputsize=per_page, start_date=start_date, end_date=end_date
    )
    return dataframe


def get_min_timestamp(ticker, interval):
    all_files = sorted(h.get_ticker_files(ticker, interval))
    try:
        return min([int(file.split("_")[0]) for file in all_files])
    except Exception:
        return 999999999999999


def get_max_timestamp(ticker, interval):
    all_files = sorted(h.get_ticker_files(ticker, interval))
    try:
        return max([int(file.split("_")[0]) for file in all_files])
    except Exception as e:
        print(e)
        return -999999999999999


def get_extremes_timestamps(dataframe):
    dates = dataframe.index.to_list()
    min_timestamp = int(dates[-1].timestamp())
    max_timestamp = int(dates[0].timestamp())
    return min_timestamp, max_timestamp


def get_all_data():
    for interval in h.get_intervals():
        print(f"Fetching data for bar size: {interval}")
        for ticker in data_api.get_stocks_list():
            print(f"{ticker}")
            flag = True
            if not os.path.exists(f"raw_data/{ticker}/{interval}"):
                print("\tFirst run! Creating folder!")
                os.makedirs(f"raw_data/{ticker}/{interval}")
            min_timestamp = get_min_timestamp(ticker, interval)
            while min_timestamp > (time.time() - 86400 * 730) and flag:
                try:
                    missing_days = (min_timestamp - (time.time() - 86400 * 730)) / 86400
                    print(f"\tMissing {missing_days} days of data for {ticker} on {interval} bar")
                    if min_timestamp == 999999999999999:
                        dataframe = get_bar_data(ticker=ticker, interval=interval)
                    else:
                        end_date = datetime.datetime.fromtimestamp(min_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        dataframe = get_bar_data(ticker=ticker, interval=interval, end_date=end_date)
                    min_timestamp, max_timestamp = get_extremes_timestamps(dataframe)
                    min_timestamp_date = datetime.datetime.fromtimestamp(min_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    max_timestamp_date = datetime.datetime.fromtimestamp(max_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\t\tMin timestamp {min_timestamp} ({min_timestamp_date})")
                    print(f"\t\tMax timestamp {max_timestamp} ({max_timestamp_date})")
                    dataframe.to_csv(f"raw_data/{ticker}/{interval}/{min_timestamp}_{max_timestamp}.csv")
                except Exception as e:
                    print(e)
                    if "greater value than the number of historical data points" in str(e):
                        print("\tIt's less than 2 years old.")
                        flag = False
                    elif "Data not found" in str(e):
                        print("\tIt's fucked in the API.")
                        flag = False
                    else:
                        print("\t\texited")
                        exit()
                finally:
                    print("Sleeping to avoid rate limiting")
                    time.sleep(62)
            print("\tWe already fetched all the data for the ticker!")


if __name__ == "__main__":
    while True:
        try:
            get_all_data()
        except Exception as e:
            print("Error: ", e)
