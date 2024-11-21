# Standard Library
from time import time

# Third party dependencies
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Current project dependencies
import data_api
import file_combiner as fc
import helper as h


def current_data(ticker, interval):
    raw_current_data = data_api.get_time_series(symbol=ticker, interval=interval, outputsize=20)
    processed_current_data = create_final_vector(raw_current_data, for_training=False)
    return scale_one(processed_current_data)


def create_final_vector(df, for_training=True):
    new_df = pd.DataFrame()

    for bars_back in [-1, -2, -3, -4, -10, -15]:
        abs_bars = abs(bars_back)
        bars_back = bars_back if for_training else bars_back + 1
        new_df[f"{abs_bars}b_before_close"] = (df["close"].shift(bars_back) - df["close"]) / df["close"]
        new_df[f"{abs_bars}b_before_macd"] = (df["macd"].shift(bars_back) - df["macd"]) / df["macd"]
        new_df[f"{abs_bars}b_before_macd_signal"] = (df["macd_signal"].shift(bars_back) - df["macd_signal"]) / df[
            "macd_signal"
        ]
        new_df[f"{abs_bars}b_before_macd_hist"] = (df["macd_hist"].shift(bars_back) - df["macd_hist"]) / df["macd_hist"]
        new_df[f"{abs_bars}b_before_rsi"] = (df["rsi"].shift(bars_back) - df["rsi"]) / df["rsi"]
        new_df[f"{abs_bars}b_before_ema1"] = (df["ema1"].shift(bars_back) - df["ema1"]) / df["ema1"]
        new_df[f"{abs_bars}b_before_ema2"] = (df["ema2"].shift(bars_back) - df["ema2"]) / df["ema2"]
        new_df[f"{abs_bars}b_before_sma"] = (df["sma"].shift(bars_back) - df["sma"]) / df["sma"]
        new_df[f"{abs_bars}b_before_vwap"] = (df["vwap"].shift(bars_back) - df["vwap"]) / df["vwap"]
    new_df.index = df.index
    new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    new_df.dropna(inplace=True)

    return new_df


def create_training():
    all_tickers = h.get_tickers()
    all_intervals = h.get_intervals()

    for ticker in all_tickers:
        for interval in all_intervals:
            df = pd.read_csv(fc.get_ticker_file(ticker, interval), index_col=0)
            new_df = create_final_vector(df)
            new_df.to_csv(h.get_previous_combined(ticker, interval))
    return new_df


def scale_all_data():
    print("Scaling data...")
    all_tickers = h.get_tickers()
    all_intervals = h.get_intervals()

    scaler = MinMaxScaler(feature_range=(-1, 1))

    print("Creating the scaler and fitting data.")
    counter = 0
    for ticker in all_tickers:
        for interval in all_intervals:
            df = pd.read_csv(h.get_previous_combined(ticker, interval), index_col=0)
            numpy_array = df.to_numpy()
            model = scaler.fit(numpy_array)
            counter += 1
            print(f"Fitted {counter} of {len(all_tickers)*len(all_intervals)}")
    print("Done fitting data.")

    print("Transforming the data...")
    counter = 0
    for ticker in all_tickers:
        for interval in all_intervals:
            df = pd.read_csv(h.get_previous_combined(ticker, interval), index_col=0)
            numpy_array = df.to_numpy()
            scaled_data = model.transform(numpy_array)
            scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
            scaled_df.to_csv(h.get_scaled_previous_combined(ticker, interval))
            counter += 1
            print(f"Scaled {counter} of {len(all_tickers)*len(all_intervals)}")
    print("Done scaling data.")

    print("Storing the scaler model...")
    dump(h.get_new_scaler())
    print("Done storing the scaler model.")

    print("Finished.")


def scale_one(df):
    scaler = load(h.get_latest_scaler())
    numpy_array = df.to_numpy()
    scaled_data = scaler.transform(numpy_array)
    return pd.Dataframe(scaled_data, columns=df.columns, index=df.index)


def create_labels_for_all_bars():
    all_tickers = h.get_tickers()
    all_intervals = h.get_intervals()

    print("Creating labels for bars...")

    Parallel(n_jobs=4)(
        delayed(create_labels_for_each_bar)(ticker, interval) for ticker in all_tickers for interval in all_intervals
    )


def create_labels_for_each_bar(ticker, interval):
    max_number_of_bars = 50

    print(f"{ticker} {interval} starting...")
    trade_outcomes = pd.read_csv(h.get_trade_outcomes_file(ticker, interval, max_number_of_bars), index_col=0)

    trade_outcomes = trade_outcomes.reset_index(names=["datetime"])
    grouped_trade_outcomes = trade_outcomes.groupby(["ticker", "interval", "max_bar", "datetime"])

    labels = {}

    for name, group in grouped_trade_outcomes:
        return_pcts = list()
        stoplosses = list()
        take_profits = list()
        bars_in_market = list()

        for i, row in group.iterrows():
            return_pcts.append(row["return_pct"])
            stoplosses.append(row["stop_loss"])
            bars_in_market.append(row["bars_in_market"])
            take_profits.append(row["take_profit"])

        avg_return_pct = sum(return_pcts) / len(return_pcts)
        avg_stoploss = sum(stoplosses) / len(stoplosses)
        avg_bars_in_market = sum(bars_in_market) / len(bars_in_market)
        avg_takeprofit = sum(take_profits) / len(take_profits)

        labels[name[3]] = (
            "bad" if avg_return_pct < 0 else "good",
            avg_stoploss,
            avg_bars_in_market,
            avg_takeprofit,
        )

    scaled_data = pd.read_csv(h.get_scaled_previous_combined(ticker, interval), index_col=0)
    scaled_data = scaled_data.reset_index(names=["datetime"])

    def label_function(x):
        if x["datetime"] in labels.keys():
            return labels[x["datetime"]]
        else:
            return ("noop", 0, 0, 0)

    scaled_data["label"] = scaled_data.apply(lambda x: label_function(x)[0], axis=1)
    scaled_data["avg_stoploss"] = scaled_data.apply(lambda x: label_function(x)[1], axis=1)
    scaled_data["avg_bars_in_market"] = scaled_data.apply(lambda x: label_function(x)[2], axis=1)
    scaled_data["avg_takeprofit"] = scaled_data.apply(lambda x: label_function(x)[3], axis=1)

    scaled_data.to_csv(h.get_scaled_labeled(ticker, interval, max_number_of_bars), index=False)

    print(f"{ticker} {interval} done!")


if __name__ == "__main__":
    create_labels_for_all_bars()

    # df = pd.read_csv("test_train_data.csv", index_col=0)
    # print(df.head())
    # scaled_np_array = scale_stuff_2(df)
    # print(scaled_np_array)

    # scaled_df = pd.DataFrame(scaled_np_array, columns=df.columns, index=df.index)
    # print(scaled_df.head())

    # scaled_df.to_csv("scaled_test_train_data.csv")
