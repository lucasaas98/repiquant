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

# def current_data(ticker, interval):
#     raw_current_data = data_api.get_time_series(symbol=ticker, interval=interval, outputsize=20)
#     processed_current_data = create_final_vector(raw_current_data, for_training=False)
#     return scale_one(processed_current_data)


def create_final_vector(df, for_training=True):
    new_df = pd.DataFrame()

    for bars_back in [1, 2, 3, 4, 10, 15]:
        abs_bars = abs(bars_back)
        # bars_back = bars_back if for_training else bars_back + 1
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
    all_tickers = data_api.get_actionable_stocks_list()
    all_intervals = h.get_intervals()

    Parallel(n_jobs=4)(
        delayed(create_data_vector)(ticker, interval) for ticker in all_tickers for interval in all_intervals
    )


def create_data_vector(ticker, interval):
    print(f"Creating data vector for {ticker} at {interval}")
    df = pd.read_parquet(h.get_ticker_file_parquet(ticker, interval), engine="fastparquet")
    new_df = create_final_vector(df)
    new_df.to_parquet(h.get_previous_combined_parquet(ticker, interval))


def scale_all_data():
    print("Scaling data...")
    all_tickers = data_api.get_actionable_stocks_list()
    all_intervals = h.get_intervals()

    scaler = MinMaxScaler(feature_range=(-1, 1))

    print("Creating the scaler and fitting data.")
    counter = 0
    for ticker in all_tickers:
        for interval in all_intervals:
            df = pd.read_parquet(h.get_previous_combined_parquet(ticker, interval), engine="fastparquet")
            numpy_array = df.to_numpy()
            model = scaler.fit(numpy_array)
            counter += 1
            print(f"Fitted {counter} of {len(all_tickers)*len(all_intervals)}")
    print("Done fitting data.")

    print("Transforming the data...")
    counter = 0
    for ticker in all_tickers:
        for interval in all_intervals:
            df = pd.read_parquet(h.get_previous_combined_parquet(ticker, interval), engine="fastparquet")
            numpy_array = df.to_numpy()
            scaled_data = model.transform(numpy_array)
            scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
            scaled_df.to_parquet(h.get_scaled_previous_combined_parquet(ticker, interval), engine="fastparquet")
            counter += 1
            print(f"Scaled {counter} of {len(all_tickers)*len(all_intervals)}")
    print("Done scaling data.")

    print("Storing the scaler model...")
    dump(model, h.get_new_scaler())
    print("Done storing the scaler model.")

    print("Finished.")


def scale_using_previous_scaler(scaler):
    print("Scaling data...")
    all_tickers = data_api.get_actionable_stocks_list()
    all_intervals = h.get_intervals()

    model = load("models/scalers/1733583214_scaler.gzip")

    print("Scaling the data...")
    counter = 0
    for ticker in all_tickers:
        for interval in all_intervals:
            df = pd.read_parquet(h.get_previous_combined_parquet(ticker, interval), engine="fastparquet")
            numpy_array = df.to_numpy()
            scaled_data = model.transform(numpy_array)
            scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
            scaled_df.to_parquet(h.get_scaled_previous_combined_parquet(ticker, interval), engine="fastparquet")
            counter += 1
            print(f"Scaled {counter} of {len(all_tickers)*len(all_intervals)}")
    print("Done scaling data.")


def scale_one(df):
    scaler = load(h.get_latest_scaler())
    numpy_array = df.to_numpy()
    scaled_data = scaler.transform(numpy_array)
    return pd.Dataframe(scaled_data, columns=df.columns, index=df.index)


def create_labels_for_all_bars(short=False):
    all_tickers = data_api.get_actionable_stocks_list()
    # all_intervals = h.get_intervals()

    print("Creating labels for bars...")

    Parallel(n_jobs=4)(
        delayed(create_labels_for_each_bar)(ticker, interval, short) for ticker in all_tickers for interval in ["5min"]
    )


def create_labels_for_all_bars_both_sides():
    all_tickers = data_api.get_actionable_stocks_list()
    all_intervals = h.get_intervals()

    print("Creating labels for bars...")

    Parallel(n_jobs=4)(
        delayed(create_labels_for_each_bar_both_sides)(ticker, interval)
        for ticker in all_tickers
        for interval in all_intervals
    )


def label_return(return_pct):
    if return_pct > 5:
        return "very good"
    elif return_pct < -5:
        return "very bad"
    else:
        return "noop"


def label_return_short(return_pct):
    if return_pct < -5:
        return "very bad"
    elif return_pct < -2:
        return "bad"
    else:
        return "noop"


def create_labels_for_each_bar(ticker, interval, short=False):
    max_number_of_bars = 50

    print(f"{ticker} {interval} starting...")
    trade_outcomes = pd.read_parquet(
        h.get_trade_outcomes_file_parquet(ticker, interval, max_number_of_bars), engine="fastparquet"
    )

    trade_outcomes = trade_outcomes.reset_index(names=["datetime"])
    grouped_trade_outcomes = trade_outcomes.groupby(["ticker", "interval", "max_bar", "datetime"])

    labels = {}

    label_returner = label_return_short if short else label_return

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
        # max_bars_in_market = max(bars_in_market)
        avg_takeprofit = sum(take_profits) / len(take_profits)

        labels[name[3]] = (
            label_returner(avg_return_pct),
            avg_return_pct,
            avg_stoploss,
            avg_bars_in_market,
            avg_takeprofit,
        )

    scaled_data = pd.read_parquet(h.get_scaled_previous_combined_parquet(ticker, interval), engine="fastparquet")
    scaled_data = scaled_data.reset_index(names=["datetime"])

    def label_function(x):
        if x["datetime"] in labels.keys():
            return labels[x["datetime"]]
        else:
            return ("noop", 0, 0, 0, 0)

    scaled_data["label"] = scaled_data.apply(lambda x: label_function(x)[0], axis=1)
    scaled_data["avg_return_pct"] = scaled_data.apply(lambda x: label_function(x)[1], axis=1)
    scaled_data["avg_stoploss"] = scaled_data.apply(lambda x: label_function(x)[2], axis=1)
    scaled_data["avg_bars_in_market"] = scaled_data.apply(lambda x: label_function(x)[3], axis=1)
    scaled_data["avg_takeprofit"] = scaled_data.apply(lambda x: label_function(x)[4], axis=1)

    scaled_data.to_parquet(
        h.get_scaled_labeled_parquet(ticker, interval, short=short), index=False, engine="fastparquet"
    )

    print(f"{ticker} {interval} done!")


def label_return_both_sides(return_pct):
    if return_pct < -5:
        return "very bad"
    elif return_pct < -2:
        return "bad"
    elif return_pct > 5:
        return "very good"
    elif return_pct > 2:
        return "good"
    else:
        return "noop"


def create_labels_for_each_bar_both_sides(ticker, interval):
    max_number_of_bars = 50

    print(f"{ticker} {interval} starting...")
    trade_outcomes = pd.read_parquet(
        h.get_trade_outcomes_file_parquet(ticker, interval, max_number_of_bars), engine="fastparquet"
    )

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
        # max_bars_in_market = max(bars_in_market)
        avg_takeprofit = sum(take_profits) / len(take_profits)

        labels[name[3]] = (
            label_return_both_sides(avg_return_pct),
            avg_return_pct,
            avg_stoploss,
            avg_bars_in_market,
            avg_takeprofit,
        )

    scaled_data = pd.read_parquet(h.get_scaled_previous_combined_parquet(ticker, interval), engine="fastparquet")
    scaled_data = scaled_data.reset_index(names=["datetime"])

    def label_function(x):
        if x["datetime"] in labels.keys():
            return labels[x["datetime"]]
        else:
            return ("noop", 0, 0, 0, 0)

    scaled_data["label"] = scaled_data.apply(lambda x: label_function(x)[0], axis=1)
    scaled_data["avg_return_pct"] = scaled_data.apply(lambda x: label_function(x)[1], axis=1)
    scaled_data["avg_stoploss"] = scaled_data.apply(lambda x: label_function(x)[2], axis=1)
    scaled_data["avg_bars_in_market"] = scaled_data.apply(lambda x: label_function(x)[3], axis=1)
    scaled_data["avg_takeprofit"] = scaled_data.apply(lambda x: label_function(x)[4], axis=1)

    scaled_data.to_parquet(h.get_scaled_labeled_both_sides_parquet(ticker, interval), index=False, engine="fastparquet")

    print(f"{ticker} {interval} done!")


if __name__ == "__main__":
    create_labels_for_all_bars()
