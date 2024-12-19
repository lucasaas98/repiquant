# Standard Library
from time import time

# Third party dependencies
# import pandas as pd
import polars as pl
from joblib import Parallel, delayed, dump, load
from sklearn.preprocessing import MinMaxScaler

# Current project dependencies
import data_api
import file_combiner as fc
import helper as h


def create_final_vector(df, for_training=True):
    bars_back_values = [1, 2, 3, 4, 10, 15]

    for abs_bars in bars_back_values:
        df = df.with_columns(
            ((pl.col("close").shift(abs_bars) - pl.col("close")) / pl.col("close")).alias(f"{abs_bars}b_before_close"),
            ((pl.col("macd").shift(abs_bars) - pl.col("macd")) / pl.col("macd")).alias(f"{abs_bars}b_before_macd"),
            ((pl.col("macd_signal").shift(abs_bars) - pl.col("macd_signal")) / pl.col("macd_signal")).alias(
                f"{abs_bars}b_before_macd_signal"
            ),
            ((pl.col("macd_hist").shift(abs_bars) - pl.col("macd_hist")) / pl.col("macd_hist")).alias(
                f"{abs_bars}b_before_macd_hist"
            ),
            ((pl.col("rsi").shift(abs_bars) - pl.col("rsi")) / pl.col("rsi")).alias(f"{abs_bars}b_before_rsi"),
            ((pl.col("ema1").shift(abs_bars) - pl.col("ema1")) / pl.col("ema1")).alias(f"{abs_bars}b_before_ema1"),
            ((pl.col("ema2").shift(abs_bars) - pl.col("ema2")) / pl.col("ema2")).alias(f"{abs_bars}b_before_ema2"),
            ((pl.col("sma").shift(abs_bars) - pl.col("sma")) / pl.col("sma")).alias(f"{abs_bars}b_before_sma"),
            ((pl.col("vwap").shift(abs_bars) - pl.col("vwap")) / pl.col("vwap")).alias(f"{abs_bars}b_before_vwap"),
        )

    df = df.select(
        pl.exclude(
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "macd",
                "macd_signal",
                "macd_hist",
                "rsi",
                "ema1",
                "ema2",
                "sma",
                "vwap",
            ]
        )
    ).with_columns(pl.col("*").replace([float("inf"), float("-inf")], None))

    return df.drop_nulls()


def create_training():
    all_tickers = data_api.get_actionable_stocks_list()
    all_intervals = h.get_intervals()

    Parallel(n_jobs=16)(
        delayed(create_data_vector)(ticker, interval) for ticker in all_tickers for interval in all_intervals
    )


def create_data_vector(ticker, interval):
    print(f"Creating data vector for {ticker} at {interval}")
    df = pl.read_parquet(h.get_ticker_file_parquet(ticker, interval))
    new_df = create_final_vector(df)
    new_df.write_parquet(h.get_previous_combined_parquet(ticker, interval))


def scale_all_data():
    print("Scaling data...")
    all_tickers = data_api.get_actionable_stocks_list()
    all_intervals = h.get_intervals()

    scaler = MinMaxScaler(feature_range=(-1, 1))

    print("Creating the scaler and fitting data.")
    counter = 0
    for ticker in all_tickers:
        for interval in all_intervals:
            numpy_array = (
                pl.read_parquet(h.get_previous_combined_parquet(ticker, interval))
                .select(pl.exclude("datetime"))
                .to_numpy()
            )
            model = scaler.fit(numpy_array)
            counter += 1
            print(f"Fitted {counter} of {len(all_tickers)*len(all_intervals)}")
    print("Done fitting data.")

    print("Transforming the data...")
    counter = 0
    for ticker in all_tickers:
        for interval in all_intervals:
            df = pl.read_parquet(h.get_previous_combined_parquet(ticker, interval))
            with_exclusion = df.select(pl.exclude("datetime"))
            scaled_data = model.transform(with_exclusion.to_numpy())
            scaled_df = pl.from_numpy(scaled_data, schema=with_exclusion.columns, orient="row")
            scaled_df.insert_column(0, df.select(pl.col("datetime")).to_series())
            scaled_df.write_parquet(h.get_scaled_previous_combined_parquet(ticker, interval))
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

    model = load(f"models/scalers/{scaler}.gzip")

    print("Scaling the data...")
    counter = 0
    for ticker in all_tickers:
        for interval in all_intervals:
            df = pl.read_parquet(h.get_previous_combined_parquet(ticker, interval))
            with_exclusion = df.select(pl.exclude("datetime"))
            scaled_data = model.transform(with_exclusion.to_numpy())
            scaled_df = pl.from_numpy(scaled_data, schema=with_exclusion.columns, orient="row")
            scaled_df.insert_column(0, df.select(pl.col("datetime")).to_series())
            scaled_df.write_parquet(h.get_scaled_previous_combined_parquet(ticker, interval))
            counter += 1
            print(f"Scaled {counter} of {len(all_tickers)*len(all_intervals)}")
    print("Done scaling data.")


def create_labels_for_all_bars(short=False):
    all_tickers = data_api.get_actionable_stocks_list()
    # all_intervals = h.get_intervals()

    print("Creating labels for bars...")

    Parallel(n_jobs=16)(
        delayed(create_labels_for_each_bar)(ticker, interval, short) for ticker in all_tickers for interval in ["5min"]
    )


def create_labels_for_all_bars_both_sides():
    all_tickers = data_api.get_actionable_stocks_list()
    all_intervals = h.get_intervals()

    print("Creating labels for bars...")

    Parallel(n_jobs=16)(
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
    label_returner = label_return_short if short else label_return
    trade_outcomes = pl.read_parquet(h.get_trade_outcomes_file_parquet(ticker, interval, max_number_of_bars))

    df = trade_outcomes.sql(
        """
        SELECT
            datetime,
            AVG(return_pct) as avg_return_pct,
            AVG(stop_loss) as avg_stoploss,
            AVG(bars_in_market) as avg_bars_in_market,
            AVG(take_profit) as avg_takeprofit
        FROM self
        GROUP BY "datetime"
    """
    ).with_columns(
        pl.col("avg_return_pct").map_elements(lambda pct: label_returner(pct), return_dtype=pl.String).alias("label"),
    )

    scaled_data = pl.read_parquet(h.get_scaled_previous_combined_parquet(ticker, interval))

    result = scaled_data.join(df, on="datetime", how="left").with_columns(
        label=pl.when(pl.col("label").is_null()).then(pl.lit("noop")).otherwise(pl.col("label")),
        avg_return_pct=pl.when(pl.col("avg_return_pct").is_null())
        .then(pl.lit(0.0))
        .otherwise(pl.col("avg_return_pct")),
        avg_stoploss=pl.when(pl.col("avg_stoploss").is_null()).then(pl.lit(0.0)).otherwise(pl.col("avg_stoploss")),
        avg_bars_in_market=pl.when(pl.col("avg_bars_in_market").is_null())
        .then(pl.lit(0.0))
        .otherwise(pl.col("avg_bars_in_market")),
        avg_takeprofit=pl.when(pl.col("avg_takeprofit").is_null())
        .then(pl.lit(0.0))
        .otherwise(pl.col("avg_takeprofit")),
    )

    result.write_parquet(h.get_scaled_labeled_parquet(ticker, interval, short=short))

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
    trade_outcomes = pl.read_parquet(h.get_trade_outcomes_file_parquet(ticker, interval, max_number_of_bars))

    df = trade_outcomes.sql(
        """
        SELECT
            datetime,
            AVG(return_pct) as avg_return_pct,
            AVG(stop_loss) as avg_stoploss,
            AVG(bars_in_market) as avg_bars_in_market,
            AVG(take_profit) as avg_takeprofit
        FROM self
        GROUP BY "datetime"
    """
    ).with_columns(
        pl.col("avg_return_pct")
        .map_elements(lambda pct: label_return_both_sides(pct), return_dtype=pl.String)
        .alias("label"),
    )

    scaled_data = pl.read_parquet(h.get_scaled_previous_combined_parquet(ticker, interval))

    result = scaled_data.join(df, on="datetime", how="left").with_columns(
        label=pl.when(pl.col("label").is_null()).then(pl.lit("noop")).otherwise(pl.col("label")),
        avg_return_pct=pl.when(pl.col("avg_return_pct").is_null())
        .then(pl.lit(0.0))
        .otherwise(pl.col("avg_return_pct")),
        avg_stoploss=pl.when(pl.col("avg_stoploss").is_null()).then(pl.lit(0.0)).otherwise(pl.col("avg_stoploss")),
        avg_bars_in_market=pl.when(pl.col("avg_bars_in_market").is_null())
        .then(pl.lit(0.0))
        .otherwise(pl.col("avg_bars_in_market")),
        avg_takeprofit=pl.when(pl.col("avg_takeprofit").is_null())
        .then(pl.lit(0.0))
        .otherwise(pl.col("avg_takeprofit")),
    )

    result.write_parquet(h.get_scaled_labeled_both_sides_parquet(ticker, interval))
    print(f"{ticker} {interval} done!")


if __name__ == "__main__":
    create_labels_for_all_bars()
