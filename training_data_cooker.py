# Standard Library
import os

# Third party dependencies
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

# Current project dependencies
import data_api
import engine
import helper as h


def classify_trade(close, trade_take_profit_price, trade_stop_loss_price, shorting):
    result = None
    if close >= trade_take_profit_price and not shorting:
        result = True
    elif close <= trade_stop_loss_price and not shorting:
        result = False
    elif close <= trade_take_profit_price and shorting:
        result = True
    elif close >= trade_stop_loss_price and shorting:
        result = False
    return result


def label_data_simpler(ticker, interval, max_number_of_bars):
    dataframe = pl.read_parquet(h.get_ticker_file_parquet(ticker, interval), columns=["datetime", "close"])

    numpy_stuff = engine.shift_return_pct(dataframe).to_numpy()
    results = dict()
    results["datetime"] = list()
    results["sl"] = list()
    results["tp"] = list()
    results["bars_in_market"] = list()
    results["win_loss"] = list()
    results["return_pct"] = list()
    for sl, tp, row in [
        (sl, tp, row)
        for sl in [0.5, 1, 2, 3, 4, 5]
        for tp in [
            1.01,
            1.02,
            1.03,
            1.04,
            1.05,
            1.06,
            1.07,
            1.08,
            1.09,
            1.10,
            -1.01,
            -1.02,
            -1.03,
            -1.04,
            -1.05,
            -1.06,
            -1.07,
            -1.08,
            -1.09,
            -1.10,
        ]
        for row in numpy_stuff
    ]:
        shorting = tp < 0
        trade_stop_loss_price = row[1] / (1 - sl / 100) if shorting else row[1] * (1 - sl / 100)
        trade_take_profit_price = row[1] / tp if shorting else row[1] * tp

        for i in range(50):
            if row[i + 2]:
                trade_class = classify_trade(row[i + 2], trade_take_profit_price, trade_stop_loss_price, shorting)
                if trade_class is not None:
                    results["datetime"].append(row[0])
                    results["sl"].append(sl)
                    results["tp"].append(tp)
                    results["bars_in_market"].append(i)
                    results["win_loss"].append(trade_class)
                    results["return_pct"].append((((row[i + 2] - row[1]) / row[1]) * 100))
                    break
    return pl.from_dict(results)


def calculate_all_trade_outcomes_to_dataframe():
    all_tickers = data_api.get_actionable_stocks_list()
    # all_intervals = h.get_intervals()

    Parallel(n_jobs=8)(
        delayed(calculate_trade_outcomes_to_dataframe)(ticker, interval)
        for ticker in all_tickers
        for interval in ["5min"]
    )


def calculate_trade_outcomes_to_dataframe(ticker, interval):
    if not os.path.exists(f"processed_data/{ticker}/{interval}"):
        os.makedirs(f"processed_data/{ticker}/{interval}")

    for max_number_of_bars in [50]:
        print(f"Calculating outcomes of {interval} for {ticker}. max_bars: {max_number_of_bars}")
        df = (
            label_data_simpler(ticker, interval, max_number_of_bars)
            .rename({"sl": "stop_loss"})
            .rename({"tp": "take_profit"})
        )

        df.write_parquet(h.get_trade_outcomes_file_parquet(ticker, interval, max_bar=50))


if __name__ == "__main__":
    calculate_all_trade_outcomes_to_dataframe()
