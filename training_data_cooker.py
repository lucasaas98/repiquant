# Standard Library
import os

# Third party dependencies
import pandas as pd
from joblib import Parallel, delayed

# Current project dependencies
import data_api
import engine
import helper as h


def label_data(ticker, interval, max_number_of_bars):
    dataframe = pd.read_csv(h.get_ticker_file(ticker, interval), index_col=0)
    closes = dataframe["close"].values
    results = dict()
    results["draw"] = list()
    results["win"] = list()
    results["loss"] = list()

    for i in range(len(dataframe) - max_number_of_bars):
        new_close_prices = closes[i:]
        for stop_loss in [1, 2, 3, 4, 5, 6, 8, 10]:
            for take_profit_ratio in [1.05, 1.1, 1.2, 1.5, 2, 3, 5]:
                (win, return_pct, bars_in_market) = engine.calculate_trade_outcome(
                    close_prices=new_close_prices,
                    stop_loss_pct=stop_loss,
                    take_profit_ratio=take_profit_ratio,
                    max_number_of_bars=max_number_of_bars,
                )
                if win is None:
                    results["draw"].append(i)
                elif win:
                    results["win"].append((i, return_pct, bars_in_market, stop_loss, take_profit_ratio))
                else:
                    results["loss"].append((i, return_pct, bars_in_market, stop_loss, take_profit_ratio))
    return results


def label_data_simpler(ticker, interval, max_number_of_bars):
    dataframe = pd.read_csv(h.get_ticker_file(ticker, interval), index_col=0)
    closes = dataframe["close"].values
    indexes = dataframe.index.tolist()

    closes_and_indexes = [(closes[(i + 1) :], indexes[i]) for i in range(len(dataframe) - max_number_of_bars - 1)]

    results = list()

    for [new_close_prices, new_index] in closes_and_indexes:
        for stop_loss in [1, 2, 3, 4, 5, 6, 8, 10]:
            for take_profit_ratio in [1.05, 1.1, 1.2, 1.5, 2, 3, 5]:
                (win, return_pct, bars_in_market) = engine.calculate_trade_outcome(
                    close_prices=new_close_prices,
                    stop_loss_pct=stop_loss,
                    take_profit_ratio=take_profit_ratio,
                    max_number_of_bars=max_number_of_bars,
                )
                if win is not None:
                    results.append((new_index, return_pct, bars_in_market, stop_loss, take_profit_ratio))

    return results


def calculate_trade_outcomes():
    all_tickers = h.get_tickers()
    all_intervals = h.get_intervals()

    for ticker in all_tickers:
        for interval in all_intervals:
            for max_number_of_bars in [5, 10, 20, 50, 60, 70, 100]:
                print(f"Calculating outcomes of {interval} for {ticker}. max_bars: {max_number_of_bars}")
                results = label_data(ticker, interval, max_number_of_bars)

                # analyze wins

                winner_stop_loss_sum = 0
                winner_take_profit_ratio = 0
                winner_time_in_market = 0

                most_profit = None
                print(f"\tnumber of wins: {len(results['win'])}")

                for win in results["win"]:
                    (i, win_return_pct, win_bars_in_market, win_stop_loss, win_take_profit_ratio) = win
                    winner_stop_loss_sum += win_stop_loss
                    winner_take_profit_ratio += win_take_profit_ratio
                    winner_time_in_market += win_bars_in_market
                    if not most_profit or most_profit[0] < win_return_pct:
                        most_profit = (win_return_pct, win_stop_loss, win_take_profit_ratio, win_bars_in_market)

                num_wins = len(results["win"])
                avg_winner_stop_loss = winner_stop_loss_sum / num_wins
                avg_winner_take_profit_ratio = winner_take_profit_ratio / num_wins
                avg_winner_time_in_market = winner_time_in_market / num_wins

                print("\tAverage winning stop_loss: ", avg_winner_stop_loss, "%")
                print("\tAverage winning take_profit_ratio: ", avg_winner_take_profit_ratio)
                print(
                    "\tAverage winning time in market: ",
                    avg_winner_time_in_market,
                    "bars",
                )
                print(
                    f"\tBest performance: {most_profit[0]}% -> sl: {most_profit[1]}, tpr: {most_profit[2]}, bars_in_market: {most_profit[3]}"
                )

                # analyze losses
                loser_stop_loss_sum = 0
                loser_take_profit_ratio = 0
                loser_time_in_market = 0

                least_profit = None
                print(f"\n\tnumber of losses: {len(results['loss'])}")

                for loss in results["loss"]:
                    (i, loss_return_pct, loss_bars_in_market, loss_stop_loss, loss_take_profit_ratio) = loss
                    loser_stop_loss_sum += loss_stop_loss
                    loser_take_profit_ratio += loss_take_profit_ratio
                    loser_time_in_market += loss_bars_in_market
                    if not least_profit or least_profit[0] > loss_return_pct:
                        least_profit = (loss_return_pct, loss_stop_loss, loss_take_profit_ratio, loss_bars_in_market)

                num_losses = len(results["loss"])
                avg_loser_stop_loss = loser_stop_loss_sum / num_losses
                avg_loser_take_profit_ratio = loser_take_profit_ratio / num_losses
                avg_loser_time_in_market = loser_time_in_market / num_losses

                print("\tAverage losing stop_loss: ", avg_loser_stop_loss, "%")
                print("\tAverage losing take_profit_ratio: ", avg_loser_take_profit_ratio)
                print("\tAverage losing time in market: ", avg_loser_time_in_market, "bars")
                print(
                    f"\tWorst performance: {least_profit[0]}% -> sl: {least_profit[1]}, tpr: {least_profit[2]}, bars_in_market: {least_profit[3]}"
                )

                print("\n\n")
            print("\n\n\n")


def for_loop_trade_results(results, ticker, interval):
    for max_number_of_bars in [5, 10, 20, 50, 60, 70, 100]:
        print(f"Calculating outcomes of {interval} for {ticker}. max_bars: {max_number_of_bars}")
        results.append((ticker, interval, max_number_of_bars, label_data_simpler(ticker, interval, max_number_of_bars)))


def calculate_all_trade_outcomes_to_dataframe():
    all_tickers = data_api.get_actionable_stocks_list()
    all_intervals = h.get_intervals()

    Parallel(n_jobs=4)(
        delayed(calculate_trade_outcomes_to_dataframe)(ticker, interval)
        for ticker in all_tickers
        for interval in all_intervals
    )


def calculate_trade_outcomes_to_dataframe(ticker, interval):
    if not os.path.exists(f"processed_data/{ticker}/{interval}"):
        os.makedirs(f"processed_data/{ticker}/{interval}")

    for max_number_of_bars in [5, 10, 20, 50, 60, 70, 100]:
        print(f"Calculating outcomes of {interval} for {ticker}. max_bars: {max_number_of_bars}")
        results = []

        results.append((ticker, interval, max_number_of_bars, label_data_simpler(ticker, interval, max_number_of_bars)))

        # Create an empty DataFrame
        tickers = list()
        intervals = list()
        max_bar_list = list()
        return_pcts = list()
        bars_in_market = list()
        stop_losses = list()
        take_profit_ratios = list()
        indexes = list()

        # Add data to the DataFrame
        for i, result_i in enumerate(results):
            ticker, interval, max_bars, result = result_i

            for trade in result:
                timestamp, trade_return_pct, trade_bars_in_market, trade_stop_loss, trade_take_profit_ratio = trade

                tickers.append(ticker)
                intervals.append(interval)
                max_bar_list.append(max_bars)
                return_pcts.append(trade_return_pct)
                bars_in_market.append(trade_bars_in_market)
                stop_losses.append(trade_stop_loss)
                take_profit_ratios.append(trade_take_profit_ratio)
                indexes.append(timestamp)

        trade_outcomes = {
            "ticker": tickers,
            "interval": interval,
            "max_bar": max_bar_list,
            "return_pct": return_pcts,
            "bars_in_market": bars_in_market,
            "stop_loss": stop_losses,
            "take_profit": take_profit_ratios,
        }
        df = pd.DataFrame(trade_outcomes, index=indexes)

        df.to_csv(h.get_trade_outcomes_file(ticker, interval, max_bar=50))


def create_labeled_dataset():
    all_tickers = h.get_tickers()
    all_intervals = h.get_intervals()

    for ticker in all_tickers:
        for interval in all_intervals:
            for max_number_of_bars in [5, 10, 20, 50, 60, 70, 100]:
                # Create a dataframe with all the trade outcomes
                df = pd.read_csv(h.get_trade_outcomes_file(ticker, interval, max_bar=50), index_col=0)
                df = df.reset_index(names=["datetime"])

                # Compile data for each max_bar and interval combination
                grouped_df = df.groupby(["ticker", "interval", "max_bar", "datetime"])

                # Initialize an empty list to store the labeled data
                labeled_data = []

                # Iterate over each group
                for name, group in grouped_df:
                    # Get the index for logging
                    idx = group.index.tolist()[0]

                    # Determine the label for each row in this group based on return_pct
                    labels = group.apply(lambda x: "bad" if x["return_pct"] < 0 else "good", axis=1)

                    try:
                        # Combine data from all rows in this group into a single dictionary for each index
                        for i, row in group.iterrows():
                            data_dict = {
                                "ticker": name[0],
                                "interval": name[1],
                                "max_bar": name[2],
                                "datetime": name[3],
                                "take_profit": row["take_profit"],
                                "bars_in_market": row["bars_in_market"],
                                "return_pct": row["return_pct"],
                                "label": labels.loc[i],
                            }

                            # Append the dictionary to the list
                            labeled_data.append(data_dict)
                    except Exception as e:
                        print("i:", i)
                        print("row:", row)
                        print("error:", e)

                        print(f"{name} - {idx} - {name[3]}")
                        exit()

                    df = pd.DataFrame(labeled_data)
                    df.to_csv(h.get_labeled_outcomes(ticker, interval, max_number_of_bars))


if __name__ == "__main__":
    # create_labeled_dataset()
    calculate_all_trade_outcomes_to_dataframe()
    # label_data_simpler("TSLA", "5min", 50)
