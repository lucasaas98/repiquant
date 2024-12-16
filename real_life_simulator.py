# Future imports
from __future__ import absolute_import, division, print_function, unicode_literals

# Standard Library
import os.path
from random import random

# Third party dependencies
import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd

# Current project dependencies
import env
import helper as h
from consts import INTERVAL, MULTIPLE_FEED_TICKERS, TICKER

# from strategies.cnn_strategy import CNNStrategy
# from strategies.more_classes_strategy import MoreClassesStrategy
# from strategies.multiple_feeds_strategy import MultipleFeedsStrategy
# from strategies.repi_strategy import RepiStrategy
from strategies.short_and_long_strategy import ShortAndLongStrategy

# from strategies.binary_model_strategy import BinaryModelStrategy


class PandasData(bt.feed.DataBase):
    params = (
        ("datetime", -1),
        ("open", -1),
        ("high", -1),
        ("low", -1),
        ("close", -1),
        ("volume", -1),
        ("macd", -1),
        ("macd_signal", -1),
        ("macd_hist", -1),
        ("rsi", -1),
        ("ema1", -1),
        ("ema2", -1),
        ("sma", -1),
        ("vwap", -1),
    )


def run_multifeed_strategy(strategy):
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(strategy)

    for ticker in MULTIPLE_FEED_TICKERS:
        datapath = os.path.join(h.RAW_DATA_FOLDER, ticker, INTERVAL, "all_current.parquet")
        df = pd.read_parquet(datapath, engine="fastparquet")
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data, name=ticker)

    # Set our desired cash start
    cerebro.broker.setcash(30000.0)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=10)

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(btanalyzers.Returns, _name="returns")
    cerebro.addanalyzer(btanalyzers.Transactions, _name="trans")

    # Print out the starting conditions
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Run over everything
    back = cerebro.run()

    # Print out the final result
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    print(f"Sharpe Ratio: {back[0].analyzers.sharpe.get_analysis()['sharperatio']}")
    print(f"Transactions: {back[0].analyzers.trans.get_analysis()}")
    print(f"Returns: {back[0].analyzers.returns.get_analysis()['rnorm100']}")

    cerebro.plot()


def run_strategy(strategy):
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(strategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    datapath = os.path.join(h.RAW_DATA_FOLDER, TICKER, INTERVAL, "all_current.parquet")
    df = pd.read_parquet(datapath, engine="fastparquet")
    data = PandasData(dataname=df)
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(30000.0)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=10)

    # cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe")
    # cerebro.addanalyzer(btanalyzers.Returns, _name="returns")
    # cerebro.addanalyzer(btanalyzers.Transactions, _name="trans")

    # Print out the starting conditions
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # # Print out the final result
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    # print(f"Sharpe Ratio: {back[0].analyzers.sharpe.get_analysis()['sharperatio']}")
    # print(f"Returns: {back[0].analyzers.returns.get_analysis()['rnorm100']}")
    # print(f"Transactions: \n{back[0].analyzers.trans.get_analysis()}\n")

    # cerebro.plot()


if __name__ == "__main__":
    strategy = ShortAndLongStrategy
    run_strategy(strategy)
