# Future imports
from __future__ import absolute_import, division, print_function, unicode_literals

# Standard Library
import os.path

# Third party dependencies
import backtrader as bt
import pandas as pd

# Current project dependencies
import helper as h

TICKER = "TSLA"
INTERVAL = "1h"


class DataTranslator:
    def __init__(self, ticker, interval):
        self.data = None
        self.ticker = ticker
        self.interval = interval
        self.data_file = h.get_scaled_previous_combined(ticker, interval)
        self.load_data()

    def load_data(self):
        self.data = pd.read_csv(self.data_file, index_col=0)

    def get_data_for_bar(self, timestamp):
        real_index = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return self.data.loc[real_index]


class RepiStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        """Logging function fot this strategy"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.translator = DataTranslator(TICKER, INTERVAL)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    "SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log("Close, %.2f" % self.dataclose[0])
        # self.log("Volume, %.2f" % self.datavolume[0])
        # self.log(f"datetime: {self.datas[0].datetime.time()}")
        data_array = self.translator.get_data_for_bar(self.datas[0].datetime.datetime(0)).to_numpy()
        print(type(data_array))

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] < self.dataclose[-1]:
                # current close less than previous close

                if self.dataclose[-1] < self.dataclose[-2]:
                    # previous close less than the previous close

                    # BUY, BUY, BUY!!! (with default parameters)
                    self.log("BUY CREATE, %.2f" % self.dataclose[0])

                    # Keep track of the created order to avoid a 2nd order
                    self.order = self.buy()

        else:

            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + 5):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log("SELL CREATE, %.2f" % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()


if __name__ == "__main__":
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(RepiStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    datapath = os.path.join(h.RAW_DATA_FOLDER, TICKER, INTERVAL, "all_current.csv")

    # Create a Data Feed
    data = bt.feeds.GenericCSVData(dataname=datapath, timeframe=bt.TimeFrame.Minutes, compression=60)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Print out the starting conditions
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
