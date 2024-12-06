# Standard Library
from random import random

# Third party dependencies
import backtrader as bt

# Current project dependencies
from consts import INTERVAL, MULTIPLE_FEED_TICKERS, TICKER
from helper import DataTranslator


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
        self.max_time_in_trade = 28
        self.time_in_trade = 0
        self.stop_loss_pct = 5
        self.take_profit_ratio = 1.06

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
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # data_array = self.translator.get_increments_for_bar(self.datas[0].datetime.datetime(0)).to_numpy()
        close = self.translator.get_close_for_bar(self.datas[0].datetime.datetime(0)).to_numpy()[3]
        self.log("Close, %.2f" % close)

        # Check if we are in the market
        if not self.position and random() > 0.5:

            # BUY, BUY, BUY!!! (with default parameters)
            self.log("BUY CREATE, %.2f" % close)

            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy()
            self.stop_loss_price = close * (1 - self.stop_loss_pct / 100)
            self.take_profit_price = close * self.take_profit_ratio
        elif self.position:
            # 28 bars in the market and we sell, 1.06 profit ratio and we sell, 5% down and we sell
            if (
                (len(self) >= (self.bar_executed + 28))
                or (close >= self.take_profit_price)
                or (close <= self.stop_loss_price)
            ):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log("SELL CREATE, %.2f" % close)

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
