# Third party dependencies
import backtrader as bt
import keras
import numpy as np

# Current project dependencies
from consts import INTERVAL, TICKER
from helper import DataTranslator


class ShortAndLongStrategy(bt.Strategy):
    def __init__(self):
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.translator = DataTranslator(TICKER, INTERVAL)
        self.max_time_in_trade = 28
        self.time_in_trade = 0
        self.stop_loss_pct = 2
        self.long_model = None
        self.short_model = None
        self.shorting = False
        self.load_models()

    def convert_long_prediction_to_label(self, prediction):
        prediction = prediction[0]
        prediction_class = np.argmax(prediction)

        if prediction_class == 2:
            return "very good"
        elif prediction_class == 1:
            return "good"
        elif prediction_class == 0:
            return "noop"

    def convert_short_prediction_to_label(self, prediction):
        prediction = prediction[0]
        prediction_class = np.argmax(prediction)

        if prediction_class == 2:
            return "very bad"
        elif prediction_class == 1:
            return "bad"
        elif prediction_class == 0:
            return "noop"

    def log(self, txt, dt=None):
        """Logging function fot this strategy"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print("%s, %s" % (dt.isoformat(), txt))

    def load_models(self):
        self.long_model = keras.models.load_model("models/classifiers/1733351489_model_5min_more_classes.keras")
        self.short_model = keras.models.load_model("models/classifiers/1733351489_model_5min_more_classes_short.keras")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "%s - BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (self.position_ticker, order.executed.price, order.executed.value, order.executed.comm)
                )

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    "%s - SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (self.position_ticker, order.executed.price, order.executed.value, order.executed.comm)
                )

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log("%s - OPERATION PROFIT, GROSS %.2f, NET %.2f" % (self.position_ticker, trade.pnl, trade.pnlcomm))

    def next(self):
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        data_array = self.translator.get_increments_for_bar(self.datas[0].datetime.datetime(0))
        if data_array is None:
            self.log("increments missing!!")
            return
        else:
            data_array = data_array.to_numpy()

        close = self.translator.get_close_for_bar(self.datas[0].datetime.datetime(0)).to_numpy()[3]
        if not close:
            self.log("missing a close!!")
            return
        self.log("Close, %.2f" % close)

        reshaped_data = np.reshape(data_array, (1, 54))

        long_prediction = self.long_model(reshaped_data)
        long_label = self.convert_long_prediction_to_label(np.array(long_prediction))
        self.log(f"Long prediction, {long_label}")

        short_prediction = self.short_model(reshaped_data)
        short_label = self.convert_short_prediction_to_label(np.array(short_prediction))
        self.log(f"Short prediction, {short_label}")

        # Check if we are in the market
        if not self.position:
            if long_label == "very good":
                self.log(f"{TICKER} - BUY CREATE, %.2f" % close)
                self.order = self.buy()
                self.stop_loss_price = close * (1 - self.stop_loss_pct / 100)
                self.take_profit_price = close * 1.05
                self.shorting = False
            elif long_label == "good":
                self.log(f"{TICKER} - BUY CREATE, %.2f" % close)
                self.order = self.buy()
                self.stop_loss_price = close * (1 - self.stop_loss_pct / 100)
                self.take_profit_price = close * 1.02
                self.shorting = False
            elif short_label == "very bad":
                self.log(f"{ticker} - SELL CREATE, %.2f" % close)
                self.order = self.sell()
                self.stop_loss_price = close * (1 + self.stop_loss_pct / 100)
                self.take_profit_price = close * 0.95
                self.shorting = True
            elif short_label == "bad":
                self.log(f"{ticker} - SELL CREATE, %.2f" % close)
                self.order = self.sell()
                self.stop_loss_price = close * (1 + self.stop_loss_pct / 100)
                self.take_profit_price = close * 0.98
                self.shorting = True
        elif self.position and not self.shorting:
            if (
                (len(self) >= (self.bar_executed + 40))
                or (close >= self.take_profit_price)
                or (close <= self.stop_loss_price)
            ):
                self.log("SELL CREATE, %.2f" % close)
                self.order = self.sell()
        elif self.position and self.shorting:
            if (
                (len(self) >= (self.bar_executed + 40))
                or (close <= self.take_profit_price)
                or (close >= self.stop_loss_price)
            ):
                self.log("BUY CREATE, %.2f" % close)
                self.order = self.buy()
