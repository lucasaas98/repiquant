# Third party dependencies
import backtrader as bt
import keras
import numpy as np

# Current project dependencies
from consts import INTERVAL, MULTIPLE_FEED_TICKERS, TICKER
from helper import DataTranslator


class MoreClassesStrategy(bt.Strategy):

    def convert_prediction_to_label(self, prediction):
        prediction = prediction[0]
        prediction_class = np.argmax(prediction)

        # if return_pct > 5:
        #     return "very good"
        # elif return_pct > 3:
        #     return "good"
        # elif return_pct > 1:
        #     return "ok"
        # elif return_pct < 1 and return_pct > -1:
        #     return "noop"
        # elif return_pct <= -1 and return_pct > -3:
        #     return "bad"
        # elif return_pct <= -3:
        #     return "very bad"

        if prediction_class == 2:
            return "very good"
        elif prediction_class == 1:
            return "good"
        elif prediction_class == 0:
            return "noop"

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
        self.stop_loss_pct = 3
        self.take_profit_ratio = 1.03
        self.model = None
        self.load_model()

    def load_model(self):
        # with tf.device("/device:GPU:0"):
        self.model = keras.models.load_model("models/classifiers/1733351489_model_5min_more_classes.keras")

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

        data_array = self.translator.get_increments_for_bar(self.datas[0].datetime.datetime(0))
        if data_array is None:
            self.log("increments missing!!")
            return
        else:
            data_array = data_array.to_numpy()
        # self.log(f"data_array, {data_array}")

        close = self.translator.get_close_for_bar(self.datas[0].datetime.datetime(0)).to_numpy()[3]
        if not close:
            self.log("missing a close!!")
            return
        self.log("Close, %.2f" % close)

        # data_array.reshape((1, 54))
        # self.log("data_array2, %.2f" % data_array)
        # x_tensor = tf.convert_to_tensor(data_array, dtype=tf.float32)
        # self.log(f"x_tensor, {x_tensor}")
        reshaped_data = np.reshape(data_array, (1, 54))

        prediction = self.model(reshaped_data)
        label = self.convert_prediction_to_label(np.array(prediction))
        self.log(f"Prediction, {label}")

        # # Check if we are in the market
        if not self.position:
            self.position_size = close
            if label == "very good":
                self.log("BUY CREATE, %.2f" % close)

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()
                self.stop_loss_price = close * (1 - self.stop_loss_pct / 100)
                self.take_profit_price = close * 1.05
            elif label == "good":
                self.log("BUY CREATE, %.2f" % close)

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()
                self.stop_loss_price = close * (1 - self.stop_loss_pct / 100)
                self.take_profit_price = close * 1.02
        elif self.position:
            # 28 bars in the market and we sell, 1.06 profit ratio and we sell, 5% down and we sell
            if (
                (len(self) >= (self.bar_executed + 28))
                or (close >= self.take_profit_price)
                or (close <= self.stop_loss_price)
                # or (label in ["bad", "very bad"])
            ):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log("SELL CREATE, %.2f" % close)

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
