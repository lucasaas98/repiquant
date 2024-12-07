# Standard Library
import json
from datetime import datetime
from time import sleep

# Third party dependencies
import keras
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler

# Current project dependencies
import data_api as api
import trading_api as alpaca
from consts import INTERVAL, MULTIPLE_FEED_EXCHANGE, MULTIPLE_FEED_TICKERS
from data_vectorizer import create_final_vector
from notifier import notify_trade


class PaperTrader:

    def __init__(self):
        # class variables
        self.model = None
        self.scaler = None
        self.balance = None
        self.pending_order = None
        self.position = False
        self.ticker_predictions = dict()
        self.current_bar = 0

        # load everything
        self.load_models()
        self.pre_cycle()

        self.log(f"Starting with balance: {self.balance}")

    def log(self, txt, notify=False):
        dt = datetime.now()
        print("%d - %s, %s" % (self.current_bar, dt.isoformat(), txt))
        if notify:
            notify_trade(txt)

    def load_models(self):
        self.model = keras.models.load_model("models/classifiers/1733351489_model_5min_more_classes.keras")
        self.scaler = load("models/scalers/1732994464_scaler.gzip")

    def convert_prediction_to_label(self, prediction):
        prediction = prediction[0]
        prediction_class = np.argmax(prediction)

        if prediction_class == 2:
            return "very good"
        elif prediction_class == 1:
            return "good"
        elif prediction_class == 0:
            return "noop"

    def get_best_ticker(self):
        best_ticker = None
        best_label = None
        for ticker, (label, prediction, _raw_df) in self.ticker_predictions.items():
            if label == "very good":
                best_ticker = ticker
                best_label = label
                break
            if label == "good" and best_label is None:
                best_ticker = ticker
                best_label = label
        return best_ticker, best_label

    def pre_cycle(self):
        self.account = alpaca.get_account()
        self.balance = self.account.cash
        if self.pending_order:
            alpaca_order = alpaca.get_order_by_id(self.pending_order.id)
            self.pending_order = None if alpaca_order.status == "filled" else alpaca_order
            if not self.pending_order:
                self.log(
                    f"-- {alpaca_order.symbol} -- Order {alpaca_order.id} filled at {alpaca_order.filled_at}. Bought {alpaca_order.filled_qty} at {alpaca_order.filled_avg_price} for a total of ${alpaca_order.filled_qty * alpaca_order.filled_avg_price}",
                    notify=True,
                )
        all_positions = alpaca.get_all_positions()
        old_position = self.position
        self.position = all_positions[0] if len(all_positions) != 0 else None
        if old_position != self.position:
            self.log(f"Current balance: {self.balance}")

    def cycle(self):
        if not self.position and not self.pending_order:
            for ticker in MULTIPLE_FEED_TICKERS:
                raw_data_df = api.get_time_series(symbol=ticker, interval=INTERVAL, outputsize=20)
                raw_data_df = raw_data_df[::-1]

                increments_df = create_final_vector(raw_data_df)
                current_bar_vector = increments_df.to_numpy()[-1]

                scaled_data = self.scaler.transform(current_bar_vector.reshape(1, -1))
                prediction = self.model(scaled_data)
                label = self.convert_prediction_to_label(np.array(prediction))

                self.ticker_predictions[ticker] = (label, prediction, raw_data_df)
            best_ticker, best_label = self.get_best_ticker()
            if best_ticker:
                best_ticker_close = self.ticker_predictions[best_ticker][2].to_numpy()[-1][3]
            if best_label == "very good":
                self.log(f"Found a {best_label} position with ticker {best_ticker}")
                self.pending_order = alpaca.create_bracket_order(
                    best_ticker, best_ticker_close, 5, 5, 2, float(self.balance)
                )
            elif best_label == "good":
                self.log(f"Found a {best_label} position with ticker {best_ticker}")
                self.pending_order = alpaca.create_bracket_order(
                    best_ticker, best_ticker_close, 5, 2, 2, float(self.balance)
                )

    def print_bar_result(self):
        new_dict = dict()
        for ticker, (label, prediction, raw_data_df) in self.ticker_predictions.items():
            new_dict[ticker] = label
        self.log(json.dumps(new_dict))

    def run(self):
        market_state = api.get_market_open(MULTIPLE_FEED_EXCHANGE)
        market_open = market_state[0]["is_market_open"]
        while True:
            if market_open:
                self.current_bar += 1
                self.log(f"Starting bar: {self.current_bar}")
                self.pre_cycle()
                self.cycle()
                self.print_bar_result()
                self.log("Sleeping 5 mins")
            else:
                self.log("Market is closed! Sleeping 5 mins")
            sleep(300)


if __name__ == "__main__":
    # Third party dependencies
    from dotenv import load_dotenv

    load_dotenv()
    paper_trader = PaperTrader()
    paper_trader.run()
