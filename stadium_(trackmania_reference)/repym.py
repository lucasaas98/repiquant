# Standard Library
import json
import os
import random
from collections import deque

# Third party dependencies
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

RAW_DATA_FOLDER = "raw_data"
PROCESSED_DATA_FOLDER = "processed_data"
TICKER = "TSLA"
INTERVAL = "5min"
TICKERS = ["NVDA", "META", "MSFT", "TSLA", "AAPL", "AMZN"]
EPISODE_LENGTH = 50


class Repym(gym.Env):
    def __init__(self, to_print=False):
        super().__init__()
        self.starting_balance = 10000
        self.balance = 10000
        self.current_bar = 0
        self.commission_pct = 0.001

        self.trade_start_cash = 0
        self.trade_start = 0
        self.trade_best_out_bar = 0
        self.assets_data = {}
        self.portfolio_history = {}
        self.daily_portfolio_values = deque([self.starting_balance] * 10, maxlen=10)

        self.asset = None
        # self.training_data = None
        self.real_data = None
        self.real_indexes = None
        self.starting_bar = None
        self.shorting = False

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(654,),
            dtype=np.float32,
        )
        self.print = to_print

        # buy, sell, noop
        self.action_space = spaces.Discrete(3)

        # Initialize state as a numpy array
        self.observation = np.zeros((654,), dtype=np.float32)

    def step(self, action):
        self.current_bar += 1

        # Update state based on action
        ret_action = self.update_state(action)

        # Accumulate reward
        reward = self.calculate_reward(ret_action)
        # self.reward_buffer.append(reward)
        terminated = bool(self.is_done())

        # update portfolio
        if not terminated:
            self.daily_portfolio_values.append(self.assets_data[self.asset]["total_value"])

        truncated = False
        info = {}
        return self.observation, reward, terminated, truncated, info

    def calculate_commission(self, trade_value):
        return trade_value * self.commission_pct

    def buy(self, asset, price):
        if self.assets_data[self.asset]["positions"] < 0:  # Cover Short
            if self.print:
                print("Covering Short!")
            shares_to_cover = -self.assets_data[self.asset]["positions"]
            required_cash = abs(shares_to_cover) * price
            self.assets_data[self.asset]["cash"] += required_cash
            self.assets_data[self.asset]["positions"] += shares_to_cover
            self.shorting = False
            return 0
        elif self.assets_data[asset]["cash"] > 0:  # Buy
            if self.print:
                print("Going Long!")
            trade_value = self.assets_data[asset]["cash"]
            commission = self.calculate_commission(trade_value)
            shares_to_buy = (trade_value - commission) / price
            self.assets_data[asset]["positions"] += shares_to_buy
            self.assets_data[asset]["cash"] -= trade_value
            self.trade_start_cash = trade_value
            self.trade_start = self.current_bar
            return 1

    def sell(self, asset, price):
        if self.assets_data[asset]["positions"] > 0:  # Sell
            if self.print:
                print("Exiting Long!")
            trade_value = self.assets_data[asset]["positions"] * price
            commission = self.calculate_commission(trade_value)
            self.assets_data[asset]["cash"] += trade_value - commission
            self.assets_data[asset]["positions"] = 0
            return 2
        elif self.assets_data[self.asset]["cash"] > 0:  # Short
            if self.print:
                print("Shortinqg!")
            trade_value = self.assets_data[asset]["cash"]
            shares_to_short = -abs(trade_value / price)
            self.assets_data[asset]["positions"] += shares_to_short
            self.assets_data[asset]["cash"] -= trade_value
            self.shorting = True
            self.trade_start_cash = trade_value
            self.trade_start = self.current_bar
            return 3

    def update_portfolio(self, asset, price):
        self.assets_data[asset]["position_value"] = abs(self.assets_data[asset]["positions"]) * price
        self.assets_data[asset]["total_value"] = (
            self.assets_data[asset]["cash"] + self.assets_data[asset]["position_value"]
        )
        self.portfolio_history[asset].append(self.assets_data[asset]["total_value"])

    def reset(self, seed=None, options=None):
        self.asset = TICKER
        # random.choice(TICKERS)

        # self.training_data = pd.read_parquet(
        #     os.path.join(PROCESSED_DATA_FOLDER, self.asset, INTERVAL, "scaled_previous_combined_by_datetime.parquet"),
        #     engine="fastparquet",
        # )
        self.trade_outcomes = pd.read_parquet(
            os.path.join(PROCESSED_DATA_FOLDER, self.asset, INTERVAL, "50_max_bars_trade_outcomes.parquet"),
            engine="fastparquet",
        ).set_index("datetime")

        self.real_data = pd.read_parquet(
            os.path.join(RAW_DATA_FOLDER, self.asset, INTERVAL, "all_current.parquet"),
            engine="fastparquet",
        ).set_index("datetime")
        self.real_indexes = self.real_data.index.to_numpy()
        self.starting_bar = EPISODE_LENGTH + 1
        # int((random.random() * (len(self.real_indexes) - 2 * EPISODE_LENGTH)) + EPISODE_LENGTH)

        self.reward = 0
        self.assets_data[self.asset] = {
            "cash": self.starting_balance,
            "positions": 0,
            "position_value": 0,
            "total_value": 0,
        }
        self.portfolio_history[self.asset] = []
        self.current_bar = self.starting_bar
        self.shorting = False
        self.trade_start_cash = 0
        self.trade_start = 0

        # current_date = self.real_indexes[self.current_bar]
        # increments_data = self.training_data.loc[current_date].to_numpy() * 100

        indexes = [self.real_indexes[self.current_bar - i] for i in range(EPISODE_LENGTH)]

        self.observation = np.concatenate(
            (
                np.concatenate(self.real_data.loc[indexes].to_numpy()),
                np.array(
                    [
                        self.assets_data[self.asset]["cash"],
                        self.assets_data[self.asset]["positions"],
                        self.assets_data[self.asset]["position_value"],
                        self.assets_data[self.asset]["total_value"],
                    ]
                ),
            )
        )

        return self.observation, {}

    def render(self, mode="human"):
        # print("observation:", self.observation)
        print(
            f"\nstarting_bar {self.starting_bar}\tcurrent_bar {self.current_bar}\tticker {self.asset}",
        )
        print("self.daily_portfolio_values:", self.daily_portfolio_values)
        print("self.assets_data:", self.assets_data)

    def update_state(self, action):
        current_date = self.real_indexes[self.current_bar]
        ohlc_data = self.real_data.loc[current_date].to_numpy()
        # row = self.real_data.loc[current_date]
        # print(row.head())
        close = ohlc_data[3]
        ret_action = 4
        # print(action)
        # print(type(action))
        if action == 0:
            ret_action = self.buy(self.asset, close)
        elif action == 1:
            ret_action = self.sell(self.asset, close)
        elif action == 2:
            pass

        self.update_portfolio(self.asset, close)

        # ohlc_datas = []
        # for i in range(EPISODE_LENGTH):
        #     ohlc_datas.append(
        #         self.real_data.drop(["datetime"], axis=1).loc[self.real_indexes[self.current_bar - i]].to_numpy()
        #     )
        indexes = [self.real_indexes[self.current_bar - i] for i in range(EPISODE_LENGTH)]

        self.observation = np.concatenate(
            (
                np.concatenate(self.real_data.loc[indexes].to_numpy()),
                np.array(
                    [
                        self.assets_data[self.asset]["cash"],
                        self.assets_data[self.asset]["positions"],
                        self.assets_data[self.asset]["position_value"],
                        self.assets_data[self.asset]["total_value"],
                    ]
                ),
            )
        )
        return ret_action

    def calculate_reward(self, action):
        # because we are training in the past we can attribute rewards based on action + potential outcome of action
        # meaning we should create a map where we see what the potential max profit for a certain timeframe is and what are the actions to achieve it
        # if we are late to changing direction we must remove from the potential outcome of the action.
        # if we too early to change direction we must keep the potential outcome but we will deduce whatever amount we lost by being early
        # if we realize the profit we give big reward.

        # current_total_value = self.assets_data[self.asset]["total_value"]
        # bar_diff = self.daily_portfolio_values[-1] - self.daily_portfolio_values[-2]
        # base_reward = bar_diff
        current_date = self.real_indexes[self.current_bar]
        # print(current_date)

        current_day_trade_outcomes = self.trade_outcomes.loc[current_date]
        relevant_trade_outcomes = current_day_trade_outcomes[
            (
                current_day_trade_outcomes["take_profit"] < 0
                if self.shorting
                else current_day_trade_outcomes["take_profit"] > 0
            )
        ].to_numpy()
        # relevant_trade_outcomes.loc[:, "bars_in_market"].mean()
        # columns = relevant_trade_outcomes.columns
        reward = 0

        # print(f"action: {action}, shorting: {self.shorting}, cash: {self.assets_data[self.asset]['cash']}")
        # print(f"relevant_trade_outcomes: {relevant_trade_outcomes}")

        if action == 0:  # cover short
            diff = self.current_bar - self.trade_start - self.trade_best_out_bar
            reward += 500 if diff == 0 else (1 / (abs(diff) + 1)) * 250
            reward += self.assets_data[self.asset]["cash"] - self.trade_start_cash
            # print("0 reward")
        elif action == 1:  # longing
            min_bar_out = 52
            for idx, trade_outcome in enumerate(relevant_trade_outcomes):
                [ticker, interval, max_bar, return_pct, bars_in_market, stop_loss, take_profit] = trade_outcome
                if bars_in_market < min_bar_out:
                    min_bar_out = bars_in_market

                reward += (bars_in_market / 50) * return_pct
            self.trade_best_out_bar = min_bar_out
            # print("1 reward")
        elif action == 2:  # sell long
            diff = self.current_bar - self.trade_start - self.trade_best_out_bar
            reward += 500 if diff == 0 else (1 / (abs(diff) + 1)) * 250
            reward += self.assets_data[self.asset]["cash"] - self.trade_start_cash
            # print("2 reward")
        elif action == 3:  # shorting
            min_bar_out = 52
            for idx, trade_outcome in enumerate(relevant_trade_outcomes):
                [ticker, interval, max_bar, return_pct, bars_in_market, stop_loss, take_profit] = trade_outcome
                if bars_in_market < min_bar_out:
                    min_bar_out = bars_in_market

                reward += (bars_in_market / 50) * return_pct
            self.trade_best_out_bar = min_bar_out
            # print("3 reward")
        elif action == 4:  # Hold
            diff = self.current_bar - self.trade_start - self.trade_best_out_bar
            reward += 20 * (1 / abs(diff)) if diff < 0 else -10
            # print("4 reward")

        return reward

    def is_done(self):
        ret = (self.current_bar - self.starting_bar) == EPISODE_LENGTH
        return ret or self.assets_data[self.asset]["total_value"] <= 0

    def close(self):
        return
