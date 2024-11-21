# Third party dependencies
import pandas as pd


def calculate_trade_outcome(close_prices, stop_loss_pct, take_profit_ratio, max_number_of_bars=50):
    """
    Calculate the outcome of a trade given the close stock market prices,
    stop loss percentage, and take profit ratio.

    Parameters:
        close_prices (list or Series): Close stock market prices.
        stop_loss_pct (float): Stop loss percentage (e.g. 5.0 for 5%).
        take_profit_ratio (float): Take profit ratio (e.g. 1.2 for a 20% gain).

    Returns:
        tuple: (win/loss, return pct, bars in market)
    """
    # Calculate stop loss price
    stop_loss_price = close_prices[0] * (1 - stop_loss_pct / 100)

    # Calculate take profit price
    take_profit_price = close_prices[0] * take_profit_ratio

    # Check if trade would result in a win or loss over the next max_number_of_bars bars
    for idx, current_close_price in enumerate(close_prices[1 : (max_number_of_bars + 1)]):
        if current_close_price >= take_profit_price:
            return True, ((current_close_price - close_prices[0]) / close_prices[0]) * 100, idx + 1
        elif current_close_price <= stop_loss_price:
            return False, ((current_close_price - close_prices[0]) / close_prices[0]) * 100, idx + 1

    # If no win or loss condition is met over the next max_number_of_bars bars
    return None, None, max_number_of_bars


if __name__ == "__main__":
    print("Running main")

    # Current project dependencies
    import file_combiner as fc

    data_file = fc.get_ticker_file("TSLA", "5min")
    data = pd.read_csv(data_file, index_col=0)

    stop_loss_pct = 5.0
    take_profit_ratio = 1.2

    win, return_pct, outcome = calculate_trade_outcome(data["close"].values, stop_loss_pct, take_profit_ratio)

    print(f"Trade Outcome: {outcome} ({return_pct:.2f}%)")
