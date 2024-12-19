# Third party dependencies
import numpy as np
import polars as pl
from tqdm import tqdm


def shift_return_pct(df):
    df = df.lazy()
    for i in range(1, 51):
        df = df.with_columns(pl.col("close").shift(-i).alias(f"{i}"))
    return df.collect()


def get_trade_outcome(df):
    df = df.with_columns(
        (pl.when(pl.col("tp") < 0).then(1).otherwise(0).cast(pl.Boolean).alias("shorting")),
    ).with_columns(
        pl.when(pl.col("shorting"))
        .then(pl.col("close") / (1 - pl.col("sl") / 100))
        .otherwise(pl.col("close") * (1 - pl.col("sl") / 100))
        .alias("trade_stop_loss_price"),
        pl.when(pl.col("shorting"))
        .then(pl.col("close") / pl.col("tp"))
        .otherwise(pl.col("close") * pl.col("tp"))
        .alias("trade_take_profit_price"),
    )

    for i in range(1, 51):
        df = df.with_columns(
            pl.when((pl.col(f"{i}") >= pl.col("trade_take_profit_price")) & pl.col("shorting").not_())
            .then((((pl.col(f"{i}") - pl.col("close")) / pl.col("close")) * 100))
            .when((pl.col(f"{i}") <= pl.col("trade_stop_loss_price")) & pl.col("shorting").not_())
            .then((((pl.col(f"{i}") - pl.col("close")) / pl.col("close")) * 100))
            .when((pl.col(f"{i}") <= pl.col("trade_take_profit_price")) & pl.col("shorting"))
            .then((((pl.col(f"{i}") - pl.col("close")) / pl.col("close")) * 100))
            .when((pl.col(f"{i}") >= pl.col("trade_stop_loss_price")) & pl.col("shorting"))
            .then((((pl.col(f"{i}") - pl.col("close")) / pl.col("close")) * 100))
            .otherwise(None)
            .alias("return_pct"),
        )

    all_columns = df.columns
    null_columns = df.select(pl.all().is_null()).columns
    min_column = min([x for x in all_columns if x not in null_columns] or [-1])
    df = df.with_columns(pl.lit(int(min_column)).alias("bars_in_market"))
    return df.select("datetime", "sl", "tp", "bars_in_market", "return_pct")


def calculate_multiple_trade_outcomes(df, stop_loss_percentages, take_profit_ratios):
    sl_df = pl.DataFrame({"sl": stop_loss_percentages})
    tp_df = pl.DataFrame({"tp": take_profit_ratios})

    combined_df = df.join(sl_df, how="cross")
    combined_df = combined_df.join(tp_df, how="cross")
    # combined_df = get_trade_outcome(combined_df)
    datetimes = combined_df.select("datetime").to_numpy()

    result = None
    for idx, datetime in enumerate(tqdm(datetimes)):
        if idx > (len(datetimes) - 50):
            break
        df = combined_df.filter((pl.col("datetime") >= datetime) & (pl.col("datetime") < datetimes[idx + 50]))

        if idx == 0:
            result = get_trade_outcome(df)
        else:
            df = get_trade_outcome(df)
            result = pl.concat([result, df])

    return result


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
    short = False
    if take_profit_ratio < 0:
        short = True
        take_profit_ratio = -(take_profit_ratio)

    # Calculate stop loss price
    stop_loss_price = (
        close_prices[0] * (1 - stop_loss_pct / 100) if not short else close_prices[0] / (1 - stop_loss_pct / 100)
    )

    # Calculate take profit price
    take_profit_price = close_prices[0] * take_profit_ratio if not short else close_prices[0] / take_profit_ratio

    # Check if trade would result in a win or loss over the next max_number_of_bars bars
    for idx, current_close_price in enumerate(close_prices[1 : (max_number_of_bars + 1)]):
        return_pct = ((current_close_price - close_prices[0]) / close_prices[0]) * 100
        if current_close_price >= take_profit_price and not short:
            return True, return_pct, idx + 1
        elif current_close_price <= stop_loss_price and not short:
            return False, return_pct, idx + 1
        elif current_close_price <= take_profit_price and short:
            return True, return_pct, idx + 1
        elif current_close_price >= stop_loss_price and short:
            return False, return_pct, idx + 1
    # If no win or loss condition is met over the next max_number_of_bars bars
    return None, None, max_number_of_bars


if __name__ == "__main__":
    print("Running main")

    # Current project dependencies
    import file_combiner as fc

    data_file = fc.get_ticker_file_parquet("TSLA", "5min")
    data = pl.read_parquet(data_file, columns=["close"])

    stop_loss_pct = 5.0
    take_profit_ratio = 1.2
    closes = data.select("close").to_numpy().flatten()

    win, return_pct, outcome = calculate_trade_outcome(closes, stop_loss_pct, take_profit_ratio)

    print(f"Trade Outcome: {outcome} ({return_pct:.2f}%)")
