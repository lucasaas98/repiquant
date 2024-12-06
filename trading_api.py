# Standard Library
import os

# Third party dependencies
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetAssetsRequest,
    GetOrderByIdRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)


## Account ##
def get_account():
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    return trading_client.get_account()


## Assets ##


def get_all_assets(asset_class=AssetClass.US_EQUITY):
    """
    Get all assets for a specific asset class.

    Parameters:
        asset_class (AssetClass, optional): The asset class to retrieve. Defaults to AssetClass.US_EQUITY.

    Returns:
        List[Asset]: A list of all assets for the specified asset class.
    """
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    search_params = GetAssetsRequest(asset_class=asset_class)

    assets = trading_client.get_all_assets(search_params)

    return assets


## Orders ##


def create_bracket_order(symbol, close, position_size, take_profit_pct, stop_loss_pct, balance):
    # quantity = (balance * position_size%) / unit_price
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    qty = int((balance * position_size / 100) / close)
    stop_loss_price = close * (1 - stop_loss_pct / 100)
    take_profit_price = close * (1 + take_profit_pct / 100)

    bracket_order_data = MarketOrderRequest(
        symbol="SPY",
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_loss_price),
    )

    bracket_order = trading_client.submit_order(order_data=bracket_order_data)

    return bracket_order


def create_market_order(symbol, qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY):
    """
    Create a market order for the specified symbol and quantity.

    Parameters:
        symbol (str): The symbol of the asset to trade.
        qty (int): The quantity of the asset to trade.
        side (OrderSide, optional): The side of the order to place. Defaults to OrderSide.BUY.
        time_in_force (TimeInForce, optional): The time in force for the order. Defaults to TimeInForce.DAY.

    Returns:
        MarketOrder: The market order that was placed.
    """
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    market_order_data = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=time_in_force)

    market_order = trading_client.submit_order(order_data=market_order_data)

    return market_order


def create_limit_order(symbol, limit_price, qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY):
    """
    Create a limit order for the specified symbol, limit price, and quantity.

    Parameters:
        symbol (str): The symbol of the asset to trade.
        limit_price (float): The limit price of the order.
        qty (int): The quantity of the asset to trade.
        side (OrderSide, optional): The side of the order to place. Defaults to OrderSide.BUY.
        time_in_force (TimeInForce, optional): The time in force for the order. Defaults to TimeInForce.DAY.

    Returns:
        LimitOrder: The limit order that was placed.
    """
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    limit_order_data = LimitOrderRequest(
        symbol=symbol,
        limit_price=limit_price,
        qty=qty,
        side=side,
        time_in_force=time_in_force,
    )

    limit_order = trading_client.submit_order(order_data=limit_order_data)

    return limit_order


def get_all_orders(status=QueryOrderStatus.OPEN, side=OrderSide.SELL):
    """
    Get all orders for the specified status and side.

    Parameters:
        status (QueryOrderStatus, optional): The status of the orders to retrieve. Defaults to QueryOrderStatus.OPEN.
        side (OrderSide, optional): The side of the orders to retrieve. Defaults to OrderSide.SELL.

    Returns:
        List[Order]: A list of all orders for the specified status and side.
    """
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    request_params = GetOrdersRequest(status=status, side=side)

    orders = trading_client.get_orders(filter=request_params)

    return orders


def get_order_by_id(id):
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    order = trading_client.get_order_by_id(order_id=id)

    return order


def cancel_all_orders():
    """
    Cancel all open orders.

    Returns:
        List[CancelStatus]: A list of the statuses for each order that was canceled.
    """
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    cancel_statuses = trading_client.cancel_orders()
    return cancel_statuses


## Positions ##


def get_all_positions():
    """
    Get all positions for the current account.

    Returns:
        List[Position]: A list of all positions for the current account.
    """
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    all_positions = trading_client.get_all_positions()
    return all_positions


def close_all_positions():
    """
    Close all open positions by selling them at market price.
    Returns:
        List[CloseStatus]: A list of the statuses for each position that was closed.
    """
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    result = trading_client.close_all_positions(cancel_orders=True)
    return result
