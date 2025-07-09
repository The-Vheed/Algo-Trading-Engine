import MetaTrader5 as mt5
import pandas as pd

# import pandas_ta as ta
import time, datetime, os
import pytz
from datetime import timedelta

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Path, Depends
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from schemas import (
    BaseRequest,
    OrderRequest,
    PriceDataRequest,
    ClosePositionsRequest,
    GitOperationRequest,
    OrderResponse,
    ClosePositionsResponse,
    PriceDataResponse,
    AccountInfoResponse,
    GitOperationResponse,
    TimeFrameEnum,
    SymbolEnum,
)

load_dotenv()


# set time zone to UTC
timezone = pytz.timezone("Etc/UTC")
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}
TIMEFRAME = None

V25 = "Volatility 25 Index"
SI = "Step Index"
UJ = "USDJPY"
EU = "EURUSD"

default_symbol = V25


comment_map = {"buy": "BOT BUY", "sell": "BOT SELL"}


# Function to prepare technical indicators for historical stock price data
def prep_indicators(hist: pd.DataFrame):
    # Extract the "close" prices from the historical data and reshape for scaling
    # fit_data = hist["close"].values
    # fit_data = fit_data.reshape((fit_data.shape[0], 1))

    # Fit the scaler on the "close" prices to calculate the scaling parameters
    # scaler_name = 'scaler.pkl'
    # try:
    #     scaler = pkl.load(open(scaler_name, 'rb'))
    # except:
    #     # Initialize a MinMaxScaler object for scaling data between 0 and 1
    #     scaler = MinMaxScaler(copy=False)
    #     scaler.fit(fit_data)
    #     pkl.dump(scaler, open(scaler_name, 'wb'))

    # # Scale the relevant columns using the previously fitted scaler
    # scale(scaler, hist, "open")
    # scale(scaler, hist, "high")
    # scale(scaler, hist, "low")
    # scale(scaler, hist, "close")

    # Calculate the Relative Strength Index (RSI) using the pandas_ta library
    # hist["RSI"] = ta.rsi(hist["close"], 14) / 100

    # Calculate the Ichimoku indicator (not used in the current version)
    # ichi = ta.ichimoku(
    #     hist["high"],
    #     hist["low"],
    #     hist["close"],
    #     lookahead=False,
    #     tenkan=10,
    #     kijun=30,
    #     senkou=32,
    # )
    # Concatenate the Ichimoku data to the historical data DataFrame
    # hist = pd.concat([hist, ichi[0]], axis=1, join="outer")

    # Calculate the Bollinger Bands
    # boll = ta.bbands(hist["close"], length=30, std=2, mamode="ema")
    # boll2 = ta.bbands(hist["close"], length=50, std=3, mamode="ema")
    # # boll = ta.bbands(hist["close"], length=50, std=4, mamode="sma")
    # hist["MA500"] = ta.ema(hist["close"], length=500)

    # # Concatenate the Bollinger Bands data to the historical data DataFrame (commented out for now)
    # hist = pd.concat([hist, boll], axis=1, join="outer")
    # hist = pd.concat([hist, boll2], axis=1, join="outer")

    # # Create a BBM trend indicator
    # hist["BBM_DIFF"] = hist["BBM_50_3.0"] - hist["BBM_30_2.0"]
    # hist.drop(columns=["BBM_50_3.0", "BBM_30_2.0"], inplace=True)

    # Check chart patterns
    # hist["ENGULF"] = (
    #    talib.CDLENGULFING(hist["open"], hist["high"],
    #                       hist["low"], hist["close"]) / 100
    # )
    # hist["DOJISTAR"] = (
    #    talib.CDLDOJISTAR(hist["open"], hist["high"],
    #                      hist["low"], hist["close"]) / 100
    # )
    # hist["ESTAR"] = (
    #    talib.CDLEVENINGSTAR(hist["open"], hist["high"],
    #                         hist["low"], hist["close"])
    #    / 100
    # )
    # hist["MSTAR"] = (
    #    talib.CDLMORNINGSTAR(hist["open"], hist["high"],
    #                         hist["low"], hist["close"])
    #    / 100
    # )
    # hist["SHOOTSTAR"] = (
    #    talib.CDLSHOOTINGSTAR(
    #        hist["open"], hist["high"], hist["low"], hist["close"])
    #    / 100
    # )

    # fig5 = make_subplots()
    # fig5.add_trace(go.Candlestick(x=hist.index,
    #                               open=hist['open'],
    #                               high=hist['high'],
    #                               low=hist['low'],
    #                               close=hist['close'],
    #                              ))

    # fig5.add_trace(go.Scatter(x=hist.index,y=hist['BBM_30_2.0'], mode='lines'))
    # fig5.add_trace(go.Scatter(x=hist.index,y=hist['BBL_30_2.0'], mode='lines+markers'))
    # fig5.add_trace(go.Scatter(x=hist.index,y=hist['BBU_30_2.0'], mode='lines+markers'))

    # fig5.add_trace(go.Scatter(x=hist.index,y=hist['BBM_50_3.0'], mode='lines'))
    # fig5.add_trace(go.Scatter(x=hist.index,y=hist['BBL_50_3.0'], mode='lines+markers'))
    # fig5.add_trace(go.Scatter(x=hist.index,y=hist['BBU_50_3.0'], mode='lines+markers'))

    # fig5.update_layout(xaxis_rangeslider_visible=False)

    # fig5.show()

    # drop extra candle stick data
    # hist.drop(columns=["open"], inplace=True)

    # print(hist.columns)

    return hist.dropna()


def calculate_lot_size(symbol, stop_loss_price, stop_loss_amount):
    """
    Calculate the lot size dynamically based on risk amount and stop-loss price.

    Parameters:
    - symbol: The trading pair (e.g., "EURUSD").
    - stop_loss_price: The price at which the stop-loss is set.
    - stop_loss_amount: The maximum risk amount in account currency.

    Returns:
    - The lot size to use or None if the calculation fails.
    """
    # Get symbol information
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None

    # Get the current price for the symbol
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick data for {symbol}")
        return None

    # Determine whether this is a BUY or SELL order
    if tick.ask > stop_loss_price:
        distance_points = (tick.ask - stop_loss_price) / symbol_info.point  # For BUY
    else:
        distance_points = (stop_loss_price - tick.bid) / symbol_info.point  # For SELL

    if distance_points <= 0:
        print("Invalid stop-loss price; must be further away from the current price.")
        return None

    # Calculate point value and loss per lot
    tick_value = symbol_info.trade_tick_value
    point_value = tick_value / symbol_info.trade_tick_size
    loss_per_lot = distance_points * point_value

    # Calculate the lot size
    lot_size = stop_loss_amount / loss_per_lot
    return lot_size


# Function to download historical data for a specific currency pair
def download_data(symbol, timeframe, start_date=None, end_date=None, count=None):
    """
    Download historical price data from MetaTrader 5.

    Args:
        symbol (str): Trading symbol (e.g. "Volatility 25 Index")
        timeframe (int): MT5 timeframe constant
        start_date (datetime, optional): Start date for historical data
        end_date (datetime, optional): End date for historical data
        count (int, optional): Number of candles to retrieve (alternative to date range)

    Returns:
        pd.DataFrame: DataFrame with price data
    """
    try:
        # Copy historical data for the specified symbol, timeframe, and date range
        if count:
            history = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        else:
            history = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

        # Convert to DataFrame
        df = pd.DataFrame(history)

        # Add technical indicators if needed
        if df.shape[0] > 20:  # Only add indicators if we have enough data
            df = prep_indicators(df)

        return df
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()


def find_filling_mode(symbol):
    """
    The MetaTrader5 library doesn't find the filling mode correctly for a lot of brokers
    """
    for i in range(2):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": mt5.symbol_info(symbol).volume_min,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(symbol).ask,
            "type_filling": i,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_check(request)

        if result.comment == "Done":
            break

    return i


def get_symbol_precision(symbol):
    """
    Get the number of decimal points (precision) for a trading symbol.

    Parameters:
    - symbol: The trading pair (e.g., "EURUSD").

    Returns:
    - The number of decimal places for the symbol or None if failed.
    """
    # Get symbol information
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None

    # Determine precision based on the point size
    point_size = symbol_info.point
    if point_size == 0:
        print(f"Point size is zero for {symbol}")
        return None

    # Calculate the number of decimal places
    precision = abs(int(round(-1 * (mt5.log10(point_size)))))
    return precision


# Function to open a market buy position
def open_buy(request, stop_loss, take_profit, stop_loss_amount, trailing=0):
    # ! YOU NEED TO HAVE THE SYMBOL IN THE MARKET WATCH TO OPEN OR CLOSE A POSITION
    symbol = request.symbol
    comment = request.comment
    price = request.price

    selected = mt5.symbol_select(symbol)
    if not selected:
        print(
            f"\nERROR - Failed to select '{symbol}' in MetaTrader 5 with error :",
            mt5.last_error(),
        )

    # Extract filling_mode
    filling_type = find_filling_mode(symbol)
    lot = calculate_lot_size(symbol, stop_loss, stop_loss_amount)
    if not lot:
        return None

    # calculate levels from pct
    # tp_price = (1 + take_profit / 100) * price
    # sl_price = (1 - stop_loss / 100) * price

    min_vol = 0

    # if symbol == V25:
    #     min_vol = 0.5
    # elif symbol == SI:
    #     min_vol = 0.1
    # elif symbol == UJ:
    #     min_vol = 0.5

    trailing_amount = round(
        abs(price - stop_loss) * trailing, get_symbol_precision(request.symbol)
    )

    # Create a request to open a market buy order
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": max(round(float(lot), 2), min_vol),
        "type": mt5.ORDER_TYPE_BUY,
        "sl": stop_loss,
        "tp": take_profit,
        "comment": comment_map["buy"] + f" | {trailing_amount} " + f" | {comment}",
        "type_filling": filling_type,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    # Send the order request and get the order details
    order = mt5.order_send(request)

    return order


# Function to open a market sell position
def open_sell(request, stop_loss, take_profit, stop_loss_amount, trailing=0):
    # ! YOU NEED TO HAVE THE SYMBOL IN THE MARKET WATCH TO OPEN OR CLOSE A POSITION
    symbol = request.symbol
    comment = request.comment
    price = request.price

    selected = mt5.symbol_select(symbol)
    if not selected:
        print(
            f"\nERROR - Failed to select '{symbol}' in MetaTrader 5 with error :",
            mt5.last_error(),
        )

    # Extract filling_mode
    filling_type = find_filling_mode(symbol)
    lot = calculate_lot_size(symbol, stop_loss, stop_loss_amount)
    if not lot:
        return None

    min_vol = 0

    # if symbol == V25:
    #     min_vol = 0.5
    # elif symbol == SI:
    #     min_vol = 0.1
    # elif symbol == UJ:
    #     min_vol = 0.5

    trailing_amount = round(
        abs(price - stop_loss) * trailing, get_symbol_precision(request.symbol)
    )

    # Create a request to open a market sell orders
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": max(round(float(lot), 2), min_vol),
        "type": mt5.ORDER_TYPE_SELL,
        "sl": stop_loss,
        "tp": take_profit,
        "comment": comment_map["sell"] + f" | {trailing_amount} " + f" | {comment}",
        "type_filling": filling_type,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    # Send the order request and get the order details
    order = mt5.order_send(request)

    return order


# Function to close all open positions
def close_all_positions():
    # Get all open positions
    positions = mt5.positions_get()

    results = []

    # Close each open position
    for position in positions:
        if not str(position.comment).split(" | ")[0] in comment_map.values():
            continue

        tick = mt5.symbol_info_tick(position.symbol)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if position.type == 1 else tick.bid,
            "comment": "BOT CLOSE ALL",
        }

        results.append(mt5.order_send(request))

    return results


# Function to get account balance information
def get_account_info():
    # Get account information
    account_info = mt5.account_info()
    return account_info


def get_account_balance():
    return get_account_info().balance + 0


# Function to get all available trading symbols
def get_trading_symbols():
    # Get all symbols available for trading
    symbols = mt5.symbols_get()
    return symbols


# Function to get all open positions
def get_open_positions(symbol):
    # Get all open positions
    positions = mt5.positions_get(symbol=symbol)
    return positions


def get_last_pnl():
    # Fetch last closed trade
    closed_trades = mt5.history_deals_get(ticket=0, count=1)

    if closed_trades:
        last_trade = closed_trades[0]

        # Extract time and profit
        last_trade_time = last_trade.time  # Adjust timeframe as needed
        last_trade_profit = last_trade.profit

        return last_trade_time, last_trade_profit
    else:
        return None, 1000  # No closed trades found


# API side
# Static access key for authentication
ACCESS_KEY = os.environ["ACCESS_KEY"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to MetaTrader 5
    mt5.initialize()

    print(get_account_info())

    # Handle requests
    yield

    # print(close_all_positions())
    # Disconnect from MetaTrader 5
    mt5.shutdown()


app = FastAPI(
    lifespan=lifespan,
    title="Trading Bot API",
    description="API for automated trading operations via MetaTrader 5",
    version="1.0.0",
    docs_url="/docs",
)


# Authentication dependency
def verify_access_key(request: BaseRequest):
    if request.access_key != ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid access key")


# Endpoint to execute a buy order
@app.post(
    "/buy",
    response_model=OrderResponse,
    summary="Execute a buy order",
    tags=["Trading"],
)
def execute_buy(request: OrderRequest, _=Depends(verify_access_key)):
    balance = get_account_balance()
    sl_amount = balance * (request.risk / 100)

    # Execute a market buy order
    order = open_buy(
        request,
        request.sl,
        request.tp,
        sl_amount,
        request.trailing,
    )

    response = {
        "status": True,
        "message": "Buy order executed",
        "order_details": (
            order._asdict() if hasattr(order, "_asdict") else dict(vars(order))
        ),
    }

    if str(order.retcode) != "10009":
        response["status"] = False
        response["message"] = "An error occurred"

    return response


# Endpoint to execute a sell order
@app.post(
    "/sell",
    response_model=OrderResponse,
    summary="Execute a sell order",
    tags=["Trading"],
)
def execute_sell(request: OrderRequest, _=Depends(verify_access_key)):
    balance = get_account_balance()
    sl_amount = balance * (request.risk / 100)

    # Execute a market sell order
    order = open_sell(
        request,
        request.sl,
        request.tp,
        sl_amount,
        request.trailing,
    )

    response = {
        "status": True,
        "message": "Sell order executed",
        "order_details": (
            order._asdict() if hasattr(order, "_asdict") else dict(vars(order))
        ),
    }

    if str(order.retcode) != "10009":
        response["status"] = False
        response["message"] = "An error occurred"

    return response


# Endpoint to close all positions
@app.post(
    "/close_all",
    response_model=ClosePositionsResponse,
    summary="Close all open positions",
    tags=["Trading"],
)
def execute_close_all(request: ClosePositionsRequest, _=Depends(verify_access_key)):
    # Cancel all previous orders
    results = close_all_positions()

    # Convert results to dict for serialization
    processed_results = []
    for result in results:
        processed_results.append(
            result._asdict() if hasattr(result, "_asdict") else dict(vars(result))
        )

    return {
        "status": True,
        "message": "Close all positions executed",
        "closed_positions": processed_results,
    }


# Unified endpoint for price data
@app.post(
    "/price_data",
    response_model=PriceDataResponse,
    summary="Get price data for specified symbol and timeframes",
    tags=["Market Data"],
)
def get_price_data(request: PriceDataRequest, _=Depends(verify_access_key)):

    # Handle trailing stops if appropriate
    if not request.start_time and not request.count:
        # Update trailing stop logic for positions
        positions = get_open_positions(request.symbol)
        tick = mt5.symbol_info_tick(request.symbol)
        if tick is None:
            raise HTTPException(
                status_code=500, detail=f"Failed to get tick data for {request.symbol}"
            )

        # Loop through positions and adjust the stop loss
        for position in positions:
            try:
                # Parse the trailing distance from the comment
                trailing_parts = str(position.comment).split(" | ")
                if len(trailing_parts) > 1:
                    trailing_distance = float(trailing_parts[1])
                else:
                    continue

                # Calculate new stop loss
                if position.type == mt5.ORDER_TYPE_BUY:
                    new_stop_loss = tick.bid - trailing_distance
                    if position.sl is None or new_stop_loss > position.sl:
                        stop_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position.ticket,
                            "symbol": request.symbol,
                            "sl": new_stop_loss,
                            "tp": position.tp,  # Keep existing take profit
                        }
                        result = mt5.order_send(stop_request)
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"Failed to modify stop loss: {result.retcode}")
                        else:
                            print(
                                f"Stop loss updated for BUY position {position.ticket}: {new_stop_loss}"
                            )

                elif position.type == mt5.ORDER_TYPE_SELL:
                    new_stop_loss = tick.ask + trailing_distance
                    if position.sl is None or new_stop_loss < position.sl:
                        stop_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position.ticket,
                            "symbol": request.symbol,
                            "sl": new_stop_loss,
                            "tp": position.tp,  # Keep existing take profit
                        }
                        result = mt5.order_send(stop_request)
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"Failed to modify stop loss: {result.retcode}")
                        else:
                            print(
                                f"Stop loss updated for SELL position {position.ticket}: {new_stop_loss}"
                            )
            except Exception as e:
                print(f"Error updating trailing stop: {e}")

    # Initialize response
    response = {"symbol": request.symbol, "data": {}}

    # Set default time ranges for different timeframes if not specified
    time_ranges = {
        TimeFrameEnum.M1: 60,
        TimeFrameEnum.M5: 300,
        TimeFrameEnum.M15: 900,
        TimeFrameEnum.H1: 3600,
        TimeFrameEnum.H4: 14400,
        TimeFrameEnum.D1: 30,
        TimeFrameEnum.W1: 52,
        TimeFrameEnum.MN1: 12,
    }

    # Process each requested timeframe
    for tf in request.timeframes:
        timeframe = TIMEFRAMES[tf]

        # Determine how to fetch data based on parameters
        if request.count:
            # Use count parameter
            data = download_data(request.symbol, timeframe, count=request.count)
        elif request.start_time:
            # Use specified start and end time
            end_time = request.end_time or datetime.datetime.now(timezone)
            data = download_data(
                request.symbol, timeframe, request.start_time, end_time
            )
        else:
            # Use default time range based on timeframe
            end_date = datetime.datetime.now(timezone)
            days_back = time_ranges.get(tf, 300)  # Default to 300 (M5 default)
            start_date = end_date - timedelta(days=days_back)
            data = download_data(request.symbol, timeframe, start_date, end_date)

        # Add data to response
        response["data"][tf] = data.to_dict("split")

    return response


# Endpoint to get account information and positions bias
@app.post(
    "/account_info",
    response_model=AccountInfoResponse,
    summary="Get account information including balance and positions bias",
    tags=["Trading Analysis"],
)
def get_account_info_api(request: BaseRequest, _=Depends(verify_access_key)):
    # Get account balance
    balance = get_account_balance()

    # Get open positions for all symbols (no specific symbol filter needed)
    positions = mt5.positions_get()
    last_time, last_profit = get_last_pnl()

    buys = 0
    sells = 0

    for position in positions:
        comment_parts = str(position.comment).split(" | ")
        if len(comment_parts) > 0:
            if comment_parts[0] == comment_map["buy"]:
                buys += 1
            elif comment_parts[0] == comment_map["sell"]:
                sells += 1

    positions_bias = None

    if buys > sells:
        positions_bias = "BUY"
    elif sells > buys:
        positions_bias = "SELL"

    return {
        "bias": positions_bias,
        "positions_count": len(positions),
        "last_trade": [last_time, last_profit],
        "balance": balance,
    }


# Git operations endpoints
@app.post(
    "/pull",
    response_model=GitOperationResponse,
    summary="Pull latest changes from git repository",
    tags=["Repository Management"],
)
def pull(request: GitOperationRequest, _=Depends(verify_access_key)):
    try:
        os.system("git add .")
        os.system("git stash")
        os.system("git pull")
        return {"status": True}
    except Exception as e:
        return {"status": False}


@app.post(
    "/push",
    response_model=GitOperationResponse,
    summary="Push local changes to git repository",
    tags=["Repository Management"],
)
def push(request: GitOperationRequest, _=Depends(verify_access_key)):
    try:
        # Uncomment the following line to enable automated commits and pushes
        # os.system("git add . && git commit -a -m 'Automated commit' && git push")
        return {"status": True}
    except Exception as e:
        return {"status": False}


if __name__ == "__main__":
    # Run the server on port 8080
    import uvicorn

    # Python Program to Get IP Address
    import socket

    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)

    print("Your Computer Name is:" + hostname)
    print("Your Computer IP Address is:" + IPAddr)
    # uvicorn.run("windows_server:app", host="0.0.0.0", port=80, reload=True)
    uvicorn.run("windows_server:app", host="0.0.0.0", port=8000, reload=True)

    # mt5.initialize()
    # print("Symbol:", get_trading_symbols()[0])

    # Define the symbol and timeframe
    # symbol = V25
    # timeframe = TIMEFRAME  # 1 minute

    # # Calculate the start and end dates (2 months ago to the current date)
    # end_date = datetime.datetime.now(timezone)
    # start_date = end_date - datetime.timedelta(days=60)  # 2 months

    # # Download historical data
    # historical_data = download_data(symbol, timeframe, start_date, end_date)

    # # Print the downloaded data
    # print("\n\nHistorical data:\n", historical_data.iloc[:10, :])

    # # Save output data
    # historical_data.to_csv(f"{symbol} {timeframe}.csv")
