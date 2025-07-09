from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union, Any
from enum import Enum
from datetime import datetime


class TimeFrameEnum(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"


class SymbolEnum(str, Enum):
    V25 = "Volatility 25 Index"
    SI = "Step Index"
    UJ = "USDJPY"
    EU = "EURUSD"
    SPX = "US SP 500"


class BaseRequest(BaseModel):
    access_key: str = Field(..., description="API access key for authentication")


class OrderRequest(BaseRequest):
    symbol: str = Field(SymbolEnum.V25, description="Trading symbol")
    price: float = Field(
        0.0, description="Current price (optional, used for reference)"
    )
    comment: str = Field("", description="Comment for the order")
    tp: float = Field(1.0, description="Take profit value")
    sl: float = Field(0.2, description="Stop loss value")
    trailing: float = Field(0.0, description="Trailing stop value (0 for no trailing)")
    risk: float = Field(2.0, description="Risk percentage of account balance")


class PriceDataRequest(BaseRequest):
    symbol: str = Field(SymbolEnum.V25, description="Trading symbol")
    timeframes: List[TimeFrameEnum] = Field(
        default=[TimeFrameEnum.M5], description="List of timeframes to fetch"
    )
    start_time: Optional[datetime] = Field(
        None, description="Start time for historical data (UTC)"
    )
    end_time: Optional[datetime] = Field(
        None,
        description="End time for historical data (UTC), defaults to current time if not provided",
    )
    count: Optional[int] = Field(
        None, description="Number of candles to fetch (alternative to start_time)"
    )


class ClosePositionsRequest(BaseRequest):
    pass


class GitOperationRequest(BaseRequest):
    pass


class OrderResponse(BaseModel):
    status: bool = Field(..., description="Order execution status")
    message: str = Field(..., description="Response message")
    order_details: Dict[str, Any] = Field(..., description="MetaTrader order details")


class ClosePositionsResponse(BaseModel):
    status: bool = Field(..., description="Operation status")
    message: str = Field(..., description="Response message")
    closed_positions: List[Dict[str, Any]] = Field(
        ..., description="Details of closed positions"
    )


class PriceDataResponse(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    data: Dict[str, Dict[str, Any]] = Field(
        ..., description="Price data organized by timeframe"
    )
    balance: Optional[float] = Field(None, description="Current account balance")


class PositionsBiasResponse(BaseModel):
    bias: Optional[str] = Field(
        None, description="Trading bias based on open positions (BUY, SELL, or None)"
    )
    positions_count: int = Field(..., description="Number of open positions")
    last_trade: List = Field(..., description="Last trade time and profit")


class AccountInfoResponse(BaseModel):
    bias: Optional[str] = Field(
        None, description="Trading bias based on open positions (BUY, SELL, or None)"
    )
    positions_count: int = Field(..., description="Number of open positions")
    last_trade: List = Field(..., description="Last trade time and profit")
    balance: float = Field(..., description="Current account balance")


class GitOperationResponse(BaseModel):
    status: bool = Field(..., description="Operation status")
