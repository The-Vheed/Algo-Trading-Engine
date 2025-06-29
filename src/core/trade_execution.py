import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date, timedelta
import pandas as pd
import uuid

from src.utils.logger import Logger

logger = Logger(__name__)


class TradeExecutor:
    """
    Handles execution of trades based on generated signals with risk management.
    Supports both live trading (via trading API) and backtesting.
    """

    def __init__(
        self, config: Dict[str, Any], trading_api=None, is_backtesting: bool = False
    ):
        """
        Initialize the TradeExecutor.

        Args:
            config: Configuration dictionary with strategy and risk settings
            trading_api: Trading API object (required for live trading)
            is_backtesting: Whether running in backtesting mode
        """
        self.config = config
        self.trading_api = trading_api
        self.is_backtesting = is_backtesting

        # Extract risk management settings
        self.risk_settings = config.get("strategy", {}).get("risk_management", {})
        self.risk_per_trade = self.risk_settings.get("risk_per_trade", 2.0)
        self.max_concurrent_trades = self.risk_settings.get("trade_management", {}).get(
            "max_concurrent_trades", 3
        )
        self.allow_opposing_trades = self.risk_settings.get("trade_management", {}).get(
            "allow_opposing_trades", False
        )

        # Extract drawdown protection settings
        self.drawdown_protection = self.risk_settings.get("drawdown_protection", {})
        self.daily_loss_limit = self.drawdown_protection.get("daily_loss_limit", {})
        self.weekly_loss_limit = self.drawdown_protection.get("weekly_loss_limit", {})
        self.monthly_loss_limit = self.drawdown_protection.get("monthly_loss_limit", {})
        self.max_consecutive_losses = self.drawdown_protection.get(
            "max_consecutive_losses", 5
        )

        # Initialize backtesting variables if in backtesting mode
        if is_backtesting:
            backtesting_config = config.get("backtesting", {})
            self.initial_balance = backtesting_config.get("initial_balance", 10000.0)
            self.current_balance = self.initial_balance
            self.commission_per_lot = backtesting_config.get("commission_per_lot", 7.0)
            self.spread_pips = backtesting_config.get("spread_pips", 1.2)
            self.positions = []  # List of open positions for backtesting
            self.closed_positions = []  # Historical closed positions

        # Initialize with default values - will be updated with actual data timestamps
        default_date = date(2000, 1, 1)  # Default date that will be replaced

        # Performance tracking for risk management
        self.performance = {
            "daily": {"date": default_date, "pnl": 0.0, "trades": []},
            "weekly": {"week": 1, "year": 2000, "pnl": 0.0, "trades": []},
            "monthly": {"month": 1, "year": 2000, "pnl": 0.0, "trades": []},
        }

        # Track consecutive losses
        self.consecutive_losses = 0

        # Flag to track if performance periods have been initialized with actual data
        self.performance_initialized = False

        logger.info(
            f"TradeExecutor initialized. Mode: {'Backtesting' if is_backtesting else 'Live Trading'}"
        )

    def _get_week_number(self, dt: datetime) -> Tuple[int, int]:
        """
        Get ISO week number and year for the given datetime.

        Returns:
            Tuple of (week_number, year)
        """
        return dt.isocalendar()[1], dt.isocalendar()[0]

    async def execute_signal(
        self,
        signal: Dict[str, Any],
        exit_parameters: Dict[str, Any],
        current_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Execute a trading signal with risk management checks.

        Args:
            signal: Trading signal with type, timestamp, price, etc.
            exit_parameters: Dictionary with stop loss and take profit levels
            current_time: Current datetime from candlestick data (or system time if None)

        Returns:
            Dictionary with execution results
        """
        # Update performance tracking periods using provided timestamp
        self._update_performance_periods(
            current_time or signal.get("timestamp", datetime.now())
        )

        # Check if we can place the trade based on risk management rules
        can_trade, rejection_reason = self._check_risk_management(signal)
        if not can_trade:
            logger.warning(f"Trade rejected: {rejection_reason}")
            return {"success": False, "message": f"Trade rejected: {rejection_reason}"}

        # Adjust price for spread if backtesting
        adjusted_price = signal["price"]
        if self.is_backtesting:
            pip_size = 0.0001 if signal.get("symbol", "").endswith("USD") else 0.01
            spread_amount = self.spread_pips * pip_size

            if signal["type"] == "BUY":
                adjusted_price += spread_amount  # Buy at ask price (higher)
            else:
                adjusted_price -= spread_amount  # Sell at bid price (lower)

        # Calculate position size based on risk
        position_size = self._calculate_position_size(
            signal.get("symbol", ""),
            adjusted_price,
            exit_parameters.get(
                "stop_loss",
                (
                    adjusted_price * 0.99
                    if signal["type"] == "BUY"
                    else adjusted_price * 1.01
                ),
            ),
            self.risk_per_trade,
        )

        # Execute the trade (differently depending on backtest vs live)
        if self.is_backtesting:
            return self._execute_backtest_trade(
                signal, exit_parameters, position_size, adjusted_price
            )
        else:
            # Live trading via API
            if not self.trading_api:
                return {
                    "success": False,
                    "message": "No trading API available for live execution",
                }

            if signal["type"] == "BUY":
                result = await self.trading_api.execute_buy_order(
                    symbol=signal.get("symbol", ""),
                    price=signal["price"],
                    comment=f"Signal: {signal.get('source', 'Strategy')}",
                    tp=exit_parameters.get("take_profit"),
                    sl=exit_parameters.get("stop_loss"),
                    trailing=0.0,  # Could be configurable
                    risk=self.risk_per_trade,
                )
            else:  # SELL
                result = await self.trading_api.execute_sell_order(
                    symbol=signal.get("symbol", ""),
                    price=signal["price"],
                    comment=f"Signal: {signal.get('source', 'Strategy')}",
                    tp=exit_parameters.get("take_profit"),
                    sl=exit_parameters.get("stop_loss"),
                    trailing=0.0,  # Could be configurable
                    risk=self.risk_per_trade,
                )

            # Process the result
            if result.get("success", False):
                logger.info(
                    f"Successfully executed {signal['type']} order for {signal.get('symbol', '')}"
                )

                # Update performance tracking
                self._record_trade_opened(signal, exit_parameters, position_size)

            return result

    def _execute_backtest_trade(
        self,
        signal: Dict[str, Any],
        exit_parameters: Dict[str, Any],
        position_size: float,
        adjusted_price: float,
    ) -> Dict[str, Any]:
        """
        Execute a trade in backtesting mode.

        Args:
            signal: Trading signal
            exit_parameters: SL/TP parameters
            position_size: Calculated position size
            adjusted_price: Price adjusted for spread

        Returns:
            Execution result
        """
        # Create a unique trade ID
        trade_id = str(uuid.uuid4())

        # Create the position object
        position = {
            "id": trade_id,
            "symbol": signal.get("symbol", ""),
            "type": signal["type"],  # BUY or SELL
            "open_price": adjusted_price,
            "open_time": signal.get("timestamp", datetime.now()),
            "size": position_size,
            "stop_loss": exit_parameters.get("stop_loss"),
            "take_profit": exit_parameters.get("take_profit"),
            "status": "OPEN",
            "pnl": 0.0,
            "close_price": None,
            "close_time": None,
            "close_reason": None,
        }

        # Add to open positions
        self.positions.append(position)

        # Update performance tracking
        self._record_trade_opened(signal, exit_parameters, position_size)

        logger.info(
            f"Backtest: Opened {signal['type']} position of {position_size} units for {signal.get('symbol', '')} at {adjusted_price}"
        )

        return {
            "success": True,
            "message": f"Backtest: Opened {signal['type']} position",
            "order_details": position,
            "trade_id": trade_id,
        }

    def update(
        self, current_data: Dict[str, pd.DataFrame], timeframe: str, symbol: str
    ) -> List[Dict[str, Any]]:
        """
        Update open positions with latest market data (primarily for backtesting).
        Checks if any SL/TP levels have been hit.

        Args:
            current_data: Dictionary of DataFrames by timeframe
            timeframe: Primary timeframe to use for checking positions
            symbol: Current trading symbol

        Returns:
            List of positions that were closed during this update
        """
        if not self.is_backtesting and not self.positions:
            return []

        if timeframe not in current_data:
            logger.warning(
                f"Cannot update positions: timeframe {timeframe} not in current data"
            )
            return []

        df = current_data[timeframe]
        if df.empty:
            return []

        # Use the symbol parameter directly instead of extracting from data or DataFrame
        current_symbol = symbol

        # Get the latest candle
        latest_candle = df.iloc[-1]
        current_time = df.index[-1]

        # Update performance tracking periods using candlestick timestamp
        self._update_performance_periods(current_time)

        closed_positions = []
        remaining_positions = []

        # Check each position
        for position in self.positions:
            # Skip if symbol doesn't match
            if position["symbol"] != current_symbol:
                remaining_positions.append(position)
                continue

            # Check if position was closed
            position_closed = False
            close_price = None
            close_reason = None

            # Check for stop loss hit
            if position["type"] == "BUY":
                # For long positions
                if latest_candle["Low"] <= position["stop_loss"]:
                    close_price = position["stop_loss"]  # Assume filled at exact SL
                    close_reason = "stop_loss"
                    position_closed = True
                elif latest_candle["High"] >= position["take_profit"]:
                    close_price = position["take_profit"]  # Assume filled at exact TP
                    close_reason = "take_profit"
                    position_closed = True
            else:  # SELL position
                # For short positions
                if latest_candle["High"] >= position["stop_loss"]:
                    close_price = position["stop_loss"]
                    close_reason = "stop_loss"
                    position_closed = True
                elif latest_candle["Low"] <= position["take_profit"]:
                    close_price = position["take_profit"]
                    close_reason = "take_profit"
                    position_closed = True

            if position_closed and close_price is not None:
                # Close the position
                closed_position = position.copy()
                closed_position["close_price"] = close_price
                closed_position["close_time"] = current_time
                closed_position["close_reason"] = close_reason
                closed_position["status"] = "CLOSED"

                # Calculate P&L
                pip_size = 0.0001 if position["symbol"].endswith("USD") else 0.01

                if position["type"] == "BUY":
                    pips = (close_price - position["open_price"]) / pip_size
                else:  # SELL
                    pips = (position["open_price"] - close_price) / pip_size

                # Calculate monetary profit/loss
                pnl = pips * position["size"] * 10  # Assuming $10 per pip per lot

                # Apply commission
                commission = (
                    position["size"] / 100000
                ) * self.commission_per_lot  # Assuming size in units
                pnl -= commission

                closed_position["pnl"] = pnl

                # Update account balance
                self.current_balance += pnl

                # Record the closed trade
                self.closed_positions.append(closed_position)
                closed_positions.append(closed_position)

                # Update performance metrics
                self._record_trade_closed(closed_position)

                logger.info(
                    f"Backtest: Closed {position['type']} position for {position['symbol']} at {close_price}. "
                    f"Reason: {close_reason}. P&L: {pnl:.2f}. New balance: {self.current_balance:.2f}"
                )
            else:
                # Position still open
                remaining_positions.append(position)

        # Update positions list
        self.positions = remaining_positions

        return closed_positions

    async def close_all_positions(self) -> Dict[str, Any]:
        """
        Close all open positions.

        Returns:
            Result dictionary with closed positions
        """
        if self.is_backtesting:
            return self._backtest_close_all()
        else:
            # Live trading via API
            if not self.trading_api:
                return {"success": False, "message": "No trading API available"}

            result = await self.trading_api.close_all_positions()

            if result.get("success", False):
                logger.info("Successfully closed all positions")

                # Record closed positions in performance tracking
                closed_count = len(result.get("closed_positions", []))
                if closed_count > 0:
                    # Ideally we would get detailed position data from the API
                    # and update our performance metrics
                    pass

            return result

    def _backtest_close_all(self) -> Dict[str, Any]:
        """
        Close all positions in backtesting mode.

        Returns:
            Result with closed positions
        """
        if not self.positions:
            return {
                "success": True,
                "message": "No positions to close",
                "closed_positions": [],
            }

        closed_details = []
        total_pnl = 0.0

        # Get current prices (in real implementation, you'd get the latest market data)
        # For simplicity, we'll just use the open prices and apply a small spread
        for position in self.positions:
            # In a real implementation, you would get the current market price
            # Here we'll just simulate a small price move from the open price
            if position["type"] == "BUY":
                close_price = position["open_price"] * 0.9995  # Slight loss
            else:
                close_price = position["open_price"] * 1.0005  # Slight loss

            # Calculate P&L
            pip_size = 0.0001 if position["symbol"].endswith("USD") else 0.01

            if position["type"] == "BUY":
                pips = (close_price - position["open_price"]) / pip_size
            else:  # SELL
                pips = (position["open_price"] - close_price) / pip_size

            # Calculate monetary profit/loss
            pnl = pips * position["size"] * 10  # Assuming $10 per pip per lot

            # Apply commission
            commission = (position["size"] / 100000) * self.commission_per_lot
            pnl -= commission

            # Update closed position
            closed_position = position.copy()
            closed_position["close_price"] = close_price
            closed_position["close_time"] = datetime.now()
            closed_position["close_reason"] = "manual"
            closed_position["status"] = "CLOSED"
            closed_position["pnl"] = pnl

            # Add to closed positions
            self.closed_positions.append(closed_position)
            closed_details.append(closed_position)

            # Update performance metrics
            self._record_trade_closed(closed_position)

            total_pnl += pnl

        # Update account balance
        self.current_balance += total_pnl

        # Clear positions
        self.positions = []

        logger.info(
            f"Backtest: Closed all positions. Total P&L: {total_pnl:.2f}. New balance: {self.current_balance:.2f}"
        )

        return {
            "success": True,
            "message": f"Closed {len(closed_details)} positions with total P&L: {total_pnl:.2f}",
            "closed_positions": closed_details,
        }

    def _calculate_position_size(
        self, symbol: str, entry_price: float, stop_loss: float, risk_percentage: float
    ) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss level
            risk_percentage: Risk percentage (0-100)

        Returns:
            Position size in units (standard lots)
        """
        # Get account balance (different for backtesting vs live)
        if self.is_backtesting:
            balance = self.current_balance
        else:
            # For live trading, we would get this from the API
            balance = 10000.0  # Default value if we can't get it

        # Calculate risk amount in currency
        risk_amount = balance * (risk_percentage / 100.0)

        # Calculate pip value of stop loss
        pip_size = 0.0001 if symbol.endswith("USD") else 0.01

        if entry_price == stop_loss:
            logger.warning(
                f"Entry price equals stop loss for {symbol}. Using default position size."
            )
            return 10000  # 0.1 standard lot as default

        pip_distance = abs(entry_price - stop_loss) / pip_size

        if pip_distance == 0:
            logger.warning(
                f"Zero pip distance for {symbol}. Using default position size."
            )
            return 10000  # 0.1 standard lot as default

        # Calculate pip value
        pip_value = 10.0  # Assuming $10 per pip per standard lot (100,000 units)

        # Calculate position size in standard lots
        position_size = risk_amount / (pip_distance * pip_value)

        # Convert to units (standard lot = 100,000 units)
        position_size_units = position_size * 100000

        # Apply minimum and maximum position size constraints
        min_size = 1000  # 0.01 standard lot
        max_size = 500000  # 5 standard lots

        position_size_units = max(min_size, min(position_size_units, max_size))

        return position_size_units

    def _check_risk_management(self, signal: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a trade should be allowed based on risk management rules.

        Args:
            signal: Trading signal

        Returns:
            Tuple of (allowed, reason)
        """
        # Check max concurrent trades
        if not self.is_backtesting:
            # Get current positions count from API in live mode
            pass
        else:
            if len(self.positions) >= self.max_concurrent_trades:
                return (
                    False,
                    f"Maximum concurrent trades ({self.max_concurrent_trades}) reached",
                )

        # Check opposing trades
        if not self.allow_opposing_trades:
            # Check if we have opposite positions
            current_bias = None

            if self.is_backtesting:
                # Determine bias from open positions
                buy_count = sum(1 for p in self.positions if p["type"] == "BUY")
                sell_count = sum(1 for p in self.positions if p["type"] == "SELL")

                if buy_count > sell_count:
                    current_bias = "BUY"
                elif sell_count > buy_count:
                    current_bias = "SELL"
            else:
                # In live mode, we would get this from the API
                pass

            if current_bias is not None and current_bias != signal["type"]:
                return (
                    False,
                    f"Opposing trade not allowed. Current bias: {current_bias}",
                )

        # Check daily loss limit
        if self.daily_loss_limit.get("enabled", False):
            limit_pct = self.daily_loss_limit.get("percentage", 3.0)

            if self.is_backtesting:
                # Calculate daily loss from performance tracking
                initial_balance = self.initial_balance
                daily_loss_pct = (
                    self.performance["daily"]["pnl"] / initial_balance
                ) * 100

                if daily_loss_pct <= -limit_pct:
                    return (
                        False,
                        f"Daily loss limit reached: {daily_loss_pct:.2f}% (limit: {limit_pct}%)",
                    )
            else:
                # In live mode, we would calculate this from the API
                pass

        # Check max consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return (
                False,
                f"Maximum consecutive losses reached: {self.consecutive_losses}",
            )

        # All checks passed
        return True, ""

    def _update_performance_periods(self, current_time: Optional[datetime] = None):
        """
        Update performance tracking periods if day/week/month has changed.

        Args:
            current_time: Current datetime from candlestick data (or system time if None)
        """
        if current_time is None:
            current_time = datetime.now()

        today = current_time.date()
        current_week, current_year = self._get_week_number(current_time)
        current_month = current_time.month

        # Initialize performance tracking with first timestamp if not done yet
        if not self.performance_initialized:
            self.performance["daily"]["date"] = today
            self.performance["weekly"]["week"] = current_week
            self.performance["weekly"]["year"] = current_year
            self.performance["monthly"]["month"] = current_month
            self.performance["monthly"]["year"] = current_year
            self.performance_initialized = True
            return

        # Check if day changed
        if self.performance["daily"]["date"] != today:
            # Reset daily tracking
            self.performance["daily"] = {"date": today, "pnl": 0.0, "trades": []}

        # Check if week changed
        if (
            self.performance["weekly"]["week"] != current_week
            or self.performance["weekly"]["year"] != current_year
        ):
            # Reset weekly tracking
            self.performance["weekly"] = {
                "week": current_week,
                "year": current_year,
                "pnl": 0.0,
                "trades": [],
            }

        # Check if month changed
        if (
            self.performance["monthly"]["month"] != current_month
            or self.performance["monthly"]["year"] != current_year
        ):
            # Reset monthly tracking
            self.performance["monthly"] = {
                "month": current_month,
                "year": current_year,
                "pnl": 0.0,
                "trades": [],
            }

    def _record_trade_opened(
        self,
        signal: Dict[str, Any],
        exit_parameters: Dict[str, Any],
        position_size: float,
    ):
        """Record a new trade in performance tracking"""
        trade_record = {
            "id": str(uuid.uuid4()),
            "type": signal["type"],
            "symbol": signal.get("symbol", ""),
            "open_time": signal.get("timestamp", datetime.now()),
            "open_price": signal["price"],
            "size": position_size,
            "stop_loss": exit_parameters.get("stop_loss"),
            "take_profit": exit_parameters.get("take_profit"),
            "status": "OPEN",
        }

        # Add to daily, weekly, monthly tracking
        self.performance["daily"]["trades"].append(trade_record)
        self.performance["weekly"]["trades"].append(trade_record)
        self.performance["monthly"]["trades"].append(trade_record)

    def _record_trade_closed(self, closed_position: Dict[str, Any]):
        """Record a closed trade in performance tracking"""
        trade_id = closed_position["id"]
        pnl = closed_position["pnl"]

        # Update P&L in all tracking periods
        self.performance["daily"]["pnl"] += pnl
        self.performance["weekly"]["pnl"] += pnl
        self.performance["monthly"]["pnl"] += pnl

        # Update consecutive losses tracking
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Update trade status in all tracking periods
        for period in ["daily", "weekly", "monthly"]:
            for trade in self.performance[period]["trades"]:
                if trade["id"] == trade_id:
                    trade["status"] = "CLOSED"
                    trade["close_price"] = closed_position["close_price"]
                    trade["close_time"] = closed_position["close_time"]
                    trade["close_reason"] = closed_position["close_reason"]
                    trade["pnl"] = pnl
                    break

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        # First update periods if needed
        self._update_performance_periods()

        # Calculate metrics
        metrics = {
            "balance": self.current_balance if self.is_backtesting else None,
            "open_positions": len(self.positions) if self.is_backtesting else None,
            "total_closed_trades": (
                len(self.closed_positions) if self.is_backtesting else None
            ),
            "consecutive_losses": self.consecutive_losses,
            "daily": {
                "pnl": self.performance["daily"]["pnl"],
                "trades_count": len(self.performance["daily"]["trades"]),
                "win_rate": self._calculate_win_rate(
                    self.performance["daily"]["trades"]
                ),
            },
            "weekly": {
                "pnl": self.performance["weekly"]["pnl"],
                "trades_count": len(self.performance["weekly"]["trades"]),
                "win_rate": self._calculate_win_rate(
                    self.performance["weekly"]["trades"]
                ),
            },
            "monthly": {
                "pnl": self.performance["monthly"]["pnl"],
                "trades_count": len(self.performance["monthly"]["trades"]),
                "win_rate": self._calculate_win_rate(
                    self.performance["monthly"]["trades"]
                ),
            },
        }

        return metrics

    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from a list of trades"""
        closed_trades = [t for t in trades if t.get("status") == "CLOSED"]

        if not closed_trades:
            return 0.0

        winning_trades = sum(1 for t in closed_trades if t.get("pnl", 0) > 0)
        return (winning_trades / len(closed_trades)) * 100 if closed_trades else 0.0
