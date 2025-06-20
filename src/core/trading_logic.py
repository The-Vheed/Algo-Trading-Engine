import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from src.utils.logger import Logger

logger = Logger(__name__)


class TradingLogicEngine:
    """
    Processes market data and indicators to generate trading signals
    based on the current regime and trading rules defined in the configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TradingLogicEngine with the strategy configuration.

        Args:
            config: The strategy configuration containing trading logic rules
        """
        self.config = config
        self.trading_logic = self._parse_trading_logic_config()
        logger.info(
            f"TradingLogicEngine initialized with rules for {len(self.trading_logic)} regimes"
        )

    def _parse_trading_logic_config(self) -> Dict[str, Any]:
        """Parse and extract trading logic from the config"""
        try:
            return self.config.get("strategy", {}).get("trading_logic", {})
        except (KeyError, AttributeError):
            logger.warning("No trading logic found in config")
            return {}

    def generate_signals(
        self,
        regime: str,
        indicator_data: Dict[str, Dict[str, Dict[str, Any]]],
        price_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on the current regime and indicator values.

        Args:
            regime: Current market regime (e.g., "TRENDING", "RANGING")
            indicator_data: Dictionary with calculated indicator values
            price_data: Dictionary with price data for different timeframes

        Returns:
            Dictionary with trading signals and related information
        """
        # Default response with no signals
        result = {
            "regime": regime,
            "entry_signals": [],
            "exit_signals": [],
            "risk_management": {},
        }

        # Get trading logic for the current regime
        regime_logic = self.trading_logic.get(regime)
        if not regime_logic:
            logger.warning(f"No trading logic defined for regime: {regime}")
            return result

        # Check entry conditions for BUY and SELL
        entry_signals = self._check_entry_conditions(
            regime_logic, indicator_data, price_data
        )
        if entry_signals:
            result["entry_signals"] = entry_signals

        # Calculate exit levels for any entry signals
        exit_signals = self._calculate_exit_levels(
            regime_logic, entry_signals, indicator_data, price_data
        )
        if exit_signals:
            result["exit_signals"] = exit_signals

        # Add risk management parameters
        result["risk_management"] = self._get_risk_parameters(regime)

        return result

    def _check_entry_conditions(
        self,
        regime_logic: Dict[str, Any],
        indicator_data: Dict[str, Dict[str, Dict[str, Any]]],
        price_data: Dict[str, pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """
        Check if entry conditions are met for the current regime.

        Args:
            regime_logic: Trading logic for the current regime
            indicator_data: Dictionary with calculated indicator values
            price_data: Dictionary with price data for different timeframes

        Returns:
            List of entry signals (can be empty if no conditions are met)
        """
        entry_signals = []

        # Get entry conditions from regime logic
        entries = regime_logic.get("entries", {})

        # Check BUY conditions
        buy_conditions = entries.get("BUY", {}).get("conditions", [])
        if buy_conditions and self._evaluate_conditions(
            buy_conditions, indicator_data, price_data
        ):
            entry_signals.append(
                {
                    "type": "BUY",
                    "timestamp": pd.Timestamp.now(),
                    "price": self._get_current_price(price_data, "close"),
                }
            )
            logger.info(f"BUY signal generated at price {entry_signals[-1]['price']}")

        # Check SELL conditions
        sell_conditions = entries.get("SELL", {}).get("conditions", [])
        if sell_conditions and self._evaluate_conditions(
            sell_conditions, indicator_data, price_data
        ):
            entry_signals.append(
                {
                    "type": "SELL",
                    "timestamp": pd.Timestamp.now(),
                    "price": self._get_current_price(price_data, "close"),
                }
            )
            logger.info(f"SELL signal generated at price {entry_signals[-1]['price']}")

        return entry_signals

    def _calculate_exit_levels(
        self,
        regime_logic: Dict[str, Any],
        entry_signals: List[Dict[str, Any]],
        indicator_data: Dict[str, Dict[str, Dict[str, Any]]],
        price_data: Dict[str, pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """
        Calculate exit levels (stop loss and take profit) for entry signals.

        Args:
            regime_logic: Trading logic for the current regime
            entry_signals: List of entry signals
            indicator_data: Dictionary with calculated indicator values
            price_data: Dictionary with price data for different timeframes

        Returns:
            List of exit signals with calculated levels
        """
        exit_signals = []

        if not entry_signals:
            return exit_signals

        # Get exit logic from regime logic
        exits = regime_logic.get("exits", {})

        for entry in entry_signals:
            signal_type = entry["type"]  # BUY or SELL
            entry_price = entry["price"]

            # Get the stop loss and take profit logic for this signal type
            signal_exits = exits.get(signal_type, {})

            if not signal_exits:
                logger.warning(
                    f"No exit logic defined for {signal_type} in current regime"
                )
                continue

            # Calculate stop loss level
            stop_loss = self._calculate_stop_loss(
                signal_type,
                entry_price,
                signal_exits.get("stop_loss", {}),
                indicator_data,
            )

            # Calculate take profit level
            take_profit = self._calculate_take_profit(
                signal_type,
                entry_price,
                signal_exits.get("take_profit", {}),
                indicator_data,
            )

            exit_signals.append(
                {
                    "entry_signal_type": signal_type,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward_ratio": (
                        abs(take_profit - entry_price) / abs(stop_loss - entry_price)
                        if stop_loss != entry_price
                        else 0
                    ),
                }
            )

            logger.info(
                f"{signal_type} exit levels - SL: {stop_loss}, TP: {take_profit}"
            )

        return exit_signals

    def _calculate_stop_loss(
        self,
        signal_type: str,
        entry_price: float,
        stop_loss_config: Dict[str, Any],
        indicator_data: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> float:
        """
        Calculate the stop loss level based on the configuration.
        """
        method = stop_loss_config.get("method", "fixed")

        if method == "ATR":
            # ATR-based stop loss
            indicator_path = stop_loss_config.get("indicator", "")
            multiplier = stop_loss_config.get("multiplier", 1.5)

            # Extract the ATR value from indicator data
            atr_value = self._get_indicator_value(indicator_path, indicator_data)

            if signal_type == "BUY":
                return entry_price - (atr_value * multiplier)
            else:  # SELL
                return entry_price + (atr_value * multiplier)

        elif method == "fixed":
            # Fixed pip/point distance
            distance = stop_loss_config.get("distance", 50)
            if signal_type == "BUY":
                return entry_price - (distance * 0.0001)  # Assuming 4 decimal places
            else:  # SELL
                return entry_price + (distance * 0.0001)

        elif method == "percentage":
            # Percentage of entry price
            percentage = stop_loss_config.get("percentage", 1.0)
            if signal_type == "BUY":
                return entry_price * (1 - percentage / 100)
            else:  # SELL
                return entry_price * (1 + percentage / 100)

        else:
            logger.warning(f"Unknown stop loss method: {method}. Using default.")
            if signal_type == "BUY":
                return entry_price * 0.99  # 1% below entry
            else:  # SELL
                return entry_price * 1.01  # 1% above entry

    def _calculate_take_profit(
        self,
        signal_type: str,
        entry_price: float,
        take_profit_config: Dict[str, Any],
        indicator_data: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> float:
        """
        Calculate the take profit level based on the configuration.
        """
        method = take_profit_config.get("method", "risk_reward")

        if method == "risk_reward":
            # Risk:reward ratio based on stop loss
            ratio = take_profit_config.get("ratio", 2.0)

            # Get the stop loss config and calculate SL
            stop_loss_config = take_profit_config.get("stop_loss_config", {})
            if not stop_loss_config:
                # If no specific SL config provided for TP calculation, get from the main config
                stop_loss_config = self.trading_logic.get(signal_type, {}).get(
                    "stop_loss", {}
                )

            stop_loss = self._calculate_stop_loss(
                signal_type, entry_price, stop_loss_config, indicator_data
            )

            risk_distance = abs(entry_price - stop_loss)
            if signal_type == "BUY":
                return entry_price + (risk_distance * ratio)
            else:  # SELL
                return entry_price - (risk_distance * ratio)

        elif method == "level":
            # Specific price level (support/resistance, pivot, etc.)
            target = take_profit_config.get("target", "")
            level_path = take_profit_config.get("level", "")

            # Extract the level value from indicator data
            level_value = self._get_indicator_value(level_path, indicator_data)

            return level_value

        elif method == "fixed":
            # Fixed pip/point distance
            distance = take_profit_config.get("distance", 100)
            if signal_type == "BUY":
                return entry_price + (distance * 0.0001)  # Assuming 4 decimal places
            else:  # SELL
                return entry_price - (distance * 0.0001)

        elif method == "percentage":
            # Percentage of entry price
            percentage = take_profit_config.get("percentage", 2.0)
            if signal_type == "BUY":
                return entry_price * (1 + percentage / 100)
            else:  # SELL
                return entry_price * (1 - percentage / 100)

        else:
            logger.warning(f"Unknown take profit method: {method}. Using default.")
            if signal_type == "BUY":
                return entry_price * 1.02  # 2% above entry
            else:  # SELL
                return entry_price * 0.98  # 2% below entry

    def _evaluate_conditions(
        self,
        conditions: List[str],
        indicator_data: Dict[str, Dict[str, Dict[str, Any]]],
        price_data: Dict[str, pd.DataFrame],
    ) -> bool:
        """
        Evaluate a list of conditions against the indicator and price data.
        """
        if not conditions:
            return False

        for condition in conditions:
            # Skip empty conditions
            if not condition.strip():
                continue

            # Replace price variables with actual values
            condition = self._replace_price_variables(condition, price_data)

            # Evaluate the condition
            if not self._evaluate_condition(condition, indicator_data):
                return False

        # All conditions passed
        return True

    def _evaluate_condition(
        self, condition: str, indicator_data: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> bool:
        """
        Evaluate a single condition string against the indicator data.
        """
        logger.debug(f"Evaluating condition: {condition}")

        try:
            # Handle basic comparison operators
            if ">" in condition:
                left, right = condition.split(">")
                left_val = self._get_indicator_value(left.strip(), indicator_data)
                right_val = (
                    self._get_indicator_value(right.strip(), indicator_data)
                    if "." in right.strip()
                    else float(right.strip())
                )
                return left_val > right_val
            elif "<" in condition:
                left, right = condition.split("<")
                left_val = self._get_indicator_value(left.strip(), indicator_data)
                right_val = (
                    self._get_indicator_value(right.strip(), indicator_data)
                    if "." in right.strip()
                    else float(right.strip())
                )
                return left_val < right_val
            elif ">=" in condition:
                left, right = condition.split(">=")
                left_val = self._get_indicator_value(left.strip(), indicator_data)
                right_val = (
                    self._get_indicator_value(right.strip(), indicator_data)
                    if "." in right.strip()
                    else float(right.strip())
                )
                return left_val >= right_val
            elif "<=" in condition:
                left, right = condition.split("<=")
                left_val = self._get_indicator_value(left.strip(), indicator_data)
                right_val = (
                    self._get_indicator_value(right.strip(), indicator_data)
                    if "." in right.strip()
                    else float(right.strip())
                )
                return left_val <= right_val
            elif "==" in condition:
                left, right = condition.split("==")
                left_val = self._get_indicator_value(left.strip(), indicator_data)
                right_val = (
                    self._get_indicator_value(right.strip(), indicator_data)
                    if "." in right.strip()
                    else float(right.strip())
                )
                return left_val == right_val

            logger.warning(f"Unsupported condition format: {condition}")
            return False

        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {str(e)}")
            return False

    def _get_indicator_value(
        self, path: str, indicator_data: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Any:
        """
        Get a specific indicator value from the data structure using a path like "M1.regime_adx.adx"
        """
        # First try to parse as a float for numeric literals
        try:
            return float(path)
        except (ValueError, TypeError):
            pass

        parts = path.split(".")

        # Handle different path lengths
        if len(parts) == 1:
            # Simple value like "25"
            try:
                return float(path)
            except ValueError:
                return path
        elif len(parts) == 2:
            # Something like "indicator.value"
            indicator_name, property_name = parts
            # This is a special case that shouldn't normally happen
            logger.warning(f"Unusual indicator path format: {path}")
            return 0.0
        elif len(parts) == 3:
            # Standard format: "timeframe.indicator_name.property_name"
            timeframe, indicator_name, property_name = parts
            value = (
                indicator_data.get(timeframe, {})
                .get(indicator_name, {})
                .get(property_name, 0.0)
            )
            logger.debug(f"Retrieved value for {path}: {value}")
            return value
        elif len(parts) == 4:
            # Extended format: "timeframe.indicator_name.property_name.sub_property"
            timeframe, indicator_name, property_name, sub_property = parts

            # Get the main property
            value = (
                indicator_data.get(timeframe, {})
                .get(indicator_name, {})
                .get(property_name, {})
            )

            # If it's a dict, get the sub-property
            if isinstance(value, dict):
                value = value.get(sub_property, 0.0)
            else:
                logger.warning(
                    f"Property {property_name} is not a dict, cannot access {sub_property}"
                )
                return 0.0

            logger.debug(f"Retrieved value for {path}: {value}")
            return value
        else:
            logger.error(f"Invalid indicator path format: {path}")
            return 0.0

    def _replace_price_variables(
        self, condition: str, price_data: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Replace price variables in a condition string with actual values.
        """
        # Use the smallest timeframe for price data (assuming it's the most recent)
        timeframe = min(price_data.keys()) if price_data else None

        if not timeframe or timeframe not in price_data:
            logger.warning(f"No price data available for condition: {condition}")
            return condition

        df = price_data[timeframe]
        if df.empty:
            logger.warning(f"Empty price data for timeframe {timeframe}")
            return condition

        # Get the last row for current price data
        last_row = df.iloc[-1]
        previous_row = df.iloc[-2] if len(df) > 1 else last_row

        # Replace price variables with actual values
        replacements = {
            "price.open": str(last_row["Open"]),
            "price.high": str(last_row["High"]),
            "price.low": str(last_row["Low"]),
            "price.close": str(last_row["Close"]),
            "price.previous_close": str(previous_row["Close"]),
            "price.previous_open": str(previous_row["Open"]),
            "price.previous_high": str(previous_row["High"]),
            "price.previous_low": str(previous_row["Low"]),
            "price.entry": str(
                last_row["Close"]
            ),  # Default to current close for entry price
        }

        for var, value in replacements.items():
            condition = condition.replace(var, value)

        return condition

    def _get_current_price(
        self, price_data: Dict[str, pd.DataFrame], price_type: str = "close"
    ) -> float:
        """
        Get the current price from the price data.
        """
        # Use the smallest timeframe for price data (assuming it's the most recent)
        timeframe = min(price_data.keys()) if price_data else None

        if not timeframe or timeframe not in price_data:
            logger.warning(f"No price data available for getting current price")
            return 0.0

        df = price_data[timeframe]
        if df.empty:
            logger.warning(f"Empty price data for timeframe {timeframe}")
            return 0.0

        # Get the last row for current price data
        last_row = df.iloc[-1]

        # Map price_type to column name
        price_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
        }

        column = price_map.get(price_type.lower(), "Close")

        return last_row[column]

    def _get_risk_parameters(self, regime: str) -> Dict[str, Any]:
        """
        Get risk management parameters for the current regime.
        """
        # Get global risk management parameters
        risk_params = self.config.get("strategy", {}).get("risk_management", {}).copy()

        # Add regime-specific risk adjustments if any
        if regime == "NEUTRAL":
            position_scale = (
                self.config.get("strategy", {})
                .get("risk_management", {})
                .get("trade_management", {})
                .get("neutral_regime_position_scale", 0.5)
            )
            risk_params["position_scale"] = position_scale
        else:
            risk_params["position_scale"] = 1.0

        return risk_params
