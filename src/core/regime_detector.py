import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import re
from src.utils.logger import Logger

logger = Logger(__name__)


class RegimeDetector:
    """
    Detects the current market regime based on indicator data and configuration rules.
    Simplified version that uses series data from indicators instead of storing historical values.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RegimeDetector with the strategy configuration.

        Args:
            config: The strategy configuration containing regime logic rules
        """
        self.config = config
        self.regime_rules = self._parse_regime_config()
        # Store regimes by symbol
        self.current_regimes = {}  # {symbol: regime}
        self.regime_history = {}  # {symbol: [(timestamp, regime), ...]}

        logger.debug(f"RegimeDetector initialized with {len(self.regime_rules)} rules")
        for rule in self.regime_rules:
            logger.debug(
                f"Loaded regime rule: {rule['name']} with {len(rule['conditions'])} conditions"
            )

    def _parse_regime_config(self) -> List[Dict[str, Any]]:
        """Parse and extract regime rules from the config"""
        try:
            regime_logic = self.config.get("strategy", {}).get("regime_logic", {})
            return regime_logic.get("rules", [])
        except (KeyError, AttributeError):
            logger.warning("No regime logic rules found in config")
            return []

    def detect_regime(
        self, indicator_data: Dict[str, Dict[str, Dict[str, Any]]], symbol: str = ""
    ) -> str:
        """
        Detect the current market regime based on indicator data and configuration rules.

        Args:
            indicator_data: Dictionary with calculated indicator values
            symbol: Trading symbol to track regime for

        Returns:
            String representing the detected market regime
        """
        # Evaluate each regime rule in order
        for rule in self.regime_rules:
            regime_name = rule["name"]
            conditions = rule["conditions"]

            # Special case for 'default' regime
            if len(conditions) == 1 and conditions[0] == "default":
                logger.debug(f"Using default regime: {regime_name}")
                new_regime = regime_name
                break

            # Evaluate all conditions for this regime
            all_conditions_met = True
            for condition in conditions:
                result = self._evaluate_condition(condition, indicator_data)
                if not result:
                    all_conditions_met = False
                    break

            if all_conditions_met:
                logger.debug(f"Detected regime: {regime_name} - all conditions met")
                new_regime = regime_name
                break
        else:
            # If no regime rules matched, use default NEUTRAL
            logger.debug("No regime rules matched, using NEUTRAL as default")
            new_regime = "NEUTRAL"

        # Get current regime for this symbol
        current_regime = self.current_regimes.get(symbol, "NEUTRAL")

        # Record regime change if it's different
        if new_regime != current_regime:
            logger.info(
                f"Regime for {symbol} changed from {current_regime} to {new_regime}"
            )

            # Initialize history for this symbol if not exists
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []

            self.regime_history[symbol].append((pd.Timestamp.now(), new_regime))
            self.current_regimes[symbol] = new_regime

        return new_regime

    def _evaluate_expression(
        self, expression: str, indicator_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Evaluate a simple arithmetic expression which can contain an indicator path.
        e.g., "H1;range_sr;S1 * 0.992", "25.0", or "H1;some;value"
        """
        expression = expression.strip()

        # First try to parse as a float for numeric literals
        try:
            return float(expression)
        except (ValueError, TypeError):
            pass

        # Check for arithmetic operations (order matters for proper parsing)
        operators = ["*", "/", "+", "-"]
        for op in operators:
            if op in expression:
                # Split only on the first occurrence of the operator
                parts = expression.split(op, 1)
                left = parts[0].strip()
                right = parts[1].strip()

                # Recursively evaluate both sides
                try:
                    left_val = self._evaluate_expression(left, indicator_data)
                    right_val = self._evaluate_expression(right, indicator_data)

                    if op == "*":
                        return left_val * right_val
                    elif op == "/":
                        return left_val / right_val if right_val != 0 else float("inf")
                    elif op == "+":
                        return left_val + right_val
                    elif op == "-":
                        return left_val - right_val
                except Exception as e:
                    logger.error(
                        f"Error evaluating expression '{expression}': {str(e)}"
                    )
                    return 0.0

        # If no operators, get the indicator value
        return self._get_indicator_value(expression, indicator_data)

    def _evaluate_condition(
        self, condition: str, indicator_data: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> bool:
        """
        Evaluate a single condition string against the indicator data.

        Args:
            condition: Condition string (e.g., "M1;regime_adx;adx > 25")
            indicator_data: Dictionary with calculated indicator values

        Returns:
            Boolean result of the condition evaluation
        """
        logger.debug(f"Evaluating condition: {condition}")

        try:
            # Define all comparison operators to check (order matters for proper parsing)
            operators = [">=", "<=", "==", ">", "<"]

            for op in operators:
                if op in condition:
                    left, right = condition.split(op, 1)
                    # Evaluate both sides as potential expressions
                    left_val = self._evaluate_expression(left.strip(), indicator_data)
                    right_val = self._evaluate_expression(right.strip(), indicator_data)

                    if op == ">":
                        return left_val > right_val
                    elif op == "<":
                        return left_val < right_val
                    elif op == ">=":
                        return left_val >= right_val
                    elif op == "<=":
                        return left_val <= right_val
                    elif op == "==":
                        # Try float comparison first, fall back to string if needed
                        try:
                            if not isinstance(right_val, float):
                                right_val = float(right_val)
                            return left_val == right_val
                        except (ValueError, TypeError):
                            return str(left_val) == str(right_val)

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

        Args:
            path: Path to the indicator value
            indicator_data: Calculated indicator data

        Returns:
            The indicator value
        """
        parts = path.split(";")

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
            # value = (
            #     indicator_data.get(timeframe, {})
            #     .get(indicator_name, {})
            #     .get(property_name, 0.0)
            # )
            # logger.debug(f"Retrieved value for {path}: {value}")
            # return value
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

        # Standard 3-part path handling
        value = (
            indicator_data.get(timeframe, {})
            .get(indicator_name, {})
            .get(property_name, 0.0)
        )
        logger.debug(f"Retrieved value for {path}: {value}")
        return value

    def get_regime_history(self, symbol: str = "") -> List[tuple]:
        """
        Return the history of regime changes for a specific symbol.

        Args:
            symbol: Symbol to get regime history for

        Returns:
            List of (timestamp, regime) tuples
        """
        return self.regime_history.get(symbol, [])

    def get_current_regime(self, symbol: str = "") -> str:
        """
        Return the current detected regime for a specific symbol.

        Args:
            symbol: Symbol to get current regime for

        Returns:
            Current regime string
        """
        return self.current_regimes.get(symbol, "NEUTRAL")
