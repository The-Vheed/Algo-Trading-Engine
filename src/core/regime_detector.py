import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import re
from src.utils.logger import Logger

logger = Logger(__name__)

class RegimeDetector:
    """
    Detects the current market regime based on indicator data and configuration rules.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RegimeDetector with the strategy configuration.
        
        Args:
            config: The strategy configuration containing regime logic rules
        """
        self.config = config
        self.regime_rules = self._parse_regime_config()
        self.current_regime = "NEUTRAL"  # Default regime
        self.regime_history = []  # Keep track of regime changes
        self.historical_indicator_data = {}  # Store historical data for percentile calculations
        
        # Pre-compute percentiles for indicators that need them
        self._compute_percentiles()
        
        logger.info(f"RegimeDetector initialized with {len(self.regime_rules)} rules")
        for rule in self.regime_rules:
            logger.info(f"Loaded regime rule: {rule['name']} with {len(rule['conditions'])} conditions")
    
    def _parse_regime_config(self) -> List[Dict[str, Any]]:
        """Parse and extract regime rules from the config"""
        try:
            regime_logic = self.config.get("strategy", {}).get("regime_logic", {})
            return regime_logic.get("rules", [])
        except (KeyError, AttributeError):
            logger.warning("No regime logic rules found in config")
            return []
    
    def _compute_percentiles(self):
        """
        Pre-compute percentiles for indicators that reference them.
        This will be updated as new data comes in.
        """
        self.percentiles = {}
    
    def update_historical_data(self, indicator_data: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Update the historical indicator data used for percentile calculations.
        
        Args:
            indicator_data: The latest indicator data
        """
        # Append new data points to historical records
        for timeframe, indicators in indicator_data.items():
            if timeframe not in self.historical_indicator_data:
                self.historical_indicator_data[timeframe] = {}
                
            for indicator_name, indicator_values in indicators.items():
                if indicator_name not in self.historical_indicator_data[timeframe]:
                    self.historical_indicator_data[timeframe][indicator_name] = {}
                
                # Store data for each property (excluding series data)
                for prop, value in indicator_values.items():
                    if prop != "series":
                        if prop not in self.historical_indicator_data[timeframe][indicator_name]:
                            self.historical_indicator_data[timeframe][indicator_name][prop] = []
                        
                        self.historical_indicator_data[timeframe][indicator_name][prop].append(value)
                        
                        # Limit history size to avoid memory issues (keep last 100 values)
                        self.historical_indicator_data[timeframe][indicator_name][prop] = \
                            self.historical_indicator_data[timeframe][indicator_name][prop][-100:]
                        
                        # Calculate percentiles for this property
                        hist_values = self.historical_indicator_data[timeframe][indicator_name][prop]
                        if len(hist_values) >= 20:  # Need enough data for meaningful percentiles
                            percentile_25 = np.percentile(hist_values, 25)
                            percentile_75 = np.percentile(hist_values, 75)
                            
                            # Add percentile values to the indicator data for reference in conditions
                            if "percentile_25" not in indicator_values:
                                indicator_values["percentile_25"] = percentile_25
                            else:
                                indicator_values["percentile_25"] = percentile_25
                                
                            if "percentile_75" not in indicator_values:
                                indicator_values["percentile_75"] = percentile_75
                            else:
                                indicator_values["percentile_75"] = percentile_75
                            
                            # Also add to property-specific percentiles
                            if prop not in indicator_values:
                                indicator_values[prop] = {}
                            if isinstance(indicator_values[prop], dict):
                                indicator_values[prop]["percentile_25"] = percentile_25
                                indicator_values[prop]["percentile_75"] = percentile_75
    
    def detect_regime(self, indicator_data: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
        """
        Detect the current market regime based on indicator data and configuration rules.
        
        Args:
            indicator_data: Dictionary with calculated indicator values
            
        Returns:
            String representing the detected market regime
        """
        # Update historical data for percentile calculations
        self.update_historical_data(indicator_data)
        
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
                logger.info(f"Detected regime: {regime_name} - all conditions met")
                new_regime = regime_name
                break
        else:
            # If no regime rules matched, use default NEUTRAL
            logger.debug("No regime rules matched, using NEUTRAL as default")
            new_regime = "NEUTRAL"
        
        # Record regime change if it's different
        if new_regime != self.current_regime:
            logger.info(f"Regime changed from {self.current_regime} to {new_regime}")
            self.regime_history.append((pd.Timestamp.now(), new_regime))
            self.current_regime = new_regime
        
        return new_regime
    
    def _evaluate_condition(self, condition: str, indicator_data: Dict[str, Dict[str, Dict[str, Any]]]) -> bool:
        """
        Evaluate a single condition string against the indicator data.
        
        Args:
            condition: Condition string (e.g., "M1.regime_adx.adx > 25")
            indicator_data: Dictionary with calculated indicator values
            
        Returns:
            Boolean result of the condition evaluation
        """
        logger.debug(f"Evaluating condition: {condition}")
        
        try:
            # Handle basic comparison operators
            if ">" in condition:
                left, right = condition.split(">")
                left_val = self._get_indicator_value(left.strip(), indicator_data)
                right_val = self._get_indicator_value(right.strip(), indicator_data) if "." in right.strip() else float(right.strip())
                return left_val > right_val
            elif "<" in condition:
                left, right = condition.split("<")
                left_val = self._get_indicator_value(left.strip(), indicator_data)
                right_val = self._get_indicator_value(right.strip(), indicator_data) if "." in right.strip() else float(right.strip())
                return left_val < right_val
            elif ">=" in condition:
                left, right = condition.split(">=")
                left_val = self._get_indicator_value(left.strip(), indicator_data)
                right_val = self._get_indicator_value(right.strip(), indicator_data) if "." in right.strip() else float(right.strip())
                return left_val >= right_val
            elif "<=" in condition:
                left, right = condition.split("<=")
                left_val = self._get_indicator_value(left.strip(), indicator_data)
                right_val = self._get_indicator_value(right.strip(), indicator_data) if "." in right.strip() else float(right.strip())
                return left_val <= right_val
            elif "==" in condition:
                left, right = condition.split("==")
                left_val = self._get_indicator_value(left.strip(), indicator_data)
                right_val = self._get_indicator_value(right.strip(), indicator_data) if "." in right.strip() else right.strip()
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
    
    def _get_indicator_value(self, path: str, indicator_data: Dict[str, Dict[str, Dict[str, Any]]]) -> Any:
        """
        Get a specific indicator value from the data structure using a path like "M1.regime_adx.adx"
        
        Args:
            path: Path to the indicator value
            indicator_data: Calculated indicator data
            
        Returns:
            The indicator value
        """
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
        elif len(parts) == 4:
            # Extended format: "timeframe.indicator_name.property_name.sub_property"
            timeframe, indicator_name, property_name, sub_property = parts
            
            # Get the main property
            value = indicator_data.get(timeframe, {}).get(indicator_name, {}).get(property_name, {})
            
            # If it's a dict, get the sub-property
            if isinstance(value, dict):
                value = value.get(sub_property, 0.0)
            else:
                logger.warning(f"Property {property_name} is not a dict, cannot access {sub_property}")
                return 0.0
                
            logger.debug(f"Retrieved value for {path}: {value}")
            return value
        else:
            logger.error(f"Invalid indicator path format: {path}")
            return 0.0
            
        # Standard 3-part path handling
        value = indicator_data.get(timeframe, {}).get(indicator_name, {}).get(property_name, 0.0)
        logger.debug(f"Retrieved value for {path}: {value}")
        return value
    
    def get_regime_history(self) -> List[tuple]:
        """Return the history of regime changes"""
        return self.regime_history
    
    def get_current_regime(self) -> str:
        """Return the current detected regime"""
        return self.current_regime
