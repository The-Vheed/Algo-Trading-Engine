import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List, Any, Optional, Union


class IndicatorEngine:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the IndicatorEngine with the strategy configuration.

        Args:
            config: The strategy configuration containing indicator definitions
        """
        self.config = config
        # Extract indicators by timeframe from config
        self.indicator_definitions = self._parse_indicator_config()

    def _parse_indicator_config(self) -> Dict[str, List[Dict[str, Any]]]:
        """Parse and extract indicator definitions from the config"""
        try:
            return self.config.get("strategy", {}).get("indicators", {})
        except (KeyError, AttributeError):
            return {}

    def calculate_indicators(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Calculate all indicators defined in the config for each timeframe.

        Args:
            market_data: Dictionary with timeframe keys and DataFrame values
                         containing OHLCV data for each timeframe

        Returns:
            A nested dictionary structure:
            {timeframe: {indicator_name: {property: value}}}
        """
        results = {}

        # Process each timeframe
        for timeframe, indicators in self.indicator_definitions.items():
            if timeframe not in market_data:
                continue

            df = market_data[timeframe].copy()
            results[timeframe] = {}

            # Calculate each indicator for this timeframe
            for indicator_config in indicators:
                indicator_name = indicator_config["name"]
                indicator_type = indicator_config["type"]
                params = indicator_config.get("params", {})

                # Calculate the indicator and store results
                indicator_result = self._calculate_single_indicator(
                    df, indicator_type, params
                )

                if indicator_result is not None:
                    results[timeframe][indicator_name] = indicator_result

        return results

    def _calculate_single_indicator(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate a single indicator using pandas_ta or custom logic.

        Args:
            df: DataFrame with OHLCV data
            indicator_type: Type of indicator to calculate
            params: Parameters for the indicator calculation

        Returns:
            Dictionary with indicator properties and values

        Examples:
            ADX indicator output format:
            {
                "adx": 26.5,              # Current ADX value
                "plus_di": 25.3,          # Current DI+ value
                "minus_di": 12.1,         # Current DI- value
                "series": {               # Historical values
                    "adx": {
                        1609459200000: 24.1,
                        1609545600000: 25.3,
                        1609632000000: 26.5
                    },
                    "plus_di": { ... },
                    "minus_di": { ... }
                }
            }

            MACD indicator output format:
            {
                "macd": 0.0012,           # Current MACD line value
                "signal": 0.0008,         # Current signal line value
                "histogram": 0.0004,      # Current histogram value (MACD - signal)
                "series": {               # Historical values
                    "macd": {
                        1609459200000: 0.0010,
                        1609545600000: 0.0011,
                        1609632000000: 0.0012
                    },
                    "signal": { ... },
                    "histogram": { ... }
                }
            }
        """
        try:
            # Handle different indicator types
            if indicator_type == "ADX":
                period = params.get("period", 14)
                adx = ta.adx(df["high"], df["low"], df["close"], length=period)
                return {
                    "adx": adx[f"ADX_{period}"].iloc[-1],
                    "plus_di": adx[f"DMP_{period}"].iloc[-1],
                    "minus_di": adx[f"DMN_{period}"].iloc[-1],
                    "series": {
                        "adx": adx[f"ADX_{period}"].to_dict(),
                        "plus_di": adx[f"DMP_{period}"].to_dict(),
                        "minus_di": adx[f"DMN_{period}"].to_dict(),
                    },
                }

            elif indicator_type == "EMA":
                period = params.get("period", 20)
                ema = ta.ema(df["close"], length=period)
                return {"value": ema.iloc[-1], "series": ema.to_dict()}

            elif indicator_type == "MACD":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)

                return {
                    "macd": macd[f"MACD_{fast}_{slow}_{signal}"].iloc[-1],
                    "signal": macd[f"MACDs_{fast}_{slow}_{signal}"].iloc[-1],
                    "histogram": macd[f"MACDh_{fast}_{slow}_{signal}"].iloc[-1],
                    "series": {
                        "macd": macd[f"MACD_{fast}_{slow}_{signal}"].to_dict(),
                        "signal": macd[f"MACDs_{fast}_{slow}_{signal}"].to_dict(),
                        "histogram": macd[f"MACDh_{fast}_{slow}_{signal}"].to_dict(),
                    },
                }

            elif indicator_type == "BBW":
                period = params.get("period", 20)
                std_dev = params.get("std_dev", 2.0)
                bbands = ta.bbands(df["close"], length=period, std=std_dev)

                # Calculate bandwidth
                upper = bbands[f"BBU_{period}_{std_dev}"]
                lower = bbands[f"BBL_{period}_{std_dev}"]
                middle = bbands[f"BBM_{period}_{std_dev}"]
                bandwidth = (upper - lower) / middle

                return {
                    "width": bandwidth.iloc[-1],
                    "upper": upper.iloc[-1],
                    "middle": middle.iloc[-1],
                    "lower": lower.iloc[-1],
                    "series": {
                        "width": bandwidth.to_dict(),
                        "upper": upper.to_dict(),
                        "middle": middle.to_dict(),
                        "lower": lower.to_dict(),
                    },
                }

            elif indicator_type == "ATR":
                period = params.get("period", 14)
                atr = ta.atr(df["high"], df["low"], df["close"], length=period)

                return {"value": atr.iloc[-1], "series": atr.to_dict()}

            elif indicator_type == "STOCH":
                k_period = params.get("k_period", 14)
                d_period = params.get("d_period", 3)
                smooth = params.get("smoothing", 3)

                stoch = ta.stoch(
                    df["high"],
                    df["low"],
                    df["close"],
                    k=k_period,
                    d=d_period,
                    smooth=smooth,
                )

                return {
                    "k": stoch[f"STOCHk_{k_period}_{d_period}_{smooth}"].iloc[-1],
                    "d": stoch[f"STOCHd_{k_period}_{d_period}_{smooth}"].iloc[-1],
                    "series": {
                        "k": stoch[f"STOCHk_{k_period}_{d_period}_{smooth}"].to_dict(),
                        "d": stoch[f"STOCHd_{k_period}_{d_period}_{smooth}"].to_dict(),
                    },
                }

            elif indicator_type == "PIVOT":
                method = params.get("method", "classic")
                # Get the previous day's data for pivot calculation
                high = df["high"].iloc[-2]
                low = df["low"].iloc[-2]
                close = df["close"].iloc[-2]

                # Calculate classic pivot points
                pivot = (high + low + close) / 3
                r1 = (2 * pivot) - low
                s1 = (2 * pivot) - high
                r2 = pivot + (high - low)
                s2 = pivot - (high - low)
                r3 = high + 2 * (pivot - low)
                s3 = low - 2 * (high - pivot)

                return {
                    "P": pivot,
                    "R1": r1,
                    "S1": s1,
                    "R2": r2,
                    "S2": s2,
                    "R3": r3,
                    "S3": s3,
                }

            else:
                print(f"Warning: Indicator type {indicator_type} not implemented")
                return None

        except Exception as e:
            print(f"Error calculating {indicator_type}: {str(e)}")
            return None

    # def evaluate_condition(
    #     self, condition: str, indicator_data: Dict[str, Dict[str, Dict[str, Any]]]
    # ) -> bool:
    #     """
    #     Evaluate a condition string against the calculated indicator data.

    #     Args:
    #         condition: Condition string like "H4.regime_adx.adx > 25"
    #         indicator_data: Calculated indicator data

    #     Returns:
    #         Boolean result of the condition
    #     """
    #     # This is a simple placeholder - a real implementation would need a proper expression parser
    #     # For complex conditions with functions like percentile()
    #     try:
    #         # Example basic parsing for simple conditions
    #         if ">" in condition:
    #             left, right = condition.split(">")
    #             left_val = self._get_indicator_value(left.strip(), indicator_data)
    #             right_val = float(right.strip())
    #             return left_val > right_val
    #         elif "<" in condition:
    #             left, right = condition.split("<")
    #             left_val = self._get_indicator_value(left.strip(), indicator_data)
    #             right_val = float(right.strip())
    #             return left_val < right_val
    #         # Add more operators as needed

    #         return False
    #     except Exception as e:
    #         print(f"Error evaluating condition '{condition}': {str(e)}")
    #         return False

    # def _get_indicator_value(
    #     self, path: str, indicator_data: Dict[str, Dict[str, Dict[str, Any]]]
    # ) -> float:
    #     """
    #     Get a specific indicator value from the data structure using a path like "H4.regime_adx.adx"

    #     Args:
    #         path: Path to the indicator value
    #         indicator_data: Calculated indicator data

    #     Returns:
    #         The indicator value
    #     """
    #     parts = path.split(".")
    #     if len(parts) != 3:
    #         raise ValueError(f"Invalid indicator path: {path}")

    #     timeframe, indicator_name, property_name = parts

    #     return (
    #         indicator_data.get(timeframe, {})
    #         .get(indicator_name, {})
    #         .get(property_name, 0.0)
    #     )
