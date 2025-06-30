import pandas as pd
import numpy as np
from ta.trend import ADXIndicator, MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import StochasticOscillator
from typing import Dict, List, Any, Optional, Union
from src.utils.logger import Logger
from scipy.signal import find_peaks

logger = Logger(__name__)


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
        logger.debug(
            f"IndicatorEngine initialized with {len(self.indicator_definitions)} timeframes"
        )

    def _parse_indicator_config(self) -> Dict[str, List[Dict[str, Any]]]:
        """Parse and extract indicator definitions from the config"""
        try:
            indicators = self.config.get("strategy", {}).get("indicators", {})
            if not indicators:
                logger.warning("No indicators defined in configuration")
            else:
                for tf, inds in indicators.items():
                    logger.debug(f"Found {len(inds)} indicators for timeframe {tf}")
            return indicators
        except (KeyError, AttributeError) as e:
            logger.error(f"Error parsing indicator config: {e}")
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
        logger.debug(f"Calculating indicators for {len(market_data)} timeframes")

        # Process each timeframe
        for timeframe, indicators in self.indicator_definitions.items():
            if timeframe not in market_data:
                logger.warning(
                    f"Timeframe {timeframe} not found in market data. Available: {list(market_data.keys())}"
                )
                continue

            df = market_data[timeframe].copy()
            results[timeframe] = {}
            logger.debug(
                f"Processing {len(indicators)} indicators for timeframe {timeframe}"
            )

            # Calculate each indicator for this timeframe
            for indicator_config in indicators:
                indicator_name = indicator_config["name"]
                indicator_type = indicator_config["type"]
                params = indicator_config.get("params", {})

                logger.debug(
                    f"Calculating {indicator_name} ({indicator_type}) with params: {params}"
                )

                # Calculate the indicator and store results
                indicator_result = self._calculate_single_indicator(
                    df, indicator_type, params
                )

                if indicator_result is not None:
                    results[timeframe][indicator_name] = indicator_result
                    logger.debug(
                        f"Successfully calculated {indicator_name} for {timeframe}"
                    )
                else:
                    logger.warning(
                        f"Failed to calculate {indicator_name} for {timeframe}"
                    )

        return results

    def _calculate_single_indicator(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate a single indicator using the 'ta' library.

        Args:
            df: DataFrame with OHLCV data
            indicator_type: Type of indicator to calculate
            params: Parameters for the indicator calculation

        Returns:
            Dictionary with indicator properties and values
        """
        try:
            # Standard column renaming to ensure compatibility
            # 'ta' library expects lowercase column names
            df_lower = df.copy()
            if "Open" in df.columns:
                df_lower.columns = [col.lower() for col in df.columns]
                logger.debug(
                    f"Converted column names to lowercase: {list(df_lower.columns)}"
                )

            # Handle different indicator types
            if indicator_type == "ADX":
                period = params.get("period", 14)
                logger.debug(f"Calculating ADX with period={period}")
                adx_indicator = ADXIndicator(
                    high=df_lower["high"],
                    low=df_lower["low"],
                    close=df_lower["close"],
                    window=period,
                )

                adx = adx_indicator.adx()
                pdi = adx_indicator.adx_pos()
                ndi = adx_indicator.adx_neg()

                logger.debug(
                    f"ADX result: {adx.iloc[-1]:.2f}, +DI: {pdi.iloc[-1]:.2f}, -DI: {ndi.iloc[-1]:.2f}"
                )

                return {
                    "adx": adx.iloc[-1],
                    "plus_di": pdi.iloc[-1],
                    "minus_di": ndi.iloc[-1],
                    "series": {
                        "adx": adx.to_dict(),
                        "plus_di": pdi.to_dict(),
                        "minus_di": ndi.to_dict(),
                    },
                }

            elif indicator_type == "EMA":
                period = params.get("period", 20)
                logger.debug(f"Calculating EMA with period={period}")
                ema_indicator = EMAIndicator(close=df_lower["close"], window=period)
                ema = ema_indicator.ema_indicator()
                logger.debug(f"EMA result: {ema.iloc[-1]:.5f}")
                return {"value": ema.iloc[-1], "series": ema.to_dict()}

            elif indicator_type == "MACD":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                logger.debug(
                    f"Calculating MACD with fast={fast}, slow={slow}, signal={signal}"
                )

                macd_indicator = MACD(
                    close=df_lower["close"],
                    window_fast=fast,
                    window_slow=slow,
                    window_sign=signal,
                )

                macd_line = macd_indicator.macd()
                signal_line = macd_indicator.macd_signal()
                histogram = macd_indicator.macd_diff()

                logger.debug(
                    f"MACD result: macd={macd_line.iloc[-1]:.5f}, signal={signal_line.iloc[-1]:.5f}, histogram={histogram.iloc[-1]:.5f}"
                )

                return {
                    "macd": macd_line.iloc[-1],
                    "signal": signal_line.iloc[-1],
                    "histogram": histogram.iloc[-1],
                    "series": {
                        "macd": macd_line.to_dict(),
                        "signal": signal_line.to_dict(),
                        "histogram": histogram.to_dict(),
                    },
                }

            elif indicator_type == "BBW":
                period = params.get("period", 20)
                std_dev = params.get("std_dev", 2.0)
                logger.debug(
                    f"Calculating Bollinger Bandwidth with period={period}, std_dev={std_dev}"
                )

                bb_indicator = BollingerBands(
                    close=df_lower["close"], window=period, window_dev=std_dev
                )

                upper = bb_indicator.bollinger_hband()
                lower = bb_indicator.bollinger_lband()
                middle = bb_indicator.bollinger_mavg()

                # Calculate bandwidth
                bandwidth = (upper - lower) / middle

                # Calculate historical percentiles for bandwidth if enough data
                if len(bandwidth) >= 20:
                    width_values = bandwidth.dropna().values
                    width_percentile_25 = np.percentile(width_values, 25)
                    width_percentile_75 = np.percentile(width_values, 75)
                    logger.debug(
                        f"BBW percentiles calculated from {len(width_values)} values"
                    )
                else:
                    width_percentile_25 = bandwidth.iloc[-1] * 0.75  # Fallback
                    width_percentile_75 = bandwidth.iloc[-1] * 1.25  # Fallback
                    logger.warning(
                        f"Insufficient data ({len(bandwidth)}) for BBW percentiles, using fallback values"
                    )

                logger.debug(
                    f"BBW result: width={bandwidth.iloc[-1]:.5f}, p25={width_percentile_25:.5f}, p75={width_percentile_75:.5f}"
                )

                return {
                    "width": {
                        "value": bandwidth.iloc[-1],
                        "percentile_25": width_percentile_25,
                        "percentile_75": width_percentile_75,
                    },
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
                logger.debug(f"Calculating ATR with period={period}")
                atr_indicator = AverageTrueRange(
                    high=df_lower["high"],
                    low=df_lower["low"],
                    close=df_lower["close"],
                    window=period,
                )
                atr = atr_indicator.average_true_range()
                logger.debug(f"ATR result: {atr.iloc[-1]:.5f}")
                return {"value": atr.iloc[-1], "series": atr.to_dict()}

            elif indicator_type == "STOCH":
                k_period = params.get("k_period", 14)
                d_period = params.get("d_period", 3)
                smooth = params.get("smoothing", 3)
                logger.debug(
                    f"Calculating Stochastic with k_period={k_period}, d_period={d_period}, smoothing={smooth}"
                )

                stoch_indicator = StochasticOscillator(
                    high=df_lower["high"],
                    low=df_lower["low"],
                    close=df_lower["close"],
                    window=k_period,
                    smooth_window=smooth,
                )

                k = stoch_indicator.stoch()
                d = stoch_indicator.stoch_signal()

                logger.debug(
                    f"Stochastic result: %K={k.iloc[-1]:.2f}, %D={d.iloc[-1]:.2f}"
                )

                return {
                    "k": k.iloc[-1],
                    "d": d.iloc[-1],
                    "series": {
                        "k": k.to_dict(),
                        "d": d.to_dict(),
                    },
                }

            elif indicator_type == "PIVOT":
                method = params.get("method", "classic")
                logger.debug(f"Calculating Pivot Points with method={method}")
                # Get the previous day's data for pivot calculation
                high = df_lower["high"].iloc[-2]
                low = df_lower["low"].iloc[-2]
                close = df_lower["close"].iloc[-2]

                # Calculate classic pivot points
                pivot = (high + low + close) / 3
                r1 = (2 * pivot) - low
                s1 = (2 * pivot) - high
                r2 = pivot + (high - low)
                s2 = pivot - (high - low)
                r3 = high + 2 * (pivot - low)
                s3 = low - 2 * (high - pivot)

                logger.debug(f"Pivot result: P={pivot:.5f}, R1={r1:.5f}, S1={s1:.5f}")

                return {
                    "P": pivot,
                    "R1": r1,
                    "S1": s1,
                    "R2": r2,
                    "S2": s2,
                    "R3": r3,
                    "S3": s3,
                }

            elif indicator_type == "S&R":
                distance = params.get("distance", 10)
                prominence = params.get("prominence", 0.025)
                period = params.get("period", None)
                logger.debug(
                    f"Calculating S&R with distance={distance}, prominence={prominence}, period={period}"
                )

                # Slice the dataframe if a period is specified
                df_slice = df_lower
                if period and len(df_lower) > period:
                    df_slice = df_lower.iloc[-period:]
                    logger.debug(f"Using last {period} bars for S&R calculation")

                # Identify resistance levels (peaks in high prices)
                resistance_indices, _ = find_peaks(
                    df_slice["high"], distance=distance, prominence=prominence
                )
                resistance_prices = df_slice["high"].iloc[resistance_indices]

                # Identify support levels (peaks in inverted low prices, i.e., troughs)
                support_indices, _ = find_peaks(
                    -df_slice["low"], distance=distance, prominence=prominence
                )
                support_prices = df_slice["low"].iloc[support_indices]

                r1 = resistance_prices.iloc[-1] if len(resistance_prices) > 0 else None
                r2 = resistance_prices.iloc[-2] if len(resistance_prices) > 1 else None
                s1 = support_prices.iloc[-1] if len(support_prices) > 0 else None
                s2 = support_prices.iloc[-2] if len(support_prices) > 1 else None

                logger.debug(
                    f"Support/Resistance result: R1={r1}, R2={r2}, S1={s1}, S2={s2}"
                )

                return {
                    "R1": r1,
                    "R2": r2,
                    "S1": s1,
                    "S2": s2,
                    "series": {
                        "resistance": resistance_prices.to_dict(),
                        "support": support_prices.to_dict(),
                    },
                }

            else:
                logger.warning(f"Indicator type '{indicator_type}' not implemented")
                return None

        except Exception as e:
            logger.error(f"Error calculating {indicator_type}: {str(e)}")
            import traceback

            logger.debug(f"Indicator calculation traceback: {traceback.format_exc()}")
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
    #     )
