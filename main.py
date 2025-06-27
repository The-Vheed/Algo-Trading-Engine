import sys
import asyncio

from src.utils.logger import Logger

logger = Logger(__name__)

from src.utils.config import ConfigManager
from src.core.data_provider import DataProvider
from src.core.indicator_engine import IndicatorEngine
from src.core.regime_detector import RegimeDetector
from src.core.trading_logic import TradingLogicEngine


DEFAULT_COUNT = 2  # Fallback default


def load_trading_api(config):
    """
    Load the appropriate trading API based on configuration.

    Args:
        config (dict): Configuration dictionary

    Returns:
        Trading API object or None if not configured for live trading
    """
    # if not config.get("live", False):
    #     return None

    trading_api_config = config.get("trading_api", {})
    api_type = trading_api_config.get("type", "").lower()

    if api_type == "mt5_fastapi":
        from src.core.trading_apis.mt5_fastapi import MT5FastAPI

        api_config = trading_api_config.get("config", {})
        return MT5FastAPI(
            base_url=api_config.get("base_url", "http://127.0.0.1:8000"),
            access_key=api_config.get("access_key", ""),
        )
    # Future APIs can be added here
    # elif api_type == "deriv":
    #     from src.core.trading_apis.deriv_api import DerivAPI
    #     return DerivAPI(...)
    else:
        logger.warning(f"Unknown trading API type: {api_type}. Using default.")
        return None


def determine_counts_from_indicators(indicators_config):
    """
    Analyze indicator configurations and determine the required count of data
    for each timeframe based on the largest period parameter in any indicator.

    Args:
        indicators_config (dict): Dictionary mapping timeframes to lists of indicator configs

    Returns:
        dict: Dictionary mapping timeframes to their required data counts
    """
    timeframe_counts = {}

    if not indicators_config:
        logger.warning("No indicator configuration found, using default counts")
        return timeframe_counts

    # Multiplier to ensure we have enough historical data for calculations
    safety_multiplier = 1.1

    for timeframe, indicators in indicators_config.items():
        max_period = 0

        logger.debug(f"Analyzing indicators for timeframe {timeframe}")

        for indicator in indicators:
            params = indicator.get("params", {})
            indicator_name = indicator.get("name", "unknown")

            # Find any parameter containing 'period' in its name
            for param_name, param_value in params.items():
                if "period" in param_name.lower() and isinstance(
                    param_value, (int, float)
                ):
                    if param_value > max_period:
                        max_period = param_value
                        logger.debug(
                            f"New max period found: {max_period} from {indicator_name}.{param_name}"
                        )

            # Special case for MACD which uses fast/slow periods
            if indicator.get("type") == "MACD":
                slow_period = params.get("slow", 0)
                if slow_period > max_period:
                    max_period = slow_period
                    logger.debug(
                        f"New max period found: {max_period} from {indicator_name}.slow"
                    )

        # Set minimum required count with safety multiplier
        if max_period > 0:
            required_count = int(max_period * safety_multiplier)
            logger.info(
                f"Determined count for {timeframe}: {required_count} (based on max period {max_period})"
            )
            timeframe_counts[timeframe] = required_count
        else:
            logger.warning(
                f"No period parameters found for {timeframe}, using default {DEFAULT_COUNT}"
            )
            timeframe_counts[timeframe] = DEFAULT_COUNT

    return timeframe_counts


async def main():
    # Initialize configuration managers
    logger.info("Starting Trading System")
    logger.debug(f"Loading configs...")

    # Variable to track any resources that need cleanup
    live_provider = None
    trading_api = None

    try:
        config = ConfigManager.load_config("config/main.yaml")

        # Load trading API if configured for live trading
        trading_api = load_trading_api(config)
        logger.info(f"Loaded trading API: {type(trading_api).__name__}")

        # Extract symbols from config file
        configured_symbols = config.get("strategy", {}).get("symbols", [])
        if not configured_symbols:
            logger.error("No symbols defined in the config file")
            return

        # Extract timeframes from config file
        configured_timeframes = config.get("strategy", {}).get("timeframes", [])
        if not configured_timeframes:
            logger.error("No timeframes defined in the config file")
            return

        logger.debug(f"Using timeframes from config: {configured_timeframes}")

        # Get indicator configuration
        indicators_config = config.get("strategy", {}).get("indicators", {})

        # Determine counts based on indicator parameters
        timeframe_counts = determine_counts_from_indicators(indicators_config)

        # Ensure all configured timeframes have a count, use default if not determined from indicators
        DEFAULT_COUNT = 2
        for tf in configured_timeframes:
            if tf not in timeframe_counts:
                timeframe_counts[tf] = DEFAULT_COUNT
                logger.warning(
                    f"No indicators found for {tf}, using default count: {DEFAULT_COUNT}"
                )

        logger.debug(f"Using data counts: {timeframe_counts}")

        # Validate indicator timeframes
        indicator_timeframes = set(indicators_config.keys())
        if indicator_timeframes:
            configured_timeframes_set = set(configured_timeframes)
            # Check if any indicator timeframe is not in the configured timeframes
            invalid_timeframes = indicator_timeframes - configured_timeframes_set
            if invalid_timeframes:
                logger.warning(
                    f"Indicator timeframes not in configured timeframes: {invalid_timeframes}"
                )

            # Check if any configured timeframe is missing from indicators
            missing_timeframes = configured_timeframes_set - indicator_timeframes
            if missing_timeframes:
                logger.warning(
                    f"Configured timeframes missing from indicators: {missing_timeframes}"
                )

        # Initialize engines
        logger.debug("Initializing engines...")
        indicator_engine = IndicatorEngine(config)
        regime_detector = RegimeDetector(config)
        trading_logic_engine = TradingLogicEngine(config)

        if not config["live"]:
            logger.info("Backtesting. Loading historical data...")
            try:
                # Get start and end times from config
                backtesting_config = config.get("backtesting", {})
                start_time = backtesting_config.get("start")
                end_time = backtesting_config.get("end")
                data_source = backtesting_config.get("data_source", "csv")

                # For historical_live, we need to pass the trading API
                api_for_data = None
                if data_source == "historical_live":
                    logger.info("Using trading API for historical data")
                    api_for_data = trading_api
                    if not api_for_data:
                        # Initialize trading API if not already done
                        api_for_data = load_trading_api(config)
                        # Connect to the API
                        if api_for_data and hasattr(api_for_data, "connect"):
                            connected = await api_for_data.connect()
                            if not connected:
                                logger.error(
                                    "Failed to connect to trading API for historical data."
                                )
                                return
                            logger.info("Trading API connected for historical data.")

                data_provider = DataProvider(
                    source_type=data_source,
                    base_path="data/historical" if data_source == "csv" else "",
                    symbol=configured_symbols[0],
                    timeframes=configured_timeframes,
                    counts=timeframe_counts,  # Now passing the dictionary of counts
                    trading_api=api_for_data,
                    start_time=start_time,
                    end_time=end_time,
                )

                data_stream = data_provider.stream_data()
                async for data_dict in data_stream:
                    for tf, df in data_dict.items():
                        logger.debug(
                            f"{tf} Length: {len(df)}\n{tf} Data (Last 1):\n"
                            + str(df.tail(1))
                        )

                        # Assert length is correct
                        assert (
                            len(df) == data_provider.counts[tf]
                        ), f"Data yielded {tf} data has length {len(df)}, expected {data_provider.counts[tf]}"

                    # Calculate indicators from the current data snapshot
                    indicators = indicator_engine.calculate_indicators(data_dict)

                    # Detect market regime
                    current_regime = regime_detector.detect_regime(indicators)
                    logger.debug(f"Current market regime: {current_regime}")

                    # Generate trading signals based on the detected regime
                    signals = trading_logic_engine.generate_signals(
                        current_regime, indicators, data_dict
                    )

                    # Log entry signals if any
                    if signals["entry_signals"]:
                        logger.info(
                            f"Entry signals generated: {len(signals['entry_signals'])}"
                        )
                        for signal in signals["entry_signals"]:
                            logger.info(
                                f"  {signal['type']} signal at time {signal['timestamp']} price {signal['price']}"
                            )

                    if signals["exit_signals"]:
                        logger.info(
                            f"Exit levels calculated: {len(signals['exit_signals'])}"
                        )
                        for signal in signals["exit_signals"]:
                            logger.info(
                                f"  {signal['entry_signal_type']} - SL: {signal['stop_loss']}, TP: {signal['take_profit']}"
                            )

                    logger.debug("Calculated indicators:")
                    for timeframe, indicators_dict in indicators.items():
                        logger.debug(f"Timeframe: {timeframe}")
                        for indicator_name, indicator_values in indicators_dict.items():
                            current_values = {
                                k: v
                                for k, v in indicator_values.items()
                                if k != "series"
                            }
                            logger.debug(f"  {indicator_name}: {current_values}")

            except Exception as e:
                logger.error(f"Error during csv data stream: {e}")
                import traceback

                traceback.print_exc()

                # Disconnect the trading API if it was used for historical data
                if api_for_data and hasattr(api_for_data, "disconnect"):
                    await api_for_data.disconnect()
                    logger.debug("Disconnected trading API used for historical data.")
        else:
            logger.info("Live Trading. Loading live data...")
            try:
                # Connect the trading API before passing to DataProvider
                if trading_api and hasattr(trading_api, "connect"):
                    connected = await trading_api.connect()
                    if not connected:
                        logger.error("Failed to connect to trading API.")
                        return
                    logger.info("Trading API connected.")

                live_provider = DataProvider(
                    source_type="live",
                    trading_api=trading_api,  # Pass the trading API object
                    symbol=configured_symbols[0],
                    timeframes=configured_timeframes,
                    counts=timeframe_counts,  # Now passing the dictionary of counts
                )

                live_stream = live_provider.stream_data()
                async for data_dict in live_stream:
                    for tf, df in data_dict.items():
                        logger.debug(
                            f"{tf} Length: {len(df)}\n{tf} Data (Last 1):\n"
                            + str(df.tail(1))
                        )
                        assert (
                            len(df) == live_provider.counts[tf]
                        ), f"Live yielded {tf} data has length {len(df)}, expected {live_provider.counts[tf]}"

                    indicators = indicator_engine.calculate_indicators(data_dict)
                    current_regime = regime_detector.detect_regime(indicators)
                    logger.debug(f"Current market regime (live): {current_regime}")

                    signals = trading_logic_engine.generate_signals(
                        current_regime, indicators, data_dict
                    )

                    if signals["entry_signals"]:
                        logger.info(
                            f"Entry signals generated: {len(signals['entry_signals'])}"
                        )
                        for signal in signals["entry_signals"]:
                            logger.info(
                                f"  {signal['type']} signal at time {signal['timestamp']} price {signal['price']}"
                            )

                    if signals["exit_signals"]:
                        logger.info(
                            f"Exit levels calculated: {len(signals['exit_signals'])}"
                        )
                        for signal in signals["exit_signals"]:
                            logger.info(
                                f"  {signal['entry_signal_type']} - SL: {signal['stop_loss']}, TP: {signal['take_profit']}"
                            )

                    logger.debug("Calculated indicators (live):")
                    for timeframe, indicators_dict in indicators.items():
                        logger.debug(f"Timeframe: {timeframe}")
                        for indicator_name, indicator_values in indicators_dict.items():
                            current_values = {
                                k: v
                                for k, v in indicator_values.items()
                                if k != "series"
                            }
                            logger.debug(f"  {indicator_name}: {current_values}")

            except Exception as e:
                logger.error(f"Error during live data stream: {e}")
                import traceback

                traceback.print_exc()
            finally:
                # Disconnect the trading API directly
                if trading_api and hasattr(trading_api, "disconnect"):
                    await trading_api.disconnect()
                    logger.debug("Disconnected from trading API.")

    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt detected. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        logger.info("\nStopped Trading System")
        # Disconnect the trading API directly
        if trading_api and hasattr(trading_api, "disconnect"):
            await trading_api.disconnect()
            logger.info("Disconnected from trading API.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user. Exiting...")
        sys.exit(0)
