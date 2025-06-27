import sys
import asyncio

from src.utils.logger import Logger

logger = Logger(__name__)

from src.utils.config import ConfigManager
from src.core.data_provider import DataProvider
from src.core.indicator_engine import IndicatorEngine
from src.core.regime_detector import RegimeDetector
from src.core.trading_logic import TradingLogicEngine


def load_trading_api(config):
    """
    Load the appropriate trading API based on configuration.

    Args:
        config (dict): Configuration dictionary

    Returns:
        Trading API object or None if not configured for live trading
    """
    if not config.get("live", False):
        return None

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
        if config.get("live", False) and trading_api:
            logger.info(f"Loaded trading API: {type(trading_api).__name__}")
        elif config.get("live", False):
            logger.warning("Live trading enabled but no trading API configured")

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

        # Prepare count values for each timeframe
        default_data_count = 3
        timeframe_counts = [default_data_count] * len(configured_timeframes)

        # Validate indicator timeframes
        indicator_timeframes = set(
            config.get("strategy", {}).get("indicators", {}).keys()
        )
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
                csv_provider = DataProvider(
                    source_type=config.get("backtesting", {}).get("data_source", "csv"),
                    base_path="data/historical",
                    symbol=configured_symbols[0],
                    timeframes=configured_timeframes,
                    counts=timeframe_counts,
                )

                csv_stream = csv_provider.stream_data()
                async for data_dict in csv_stream:
                    for tf, df in data_dict.items():
                        logger.debug(
                            f"{tf} Length: {len(df)}\n{tf} Data (Last 1):\n"
                            + str(df.tail(1))
                        )

                        # Assert length is correct for csv data as well
                        assert (
                            len(df) == csv_provider.counts[tf]
                        ), f"CSV yielded {tf} data has length {len(df)}, expected {csv_provider.counts[tf]}"

                    # Calculate indicators from the current data snapshot
                    indicators = indicator_engine.calculate_indicators(data_dict)

                    # Detect market regime
                    current_regime = regime_detector.detect_regime(indicators)
                    logger.info(f"Current market regime: {current_regime}")

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
                                f"  {signal['type']} signal at price {signal['price']}"
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
                    counts=timeframe_counts,
                )
                # No need to call await live_provider.connect() here

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
                    logger.info(f"Current market regime (live): {current_regime}")

                    signals = trading_logic_engine.generate_signals(
                        current_regime, indicators, data_dict
                    )

                    if signals["entry_signals"]:
                        logger.info(
                            f"Entry signals generated: {len(signals['entry_signals'])}"
                        )
                        for signal in signals["entry_signals"]:
                            logger.info(
                                f"  {signal['type']} signal at price {signal['price']}"
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
                if live_provider and hasattr(live_provider, "disconnect"):
                    await live_provider.disconnect()
                    logger.debug("Disconnected from live data provider.")

    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt detected. Shutting down gracefully...")
        if live_provider and hasattr(live_provider, "disconnect"):
            await live_provider.disconnect()
            logger.info("Disconnected from live data provider.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        logger.info("\nStopped Trading System")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user. Exiting...")
        sys.exit(0)
