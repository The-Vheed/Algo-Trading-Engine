import sys
import asyncio

from src.utils.logger import Logger

logger = Logger(__name__)

from src.utils.config import ConfigManager
from src.core.data_provider import DataProvider
from src.core.indicator_engine import IndicatorEngine
from src.core.regime_detector import RegimeDetector


async def main():
    # Initialize configuration managers
    logger.info("Starting Trading System")
    logger.info(f"Loading configs...")
    config = ConfigManager.load_config("config/main.yaml")

    # Extract timeframes from config file
    configured_timeframes = config.get("strategy", {}).get("timeframes", [])
    if not configured_timeframes:
        logger.error("No timeframes defined in the config file")
        return

    logger.info(f"Using timeframes from config: {configured_timeframes}")

    # Prepare count values for each timeframe (using the same count for all timeframes)
    # This can be enhanced to have different counts per timeframe if needed
    default_data_count = 500  # Default count of candles to fetch per timeframe
    timeframe_counts = [default_data_count] * len(configured_timeframes)

    # Validate that indicator timeframes match the configured timeframes
    indicator_timeframes = set(config.get("strategy", {}).get("indicators", {}).keys())
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

    # Initialize indicator engine with the loaded config
    logger.info("Initializing indicator engine...")
    indicator_engine = IndicatorEngine(config)
    logger.info(
        f"Indicator definitions parsed: {len(indicator_engine.indicator_definitions)} timeframes"
    )

    # Initialize regime detector
    logger.info("Initializing regime detector...")
    regime_detector = RegimeDetector(config)

    if not config["live"]:
        logger.info("Backtesting. Loading historical data...")
        try:
            if config["backtesting"]["data_source"] == "CSV":
                csv_provider = DataProvider(
                    source_type="csv",
                    base_path="data/historical",
                    symbol="EURUSD",
                    timeframes=configured_timeframes,
                    counts=timeframe_counts,
                )
            else:
                csv_provider = DataProvider(
                    source_type="csv",
                    base_path="data/historical",
                    symbol="EURUSD",
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

                # Log some indicator values for verification
                logger.info("Calculated indicators:")
                for timeframe, indicators_dict in indicators.items():
                    logger.info(f"Timeframe: {timeframe}")
                    for indicator_name, indicator_values in indicators_dict.items():
                        # Format indicator values for logging (just the current values, not the series)
                        current_values = {
                            k: v for k, v in indicator_values.items() if k != "series"
                        }
                        logger.info(f"  {indicator_name}: {current_values}")

                # Here you would proceed with strategy execution using the detected regime

        except Exception as e:
            logger.error(f"Error during csv data stream: {e}")
            import traceback

            traceback.print_exc()
    else:
        logger.info("Live Trading. Loading live data...")
        try:
            live_provider = DataProvider(
                source_type="live",
                app_id=config["api"]["app_id"],
                api_key=config["api"]["key"],
                symbol="R_25",
                timeframes=configured_timeframes,
                counts=timeframe_counts,
            )
            await live_provider.connect()
            # logger.info("Connected to live data provider.")

            live_stream = live_provider.stream_data()
            async for data_dict in live_stream:
                for tf, df in data_dict.items():
                    logger.debug(
                        f"{tf} Length: {len(df)}\n{tf} Data (Last 1):\n"
                        + str(df.tail(1))
                    )
                    # Assert length is correct for csv data as well
                    assert (
                        len(df) == live_provider.counts[tf]
                    ), f"Live yielded {tf} data has length {len(df)}, expected {live_provider.counts[tf]}"

                # Calculate indicators from the live data snapshot
                indicators = indicator_engine.calculate_indicators(data_dict)

                # Detect market regime
                current_regime = regime_detector.detect_regime(indicators)
                logger.info(f"Current market regime (live): {current_regime}")

                # Log some indicator values for verification
                logger.info("Calculated indicators (live):")
                for timeframe, indicators_dict in indicators.items():
                    logger.info(f"Timeframe: {timeframe}")
                    for indicator_name, indicator_values in indicators_dict.items():
                        # Format indicator values for logging (just the current values, not the series)
                        current_values = {
                            k: v for k, v in indicator_values.items() if k != "series"
                        }
                        logger.info(f"  {indicator_name}: {current_values}")

                # Here you would proceed with strategy execution using the detected regime

        except Exception as e:
            logger.error(f"Error during live data stream: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Ensure we disconnect properly
            if "live_provider" in locals() and hasattr(live_provider, "disconnect"):
                await live_provider.disconnect()
                logger.info("Disconnected from live data provider.")

    # indicator_engine = IndicatorEngine()
    # regime_detector = RegimeDetector()
    # strategy_engine = StrategyEngine()
    # risk_manager = RiskManager()
    # backtest_engine = BacktestEngine(config, data_provider)
    # api_client = DerivAPIClient(config['api_token'], config['app_id'])
    # logger = Logger()

    # # Load historical data
    # historical_data = data_provider.load_csv_data(config['data_file_path'], config['timeframe'])

    # # Run backtest if enabled
    # if config['backtesting_enabled']:
    #     results = backtest_engine.run_backtest(config['start_date'], config['end_date'])
    #     report = backtest_engine.generate_report(results)
    #     logger.log(report)

    # # Start live trading if enabled
    # if config['live_trading_enabled']:
    #     # Implement live trading logic here
    #     pass

    logger.info("Stopped Trading System")


if __name__ == "__main__":
    asyncio.run(main())
