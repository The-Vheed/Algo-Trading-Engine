import sys
import asyncio
import pandas as pd
import os
import json

from src.utils.logger import Logger

logger = Logger(__name__)

from src.utils.config import ConfigManager
from src.core.data_provider import DataProvider
from src.core.indicator_engine import IndicatorEngine
from src.core.regime_detector import RegimeDetector
from src.core.trading_logic import TradingLogicEngine
from src.core.trade_execution import TradeExecutor


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
    safety_multiplier = 1.25

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
    trade_executor = None

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

        # Define primary timeframe for position updates (shortest timeframe)
        primary_timeframe = sorted(
            configured_timeframes,
            key=lambda tf: tf[0] + str(int(tf[1:]) if tf[1:].isdigit() else 0),
        )[0]
        logger.info(
            f"Using {primary_timeframe} as primary timeframe for position updates"
        )

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

        # Initialize trade executor
        is_backtesting = not config["live"]
        trade_executor = TradeExecutor(
            config=config,
            trading_api=trading_api,
            is_backtesting=is_backtesting,
        )
        logger.info(
            f"Trade executor initialized in {'backtesting' if is_backtesting else 'live'} mode"
        )

        if is_backtesting:
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
                    counts=timeframe_counts,
                    trading_api=api_for_data,
                    start_time=start_time,
                    end_time=end_time,
                )

                data_stream = data_provider.stream_data()
                async for data_dict, current_symbol in data_stream:
                    # Now current_symbol comes directly from the data provider

                    # Log data for each timeframe
                    for tf, df in data_dict.items():
                        if isinstance(
                            df, pd.DataFrame
                        ):  # All entries should be DataFrames now
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
                    current_regime = regime_detector.detect_regime(
                        indicators, current_symbol
                    )
                    logger.debug(
                        f"Current market regime for {current_symbol}: {current_regime}"
                    )

                    # Update trade executor with latest data to check SL/TP hits and regime changes
                    current_time = data_dict[primary_timeframe].index[-1]
                    closed_positions = await trade_executor.update(
                        data_dict, primary_timeframe, current_symbol, current_regime
                    )

                    # Log closed positions if any
                    if closed_positions:
                        logger.info(
                            f"Closed {len(closed_positions)} positions at {current_time}"
                        )
                        for pos in closed_positions:
                            logger.info(
                                f"  {pos['type']} position closed at {pos['close_price']}. "
                                f"Reason: {pos['close_reason']}. P&L: {pos['pnl']:.2f}"
                            )

                    # Generate trading signals based on the detected regime
                    signals = trading_logic_engine.generate_signals(
                        current_regime, indicators, data_dict, current_symbol
                    )

                    # Execute entry signals if any
                    if signals["entry_signals"]:
                        logger.info(
                            f"Entry signals generated: {len(signals['entry_signals'])}"
                        )
                        for signal in signals["entry_signals"]:
                            # Ensure symbol is set in the signal
                            if "symbol" not in signal:
                                signal["symbol"] = current_symbol

                            logger.info(
                                f"  {signal['type']} signal for {signal['symbol']} at time {signal['timestamp']} price {signal['price']}"
                            )

                            # Find matching exit signal
                            exit_params = next(
                                (
                                    exit_signal
                                    for exit_signal in signals["exit_signals"]
                                    if exit_signal["entry_signal_type"]
                                    == signal["type"]
                                ),
                                {"stop_loss": None, "take_profit": None},
                            )

                            # Execute signal
                            execution_result = await trade_executor.execute_signal(
                                signal=signal,
                                exit_parameters=exit_params,
                                current_time=current_time,
                            )

                            if execution_result.get("success", False):
                                logger.info(
                                    f"Successfully executed {signal['type']} order"
                                )
                            else:
                                logger.warning(
                                    f"Failed to execute {signal['type']} order: {execution_result.get('message', 'unknown error')}"
                                )

                    # Log exit signals if any
                    if signals["exit_signals"]:
                        logger.info(
                            f"Exit levels calculated: {len(signals['exit_signals'])}"
                        )
                        for signal in signals["exit_signals"]:
                            logger.info(
                                f"  {signal['entry_signal_type']} - SL: {signal['stop_loss']}, TP: {signal['take_profit']}"
                            )

                    # Log performance metrics periodically
                    metrics = trade_executor.get_performance_metrics()
                    logger.debug(f"Current balance: {metrics['balance']}")
                    logger.debug(f"Open positions: {metrics['open_positions']}")
                    logger.debug(f"Daily P&L: {metrics['daily']['pnl']:.2f}")

                # Close all positions at the end of backtesting
                final_result = await trade_executor.close_all_positions()
                logger.info(
                    f"Backtesting complete. Final balance: {trade_executor.current_balance:.2f}"
                )

                # --- EXPORT BACKTEST RESULTS ---
                backtest_cfg = config.get("backtesting", {})
                output_folder = backtest_cfg.get("output_folder", "backtest_results")
                save_datastore = backtest_cfg.get("save_datastore", False)
                save_trade_history = backtest_cfg.get("save_trade_history", False)

                os.makedirs(output_folder, exist_ok=True)

                # Save price data (datastore)
                if save_datastore:
                    price_data_dir = os.path.join(output_folder, "price_data")
                    os.makedirs(price_data_dir, exist_ok=True)
                    for tf, df in data_provider.data_store.items():
                        if isinstance(df, pd.DataFrame):
                            df.to_csv(os.path.join(price_data_dir, f"{tf}.csv"))
                    logger.info(f"Saved price data to {price_data_dir}")

                # Save trade history
                if save_trade_history:
                    trades_path = os.path.join(output_folder, "trade_history.json")
                    with open(trades_path, "w") as f:
                        json.dump(
                            trade_executor.closed_positions, f, indent=2, default=str
                        )
                    logger.info(f"Saved trade history to {trades_path}")

                # Show backtest metrics
                metrics = trade_executor.get_backtest_metrics()
                logger.info("Backtest Metrics Summary:")
                logger.info(f"  Initial Balance: {metrics['initial_balance']:.2f}")
                logger.info(f"  Final Balance:   {metrics['final_balance']:.2f}")
                logger.info(f"  Max Balance:     {metrics['max_balance']:.2f}")
                logger.info(f"  Min Balance:     {metrics['min_balance']:.2f}")
                logger.info(f"  ROI:             {metrics['roi_percent']:.2f}%")
                logger.info(f"  Max ROI:         {metrics['max_roi_percent']:.2f}%")
                logger.info(
                    f"  Max Drawdown:    {metrics['max_drawdown_percent']:.2f}%"
                )
                logger.info(f"  Total Trades:    {metrics['total_trades']}")
                logger.info(f"  Win Rate:        {metrics['win_rate']:.2f}%")

            except Exception as e:
                logger.error(f"Error during backtesting: {e}")
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

                # Fetch account info to get current balance and positions
                account_info = await trading_api.get_account_info()
                logger.info(f"Account balance: {account_info.get('balance', 0.0)}")
                logger.info(f"Open positions: {account_info.get('positions_count', 0)}")

                live_provider = DataProvider(
                    source_type="live",
                    trading_api=trading_api,
                    symbol=configured_symbols[0],
                    timeframes=configured_timeframes,
                    counts=timeframe_counts,
                )

                live_stream = live_provider.stream_data()
                async for data_dict, current_symbol in live_stream:
                    # Now current_symbol comes directly from the data provider

                    # Log data for each timeframe
                    for tf, df in data_dict.items():
                        if isinstance(
                            df, pd.DataFrame
                        ):  # Skip non-DataFrame entries like 'symbol'
                            logger.debug(
                                f"{tf} Length: {len(df)}\n{tf} Data (Last 1):\n"
                                + str(df.tail(1))
                            )
                            assert (
                                len(df) == live_provider.counts[tf]
                            ), f"Live yielded {tf} data has length {len(df)}, expected {live_provider.counts[tf]}"

                    # Calculate indicators
                    indicators = indicator_engine.calculate_indicators(data_dict)

                    # Detect market regime
                    current_regime = regime_detector.detect_regime(
                        indicators, current_symbol
                    )
                    logger.debug(
                        f"Current market regime (live) for {current_symbol}: {current_regime}"
                    )

                    # Update trade executor to check for regime changes
                    current_time = data_dict[primary_timeframe].index[-1]
                    closed_positions = await trade_executor.update(
                        data_dict, primary_timeframe, current_symbol, current_regime
                    )
                    if closed_positions:
                        logger.info(
                            f"Closed {len(closed_positions)} positions due to regime change."
                        )

                    # Generate trading signals
                    signals = trading_logic_engine.generate_signals(
                        current_regime, indicators, data_dict, current_symbol
                    )

                    # Execute entry signals if any
                    if signals["entry_signals"]:
                        logger.info(
                            f"Entry signals generated: {len(signals['entry_signals'])}"
                        )
                        for signal in signals["entry_signals"]:
                            logger.info(
                                f"  {signal['type']} signal at time {signal['timestamp']} price {signal['price']}"
                            )

                            # Find matching exit signal
                            exit_params = next(
                                (
                                    exit_signal
                                    for exit_signal in signals["exit_signals"]
                                    if exit_signal["entry_signal_type"]
                                    == signal["type"]
                                ),
                                {"stop_loss": None, "take_profit": None},
                            )

                            # Execute signal
                            current_time = data_dict[primary_timeframe].index[-1]
                            execution_result = await trade_executor.execute_signal(
                                signal=signal,
                                exit_parameters=exit_params,
                                current_time=current_time,
                            )

                            if execution_result.get("success", False):
                                logger.info(
                                    f"Successfully executed {signal['type']} order"
                                )
                            else:
                                logger.warning(
                                    f"Failed to execute {signal['type']} order: {execution_result.get('message', 'unknown error')}"
                                )

                    # Log calculated indicators for debugging
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

            except KeyboardInterrupt:
                logger.info(
                    "Keyboard interrupt detected. Closing positions and shutting down..."
                )
                if trade_executor:
                    await trade_executor.close_all_positions()

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
        if trade_executor:
            await trade_executor.close_all_positions()

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
