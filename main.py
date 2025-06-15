import sys
import asyncio

from src.utils.logger import Logger

logger = Logger(__name__)


from src.utils.config import ConfigManager
from src.core.data_provider import DataProvider

from src.core.indicator_engine import IndicatorEngine

# from src.core.regime_detector import RegimeDetector
# from src.core.strategy_engine import StrategyEngine
# from src.core.risk_manager import RiskManager
# from src.backtesting.engine import BacktestEngine
# from src.api.deriv_client import DerivAPIClient


async def main():
    # Initialize configuration managers
    logger.info("Starting Trading System")
    logger.info(f"Loading configs...")
    config = ConfigManager.load_config("config/main.yaml")

    if not config["live"]:
        logger.info("Backtesting. Loading historical data...")
        try:
            if config["backtesting"]["data_source"] == "CSV":
                csv_provider = DataProvider(
                    source_type="csv",
                    base_path="data/historical",
                    symbol="EURUSD",
                    timeframes=["H1", "H4"],
                    counts=[3, 3],
                )
            else:
                csv_provider = DataProvider(
                    source_type="csv",
                    base_path="data/historical",
                    symbol="EURUSD",
                    timeframes=["H1", "H4"],
                    counts=[3, 3],
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
                timeframes=["M1", "M5"],
                counts=[3, 3],
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

        except Exception as e:
            logger.error(f"Error during live data stream: {e}")
            import traceback

            traceback.print_exc()

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
