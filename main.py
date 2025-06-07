import sys

from src.utils.logger import Logger

logger = Logger(__name__)


from src.utils.config import ConfigManager
from src.core.data_provider import DataProvider

# from src.core.indicator_engine import IndicatorEngine
# from src.core.regime_detector import RegimeDetector
# from src.core.strategy_engine import StrategyEngine
# from src.core.risk_manager import RiskManager
# from src.backtesting.engine import BacktestEngine
# from src.api.deriv_client import DerivAPIClient


def main():
    # Initialize configuration managers
    logger.info("Starting Trading System")
    logger.info(f"Loading configs...")
    api_config = ConfigManager.load_config("config/api_config.yaml")
    risk_config = ConfigManager.load_config("config/risk_config.yaml")
    strategy_config = ConfigManager.load_config("config/strategy_config.yaml")

    # # Initialize components
    data_provider = DataProvider()

    if strategy_config["backtesting"]["enabled"]:
        logger.info("Backtesting. Loading historical data...")
        data = data_provider.load_csv_data("data/historical", "EURUSD", ["H1", "H4"])
        logger.debug(data["H1"].head())
    else:
        logger.info("Live Trading. Loading live data...")
        data = data_provider.load_live_data("data/historical", "EURUSD", ["H1", "H4"])
        logger.debug(data["H1"].head())

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
    main()
