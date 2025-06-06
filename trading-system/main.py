import sys
from src.utils.config import ConfigManager
from src.core.data_provider import DataProvider
from src.core.indicator_engine import IndicatorEngine
from src.core.regime_detector import RegimeDetector
from src.core.strategy_engine import StrategyEngine
from src.core.risk_manager import RiskManager
from src.backtesting.engine import BacktestEngine
from src.api.deriv_client import DerivAPIClient
from src.utils.logger import Logger

def main():
    # Initialize configuration manager
    config_manager = ConfigManager()
    config = config_manager.load_config()

    # Initialize components
    data_provider = DataProvider()
    indicator_engine = IndicatorEngine()
    regime_detector = RegimeDetector()
    strategy_engine = StrategyEngine()
    risk_manager = RiskManager()
    backtest_engine = BacktestEngine(config, data_provider)
    api_client = DerivAPIClient(config['api_token'], config['app_id'])
    logger = Logger()

    # Load historical data
    historical_data = data_provider.load_csv_data(config['data_file_path'], config['timeframe'])

    # Run backtest if enabled
    if config['backtesting_enabled']:
        results = backtest_engine.run_backtest(config['start_date'], config['end_date'])
        report = backtest_engine.generate_report(results)
        logger.log(report)

    # Start live trading if enabled
    if config['live_trading_enabled']:
        # Implement live trading logic here
        pass

if __name__ == "__main__":
    main()