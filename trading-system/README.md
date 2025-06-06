# EUR/USD Dynamic Regime-Based Trading System

## Overview

The EUR/USD Dynamic Regime-Based Trading System is a modular, configuration-driven trading platform designed to automatically switch between trend-following and range-bound strategies based on real-time market regime identification.

## Key Features

- Dynamic market regime detection (Trending/Ranging/Neutral)
- Modular indicator framework for easy extension
- YAML-based strategy configuration
- Comprehensive backtesting engine
- Real-time monitoring and alerting
- Deriv API integration for live trading
- CSV-based historical data management

## Directory Structure

```
trading-system/
├── src/
│   ├── core/
│   │   ├── data_provider.py
│   │   ├── indicator_engine.py
│   │   ├── regime_detector.py
│   │   ├── strategy_engine.py
│   │   └── risk_manager.py
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── technical.py
│   │   └── custom/
│   ├── strategies/
│   │   ├── base_strategy.py
│   │   ├── trend_strategy.py
│   │   └── range_strategy.py
│   ├── backtesting/
│   │   ├── engine.py
│   │   ├── metrics.py
│   │   └── reports.py
│   ├── api/
│   │   └── deriv_client.py
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── alerts.py
├── config/
│   ├── strategy_config.yaml
│   ├── risk_config.yaml
│   └── api_config.yaml
├── data/
│   ├── historical/
│   │   └── EURUSD/
│   │       ├── H1.csv
│   │       └── H4.csv
│   └── logs/
├── tests/
├── docs/
└── main.py
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd trading-system
   ```
3. Install the required dependencies (if any).

## Usage

1. Configure the trading strategies and risk management settings in the `config` directory.
2. Run the application using:
   ```
   python main.py
   ```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.