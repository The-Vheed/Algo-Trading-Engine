# EUR/USD Dynamic Regime-Based Trading System
## Technical Design Document

### 1. System Overview

The EUR/USD Dynamic Regime-Based Trading System is a modular, configuration-driven trading platform that automatically switches between trend-following and range-bound strategies based on real-time market regime identification.

**Key Features:**
- Dynamic market regime detection (Trending/Ranging/Neutral)
- Modular indicator framework for easy extension
- YAML-based strategy configuration
- Comprehensive backtesting engine
- Real-time monitoring and alerting
- Deriv API integration for live trading
- CSV-based historical data management

### 2. System Architecture

#### 2.1 Core Components

**Data Layer:**
- `DataProvider`: Handles CSV file reading and Deriv API data fetching
- `DataStore`: In-memory data storage with rolling window management
- `DataValidator`: Ensures data quality and completeness

**Analysis Layer:**
- `IndicatorEngine`: Modular indicator calculation framework
- `RegimeDetector`: Market regime identification logic
- `StrategyEngine`: Strategy execution based on regime and signals
- `RiskManager`: Position sizing and risk control

**Execution Layer:**
- `SignalGenerator`: Trade signal creation and validation
- `OrderManager`: Order placement and management via Deriv API
- `PositionTracker`: Active position monitoring

**Configuration Layer:**
- `ConfigManager`: YAML configuration loading and validation
- `StrategyConfig`: Strategy-specific parameters
- `RiskConfig`: Risk management settings

**Monitoring Layer:**
- `PerformanceTracker`: Trade statistics and metrics
- `AlertManager`: Notification system for important events
- `Logger`: Comprehensive logging system

#### 2.2 Data Flow

1. Historical data loaded from CSV files
2. Real-time data fetched from Deriv API
3. Indicators calculated on H4 and H1 timeframes
4. Market regime identified using ADX and optional BBW
5. Strategy signals generated based on current regime
6. Risk management applied to position sizing
7. Orders executed via Deriv API
8. Performance monitored and logged

### 3. Module Specifications

#### 3.1 Data Provider Module

```python
class DataProvider:
    def load_csv_data(file_path: str, timeframe: str) -> DataFrame
    def fetch_live_data(symbol: str, timeframe: str, count: int) -> DataFrame
    def update_csv_with_live_data(symbol: str, file_path: str) -> bool
    def validate_data_continuity(data: DataFrame) -> bool
```

**Responsibilities:**
- Load historical OHLCV data from CSV files
- Fetch real-time data from Deriv API
- Maintain data continuity and handle gaps
- Update CSV files with latest data

#### 3.2 Indicator Engine Module

```python
class IndicatorEngine:
    def register_indicator(name: str, calculator: IndicatorCalculator)
    def calculate_indicator(name: str, data: DataFrame, **params) -> Series
    def get_available_indicators() -> List[str]
    def calculate_all_required(data: DataFrame, config: dict) -> DataFrame
```

**Built-in Indicators:**
- ADX (Average Directional Index)
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- ATR (Average True Range)
- Bollinger Bands & Bandwidth

**Extensible Framework:**
- Plugin architecture for custom indicators
- Support for Fibonacci retracements/extensions
- Fair Value Gap (FVG) detection
- Order Block identification
- Support/Resistance level detection

#### 3.3 Regime Detector Module

```python
class RegimeDetector:
    def detect_regime(data: DataFrame, config: dict) -> str
    def confirm_regime_change(current: str, previous: str, bars: int) -> bool
    def get_regime_strength(data: DataFrame) -> float
```

**Regime Types:**
- `TRENDING`: ADX > 25 (with optional BBW confirmation)
- `RANGING`: ADX < 20 (with optional BBW confirmation)
- `NEUTRAL`: ADX between 20-25
- `TRANSITIONING`: Regime change in progress

#### 3.4 Strategy Engine Module

```python
class StrategyEngine:
    def load_strategy_config(config_path: str)
    def generate_signals(data: DataFrame, regime: str) -> List[Signal]
    def validate_signal(signal: Signal, current_positions: List) -> bool
    def execute_strategy(data: DataFrame) -> List[Order]
```

**Strategy Components:**
- Trend Following Strategy (ADX > 25)
- Range Trading Strategy (ADX < 20)
- Neutral Market Handling (ADX 20-25)
- Signal confirmation logic
- Entry/Exit rule validation

#### 3.5 Risk Manager Module

```python
class RiskManager:
    def calculate_position_size(account_equity: float, risk_pct: float, sl_pips: int) -> float
    def validate_risk_limits(proposed_order: Order) -> bool
    def check_drawdown_limits(current_equity: float, peak_equity: float) -> bool
    def apply_correlation_limits(symbol: str, existing_positions: List) -> bool
```

**Risk Controls:**
- Fixed percentage risk per trade (0.5-1.5%)
- Maximum pip stop-loss limits
- Daily/weekly drawdown limits
- Position correlation checks
- Account equity protection

### 4. Configuration System

#### 4.1 YAML Configuration Structure

```yaml
strategy:
  name: "EUR_USD_Dynamic_Regime"
  version: "1.0"
  
market_regime:
  primary_indicator: "ADX"
  adx_period: 14
  trending_threshold: 25
  ranging_threshold: 20
  confirmation_bars: 2
  
  volatility_filter:
    enabled: true
    indicator: "BBW"
    bb_period: 20
    bb_std_dev: 2.0
    low_vol_percentile: 25
    high_vol_percentile: 75

trending_strategy:
  timeframes:
    regime_detection: "4H"
    entry_signals: "1H"
  
  indicators:
    h4_ema_fast: 50
    h4_ema_slow: 200
    h1_ema_fast: 20
    h1_ema_slow: 50
    macd: [12, 26, 9]
    atr_period: 14
  
  entry_rules:
    require_ema_alignment: true
    require_macd_confirmation: true
    require_candlestick_pattern: false
  
  exit_rules:
    initial_rr_ratio: 2.0
    partial_profit_rr: 1.0
    partial_profit_percent: 50
    trailing_stop_method: "ATR"  # "ATR", "EMA", "PSAR"
    trailing_atr_multiplier: 2.0

ranging_strategy:
  timeframes:
    regime_detection: "4H"
    entry_signals: "1H"
  
  indicators:
    stochastic: [14, 3, 3]
    atr_period: 14
    support_resistance_method: "PIVOT"  # "MANUAL", "PIVOT", "DONCHIAN"
  
  entry_rules:
    stochastic_oversold: 20
    stochastic_overbought: 80
    require_divergence: false
    require_candlestick_pattern: true
  
  exit_rules:
    target_opposite_level: true
    minimum_rr_ratio: 1.5
    use_mid_range_tp: true

risk_management:
  position_sizing:
    method: "FIXED_FRACTIONAL"  # "FIXED_FRACTIONAL", "VOLATILITY_ADJUSTED"
    risk_per_trade_percent: 1.0
    max_sl_pips: 70
  
  account_protection:
    daily_loss_limit_percent: 3.0
    weekly_loss_limit_percent: 8.0
    max_concurrent_positions: 3
  
  correlation_limits:
    max_correlation_exposure: 0.7

monitoring:
  performance_tracking:
    enabled: true
    metrics: ["profit_factor", "sharpe_ratio", "max_drawdown", "win_rate"]
  
  alerts:
    enabled: true
    channels: ["console", "file"]  # Future: "email", "telegram"
    conditions:
      - "regime_change"
      - "position_opened"
      - "position_closed"
      - "risk_limit_breached"

backtesting:
  enabled: true
  data_source: "CSV"
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  initial_balance: 10000
  commission_per_lot: 7.0
  spread_pips: 1.2
```

### 5. Backtesting Framework

#### 5.1 Backtesting Engine

```python
class BacktestEngine:
    def __init__(self, config: dict, data_provider: DataProvider)
    def run_backtest(start_date: str, end_date: str) -> BacktestResults
    def generate_report(results: BacktestResults) -> dict
    def export_trades(results: BacktestResults, format: str = "CSV")
```

**Features:**
- Historical simulation with realistic spreads/commissions
- Walk-forward optimization capability
- Multiple timeframe analysis
- Performance metrics calculation
- Trade-by-trade analysis
- Drawdown analysis
- Monte Carlo simulation for robustness testing

#### 5.2 Performance Metrics

- **Profitability**: Net Profit, Profit Factor, Return on Investment
- **Risk Metrics**: Maximum Drawdown, Sharpe Ratio, Sortino Ratio
- **Reliability**: Win Rate, Average Win/Loss Ratio, Consecutive Losses
- **Efficiency**: Number of Trades, Average Trade Duration, Recovery Factor

### 6. Monitoring and Alerting

#### 6.1 Performance Tracker

```python
class PerformanceTracker:
    def update_trade_metrics(trade: CompletedTrade)
    def calculate_current_drawdown() -> float
    def get_performance_summary(period: str = "daily") -> dict
    def check_performance_thresholds() -> List[Alert]
```

#### 6.2 Alert Manager

```python
class AlertManager:
    def register_alert_handler(channel: str, handler: AlertHandler)
    def send_alert(alert: Alert)
    def configure_alert_conditions(conditions: dict)
```

**Alert Types:**
- Regime changes
- Position opened/closed
- Risk limits breached
- System errors
- Performance milestones

### 7. API Integration

#### 7.1 Deriv API Wrapper

```python
class DerivAPIClient:
    def __init__(self, api_token: str, app_id: str)
    def get_ticks(symbol: str, count: int) -> List[Tick]
    def get_candles(symbol: str, timeframe: str, count: int) -> List[Candle]
    def place_order(order: Order) -> OrderResponse
    def get_open_positions() -> List[Position]
    def close_position(position_id: str) -> bool
```

### 8. Extension Points

#### 8.1 Custom Indicators

New indicators can be added by implementing the `IndicatorCalculator` interface:

```python
class FibonacciLevels(IndicatorCalculator):
    def calculate(self, data: DataFrame, **params) -> DataFrame:
        # Implementation for Fibonacci retracement/extension levels
        pass

class FairValueGap(IndicatorCalculator):
    def calculate(self, data: DataFrame, **params) -> DataFrame:
        # Implementation for FVG detection
        pass

class OrderBlocks(IndicatorCalculator):
    def calculate(self, data: DataFrame, **params) -> DataFrame:
        # Implementation for order block identification
        pass
```

#### 8.2 Custom Market Structure Detectors

```python
class MarketStructureDetector:
    def detect_structure(self, data: DataFrame) -> StructureInfo:
        # Implementation for market structure analysis
        pass
```

### 9. Deployment and Operations

#### 9.1 Directory Structure

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

#### 9.2 Execution Flow

1. **Initialization**: Load configuration, initialize components
2. **Data Loading**: Load historical data, establish API connection
3. **Backtesting** (if enabled): Run historical simulation
4. **Live Trading**: Start real-time data processing and signal generation
5. **Monitoring**: Continuous performance tracking and alerting

### 10. Testing Strategy

- **Unit Tests**: Individual component testing
- **Integration Tests**: API integration and data flow testing
- **Backtesting Validation**: Historical performance verification
- **Paper Trading**: Live market testing without real money
- **Stress Testing**: System behavior under extreme market conditions

### 11. Future Enhancements

- **Multi-Symbol Support**: Extend to other forex pairs
- **Machine Learning Integration**: ML-based regime detection
- **Advanced Order Types**: OCO, trailing stops, etc.
- **Portfolio Management**: Multi-strategy coordination
- **Web Dashboard**: Real-time monitoring interface
- **Mobile Alerts**: Push notifications via mobile app

This technical design provides a solid foundation for implementing your EUR/USD trading strategy with the flexibility to extend and customize as needed.