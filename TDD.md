# EUR/USD Grand Strategy Trading System - Enhanced with Backtesting

## 1. Enhanced System Architecture

### Core Design Principles
- **Unified Components**: Combine related functionality into cohesive modules
- **Event-Driven**: Asynchronous processing with simple message passing
- **Plugin-Ready**: Easy integration of new modules (Fair Value Gap, Order Flow, etc.)
- **High Performance**: Multi-threading with efficient caching
- **Fault Tolerant**: Robust error handling and recovery
- **Backtesting Ready**: Seamless switching between live and historical data

## 2. Technology Stack

### Core Technologies
- **Python 3.11+**: Primary language
- **AsyncIO**: Asynchronous operations
- **Redis**: Caching and message broker
- **PostgreSQL**: Data storage
- **Docker**: Containerization

### Key Libraries
```python
# Core Libraries
pandas >= 2.0.0
numpy >= 1.24.0
ta-lib >= 0.4.0
asyncio
aiohttp >= 3.8.0
redis >= 4.5.0
sqlalchemy >= 2.0.0
asyncpg >= 0.28.0

# Broker APIs
oandapyV20 >= 0.7.2
ib_insync >= 0.9.86

# Backtesting
backtrader >= 1.9.76.123
vectorbt >= 0.25.0

# Monitoring
prometheus-client >= 0.16.0
structlog >= 23.1.0
```

## 3. Enhanced System Components

### 3.1 Data Service (Enhanced)
```
DataService (Unified Data Management)
├── BrokerConnector (Multi-broker WebSocket/REST)
├── BacktestConnector (Historical Data Replay)
├── DataProcessor (Validation, Normalization, Caching)
└── DatabaseManager (PostgreSQL + Redis)
```

### 3.2 Analysis Service (Same)
```
AnalysisService (All Technical Analysis)
├── IndicatorEngine (ADX, EMA, MACD, Stochastic, ATR, etc.)
├── PatternDetector (Candlesticks, S/R levels)
└── PluginManager (Fair Value Gap, Order Flow, Custom)
```

### 3.3 Strategy Service (Enhanced)
```
StrategyService (Core Trading Logic)
├── RegimeDetector (Market state identification)
├── SignalGenerator (Entry/Exit signals)
├── RiskManager (Position sizing, SL/TP)
├── TradeExecutor (Order management)
└── BacktestExecutor (Synthetic trade execution)
```

### 3.4 Backtesting Service (New)
```
BacktestingService (Historical Testing)
├── BacktestEngine (Main orchestrator)
├── PerformanceAnalyzer (Returns, Sharpe, Drawdown)
├── ReportGenerator (HTML/PDF reports)
└── ParameterOptimizer (Grid/Random search)
```

### 3.5 Monitoring Service (Enhanced)
```
MonitoringService (System Health & Performance)
├── MetricsCollector (Prometheus metrics)
├── Logger (Structured logging)
├── AlertManager (Trade notifications)
└── BacktestReporter (Backtest visualization)
```

## 4. Enhanced Core Classes and Interfaces

### 4.1 Enhanced Data Service with Backtesting
```python
from abc import ABC, abstractmethod
from enum import Enum

class DataMode(Enum):
    LIVE = "live"
    BACKTEST = "backtest"
    PAPER = "paper"

class BaseDataConnector(ABC):
    """Abstract base for all data connectors"""
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]):
        pass
    
    @abstractmethod
    async def fetch_historical(self, symbol: str, timeframe: str, periods: int) -> pd.DataFrame:
        pass
    
    @abstractmethod
    async def get_latest(self, symbol: str, timeframe: str) -> MarketData:
        pass

class DataService:
    """Enhanced unified data management with backtesting support"""
    
    def __init__(self, config: DataConfig, mode: DataMode = DataMode.LIVE):
        self.mode = mode
        self.data_processor = DataProcessor()
        self.db_manager = DatabaseManager(config.database)
        self.cache = Redis(config.redis)
        
        # Initialize appropriate connector based on mode
        if mode == DataMode.LIVE:
            self.connector = BrokerConnector(config.brokers)
        elif mode == DataMode.BACKTEST:
            self.connector = BacktestConnector(config.backtest)
        else:  # PAPER
            self.connector = PaperTradingConnector(config.brokers)
    
    async def start_data_feed(self, symbols: List[str], start_date: datetime = None, end_date: datetime = None):
        """Start data feed - live or historical based on mode"""
        if self.mode == DataMode.BACKTEST:
            await self.connector.setup_historical_replay(symbols, start_date, end_date)
        else:
            await self.connector.subscribe(symbols)
    
    async def get_next_market_data(self) -> Optional[MarketData]:
        """Get next market data point (for backtesting iteration)"""
        if self.mode == DataMode.BACKTEST:
            return await self.connector.get_next_historical_data()
        else:
            return await self.connector.get_latest_data()

class BacktestConnector(BaseDataConnector):
    """Historical data replay connector for backtesting"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.historical_data = {}
        self.current_indices = {}
        self.data_iterator = None
    
    async def setup_historical_replay(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Setup historical data for replay"""
        for symbol in symbols:
            # Load historical data from database or external source
            data = await self._load_historical_data(symbol, start_date, end_date)
            self.historical_data[symbol] = data
            self.current_indices[symbol] = 0
        
        # Create combined data iterator sorted by timestamp
        self.data_iterator = self._create_data_iterator()
    
    async def _load_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load historical data from various sources"""
        # Try cache first
        cache_key = f"hist:{symbol}:{start_date}:{end_date}"
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            return pd.read_json(cached_data)
        
        # Load from database or external API
        if self.config.data_source == "database":
            data = await self._load_from_database(symbol, start_date, end_date)
        else:
            data = await self._load_from_external_api(symbol, start_date, end_date)
        
        # Cache the data
        await self.cache.setex(cache_key, 3600, data.to_json())  # 1 hour cache
        return data
    
    def _create_data_iterator(self):
        """Create iterator that yields data chronologically across all symbols"""
        all_data = []
        for symbol, data in self.historical_data.items():
            for _, row in data.iterrows():
                all_data.append(MarketData(
                    symbol=symbol,
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    timeframe=row.get('timeframe', 'H1')
                ))
        
        # Sort by timestamp
        all_data.sort(key=lambda x: x.timestamp)
        return iter(all_data)
    
    async def get_next_historical_data(self) -> Optional[MarketData]:
        """Get next historical data point"""
        try:
            return next(self.data_iterator)
        except StopIteration:
            return None  # End of backtest data
    
    async def subscribe(self, symbols: List[str]):
        """Not used in backtest mode"""
        pass
    
    async def fetch_historical(self, symbol: str, timeframe: str, periods: int) -> pd.DataFrame:
        """Get historical data for analysis"""
        if symbol in self.historical_data:
            current_idx = self.current_indices.get(symbol, 0)
            start_idx = max(0, current_idx - periods)
            return self.historical_data[symbol].iloc[start_idx:current_idx + 1]
        return pd.DataFrame()
    
    async def get_latest(self, symbol: str, timeframe: str) -> MarketData:
        """Get current data point in backtest"""
        if symbol in self.historical_data:
            current_idx = self.current_indices.get(symbol, 0)
            if current_idx < len(self.historical_data[symbol]):
                row = self.historical_data[symbol].iloc[current_idx]
                return MarketData(
                    symbol=symbol,
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    timeframe=timeframe
                )
        return None
```

### 4.2 Enhanced Strategy Service with Backtest Executor
```python
class BaseTradeExecutor(ABC):
    """Abstract base for all trade executors"""
    
    @abstractmethod
    async def execute_trade(self, signal: TradingSignal) -> TradeResult:
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        pass
    
    @abstractmethod
    async def get_open_positions(self) -> List[Position]:
        pass

class StrategyService:
    """Enhanced core trading strategy logic with backtesting support"""
    
    def __init__(self, config: StrategyConfig, mode: DataMode = DataMode.LIVE):
        self.config = config
        self.mode = mode
        self.regime_detector = RegimeDetector(config.regime_detection)
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config.risk_management)
        
        # Initialize appropriate executor based on mode
        if mode == DataMode.LIVE:
            self.trade_executor = LiveTradeExecutor(config.execution)
        elif mode == DataMode.BACKTEST:
            self.trade_executor = BacktestExecutor(config.backtest)
        else:  # PAPER
            self.trade_executor = PaperTradeExecutor(config.execution)
    
    async def process_market_update(self, market_data: MarketData):
        """Enhanced processing with backtest support"""
        # Get historical data for analysis (lookback period)
        historical_data = await self.data_service.fetch_historical(
            market_data.symbol, 
            market_data.timeframe, 
            self.config.analysis_lookback
        )
        
        # Perform analysis
        analysis = await self.analysis_service.analyze_market_data(historical_data)
        analysis.current_price = market_data.close
        analysis.current_timestamp = market_data.timestamp
        
        # Detect market regime
        current_regime = await self.regime_detector.detect_regime(analysis)
        
        # Generate signals based on regime
        signals = await self.signal_generator.generate_signals(current_regime, analysis)
        
        # Apply risk management
        validated_signals = await self.risk_manager.validate_signals(signals, market_data)
        
        # Execute trades
        for signal in validated_signals:
            result = await self.trade_executor.execute_trade(signal)
            
            # Log trade for backtesting analysis
            if self.mode == DataMode.BACKTEST:
                await self._log_backtest_trade(signal, result, market_data)

class BacktestExecutor(BaseTradeExecutor):
    """Synthetic trade execution for backtesting"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.account = BacktestAccount(config.initial_balance)
        self.open_positions = {}
        self.trade_history = []
        self.current_market_data = {}
    
    async def execute_trade(self, signal: TradingSignal) -> TradeResult:
        """Execute synthetic trade"""
        current_price = self.current_market_data.get(signal.symbol, {}).get('close')
        if not current_price:
            return TradeResult(success=False, error="No current price data")
        
        # Apply slippage and spread
        execution_price = self._apply_execution_costs(current_price, signal.direction)
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, execution_price)
        
        if signal.action == "OPEN":
            return await self._open_position(signal, execution_price, position_size)
        elif signal.action == "CLOSE":
            return await self._close_position(signal, execution_price)
        else:
            return TradeResult(success=False, error="Unknown signal action")
    
    async def _open_position(self, signal: TradingSignal, price: float, size: float) -> TradeResult:
        """Open new position"""
        # Check if we have enough margin
        required_margin = size * price * self.config.margin_requirement
        if required_margin > self.account.available_margin:
            return TradeResult(success=False, error="Insufficient margin")
        
        # Create position
        position = BacktestPosition(
            id=f"pos_{len(self.trade_history)}",
            symbol=signal.symbol,
            direction=signal.direction,
            size=size,
            entry_price=price,
            entry_time=signal.timestamp,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        # Update account
        self.account.used_margin += required_margin
        self.account.available_margin -= required_margin
        self.open_positions[position.id] = position
        
        # Record trade
        trade_record = TradeRecord(
            position_id=position.id,
            symbol=signal.symbol,
            action="OPEN",
            direction=signal.direction,
            size=size,
            price=price,
            timestamp=signal.timestamp,
            pnl=0.0
        )
        self.trade_history.append(trade_record)
        
        return TradeResult(
            success=True,
            trade_id=position.id,
            execution_price=price,
            execution_size=size
        )
    
    async def _close_position(self, signal: TradingSignal, price: float) -> TradeResult:
        """Close existing position"""
        position_id = signal.position_id or self._find_position_to_close(signal.symbol, signal.direction)
        
        if position_id not in self.open_positions:
            return TradeResult(success=False, error="Position not found")
        
        position = self.open_positions[position_id]
        
        # Calculate PnL
        pnl = self._calculate_pnl(position, price)
        
        # Update account
        self.account.balance += pnl
        self.account.used_margin -= (position.size * position.entry_price * self.config.margin_requirement)
        self.account.available_margin += (position.size * position.entry_price * self.config.margin_requirement)
        
        # Remove position
        del self.open_positions[position_id]
        
        # Record trade
        trade_record = TradeRecord(
            position_id=position_id,
            symbol=signal.symbol,
            action="CLOSE",
            direction=signal.direction,
            size=position.size,
            price=price,
            timestamp=signal.timestamp,
            pnl=pnl,
            entry_price=position.entry_price,
            exit_price=price
        )
        self.trade_history.append(trade_record)
        
        return TradeResult(
            success=True,
            trade_id=position_id,
            execution_price=price,
            execution_size=position.size,
            pnl=pnl
        )
    
    def _apply_execution_costs(self, price: float, direction: str) -> float:
        """Apply spread and slippage"""
        spread = self.config.spread_pips * self.config.pip_value
        slippage = self.config.slippage_pips * self.config.pip_value
        
        if direction == "BUY":
            return price + spread + slippage
        else:
            return price - spread - slippage
    
    def _calculate_position_size(self, signal: TradingSignal, price: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = self.account.balance * self.config.risk_per_trade
        
        if signal.stop_loss:
            stop_distance = abs(price - signal.stop_loss)
            size = risk_amount / stop_distance
        else:
            # Use default position size
            size = self.config.default_position_size
        
        # Apply max position size limit
        max_size = self.account.balance * self.config.max_position_size_ratio / price
        return min(size, max_size)
    
    def _calculate_pnl(self, position: BacktestPosition, exit_price: float) -> float:
        """Calculate position PnL"""
        if position.direction == "BUY":
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        
        # Subtract commission
        commission = position.size * self.config.commission_per_unit
        return pnl - commission
    
    async def update_market_data(self, market_data: MarketData):
        """Update current market data for position management"""
        self.current_market_data[market_data.symbol] = {
            'close': market_data.close,
            'timestamp': market_data.timestamp
        }
        
        # Check stop loss and take profit for open positions
        await self._check_stop_loss_take_profit(market_data)
    
    async def _check_stop_loss_take_profit(self, market_data: MarketData):
        """Check and execute stop loss / take profit orders"""
        current_price = market_data.close
        positions_to_close = []
        
        for pos_id, position in self.open_positions.items():
            if position.symbol != market_data.symbol:
                continue
            
            should_close = False
            close_reason = ""
            
            # Check stop loss
            if position.stop_loss:
                if (position.direction == "BUY" and current_price <= position.stop_loss) or \
                   (position.direction == "SELL" and current_price >= position.stop_loss):
                    should_close = True
                    close_reason = "Stop Loss"
            
            # Check take profit
            if position.take_profit and not should_close:
                if (position.direction == "BUY" and current_price >= position.take_profit) or \
                   (position.direction == "SELL" and current_price <= position.take_profit):
                    should_close = True
                    close_reason = "Take Profit"
            
            if should_close:
                positions_to_close.append((pos_id, close_reason))
        
        # Close positions that hit SL/TP
        for pos_id, reason in positions_to_close:
            close_signal = TradingSignal(
                symbol=market_data.symbol,
                direction="CLOSE",
                action="CLOSE",
                position_id=pos_id,
                timestamp=market_data.timestamp,
                reason=reason
            )
            await self.execute_trade(close_signal)
    
    async def get_account_info(self) -> AccountInfo:
        """Get current account information"""
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        for position in self.open_positions.values():
            if position.symbol in self.current_market_data:
                current_price = self.current_market_data[position.symbol]['close']
                unrealized_pnl += self._calculate_pnl(position, current_price)
        
        return AccountInfo(
            balance=self.account.balance,
            equity=self.account.balance + unrealized_pnl,
            used_margin=self.account.used_margin,
            available_margin=self.account.available_margin,
            margin_level=(self.account.balance + unrealized_pnl) / self.account.used_margin * 100 if self.account.used_margin > 0 else 0
        )
    
    async def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self.open_positions.values())
    
    def get_trade_history(self) -> List[TradeRecord]:
        """Get complete trade history for analysis"""
        return self.trade_history.copy()

@dataclass
class BacktestPosition:
    id: str
    symbol: str
    direction: str  # BUY or SELL
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class BacktestAccount:
    balance: float
    used_margin: float = 0.0
    
    def __post_init__(self):
        self.initial_balance = self.balance
        self.available_margin = self.balance

@dataclass
class TradeRecord:
    position_id: str
    symbol: str
    action: str  # OPEN or CLOSE
    direction: str  # BUY or SELL
    size: float
    price: float
    timestamp: datetime
    pnl: float
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
```

### 4.3 New Backtesting Service
```python
class BacktestingService:
    """Comprehensive backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.performance_analyzer = PerformanceAnalyzer()
        self.report_generator = ReportGenerator()
        self.parameter_optimizer = ParameterOptimizer()
    
    async def run_backtest(self, 
                          strategy_config: StrategyConfig,
                          start_date: datetime,
                          end_date: datetime,
                          symbols: List[str] = None) -> BacktestResult:
        """Run a complete backtest"""
        
        symbols = symbols or ['EUR_USD']
        
        # Initialize services in backtest mode
        data_service = DataService(self.config.data_config, DataMode.BACKTEST)
        analysis_service = AnalysisService(self.config.analysis_config)
        strategy_service = StrategyService(strategy_config, DataMode.BACKTEST)
        
        # Setup data feed
        await data_service.start_data_feed(symbols, start_date, end_date)
        
        # Initialize strategy services
        strategy_service.data_service = data_service
        strategy_service.analysis_service = analysis_service
        
        # Run backtest loop
        trade_history = []
        equity_curve = []
        current_time = start_date
        
        while current_time <= end_date:
            # Get next market data
            market_data = await data_service.get_next_market_data()
            if not market_data:
                break
            
            current_time = market_data.timestamp
            
            # Update executor market data
            await strategy_service.trade_executor.update_market_data(market_data)
            
            # Process market update
            await strategy_service.process_market_update(market_data)
            
            # Record equity curve
            account_info = await strategy_service.trade_executor.get_account_info()
            equity_curve.append({
                'timestamp': current_time,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'used_margin': account_info.used_margin
            })
        
        # Get final trade history
        trade_history = strategy_service.trade_executor.get_trade_history()
        
        # Analyze performance
        performance = await self.performance_analyzer.analyze(trade_history, equity_curve)
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            trade_history=trade_history,
            equity_curve=equity_curve,
            performance=performance,
            strategy_config=strategy_config
        )
    
    async def optimize_parameters(self,
                                base_config: StrategyConfig,
                                parameter_ranges: Dict[str, List],
                                start_date: datetime,
                                end_date: datetime,
                                optimization_metric: str = "sharpe_ratio") -> OptimizationResult:
        """Optimize strategy parameters"""
        
        return await self.parameter_optimizer.optimize(
            base_config=base_config,
            parameter_ranges=parameter_ranges,
            start_date=start_date,
            end_date=end_date,
            metric=optimization_metric,
            backtest_runner=self.run_backtest
        )

class PerformanceAnalyzer:
    """Analyze backtest performance"""
    
    async def analyze(self, trades: List[TradeRecord], equity_curve: List[Dict]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if not trades or not equity_curve:
            return PerformanceMetrics()
        
        # Convert to DataFrames for easier analysis
        trades_df = pd.DataFrame([
            {
                'timestamp': trade.timestamp,
                'pnl': trade.pnl,
                'action': trade.action,
                'symbol': trade.symbol
            }
            for trade in trades if trade.action == "CLOSE"
        ])
        
        equity_df = pd.DataFrame(equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        total_pnl = trades_df['pnl'].sum()
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        # Performance ratios
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        daily_returns = equity_df['returns'].dropna()
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
        
        # Calmar ratio
        annual_return = self._calculate_annual_return(equity_df['equity'])
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            annual_return=annual_return
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
        return (excess_returns / returns.std()) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_annual_return(self, equity: pd.Series) -> float:
        """Calculate annualized return"""
        if len(equity) < 2:
            return 0.0
        
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        days = len(equity)
        annual_return = (1 + total_return) ** (252 / days) - 1
        return annual_return

@dataclass
class BacktestResult:
    start_date: datetime
    end_date: datetime
    trade_history: List[TradeRecord]
    equity_curve: List[Dict]
    performance: PerformanceMetrics
    strategy_config: StrategyConfig

@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    annual_return: float = 0.0
```

## 5. Enhanced Configuration

```yaml
# config.yaml
system:
  mode: "live"  # live, backtest, paper
  
  data_service:
    brokers:
      - name: "oanda"
        api_key: "${OANDA_API_KEY}"
        account_id: "${OANDA_ACCOUNT_ID}"
    
    database:
      url: "postgresql://user:pass@localhost:5432/trading"
    
    redis:
      url: "redis://localhost:6379"
  
  backtest:
    data_source: "database"  # database, oanda, csv
    initial_balance: 10000.0
    margin_requirement: 0.02  # 2% margin
    spread_pips: 1.5
    slippage_pips: 0.5
    commission_per_unit: 0.00002
    pip_value: 0.0001
    risk_per_trade: 0.01  # 1% risk per trade
    max_position_size_ratio: 0.1  # 10% of account per position
    default_position_size: 10000  # units

  strategy:
    analysis_lookback: 200  # periods for technical analysis
    regime_detection:
      adx_trending_threshold: 25
      adx_ranging_threshold: 20
      confirmation_periods: 2
    
    risk_management:
      max_risk_per_trade: 0.01
      max_daily_drawdown: 0.03
    
    plugins:
      fair_value_gap:
        enabled: true
        min_gap_size: 5
      
      order_flow:
        enabled: false

  optimization:
    enabled: false
    parameter_ranges:
      adx_trending_threshold: [20, 25, 30, 35]
      adx_ranging_threshold: [15, 18, 20, 22]
      ema_periods: [[10, 20], [20, 50], [50, 100]]
    optimization_metric: "sharpe_ratio"  # sharpe_ratio, calmar_ratio, profit_factor
    max_iterations: 100
```

## 6. Enhanced Deployment with Backtesting

```yaml
# docker-compose.yml
version: '3.8'
services:
  trading-system:
    build: .
    environment:
      - CONFIG_PATH=/app/config.yaml
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql://user:pass@postgres:5432/trading
      - SYSTEM_MODE=${SYSTEM_MODE:-live}  # live, backtest, paper
    depends_on:
      - redis
      - postgres
    volumes:
      - ./config.yaml:/app/config.yaml
      - ./data:/app/data  # For CSV data files
      - ./reports:/app/reports  # For backtest reports
  
  backtesting-service:
    build: .
    command: python -m backtesting.main
    environment:
      - CONFIG_PATH=/app/config.yaml
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql://user:pass@postgres:5432
      - SYSTEM_MODE=backtest
    depends_on:
      - redis
      - postgres
    volumes:
      - ./config.yaml:/app/config.yaml
      - ./data:/app/data
      - ./reports:/app/reports
  
  redis:
    image: redis:7-alpine
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: trading
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  grafana_data:
```

## 7. Enhanced Main Application Structure

```python
# main.py
class TradingSystem:
    """Enhanced main trading system orchestrator with backtesting support"""
    
    def __init__(self, config_path: str, mode: DataMode = None):
        self.config = self._load_config(config_path)
        self.mode = mode or DataMode(self.config.system.mode)
        
        # Initialize services based on mode
        self.data_service = DataService(self.config.data_service, self.mode)
        self.analysis_service = AnalysisService(self.config.analysis)
        self.strategy_service = StrategyService(self.config.strategy, self.mode)
        self.monitoring_service = MonitoringService(self.config.monitoring)
        
        # Initialize backtesting service if needed
        if self.mode == DataMode.BACKTEST:
            self.backtesting_service = BacktestingService(self.config.backtest)
    
    async def start(self):
        """Start the trading system"""
        if self.mode == DataMode.BACKTEST:
            await self.run_backtest_mode()
        else:
            await self.run_live_mode()
    
    async def run_live_mode(self):
        """Run in live trading mode"""
        # Initialize services
        await self.data_service.start()
        await self.analysis_service.start()
        await self.strategy_service.start()
        
        # Start real-time data feed
        await self.data_service.start_data_feed(['EUR_USD'])
        
        # Start main processing loop
        await self._main_loop()
    
    async def run_backtest_mode(self):
        """Run in backtesting mode"""
        logger.info("Starting backtesting mode")
        
        # Define backtest parameters
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        symbols = ['EUR_USD']
        
        # Check if optimization is enabled
        if self.config.optimization.enabled:
            await self.run_parameter_optimization(start_date, end_date, symbols)
        else:
            await self.run_single_backtest(start_date, end_date, symbols)
    
    async def run_single_backtest(self, start_date: datetime, end_date: datetime, symbols: List[str]):
        """Run a single backtest"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        result = await self.backtesting_service.run_backtest(
            strategy_config=self.config.strategy,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols
        )
        
        # Print results
        await self._print_backtest_results(result)
        
        # Generate report
        await self._generate_backtest_report(result)
    
    async def run_parameter_optimization(self, start_date: datetime, end_date: datetime, symbols: List[str]):
        """Run parameter optimization"""
        logger.info("Starting parameter optimization")
        
        optimization_result = await self.backtesting_service.optimize_parameters(
            base_config=self.config.strategy,
            parameter_ranges=self.config.optimization.parameter_ranges,
            start_date=start_date,
            end_date=end_date,
            optimization_metric=self.config.optimization.optimization_metric
        )
        
        # Print optimization results
        await self._print_optimization_results(optimization_result)
    
    async def _main_loop(self):
        """Main processing loop for live trading"""
        while True:
            try:
                # Get latest market data
                market_data = await self.data_service.get_next_market_data()
                if market_data:
                    # Process market updates
                    await self.strategy_service.process_market_update(market_data)
                
                await asyncio.sleep(1)  # Process every second
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
    
    async def _print_backtest_results(self, result: BacktestResult):
        """Print backtest results to console"""
        perf = result.performance
        
        print("\n" + "="*60)
        print("BACKTESTING RESULTS")
        print("="*60)
        print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        print(f"Total Trades: {perf.total_trades}")
        print(f"Winning Trades: {perf.winning_trades}")
        print(f"Losing Trades: {perf.losing_trades}")
        print(f"Win Rate: {perf.win_rate:.2%}")
        print(f"Total P&L: ${perf.total_pnl:.2f}")
        print(f"Gross Profit: ${perf.gross_profit:.2f}")
        print(f"Gross Loss: ${perf.gross_loss:.2f}")
        print(f"Profit Factor: {perf.profit_factor:.2f}")
        print(f"Sharpe Ratio: {perf.sharpe_ratio:.2f}")
        print(f"Calmar Ratio: {perf.calmar_ratio:.2f}")
        print(f"Max Drawdown: {perf.max_drawdown:.2%}")
        print(f"Annual Return: {perf.annual_return:.2%}")
        print("="*60)
    
    async def _generate_backtest_report(self, result: BacktestResult):
        """Generate detailed backtest report"""
        report_generator = ReportGenerator()
        
        # Generate HTML report
        html_report = await report_generator.generate_html_report(result)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/backtest_report_{timestamp}.html"
        
        with open(report_path, 'w') as f:
            f.write(html_report)
        
        logger.info(f"Backtest report saved to {report_path}")

# Backtesting CLI
class BacktestingCLI:
    """Command-line interface for backtesting"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self):
        parser = argparse.ArgumentParser(description="EUR/USD Trading System Backtesting")
        
        parser.add_argument('--config', '-c', default='config.yaml', 
                          help='Configuration file path')
        parser.add_argument('--start-date', '-s', required=True,
                          help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end-date', '-e', required=True,
                          help='End date (YYYY-MM-DD)')
        parser.add_argument('--symbols', nargs='+', default=['EUR_USD'],
                          help='Symbols to backtest')
        parser.add_argument('--optimize', action='store_true',
                          help='Run parameter optimization')
        parser.add_argument('--report-only', action='store_true',
                          help='Generate report from existing results')
        
        return parser
    
    async def run(self, args):
        """Run backtesting based on CLI arguments"""
        # Parse dates
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        # Initialize system in backtest mode
        system = TradingSystem(args.config, DataMode.BACKTEST)
        
        if args.optimize:
            await system.run_parameter_optimization(start_date, end_date, args.symbols)
        else:
            await system.run_single_backtest(start_date, end_date, args.symbols)

# Enhanced Parameter Optimizer
class ParameterOptimizer:
    """Advanced parameter optimization"""
    
    def __init__(self):
        self.optimization_history = []
    
    async def optimize(self,
                     base_config: StrategyConfig,
                     parameter_ranges: Dict[str, List],
                     start_date: datetime,
                     end_date: datetime,
                     metric: str,
                     backtest_runner) -> OptimizationResult:
        """Run parameter optimization"""
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        best_result = None
        best_metric_value = float('-inf') if self._is_maximize_metric(metric) else float('inf')
        
        total_combinations = len(param_combinations)
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing combination {i+1}/{total_combinations}: {params}")
            
            # Create modified config
            test_config = self._apply_parameters(base_config, params)
            
            # Run backtest
            try:
                result = await backtest_runner(test_config, start_date, end_date)
                
                # Extract metric value
                metric_value = self._extract_metric_value(result.performance, metric)
                
                # Check if this is the best result
                if self._is_better_result(metric_value, best_metric_value, metric):
                    best_result = result
                    best_metric_value = metric_value
                    logger.info(f"New best result: {metric} = {metric_value:.4f}")
                
                # Store in history
                self.optimization_history.append({
                    'parameters': params,
                    'metric_value': metric_value,
                    'performance': result.performance
                })
                
            except Exception as e:
                logger.error(f"Error testing parameters {params}: {e}")
                continue
        
        return OptimizationResult(
            best_parameters=self._extract_parameters(best_result.strategy_config),
            best_result=best_result,
            optimization_history=self.optimization_history,
            metric_used=metric
        )
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations"""
        import itertools
        
        keys = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _apply_parameters(self, base_config: StrategyConfig, params: Dict) -> StrategyConfig:
        """Apply parameter values to config"""
        # Create a copy of the base config
        import copy
        config = copy.deepcopy(base_config)
        
        # Apply parameters
        for param_name, param_value in params.items():
            if param_name == 'adx_trending_threshold':
                config.regime_detection.adx_trending_threshold = param_value
            elif param_name == 'adx_ranging_threshold':
                config.regime_detection.adx_ranging_threshold = param_value
            elif param_name == 'ema_periods':
                config.signal_generation.ema_short_period = param_value[0]
                config.signal_generation.ema_long_period = param_value[1]
            # Add more parameter mappings as needed
        
        return config
    
    def _extract_metric_value(self, performance: PerformanceMetrics, metric: str) -> float:
        """Extract specific metric value"""
        metric_map = {
            'sharpe_ratio': performance.sharpe_ratio,
            'calmar_ratio': performance.calmar_ratio,
            'profit_factor': performance.profit_factor,
            'total_pnl': performance.total_pnl,
            'win_rate': performance.win_rate,
            'max_drawdown': performance.max_drawdown
        }
        return metric_map.get(metric, 0.0)
    
    def _is_maximize_metric(self, metric: str) -> bool:
        """Check if metric should be maximized"""
        maximize_metrics = ['sharpe_ratio', 'calmar_ratio', 'profit_factor', 'total_pnl', 'win_rate']
        return metric in maximize_metrics
    
    def _is_better_result(self, new_value: float, current_best: float, metric: str) -> bool:
        """Check if new result is better than current best"""
        if self._is_maximize_metric(metric):
            return new_value > current_best
        else:
            return new_value < current_best

@dataclass
class OptimizationResult:
    best_parameters: Dict[str, Any]
    best_result: BacktestResult
    optimization_history: List[Dict[str, Any]]
    metric_used: str

# Report Generator
class ReportGenerator:
    """Generate detailed backtesting reports"""
    
    async def generate_html_report(self, result: BacktestResult) -> str:
        """Generate comprehensive HTML report"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtesting Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
                .metric-card { background: #f5f5f5; padding: 15px; border-radius: 8px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2196F3; }
                .metric-label { font-size: 14px; color: #666; }
                .chart-container { margin: 30px 0; }
            </style>
        </head>
        <body>
            <h1>Backtesting Report</h1>
            <h2>Period: {start_date} to {end_date}</h2>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{total_trades}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{win_rate:.1%}</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${total_pnl:.2f}</div>
                    <div class="metric-label">Total P&L</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{sharpe_ratio:.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{profit_factor:.2f}</div>
                    <div class="metric-label">Profit Factor</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{max_drawdown:.1%}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{annual_return:.1%}</div>
                    <div class="metric-label">Annual Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{calmar_ratio:.2f}</div>
                    <div class="metric-label">Calmar Ratio</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="equity-curve" style="width:100%;height:400px;"></div>
            </div>
            
            <div class="chart-container">
                <div id="monthly-returns" style="width:100%;height:300px;"></div>
            </div>
            
            <script>
                // Equity curve chart
                var equityData = {equity_data};
                var equityTrace = {{
                    x: equityData.map(d => d.timestamp),
                    y: equityData.map(d => d.equity),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Equity'
                }};
                
                Plotly.newPlot('equity-curve', [equityTrace], {{
                    title: 'Equity Curve',
                    xaxis: {{ title: 'Date' }},
                    yaxis: {{ title: 'Equity ($)' }}
                }});
                
                // Add more charts as needed
            </script>
        </body>
        </html>
        """
        
        # Format the template with actual data
        return html_template.format(
            start_date=result.start_date.strftime('%Y-%m-%d'),
            end_date=result.end_date.strftime('%Y-%m-%d'),
            total_trades=result.performance.total_trades,
            win_rate=result.performance.win_rate,
            total_pnl=result.performance.total_pnl,
            sharpe_ratio=result.performance.sharpe_ratio,
            profit_factor=result.performance.profit_factor,
            max_drawdown=result.performance.max_drawdown,
            annual_return=result.performance.annual_return,
            calmar_ratio=result.performance.calmar_ratio,
            equity_data=json.dumps(result.equity_curve)
        )

# Run the system
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="EUR/USD Trading System")
    parser.add_argument('--mode', choices=['live', 'backtest', 'paper'], 
                       default='live', help='Trading mode')
    parser.add_argument('--config', default='config.yaml', 
                       help='Configuration file')
    
    # Backtesting specific arguments
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--optimize', action='store_true', 
                       help='Run parameter optimization')
    
    args = parser.parse_args()
    
    if args.mode == 'backtest':
        if not args.start_date or not args.end_date:
            print("Error: --start-date and --end-date required for backtesting")
            sys.exit(1)
        
        # Run backtesting CLI
        cli = BacktestingCLI()
        asyncio.run(cli.run(args))
    else:
        # Run normal trading system
        system = TradingSystem(args.config, DataMode(args.mode))
        asyncio.run(system.start())
```

## 8. Usage Examples

### Running a Simple Backtest
```bash
# Run backtest for 2023
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31

# Run with parameter optimization
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31 --optimize
```

### Docker Backtesting
```bash
# Run backtest in Docker
docker-compose run backtesting-service python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31

# Run optimization
docker-compose run backtesting-service python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31 --optimize
```

This enhanced architecture provides:

1. **Seamless Mode Switching**: Same codebase works for live, paper, and backtesting
2. **Synthetic Execution**: Realistic trade execution with slippage, spread, and commissions
3. **Comprehensive Performance Analysis**: All standard metrics including Sharpe, Calmar, drawdown
4. **Parameter Optimization**: Grid search with configurable metrics
5. **Detailed Reporting**: HTML reports with interactive charts
6. **Historical Data Management**: Efficient caching and replay of historical data
7. **Risk Management**: Proper position sizing and stop-loss handling in backtests

The backtesting engine maintains the same interfaces as live trading, making it easy to validate strategies before deploying them live.