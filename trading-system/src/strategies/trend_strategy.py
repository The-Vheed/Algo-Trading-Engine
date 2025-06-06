class TrendStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.indicator_engine = IndicatorEngine()
        self.regime_detector = RegimeDetector()

    def generate_signals(self, data: DataFrame) -> List[Signal]:
        regime = self.regime_detector.detect_regime(data, self.config['market_regime'])
        signals = []

        if regime == 'TRENDING':
            # Implement trend-following logic
            if self.indicator_engine.calculate_indicator('EMA', data, period=self.config['trending_strategy']['indicators']['h1_ema_fast']) > \
               self.indicator_engine.calculate_indicator('EMA', data, period=self.config['trending_strategy']['indicators']['h1_ema_slow']):
                signals.append(Signal('BUY', data.index[-1], data['Close'].iloc[-1]))
            elif self.indicator_engine.calculate_indicator('EMA', data, period=self.config['trending_strategy']['indicators']['h1_ema_fast']) < \
                 self.indicator_engine.calculate_indicator('EMA', data, period=self.config['trending_strategy']['indicators']['h1_ema_slow']):
                signals.append(Signal('SELL', data.index[-1], data['Close'].iloc[-1]))

        return signals

    def execute_strategy(self, data: DataFrame) -> List[Order]:
        signals = self.generate_signals(data)
        orders = []

        for signal in signals:
            order = self.create_order(signal)
            orders.append(order)

        return orders

    def create_order(self, signal: Signal) -> Order:
        # Logic to create an order based on the signal
        return Order(signal.type, signal.price, self.calculate_position_size(signal))

    def calculate_position_size(self, signal: Signal) -> float:
        # Implement position sizing logic based on risk management
        return self.risk_manager.calculate_position_size(self.account_equity, self.config['risk_management']['risk_per_trade_percent'], self.config['risk_management']['max_sl_pips'])