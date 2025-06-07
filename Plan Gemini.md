Of course. Here is the complete, detailed implementation plan, incorporating all the specified components from the Technical Design Document (TDD) into a logical, step-by-step process.

---

### **The Complete Implementation Plan**

This plan is structured in five phases, designed to build the EUR/USD Dynamic Regime-Based Trading System incrementally. Each step includes its goal, the files to modify, a concrete implementation example, and a clear method for testing its functionality before proceeding.

### **Phase 1: Core Foundation & Multi-Timeframe Data**

**Goal:** Establish the project's structure and ensure it can correctly load and manage all necessary data from different timeframes as defined in the configuration.

#### **Step 1: Project Scaffolding & Configuration**
* **Goal:** Create the project's directory structure and implement the `ConfigManager`.
* **Files to Create/Modify:**
    * The full directory structure as outlined in the TDD (e.g., `src/core`, `config/`, `data/`, etc.).
    * `trading-system/main.py`
    * `trading-system/src/utils/config.py`
    * `trading-system/config/strategy_config.yaml`.
* **Implementation (`src/utils/config.py`):**
    ```python
    import yaml

    class ConfigManager:
        @staticmethod
        def load_config(path: str) -> dict:
            """Loads and validates the YAML configuration file."""
            with open(path, 'r') as f:
                # In a production system, add validation against a schema here.
                return yaml.safe_load(f)
    ```
* **How to Test:** Run `python main.py`. It should successfully load `strategy_config.yaml` and print a value from it, confirming the foundation is working.

#### **Step 2: Multi-Timeframe Data Provider**
* **Goal:** Implement the `DataProvider` to load historical data for all required timeframes (H4 for regime, H1 for entries).
* **Files to Create/Modify:** `src/core/data_provider.py`, `main.py`.
* **Implementation (`src/core/data_provider.py`):**
    ```python
    import pandas as pd
    from pandas import DataFrame
    from typing import Dict, List

    class DataProvider:
        def load_all_csv_data(self, base_path: str, symbol: str, timeframes: List[str]) -> Dict[str, DataFrame]:
            """Loads all required CSVs into a dictionary mapping timeframe to its dataframe."""
            data_frames = {}
            for tf in timeframes:
                file_path = f"{base_path}/{symbol}/{tf}.csv"
                print(f"Loading data from: {file_path}")
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                data_frames[tf] = df
            return data_frames
    ```
* **How to Test:** In `main.py`, use the `DataProvider` to load both `H4.csv` and `H1.csv` into a dictionary. Print the last few rows (`.tail()`) of each DataFrame to confirm they are loaded and parsed correctly.

---

### **Phase 2: Full Analysis Engine**

**Goal:** Build the complete data analysis pipeline, capable of calculating all required indicators and accurately identifying the market regime.

#### **Step 3: Comprehensive Indicator Engine**
* **Goal:** Flesh out the `IndicatorEngine` to calculate all indicators specified in the configuration for both trending and ranging strategies.
* **Files to Create/Modify:** `src/core/indicator_engine.py`.
* **Implementation (`src/core/indicator_engine.py`):**
    ```python
    from pandas import DataFrame
    import pandas_ta as ta # Requires: pip install pandas-ta

    class IndicatorEngine:
        def calculate_all_required(self, data: DataFrame, config: dict) -> DataFrame:
            """Calculates all indicators for a given strategy configuration."""
            # Regime Indicators
            data.ta.adx(length=config['market_regime']['adx_period'], append=True)
            if config['market_regime']['volatility_filter']['enabled']:
                data.ta.bbands(length=config['market_regime']['volatility_filter']['bb_period'], 
                               std=config['market_regime']['volatility_filter']['bb_std_dev'], append=True)
            
            # Trending Strategy Indicators
            ts_cfg = config['trending_strategy']
            data.ta.ema(length=ts_cfg['indicators']['h1_ema_fast'], append=True)
            data.ta.ema(length=ts_cfg['indicators']['h1_ema_slow'], append=True)
            macd_cfg = ts_cfg['indicators']['macd']
            data.ta.macd(fast=macd_cfg[0], slow=macd_cfg[1], signal=macd_cfg[2], append=True)
            data.ta.atr(length=ts_cfg['indicators']['atr_period'], append=True)

            # Ranging Strategy Indicators
            rs_cfg = config['ranging_strategy']
            stoch_cfg = rs_cfg['indicators']['stochastic']
            data.ta.stoch(k=stoch_cfg[0], d=stoch_cfg[1], smooth_k=stoch_cfg[2], append=True)
            
            return data.dropna()
    ```
* **How to Test:** After loading data in `main.py`, pass the H1 DataFrame to this method. Print the DataFrame's columns (`df.columns`) to verify that all new indicator columns (e.g., `ADX_14`, `EMA_20`, `MACD_12_26_9`, `STOCHk_14_3_3`) are present.

#### **Step 4: Market Regime Detector**
* **Goal:** Implement the `RegimeDetector` to classify the market state based on the calculated ADX indicator.
* **Files to Create/Modify:** `src/core/regime_detector.py`.
* **Implementation (`src/core/regime_detector.py`):**
    ```python
    from pandas import DataFrame

    class RegimeDetector:
        def detect_regime(self, data: DataFrame, config: dict) -> str:
            """Determines market regime from the latest ADX value."""
            cfg = config['market_regime']
            last_adx = data[f"ADX_{cfg['adx_period']}"].iloc[-1]
            
            if last_adx > cfg['trending_threshold']:
                return "TRENDING"
            elif last_adx < cfg['ranging_threshold']:
                return "RANGING"
            else:
                return "NEUTRAL"
    ```
* **How to Test:** In `main.py`, after calculating indicators on the H4 data, pass the resulting DataFrame to the `RegimeDetector`. Print the returned regime string to confirm the logic is working.

---

### **Phase 3: Backtesting with Full Logic**

**Goal:** Build a robust backtester that simulates the complete trading strategy, including trade execution, risk management, and performance analysis.

#### **Step 5: Strategy Engine with Signal Generation**
* **Goal:** Implement the specific entry logic for each strategy (Trend and Range) to generate concrete trade signals.
* **Files to Create/Modify:** `src/core/strategy_engine.py`, `src/strategies/trend_strategy.py`, `src/strategies/range_strategy.py`.
* **Implementation (`src/strategies/trend_strategy.py`):**
    ```python
    from pandas import DataFrame

    class TrendStrategy:
        def generate_signal(self, h1_data: DataFrame, h4_data: DataFrame, config: dict):
            """Generates a 'BUY', 'SELL', or 'HOLD' signal for the trend strategy."""
            ts_cfg = config['trending_strategy']
            latest_h1 = h1_data.iloc[-1]
            latest_h4 = h4_data.iloc[-1]

            # H4 Trend Alignment (EMA 50 > 200 for Bullish)
            h4_bullish = latest_h4[f"EMA_{ts_cfg['indicators']['h4_ema_fast']}"] > latest_h4[f"EMA_{ts_cfg['indicators']['h4_ema_slow']}"]
            
            # H1 Entry Confirmation (EMA 20 > 50 and MACD cross)
            h1_bullish_entry = latest_h1[f"EMA_{ts_cfg['indicators']['h1_ema_fast']}"] > latest_h1[f"EMA_{ts_cfg['indicators']['h1_ema_slow']}"]
            macd_confirm = latest_h1[f"MACD_{ts_cfg['indicators']['macd'][0]}_{ts_cfg['indicators']['macd'][1]}_{ts_cfg['indicators']['macd'][2]}"] > latest_h1[f"MACDh_{ts_cfg['indicators']['macd'][0]}_{ts_cfg['indicators']['macd'][1]}_{ts_cfg['indicators']['macd'][2]}"]

            if ts_cfg['entry_rules']['require_ema_alignment'] and h4_bullish and h1_bullish_entry and macd_confirm:
                return "BUY"
            # Add corresponding SELL logic here...
            
            return "HOLD"
    ```
* **How to Test:** Create a unit test for `TrendStrategy`. Manually construct small H1 and H4 DataFrames with indicator values that should trigger a buy signal and assert that the method returns `"BUY"`.

#### **Step 6: Risk Manager Implementation**
* **Goal:** Implement the `RiskManager` to calculate position sizes and stop-loss/take-profit levels according to the TDD.
* **Files to Create/Modify:** `src/core/risk_manager.py`.
* **Implementation (`src/core/risk_manager.py`):**
    ```python
    class RiskManager:
        def calculate_position_size(self, account_equity: float, risk_pct: float, sl_pips: int, pips_to_usd: float = 10.0) -> float:
            """Calculates position size using Fixed Fractional method."""
            risk_amount = account_equity * (risk_pct / 100)
            sl_cost_per_lot = sl_pips * pips_to_usd
            if sl_cost_per_lot <= 0: return 0.0
            position_size = risk_amount / sl_cost_per_lot
            return round(position_size, 2)

        def calculate_sl_tp_atr(self, entry_price: float, atr: float, rr_ratio: float, is_buy: bool, atr_multiplier: float):
            """Calculates SL/TP based on ATR."""
            sl_distance = atr * atr_multiplier
            tp_distance = sl_distance * rr_ratio
            
            stop_loss = entry_price - sl_distance if is_buy else entry_price + sl_distance
            take_profit = entry_price + tp_distance if is_buy else entry_price - tp_distance
            return stop_loss, take_profit
    ```
* **How to Test:** Write unit tests for both calculation methods with known inputs (equity, risk %, SL pips, ATR) to verify the outputs are correct.

#### **Step 7: Advanced Backtesting Engine**
* **Goal:** Upgrade the `BacktestEngine` to run a full simulation, managing a portfolio, executing trades, and tracking equity over time.
* **Files to Create/Modify:** `src/backtesting/engine.py`, `src/backtesting/metrics.py`.
* **Implementation (`src/backtesting/engine.py`):**
    ```python
    # This is a conceptual implementation of the main loop in BacktestEngine
    class BacktestEngine:
        def run_backtest(self, h4_data: DataFrame, h1_data: DataFrame):
            print("\n--- Starting Full Logic Backtest ---")
            equity = self.config['backtesting']['initial_balance']
            open_positions = []
            closed_trades = []

            # Align H1 data to H4 data by reindexing/merging
            aligned_data = pd.merge_asof(h1_data, h4_data, on='Date', direction='backward', suffixes=('_H1', '_H4'))
            aligned_data.dropna(inplace=True)

            for i in range(1, len(aligned_data)):
                current_slice = aligned_data.iloc[:i]
                latest_bar = current_slice.iloc[-1]
                
                # Check for closing open positions first
                # ... (code to check if latest_bar.High/Low hits SL/TP of any open position) ...

                # Regime Detection on H4 data
                regime = self.regime_detector.detect_regime_from_bar(latest_bar, self.config) # Method needs adjustment to take a bar not a df
                
                signal = "HOLD"
                if regime == "TRENDING" and not open_positions:
                    signal = self.trend_strategy.generate_signal_from_bar(latest_bar, self.config)
                
                if signal != "HOLD":
                    # ... (Call RiskManager to calculate size, SL, TP) ...
                    # ... (Create a trade object and add to open_positions) ...
                    # ... (Deduct spread/commission from equity) ...
            
            # After loop, generate report
            self.generate_report(closed_trades)
    ```
* **How to Test:** Run the backtest via `main.py`. The output should be a log of trades being opened and closed. At the end, it should print a basic performance summary (e.g., Final Equity, Number of Trades).

#### **Step 8: Performance Reporting**
* **Goal:** Implement the logic to calculate and display key performance metrics from the backtest results.
* **Files to Create/Modify:** `src/backtesting/metrics.py`, `src/backtesting/engine.py`.
* **Implementation (`src/backtesting/metrics.py`):**
    ```python
    def calculate_metrics(closed_trades: list, initial_balance: float) -> dict:
        """Calculates performance metrics like Profit Factor and Max Drawdown."""
        # ... logic to calculate net profit, profit factor, win rate, etc. ...
        # This can be a complex function involving equity curves.
        
        # Placeholder
        profits = sum(t['pnl'] for t in closed_trades if t['pnl'] > 0)
        losses = abs(sum(t['pnl'] for t in closed_trades if t['pnl'] < 0))
        profit_factor = profits / losses if losses > 0 else float('inf')
        
        return {'profit_factor': profit_factor, 'total_trades': len(closed_trades)}
    ```
* **How to Test:** Integrate the call to `calculate_metrics` at the end of the `BacktestEngine`'s `run_backtest` method and print the resulting dictionary of metrics.

---

### **Phase 4: Live Trading Preparation**

**Goal:** Build the components necessary to interact with the Deriv API and manage live order execution.

#### **Step 9: API Client & Order Manager**
* **Goal:** Implement the `DerivAPIClient` and `OrderManager` to handle live order placement.
* **Files to Create/Modify:** `src/api/deriv_client.py`, `src/core/order_manager.py`.
* **Implementation (`src/core/order_manager.py`):**
    ```python
    from src.api.deriv_client import DerivAPIClient

    class OrderManager:
        def __init__(self, api_client: DerivAPIClient):
            self.api = api_client

        async def place_order(self, symbol: str, order_type: str, lots: float, sl_price: float, tp_price: float):
            """Translates internal signal to an API order and executes it."""
            print(f"LIVE EXECUTION: Placing {order_type} order for {lots} lots of {symbol}.")
            # The actual API call to the Deriv wrapper
            # response = await self.api.place_order(symbol, order_type, lots, sl_price, tp_price)
            # return response
    ```
* **How to Test:** Using a **DEMO ACCOUNT**, write a small, separate test script to instantiate and call the `OrderManager`. Verify that the trade appears correctly in the Deriv trading platform.

#### **Step 10: Position Tracker**
* **Goal:** Implement the `PositionTracker` to maintain an in-memory state of all active positions.
* **Files to Create/Modify:** `src/core/position_tracker.py`.
* **Implementation (`src/core/position_tracker.py`):**
    ```python
    class PositionTracker:
        def __init__(self):
            self.active_positions = []

        def add_position(self, trade_details):
            self.active_positions.append(trade_details)

        def remove_position(self, position_id):
            self.active_positions = [p for p in self.active_positions if p['id'] != position_id]

        def get_open_positions_count(self) -> int:
            return len(self.active_positions)
    ```
* **How to Test:** Unit test the class by adding, removing, and counting positions to ensure state is managed correctly.

---

### **Phase 5: Live Operation and Monitoring**

**Goal:** Deploy the bot into a live (demo) environment and add real-time monitoring and alerting.

#### **Step 11: The Live Trading Loop**
* **Goal:** Create the main asynchronous loop that orchestrates the entire live trading process.
* **Files to Create/Modify:** `main.py`.
* **Implementation (`main.py`):**
    ```python
    import asyncio
    # ... other imports
    
    async def run_live_trader(config):
        # ... initialize all components (API client, managers, trackers) ...
        # await api_client.connect()
        
        while True:
            # max_positions = config['account_protection']['max_concurrent_positions']
            # if position_tracker.get_open_positions_count() >= max_positions:
            #     await asyncio.sleep(60)
            #     continue
                
            # 1. Fetch live data
            # 2. Calculate indicators and regime
            # 3. Generate signal
            # 4. If signal, call RiskManager and OrderManager
            # 5. Update PositionTracker
            # 6. Wait for a specified interval before next cycle
            print("Live loop running... waiting for next candle.")
            await asyncio.sleep(3600) # Wait for 1 hour for H1 candle
    ```
* **How to Test:** Run the `run_live_trader` function against a **demo account**. Monitor the logs closely to ensure it correctly fetches data, identifies regimes, and places trades according to the strategy without errors.

#### **Step 12: Monitoring and Alerting**
* **Goal:** Implement the `AlertManager` to provide real-time notifications for critical system events.
* **Files to Create/Modify:** `src/utils/alerts.py`.
* **Implementation (`src/utils/alerts.py`):**
    ```python
    class Alert:
        def __init__(self, condition: str, message: str):
            self.condition = condition
            self.message = message

    class AlertManager:
        def send_alert(self, alert: Alert):
            # For now, just print to console. Could be extended to email/Telegram.
            print(f"[ALERT] Condition: {alert.condition} | Message: {alert.message}")
    ```
* **How to Test:** In the `BacktestEngine` and the live trading loop, instantiate the `AlertManager`. Call `send_alert` whenever a key event occurs (e.g., "regime_change", "position_opened", "risk_limit_breached"). Verify that the alert messages are printed to the console as expected.