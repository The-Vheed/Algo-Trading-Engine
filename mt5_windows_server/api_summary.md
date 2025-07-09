# Trading Bot API Documentation

## Overview
This API provides endpoints for automated trading operations via MetaTrader 5. All requests require a valid access key for authentication.

Base URL: http://127.0.0.1:8000

## Authentication
All endpoints require an `access_key` parameter in the request body. This key must match the API's configured access key.

## Endpoints

### Trading Operations

#### 1. Execute a Buy Order
- **Endpoint**: POST /buy
- **Description**: Executes a market buy order with specified parameters
- **Request Body**:
  ```json
  {
    "access_key": "your_access_key",
    "symbol": "Volatility 25 Index", // or "Step Index", "USDJPY", "EURUSD"
    "price": 1234.56, // Current price (reference only)
    "comment": "Optional comment",
    "tp": 1235.56, // Take profit price
    "sl": 1233.56, // Stop loss price
    "trailing": 0.5, // Trailing stop value (0 for no trailing)
    "risk": 2.0 // Risk percentage of account balance
  }
  ```
- **Response**:
  ```json
  {
    "status": true,
    "message": "Buy order executed",
    "order_details": { /* Order execution details */ }
  }
  ```

#### 2. Execute a Sell Order
- **Endpoint**: POST /sell
- **Description**: Executes a market sell order with specified parameters
- **Request Body**: Same structure as the buy order
- **Response**: Same structure as the buy order response

#### 3. Close All Open Positions
- **Endpoint**: POST /close_all
- **Description**: Closes all open positions created by the bot
- **Request Body**:
  ```json
  {
    "access_key": "your_access_key"
  }
  ```
- **Response**:
  ```json
  {
    "status": true,
    "message": "Close all positions executed",
    "closed_positions": [ /* List of closed position details */ ]
  }
  ```

### Market Data

#### 4. Get Price Data
- **Endpoint**: POST /price_data
- **Description**: Retrieves price data for specified symbol and timeframes. Also handles trailing stop updates.
- **Request Body**:
  ```json
  {
    "access_key": "your_access_key",
    "symbol": "Volatility 25 Index",
    "timeframes": ["M5", "H1"], // Available: M1, M5, M15, M30, H1, H4, D1, W1, MN1
    "start_time": "2023-01-01T00:00:00Z", // Optional
    "end_time": "2023-01-31T00:00:00Z", // Optional
    "count": 100 // Optional, number of candles to fetch
  }
  ```
- **Response**:
  ```json
  {
    "symbol": "Volatility 25 Index",
    "data": {
      "M5": { /* Price data for M5 timeframe */ },
      "H1": { /* Price data for H1 timeframe */ }
    }
  }
  ```

### Trading Analysis

#### 5. Get Account Information
- **Endpoint**: POST /account_info
- **Description**: Returns account information including balance and current trading bias
- **Request Body**:
  ```json
  {
    "access_key": "your_access_key"
  }
  ```
- **Response**:
  ```json
  {
    "bias": "BUY", // or "SELL" or null if balanced/no positions
    "positions_count": 3,
    "last_trade": [1642528000, 123.45], // [timestamp, profit]
    "balance": 10000.00
  }
  ```

### Repository Management

#### 6. Pull Latest Changes
- **Endpoint**: POST /pull
- **Description**: Pulls the latest changes from the git repository
- **Request Body**:
  ```json
  {
    "access_key": "your_access_key"
  }
  ```
- **Response**:
  ```json
  {
    "status": true
  }
  ```

#### 7. Push Local Changes
- **Endpoint**: POST /push
- **Description**: Pushes local changes to the git repository (currently disabled in the code)
- **Request Body**:
  ```json
  {
    "access_key": "your_access_key"
  }
  ```
- **Response**:
  ```json
  {
    "status": true
  }
  ```

## Using the API with Python Requests

Example of how to call the API using Python's requests library:

```python
import requests
import json

BASE_URL = "http://127.0.0.1:8000"
ACCESS_KEY = "your_access_key"

# Example: Execute a buy order
def place_buy_order():
    endpoint = f"{BASE_URL}/buy"
    
    data = {
        "access_key": ACCESS_KEY,
        "symbol": "Volatility 25 Index",
        "price": 1000.0,
        "comment": "API Test Order",
        "tp": 1010.0,
        "sl": 990.0,
        "trailing": 0.5,
        "risk": 2.0
    }
    
    response = requests.post(endpoint, json=data)
    return response.json()

# Example: Get account information
def get_account_info():
    endpoint = f"{BASE_URL}/account_info"
    
    data = {
        "access_key": ACCESS_KEY
    }
    
    response = requests.post(endpoint, json=data)
    return response.json()
```
