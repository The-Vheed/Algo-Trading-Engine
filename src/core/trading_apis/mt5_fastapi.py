import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from src.utils.logger import Logger

logger = Logger(__name__)


class MT5FastAPI:
    """
    Client for interacting with the MT5 FastAPI trading bot.
    Provides methods for trading operations, market data, and repository management.
    All responses are parsed internally and returned in appropriate Python data structures.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", access_key: str = ""):
        """
        Initialize the MT5 FastAPI client.

        Args:
            base_url (str): Base URL of the MT5 FastAPI server
            access_key (str): Access key for API authentication
        """
        self.base_url = base_url.rstrip("/")
        self.access_key = access_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """
        Test connection to the API server.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create session if not exists
            if self.session is None:
                self.session = aiohttp.ClientSession()

            # Test with a simple account_info call instead of positions_bias
            response = await self.get_account_info()
            logger.info("Successfully connected to MT5 FastAPI")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MT5 API: {e}")
            return False

    async def disconnect(self):
        """
        Disconnect from the API server.
        """
        if self.session:
            await self.session.close()
            self.session = None
        logger.debug("Disconnected from MT5 FastAPI")

    async def _make_request(
        self, endpoint: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make an async HTTP POST request to the API.

        Args:
            endpoint (str): API endpoint
            data (Dict[str, Any]): Request payload

        Returns:
            Dict[str, Any]: Raw JSON response data
        """
        if self.session is None:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}{endpoint}"

        try:
            async with self.session.post(url, json=data) as response:
                response_text = await response.text()
                logger.debug(f"Request to {endpoint}: {data}")
                logger.debug(f"Response status: {response.status}")
                logger.debug(
                    f"Response body: {response_text[:500]}..."
                )  # First 500 chars

                if response.status == 422:
                    logger.error(f"422 Error for {endpoint}. Request data: {data}")
                    logger.error(f"Response: {response_text}")

                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error {e.status} for {endpoint}: {e.message}")
            logger.error(f"Request data: {data}")
            raise
        except Exception as e:
            logger.error(f"Error making request to {endpoint}: {e}")
            raise

    def _parse_order_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse order execution response.

        Args:
            raw_response: Raw JSON response from order endpoint

        Returns:
            Dict containing parsed order information
        """
        try:
            return {
                "success": raw_response.get("status", False),
                "message": raw_response.get("message", ""),
                "order_details": raw_response.get("order_details", {}),
                "raw_response": raw_response,
            }
        except Exception as e:
            logger.error(f"Error parsing order response: {e}")
            return {
                "success": False,
                "message": f"Error parsing response: {e}",
                "order_details": {},
                "raw_response": raw_response,
            }

    def _parse_positions_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse positions bias response.

        Args:
            raw_response: Raw JSON response from positions_bias endpoint

        Returns:
            Dict containing parsed positions information
        """
        try:
            return {
                "bias": raw_response.get("bias"),
                "positions_count": raw_response.get("positions_count", 0),
                "last_trade": raw_response.get("last_trade"),
                "balance": raw_response.get(
                    "balance", 0.0
                ),  # Add balance field from account_info
                "raw_response": raw_response,
            }
        except Exception as e:
            logger.error(f"Error parsing positions response: {e}")
            return {
                "bias": None,
                "positions_count": 0,
                "last_trade": None,
                "balance": 0.0,  # Add default balance
                "raw_response": raw_response,
            }

    def _parse_account_info_response(
        self, raw_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse account information response.

        Args:
            raw_response: Raw JSON response from account_info endpoint

        Returns:
            Dict containing parsed account information
        """
        try:
            return {
                "bias": raw_response.get("bias"),
                "positions_count": raw_response.get("positions_count", 0),
                "last_trade": raw_response.get("last_trade"),
                "balance": raw_response.get("balance", 0.0),
                "raw_response": raw_response,
            }
        except Exception as e:
            logger.error(f"Error parsing account info response: {e}")
            return {
                "bias": None,
                "positions_count": 0,
                "last_trade": None,
                "balance": 0.0,
                "raw_response": raw_response,
            }

    def _parse_price_data_response(
        self, raw_response: Dict[str, Any]
    ) -> Dict[str, Union[pd.DataFrame, float, str]]:
        """
        Parse price data response and convert to DataFrames.

        Args:
            raw_response: Raw JSON response from price_data endpoint

        Returns:
            Dict containing DataFrames for each timeframe and other metadata
        """
        try:
            result = {
                "symbol": raw_response.get("symbol", ""),
                "data": {},
                "raw_response": raw_response,
            }

            data_section = raw_response.get("data", {})

            for timeframe, tf_data in data_section.items():
                try:
                    # Convert from "split" format back to DataFrame
                    df = self._convert_split_dict_to_dataframe(tf_data)
                    if not df.empty:
                        result["data"][timeframe] = df

                        logger.debug(
                            f"Successfully converted {timeframe} data to DataFrame with {len(df)} records"
                        )
                    else:
                        logger.warning(f"Empty DataFrame created for {timeframe}")
                        result["data"][timeframe] = pd.DataFrame()
                except Exception as e:
                    logger.error(f"Error converting {timeframe} data to DataFrame: {e}")
                    result["data"][timeframe] = pd.DataFrame()

            return result
        except Exception as e:
            logger.error(f"Error parsing price data response: {e}")
            return {
                "symbol": "",
                "data": {},
                "raw_response": raw_response,
            }

    def _convert_split_dict_to_dataframe(
        self, split_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Convert data from "split" dictionary format to pandas DataFrame.

        The "split" format from pandas to_dict("split") looks like:
        {
            "index": [list of index values],
            "columns": [list of column names],
            "data": [list of lists, each inner list is a row]
        }

        Args:
            split_data: Dictionary in "split" format

        Returns:
            pandas DataFrame
        """
        try:
            if not isinstance(split_data, dict):
                logger.error(f"Expected dict for split_data, got {type(split_data)}")
                return pd.DataFrame()

            # Check if we have the required keys
            required_keys = ["index", "columns", "data"]
            if not all(key in split_data for key in required_keys):
                logger.error(
                    f"Missing required keys in split_data. Expected {required_keys}, got {list(split_data.keys())}"
                )
                return pd.DataFrame()

            # Capitalize all column names before creating the DataFrame
            capitalized_columns = []
            for col in split_data["columns"]:
                if isinstance(col, str):
                    capitalized_columns.append(col.capitalize())
                else:
                    capitalized_columns.append(col)

            logger.debug(f"Original columns: {split_data['columns']}")
            logger.debug(f"Capitalized columns: {capitalized_columns}")

            # Create DataFrame with capitalized column names - no timestamp conversion here
            df = pd.DataFrame(
                data=split_data["data"],
                index=split_data["index"],
                columns=capitalized_columns,
            )

            # Ensure index name is "Date"
            df.index.name = "Date"

            # Ensure OHLC columns are float type
            ohlc_columns = ["Open", "High", "Low", "Close"]
            for col in ohlc_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Handle Volume column
            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
            else:
                df["Volume"] = 0.0

            logger.debug(
                f"Converted split data to DataFrame: {df.shape}, columns: {df.columns.tolist()}"
            )
            return df

        except Exception as e:
            logger.error(f"Error converting split data to DataFrame: {e}")
            import traceback

            traceback.print_exc()
            return pd.DataFrame()

    def _parse_repository_response(
        self, raw_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse repository operation response.

        Args:
            raw_response: Raw JSON response from repository endpoint

        Returns:
            Dict containing parsed repository operation result
        """
        try:
            return {
                "success": raw_response.get("status", False),
                "message": raw_response.get("message", ""),
                "raw_response": raw_response,
            }
        except Exception as e:
            logger.error(f"Error parsing repository response: {e}")
            return {
                "success": False,
                "message": f"Error parsing response: {e}",
                "raw_response": raw_response,
            }

    async def execute_buy_order(
        self,
        symbol: str,
        price: float,
        comment: str = "",
        tp: Optional[float] = None,
        sl: Optional[float] = None,
        trailing: float = 0.0,
        risk: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Execute a market buy order.

        Returns:
            Dict[str, Any]: Parsed response containing order execution details
        """
        data = {
            "access_key": self.access_key,
            "symbol": symbol,
            "price": price,
            "comment": comment,
            "tp": tp,
            "sl": sl,
            "trailing": trailing,
            "risk": risk,
        }

        raw_response = await self._make_request("/buy", data)
        return self._parse_order_response(raw_response)

    async def execute_sell_order(
        self,
        symbol: str,
        price: float,
        comment: str = "",
        tp: Optional[float] = None,
        sl: Optional[float] = None,
        trailing: float = 0.0,
        risk: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Execute a market sell order.

        Returns:
            Dict[str, Any]: Parsed response containing order execution details
        """
        data = {
            "access_key": self.access_key,
            "symbol": symbol,
            "price": price,
            "comment": comment,
            "tp": tp,
            "sl": sl,
            "trailing": trailing,
            "risk": risk,
        }

        raw_response = await self._make_request("/sell", data)
        return self._parse_order_response(raw_response)

    async def close_all_positions(self) -> Dict[str, Any]:
        """
        Close all open positions created by the bot.

        Returns:
            Dict[str, Any]: Parsed response containing details of closed positions
        """
        data = {"access_key": self.access_key}
        raw_response = await self._make_request("/close_all", data)

        try:
            return {
                "success": raw_response.get("status", False),
                "message": raw_response.get("message", ""),
                "closed_positions": raw_response.get("closed_positions", []),
                "raw_response": raw_response,
            }
        except Exception as e:
            logger.error(f"Error parsing close_all response: {e}")
            return {
                "success": False,
                "message": f"Error parsing response: {e}",
                "closed_positions": [],
                "raw_response": raw_response,
            }

    async def get_price_data(
        self,
        symbol: str,
        timeframes: List[str],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        count: Optional[int] = None,
    ) -> Dict[str, Union[pd.DataFrame, float, str]]:
        """
        Retrieve price data for specified symbol and timeframes.

        Returns:
            Dict containing DataFrames for each timeframe, symbol, and balance
        """
        data = {
            "access_key": self.access_key,
            "symbol": symbol,
            "timeframes": timeframes,
        }

        # Only add optional parameters if they have values
        if start_time is not None:
            data["start_time"] = start_time
        if end_time is not None:
            data["end_time"] = end_time
        if count is not None:
            data["count"] = count

        raw_response = await self._make_request("/price_data", data)
        return self._parse_price_data_response(raw_response)

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get information about the trading account including balance and trading bias.

        Returns:
            Dict[str, Any]: Parsed account information
        """
        data = {"access_key": self.access_key}
        raw_response = await self._make_request("/account_info", data)
        return self._parse_account_info_response(raw_response)

    async def get_positions_bias(self) -> Dict[str, Any]:
        """
        Get information about current trading bias based on open positions.
        Note: This method is kept for backward compatibility and now calls get_account_info().

        Returns:
            Dict[str, Any]: Parsed trading bias information
        """
        # Call the new account_info endpoint instead
        return await self.get_account_info()

    async def pull_repository(self) -> Dict[str, Any]:
        """
        Pull latest changes from the git repository.

        Returns:
            Dict[str, Any]: Parsed response status
        """
        data = {"access_key": self.access_key}
        raw_response = await self._make_request("/pull", data)
        return self._parse_repository_response(raw_response)

    async def push_repository(self) -> Dict[str, Any]:
        """
        Push local changes to the git repository.
        Note: This endpoint is currently disabled in the API.

        Returns:
            Dict[str, Any]: Parsed response status
        """
        data = {"access_key": self.access_key}
        raw_response = await self._make_request("/push", data)
        return self._parse_repository_response(raw_response)
