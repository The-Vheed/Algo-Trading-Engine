from src.utils.logger import Logger

logger = Logger(__name__)

import os
import random
import time
import asyncio
import pandas as pd
from pandas import DataFrame
from typing import List, Dict, Union, AsyncGenerator, Generator, Tuple, Optional, Any
from pandas import DataFrame, Series, concat
from datetime import datetime, timedelta

# Remove deriv_api import
# from deriv_api import DerivAPI


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataProvider:
    """
    Provides data either from CSV files (for backtesting) or from a live API.
    It can stream data indefinitely, yielding the latest available data for
    each configured timeframe.
    """

    def __init__(
        self,
        source_type: str,  # 'csv', 'live', or 'historical_live'
        symbol: str = "EURUSD",
        timeframes: List[str] = ["H1", "H4"],
        counts: List[int] = [1250, 500],
        trading_api: Optional[object] = None,  # Trading API object (MT5FastAPI, etc.)
        base_path: str = "",
        timeout: int = 30,
        retries: int = 3,
        initial_fetch_timeout: int = 60,  # Max time to wait for initial live data
        live_polling_interval: int = 10,  # How often to poll for new live data (seconds)
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> None:
        """
        Initializes the DataProvider.

        Args:
            source_type (str): 'csv' for backtesting from CSV files, 'live' for real-time data from API,
                               'historical_live' for historical data from API.
            symbol (str): The trading symbol (e.g., "EURUSD").
            timeframes (List[str]): List of timeframes to load/stream (e.g., ["H1", "H4"]).
            counts (List[int]): List of historical data counts corresponding to timeframes.
            trading_api (Optional[object]): Trading API object for live data (MT5FastAPI, etc.)
            base_path (str): Base path for CSV files (required for 'csv' source_type).
            timeout (int): API request timeout in seconds.
            retries (int): Number of retries for API requests.
            initial_fetch_timeout (int): Max time to wait for initial live data in seconds.
            live_polling_interval (int): How often to poll for new live data in seconds.
            start_time (Optional[str]): Start time for historical data in ISO format (YYYY-MM-DDTHH:MM:SS).
            end_time (Optional[str]): End time for historical data in ISO format (YYYY-MM-DDTHH:MM:SS).
        """
        if source_type not in ["csv", "live", "historical_live"]:
            raise ValueError("source_type must be 'csv', 'live', or 'historical_live'")

        if len(timeframes) != len(counts):
            raise ValueError("Lengths of 'timeframes' and 'counts' must be equal.")

        self.source_type = source_type
        self.symbol = symbol
        self.timeframes = timeframes
        self.counts = dict(zip(timeframes, counts))  # Map timeframe to its count
        self.trading_api = trading_api
        self.timeout = timeout
        self.retries = retries
        self.initial_fetch_timeout = initial_fetch_timeout
        self.live_polling_interval = live_polling_interval
        self.start_time = start_time
        self.end_time = end_time
        self.base_path = base_path

        self.is_connected: bool = False  # Track connection status
        self.data_loaded: bool = False  # Track if data has been loaded

        self.data_store: Dict[str, pd.DataFrame] = {}
        self.last_fetched_time: Dict[str, datetime] = {}
        self.next_fetch_time: Dict[str, datetime] = {}

        # Validate base_path for CSV mode (but don't load data yet)
        if self.source_type == "csv":
            if not base_path or not os.path.isdir(base_path):
                raise ValueError(
                    f"For 'csv' source_type, 'base_path' must be a valid directory. Got: {base_path}"
                )
        elif self.source_type == "historical_live" and not trading_api:
            raise ValueError(
                "trading_api must be provided for historical_live data source"
            )
        elif self.source_type == "live" and not trading_api:
            raise ValueError("trading_api must be provided for live data source")

    async def stream_data(self) -> AsyncGenerator[Dict[str, pd.DataFrame], None]:
        """
        Yields the latest data for each timeframe.
        In CSV/historical_live mode, it simulates a stream. In live mode, it queries the API.
        """
        # Load data if not already loaded
        if not self.data_loaded:
            if self.source_type == "csv":
                self.__load_initial_csv_data()
                self.data_loaded = True
            elif self.source_type == "historical_live":
                await self.__load_historical_data_from_api()
                self.data_loaded = True

        if self.source_type in ["csv", "historical_live"]:
            # Use async for to iterate over the async generator __stream_datastore
            async for data in self.__stream_datastore():
                yield data
        elif self.source_type == "live":
            # Use async for to iterate over the async generator __stream_live_data
            async for data in self.__stream_live_data():
                yield data

    def __load_initial_csv_data(self) -> None:
        """
        Loads all required CSVs into an internal dictionary mapping timeframe to its dataframe.
        This is done only once at initialization for CSV mode.
        """
        for tf in self.timeframes:
            file_path = os.path.join(self.base_path, self.symbol, f"{tf}.csv")
            logger.debug(f"Loading data from: {file_path}")
            try:
                df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
                self.validate_data(df, tf)
                self.data_store[tf] = df
                logger.debug(f"Loaded {len(df)} records for {tf} from {file_path}")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"CSV file not found for {self.symbol}/{tf} at {file_path}"
                )
            except Exception as e:
                raise RuntimeError(f"Error loading CSV for {self.symbol}/{tf}: {e}")

        # Determine the lowest common starting point for streaming
        # Find the earliest start date among all loaded CSVs
        if self.data_store:
            earliest_start = min(df.index.min() for df in self.data_store.values())
            self.current_stream_time = earliest_start
            logger.info(
                f"CSV data loaded. Stream will start from: {self.current_stream_time}"
            )
        else:
            self.current_stream_time = None
            logger.warning("No CSV data loaded.")

    def validate_data(self, df: pd.DataFrame, tf: str):
        """
        Validates the structure and integrity of a DataFrame.
        """
        if df.empty:
            logger.warning(f"DataFrame for {tf} is empty.")
            return

        # Check required columns
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(
                    f"Missing column '{col}' in {tf} data. Columns found: {df.columns.tolist()}"
                )

        # Check index type and name
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"Index for {tf} data is not a DatetimeIndex.")
        if df.index.name != "Date":
            logger.warning(f"Index name for {tf} is not 'Date'. Renaming it.")
            df.index.name = "Date"

        # Check date continuity
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"Date index not sorted in {tf} data.")
        if df.index.has_duplicates:
            raise ValueError(f"Duplicate dates found in {tf} data.")

        # Ensure numeric columns are numeric (e.g., O, H, L, C, V)
        for col in [
            c for c in REQUIRED_COLUMNS if c != "Date"
        ]:  # Skip 'Date' as it's the index
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' in {tf} data is not numeric.")

    def __get_timeframe_interval(self, tf: str) -> timedelta:
        """Helper to convert timeframe string to timedelta for live polling."""
        if tf.startswith("S"):
            return timedelta(seconds=int(tf[1:]))
        elif tf.startswith("M"):
            return timedelta(minutes=int(tf[1:]))
        elif tf.startswith("H"):
            return timedelta(hours=int(tf[1:]))
        elif tf.startswith("D"):
            return timedelta(days=int(tf[1:]))
        # Add more as needed for other timeframes (W, M etc.)
        else:
            raise ValueError(f"Unsupported timeframe format: {tf}")

    async def __load_historical_data_from_api(self) -> None:
        """
        Loads historical data from the trading API based on specified start and end times.
        This is used when source_type is 'historical_live'.
        """
        if not self.trading_api:
            raise ValueError("Trading API must be provided for historical data loading")

        logger.info(f"Loading historical data from trading API for {self.symbol}")
        logger.info(f"Time range: {self.start_time} to {self.end_time}")

        for tf in self.timeframes:
            try:
                logger.debug(f"Fetching historical data for {tf}...")
                response = await self.trading_api.get_price_data(
                    symbol=self.symbol,
                    timeframes=[tf],
                    start_time=self.start_time,
                    end_time=self.end_time,
                    count=None,  # Let the API determine count based on date range
                )

                if response and "data" in response and tf in response["data"]:
                    df = response["data"][tf]
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # Use the centralized timestamp parser
                        df = self._parse_timestamps(df)
                        self.validate_data(df, tf)
                        self.data_store[tf] = df
                        logger.info(
                            f"Loaded {len(df)} records for {tf} from trading API"
                        )
                    else:
                        logger.warning(
                            f"Empty or invalid DataFrame for {tf} from trading API"
                        )
                else:
                    logger.warning(
                        f"Failed to load historical data for {tf} from trading API"
                    )
            except Exception as e:
                logger.error(
                    f"Error loading historical data for {tf} from trading API: {e}"
                )
                raise RuntimeError(
                    f"Error loading historical data for {self.symbol}/{tf}: {e}"
                )

        # Determine the earliest common starting point for streaming
        if self.data_store:
            earliest_start = min(df.index.min() for df in self.data_store.values())
            self.current_stream_time = earliest_start
            logger.info(
                f"Historical data loaded from API. Stream will start from: {self.current_stream_time}"
            )
        else:
            self.current_stream_time = None
            logger.warning("No historical data loaded from API.")

    async def __stream_datastore(self) -> AsyncGenerator[Dict[str, pd.DataFrame], None]:
        """
        Streams data from loaded datastore (CSV or historical API data).
        Yields dataframes for each timeframe, with lengths equal to their respective counts,
        ensuring the end-timestamps are synchronized as much as possible.
        No data is yielded until all timeframes have at least their `count` number of records.
        """
        if not self.data_store:
            logger.error("No data available in data store to stream.")
            return

        sorted_timeframes = sorted(
            self.timeframes, key=lambda tf: self.__get_timeframe_interval(tf)
        )
        if not sorted_timeframes:
            logger.warning("No timeframes specified for streaming.")
            return

        master_step_interval = self.__get_timeframe_interval(sorted_timeframes[0])

        earliest_overall_data_start = min(
            df.index.min() for df in self.data_store.values()
        )
        latest_overall_data_end = max(df.index.max() for df in self.data_store.values())

        current_master_time = earliest_overall_data_start
        logger.info(
            f"Data Store Stream initialized. Master clock starting from: {current_master_time}. "
            f"Will advance by {master_step_interval}."
        )

        # --- Warm-up / Initial Data Collection Phase ---
        # Advance the master clock until all timeframes can provide full 'counts' data
        initial_yield_achieved = False
        # warmup_start_time = datetime.now()

        while (
            not initial_yield_achieved
            and current_master_time <= latest_overall_data_end
        ):
            all_timeframes_ready = True
            temp_data_for_check = {}

            for tf in sorted_timeframes:
                df = self.data_store[tf]
                required_count = self.counts[tf]

                end_idx_pos = df.index.searchsorted(current_master_time, side="right")

                # Check if enough data exists for this timeframe to meet the count
                # This ensures we have at least 'required_count' points ending at end_idx_pos
                if end_idx_pos < required_count:
                    all_timeframes_ready = False
                    logger.debug(
                        f"Warm-up: {tf} has only {end_idx_pos} records, needs {required_count}. "
                        f"Not yet ready. Current master time: {current_master_time}"
                    )
                    break  # Break inner loop, as one TF isn't ready

                # If ready, slice the data to form the current view
                start_idx_pos = max(0, end_idx_pos - required_count)
                sliced_df = df.iloc[start_idx_pos:end_idx_pos].copy()

                if sliced_df.empty or sliced_df.index.max() > current_master_time:
                    # This could happen if the master time is ahead of data or data is sparse
                    all_timeframes_ready = False
                    logger.debug(
                        f"Warm-up: {tf} sliced data is empty or too new for {current_master_time}. Not ready."
                    )
                    break  # Break inner loop, as this TF isn't providing valid data yet

                temp_data_for_check[tf] = (
                    sliced_df  # Store for the potential first yield
                )

            if all_timeframes_ready:
                # All timeframes are now ready to provide data of required count
                yield temp_data_for_check
                logger.debug(
                    f"Initial data yielded at {current_master_time} after warm-up."
                )
                initial_yield_achieved = True
            else:
                # Not all timeframes are ready, advance the master clock and try again
                current_master_time += master_step_interval

            # Add a safety break for very long warm-up phases, though typically not needed
            if current_master_time > latest_overall_data_end:
                logger.error(
                    "Warm-up failed: Reached end of data without all timeframes meeting 'count' requirements."
                )
                break

        if not initial_yield_achieved:
            logger.critical(
                "Data stream could not find a starting point where all timeframes met their 'count' requirements. Stream will not yield any data."
            )
            return  # Exit if warm-up failed

        # --- Main Streaming Loop (continues from where warm-up left off) ---
        while current_master_time <= latest_overall_data_end:
            yielded_data_this_round = {}
            # at_least_one_tf_updated = False

            for tf in sorted_timeframes:
                df = self.data_store[tf]
                required_count = self.counts[tf]

                end_idx_pos = df.index.searchsorted(current_master_time, side="right")

                # If there isn't enough data from this point backwards, we simply don't include it in this yield
                if end_idx_pos < required_count:
                    logger.debug(
                        f"{tf} does not have enough records ({end_idx_pos}) to meet count ({required_count}) at {current_master_time}. Skipping for this yield."
                    )
                    continue  # Skip this timeframe for the current yield, but don't stop the whole stream

                start_idx_pos = max(0, end_idx_pos - required_count)
                sliced_df = df.iloc[start_idx_pos:end_idx_pos].copy()

                if not sliced_df.empty and sliced_df.index.max() <= current_master_time:
                    yielded_data_this_round[tf] = sliced_df
                    # at_least_one_tf_updated = True
                    logger.debug(
                        f"Prepared {len(sliced_df)} records for {tf} ending at {sliced_df.index.max()}. (Target: {current_master_time})"
                    )
                else:
                    logger.debug(
                        f"No valid data slice for {tf} at {current_master_time}. Empty: {sliced_df.empty}, Max Time: {sliced_df.index.max() if not sliced_df.empty else 'N/A'}"
                    )

            if (
                yielded_data_this_round
            ):  # Only yield if data was prepared for at least one timeframe
                yield yielded_data_this_round
                logger.debug(
                    f"Yielded data ending at {current_master_time}"
                )  # Changed to debug
            else:
                logger.debug(
                    f"No new data to yield for any timeframe at {current_master_time}. Advancing master clock."
                )

            # Advance the master clock for the next iteration
            current_master_time += master_step_interval

        logger.debug(
            "Data stream finished: Reached end of available historical data."
        )  # Changed to debug

    async def __stream_live_data(self) -> AsyncGenerator[Dict[str, pd.DataFrame], None]:
        """
        Streams live data by querying the trading API.
        """
        logger.debug("Starting live data stream...")

        # --- Initial Data Fetch Phase ---
        initial_fetch_successful = False
        fetch_start_time = time.time()

        while time.time() - fetch_start_time < self.initial_fetch_timeout:
            try:
                initial_data = await self.__fetch_data_from_trading_api(
                    self.symbol, self.timeframes, self.counts
                )

                all_initial_data_ready = True
                for tf in self.timeframes:
                    df = initial_data.get(tf)
                    if df is None or df.empty or len(df) < self.counts[tf]:
                        logger.warning(
                            f"Initial fetch for {tf} is incomplete or empty. "
                            f"Got {len(df) if df is not None else 0} records, "
                            f"needed {self.counts[tf]}."
                        )
                        all_initial_data_ready = False
                        break

                if all_initial_data_ready:
                    for tf, df in initial_data.items():
                        self.data_store[tf] = df
                        self.last_fetched_time[tf] = df.index.max()
                        self.next_fetch_time[tf] = (
                            df.index.max() + self.__get_timeframe_interval(tf)
                        )
                        logger.debug(
                            f"Initialized {tf} data ending at {self.last_fetched_time[tf]}"
                        )
                    initial_fetch_successful = True
                    logger.debug("All initial live data fetches complete.")
                    break
                else:
                    logger.warning(
                        "Not all timeframes ready after initial fetch. Retrying..."
                    )
                    await asyncio.sleep(self.live_polling_interval)

            except Exception as e:
                logger.warning(
                    f"Initial live data fetch failed: {e}. Retrying in {self.live_polling_interval}s..."
                )
                await asyncio.sleep(self.live_polling_interval)

        if not initial_fetch_successful:
            raise RuntimeError(
                "Failed to fetch initial live data for all timeframes after multiple retries."
            )

        # Yield the initially fetched data snapshot
        yield self.data_store.copy()
        logger.debug("Initial live data snapshot yielded.")

        # Determine the smallest interval for the master clock step
        if not self.timeframes:
            logger.warning("No timeframes specified for streaming.")
            return

        sorted_timeframes = sorted(
            self.timeframes, key=lambda tf: self.__get_timeframe_interval(tf)
        )
        master_step_interval = self.__get_timeframe_interval(sorted_timeframes[0])

        latest_initial_data_end = max(self.last_fetched_time.values())
        current_master_time = latest_initial_data_end + master_step_interval

        while True:
            updated_any = False

            for tf in sorted_timeframes:
                interval = self.__get_timeframe_interval(tf)
                required_count = self.counts[tf]

                if current_master_time >= self.next_fetch_time[tf]:
                    try:
                        new_data = (
                            await self.__fetch_data_from_trading_api_for_timeframe(
                                self.symbol, tf, required_count
                            )
                        )

                        if not new_data.empty:
                            new_data_max_time = new_data.index.max()
                            last_fetched_dt = self.last_fetched_time.get(tf)

                            logger.debug(
                                f"Data for {tf}. Latest bar ends at: {new_data_max_time}"
                            )

                            if (
                                last_fetched_dt is None
                                or new_data_max_time > last_fetched_dt
                            ):
                                self.data_store[tf] = new_data.iloc[-required_count:]
                                self.last_fetched_time[tf] = self.data_store[
                                    tf
                                ].index.max()
                                updated_any = True
                        else:
                            logger.debug(f"No new data available for {tf}")
                    except Exception as e:
                        logger.error(f"Error fetching live data for {tf}: {e}")

                    self.next_fetch_time[tf] = self.next_fetch_time[tf] + interval

            if updated_any:
                yield self.data_store.copy()
                logger.debug(
                    f"Yielded live data snapshot at master time {current_master_time}"
                )

            current_master_time += master_step_interval

            # Calculate sleep duration
            current_time = datetime.now()
            smallest_tf = sorted_timeframes[0]
            smallest_interval = self.__get_timeframe_interval(smallest_tf)

            current_time_ts = current_time.timestamp()
            interval_seconds = smallest_interval.total_seconds()
            rounded_ts = (current_time_ts // interval_seconds) * interval_seconds

            next_interval_start = datetime.fromtimestamp(rounded_ts + interval_seconds)
            target_wake_time = next_interval_start + timedelta(seconds=3)

            sleep_duration = (target_wake_time - current_time).total_seconds()
            sleep_duration = max(0.01, sleep_duration)

            logger.info(
                f"Next candle closes at {next_interval_start}, "
                f"targeting wake at {target_wake_time} "
                f"(sleeping {sleep_duration:.3f}s)"
            )
            await asyncio.sleep(sleep_duration)

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse timestamps in DataFrame.
        Centralizes all timestamp conversion in one place to ensure consistency.

        Args:
            df: DataFrame with potential timestamp data

        Returns:
            DataFrame with properly parsed timestamps
        """
        if df is None or df.empty:
            return df

        # Handle the Time column - this is our actual timestamp data
        if "Time" in df.columns:
            try:
                # Convert Time column to datetime objects if it contains numeric timestamps
                if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
                    df["Time"] = df["Time"].apply(
                        lambda x: (
                            datetime.fromtimestamp(x)
                            if isinstance(x, (int, float))
                            else x
                        )
                    )
                    logger.debug("Converted 'Time' column to datetime objects")

                # Set the Time column as the index
                df = df.set_index("Time")
                df.index.name = "Date"  # Rename index for consistency
                logger.debug("Set 'Time' column as DataFrame index with name 'Date'")
            except Exception as e:
                logger.warning(f"Failed to process Time column: {e}")

        return df

    async def __fetch_data_from_trading_api(
        self, symbol: str, timeframes: List[str], counts: Dict[str, int]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch initial batch of historical price data for all timeframes using the trading API.
        """
        try:
            # Convert counts dict to a single count (use max for initial fetch)
            max_count = max(counts.values())

            logger.debug(
                f"Fetching data for symbol: {symbol}, timeframes: {timeframes}, count: {max_count}"
            )

            # The trading API now returns parsed response with DataFrames
            response = await self.trading_api.get_price_data(
                symbol=symbol, timeframes=timeframes, count=max_count
            )

            logger.debug(
                f"API response keys: {response.keys() if response else 'None'}"
            )

            all_timeframe_data = {}
            if response and "data" in response:
                logger.debug(
                    f"Available timeframes in response: {list(response['data'].keys())}"
                )
                for tf in timeframes:
                    if tf in response["data"]:
                        df = response["data"][tf]  # Already a DataFrame
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            # Use the centralized timestamp parser
                            df = self._parse_timestamps(df)

                            # Trim to requested count for this timeframe
                            required_count = counts.get(tf, max_count)
                            df = df.iloc[-required_count:]
                            all_timeframe_data[tf] = df
                            logger.debug(
                                f"Successfully processed {len(df)} records for {tf}"
                            )
                        else:
                            logger.warning(f"Empty or invalid DataFrame for {tf}")
                    else:
                        logger.warning(f"Timeframe {tf} not found in API response")
            else:
                logger.error(f"Invalid API response format: {response}")

            return all_timeframe_data
        except Exception as e:
            logger.error(f"Error fetching data from trading API: {e}")
            import traceback

            traceback.print_exc()
            return {}

    async def __fetch_data_from_trading_api_for_timeframe(
        self, symbol: str, timeframe: str, count: int
    ) -> pd.DataFrame:
        """
        Fetch latest price data for a specific timeframe using the trading API.
        """
        try:
            # The trading API now returns parsed response with DataFrames
            response = await self.trading_api.get_price_data(
                symbol=symbol, timeframes=[timeframe], count=count
            )

            if response and "data" in response and timeframe in response["data"]:
                df = response["data"][timeframe]  # Already a DataFrame
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Use the centralized timestamp parser
                    df = self._parse_timestamps(df)
                    return df
                else:
                    logger.warning(
                        f"Expected DataFrame for {timeframe}, got {type(df)}"
                    )

            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching {timeframe} data from trading API: {e}")
            return pd.DataFrame()
