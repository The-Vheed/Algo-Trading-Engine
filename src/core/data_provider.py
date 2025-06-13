from src.utils.logger import Logger

logger = Logger(__name__)

import os
import random
import time
import asyncio
import pandas as pd
from pandas import DataFrame
from typing import List, Dict, Union, AsyncGenerator, Generator, Tuple, Optional
from pandas import DataFrame, Series, concat
from datetime import datetime, timedelta

from deriv_api import DerivAPI


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataProvider:
    """
    Provides data either from CSV files (for backtesting) or from a live API.
    It can stream data indefinitely, yielding the latest available data for
    each configured timeframe.
    """

    def __init__(
        self,
        source_type: str,  # 'csv' or 'live'
        symbol: str = "EURUSD",
        timeframes: List[str] = ["H1", "H4"],
        counts: List[int] = [1250, 500],
        api_key: str = "",
        app_id: int = 1089,  # Default to 1089 for testing if not provided
        base_path: str = "",
        timeout: int = 30,
        retries: int = 3,
        initial_fetch_timeout: int = 60,  # Max time to wait for initial live data
        live_polling_interval: int = 10,  # How often to poll for new live data (seconds)
    ) -> None:
        """
        Initializes the DataProvider.

        Args:
            source_type (str): 'csv' for backtesting from CSV files, 'live' for real-time data from API.
            symbol (str): The trading symbol (e.g., "EURUSD").
            timeframes (List[str]): List of timeframes to load/stream (e.g., ["H1", "H4"]).
            counts (List[int]): List of historical data counts corresponding to timeframes.
                                e.g., [1250, 500] for ["H1", "H4"].
            api_key (str): API key for live data access.
            app_id (int): Application ID for live data access. Defaults to 1089.
            base_path (str): Base path for CSV files (required for 'csv' source_type).
            timeout (int): API request timeout in seconds.
            retries (int): Number of retries for API requests.
            initial_fetch_timeout (int): Max time to wait for initial live data in seconds.
            live_polling_interval (int): How often to poll for new live data in seconds.
        """
        if source_type not in ["csv", "live"]:
            raise ValueError("source_type must be 'csv' or 'live'")

        if len(timeframes) != len(counts):
            raise ValueError("Lengths of 'timeframes' and 'counts' must be equal.")

        self.source_type = source_type
        self.symbol = symbol
        self.timeframes = timeframes
        self.counts = dict(zip(timeframes, counts))  # Map timeframe to its count
        self.api_key = api_key
        self.app_id = app_id
        self.timeout = timeout
        self.retries = retries
        self.initial_fetch_timeout = initial_fetch_timeout
        self.live_polling_interval = live_polling_interval

        self.deriv_api: Optional[DerivAPI] = None  # Will be set in connect()
        self.is_connected: bool = False  # Track connection status

        self.data_store: Dict[str, pd.DataFrame] = (
            {}
        )  # Internal store for current data views
        self.last_fetched_time: Dict[str, datetime] = (
            {}
        )  # Track last data point fetched for each TF
        self.next_fetch_time: Dict[str, datetime] = (
            {}
        )  # Next allowed fetch time for live data

        if self.source_type == "csv":
            self.base_path = base_path
            if not base_path or not os.path.isdir(base_path):
                raise ValueError(
                    f"For 'csv' source_type, 'base_path' must be a valid directory. Got: {base_path}"
                )
            self.__load_initial_csv_data()  # Assuming this is a synchronous method

    async def connect(self):
        """
        Connects to the Deriv API and authorizes the client if source_type is 'live'.
        This method should be called asynchronously before fetching live data.
        """
        if self.source_type == "live" and not self.is_connected:
            logger.debug(f"Initializing Deriv API client with endpoint")

            try:
                self.deriv_api = DerivAPI(app_id=self.app_id, timeout=10)
                authorize_response = await self.deriv_api.authorize(self.api_key)

                if not authorize_response or authorize_response.get("error"):
                    error_message = authorize_response.get("error", {}).get(
                        "message", "Unknown authorization error"
                    )
                    raise RuntimeError(
                        f"Failed to authorize with Deriv API. Error: {error_message}"
                    )

                self.is_connected = True
                logger.info(
                    f"Deriv API client authorized for account: {authorize_response['authorize']['loginid']}"
                )

                # Initialize next_fetch_time to allow immediate fetch
                for tf in self.timeframes:
                    self.next_fetch_time[tf] = datetime.min  # Allow immediate fetch
            except Exception as e:
                logger.error(f"Failed to connect and authorize Deriv API: {e}")
                if self.deriv_api:
                    await self.deriv_api.disconnect()
                self.is_connected = False
                raise  # Re-raise the exception to indicate connection failure
        elif self.source_type == "live" and self.is_connected:
            logger.info("Deriv API client already connected.")
        elif self.source_type == "csv":
            logger.info("Data source is CSV, no API connection needed.")

    async def disconnect(self):
        """
        Disconnects the Deriv API client if it's connected.
        """
        if self.deriv_api and self.is_connected:
            await self.deriv_api.disconnect()
            self.is_connected = False
            logger.info("Disconnected from Deriv API.")

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

    async def stream_data(self) -> AsyncGenerator[Dict[str, pd.DataFrame], None]:
        """
        Yields the latest data for each timeframe.
        In CSV mode, it simulates a stream. In live mode, it queries the API.
        """
        if self.source_type == "csv":
            # Use async for to iterate over the async generator __stream_csv_data
            async for data in self.__stream_csv_data():
                yield data
        elif self.source_type == "live":
            # Use async for to iterate over the async generator __stream_live_data
            async for data in self.__stream_live_data():
                yield data

    async def __stream_csv_data(self) -> AsyncGenerator[Dict[str, pd.DataFrame], None]:
        """
        Streams historical data from loaded CSVs.
        Yields dataframes for each timeframe, with lengths equal to their respective counts,
        ensuring the end-timestamps are synchronized as much as possible.
        No data is yielded until all timeframes have at least their `count` number of records.
        """
        if not self.data_store:
            logger.error("No CSV data available to stream.")
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
            f"CSV Stream initialized. Master clock starting from: {current_master_time}. "
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
                    f"Initial CSV data yielded at {current_master_time} after warm-up."
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
                "CSV stream could not find a starting point where all timeframes met their 'count' requirements. Stream will not yield any data."
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
                logger.info(f"Yielded CSV data ending at {current_master_time}")
            else:
                logger.debug(
                    f"No new data to yield for any timeframe at {current_master_time}. Advancing master clock."
                )

            # Advance the master clock for the next iteration
            current_master_time += master_step_interval

        logger.info(
            "CSV data stream finished: Reached end of available historical data."
        )

    async def __stream_live_data(
        self,
    ) -> AsyncGenerator[Dict[str, pd.DataFrame], None]:
        """
        Streams live data by querying the API, optimized with a master clock
        and intelligent polling based on timeframe frequencies.
        Only yields when new data is received for any timeframe.
        Uses dynamic sleep timing aligned with the next timeframe interval.
        """
        logger.info("Starting live data stream with master clock...")

        # --- Initial Data Fetch Phase ---
        initial_fetch_successful = False
        fetch_start_time = time.time()

        while time.time() - fetch_start_time < self.initial_fetch_timeout:
            try:
                initial_data = await self.__fetch_data_from_api(
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
                        logger.info(
                            f"Initialized {tf} data ending at {self.last_fetched_time[tf]}"
                        )
                    initial_fetch_successful = True
                    logger.info("All initial live data fetches complete.")
                    break
                else:
                    logger.warning(
                        "Not all timeframes ready after initial fetch. Retrying..."
                    )
                    time.sleep(self.live_polling_interval)

            except Exception as e:
                logger.warning(
                    f"Initial live data fetch failed: {e}. Retrying in {self.live_polling_interval}s..."
                )
                time.sleep(self.live_polling_interval)

        if not initial_fetch_successful:
            raise RuntimeError(
                "Failed to fetch initial live data for all timeframes after multiple retries."
            )

        # Yield the initially fetched data snapshot
        yield self.data_store.copy()
        logger.info("Initial live data snapshot yielded.")

        # Determine the smallest interval for the master clock step
        if not self.timeframes:
            logger.warning("No timeframes specified for streaming.")
            return

        sorted_timeframes = sorted(
            self.timeframes, key=lambda tf: self.__get_timeframe_interval(tf)
        )
        master_step_interval = self.__get_timeframe_interval(sorted_timeframes[0])
        logger.info(f"Master clock will advance by {master_step_interval}.")

        # Align master clock to the next interval after the latest initial data
        latest_initial_data_end = max(self.last_fetched_time.values())
        current_master_time = latest_initial_data_end + master_step_interval

        logger.info(f"Master clock initialized to: {current_master_time}.")

        # --- Main Streaming Loop (Live Data) ---
        while True:
            current_system_time = datetime.now()
            updated_any = False

            for tf in sorted_timeframes:
                interval = self.__get_timeframe_interval(tf)
                required_count = self.counts[tf]

                # Only fetch if master clock has reached or passed the next fetch time for this TF
                if current_master_time >= self.next_fetch_time[tf]:
                    try:
                        new_data = await self.__fetch_data_from_api_for_timeframe(
                            self.symbol, tf, required_count
                        )

                        if not new_data.empty:
                            new_data_max_time = new_data.index.max()
                            last_fetched_dt = self.last_fetched_time.get(tf)

                            if (
                                last_fetched_dt is None
                                or new_data_max_time > last_fetched_dt
                            ):
                                self.data_store[tf] = new_data.iloc[-required_count:]
                                self.last_fetched_time[tf] = self.data_store[
                                    tf
                                ].index.max()
                                updated_any = True
                                logger.debug(
                                    f"New data for {tf}. Latest bar ends at: {new_data_max_time}"
                                )
                            else:
                                logger.debug(
                                    f"No newer data from API for {tf}. "
                                    f"Latest: {new_data_max_time}. Existing: {last_fetched_dt}"
                                )
                        else:
                            logger.debug(
                                f"API returned empty DataFrame for {tf}. No new data."
                            )
                    except Exception as e:
                        logger.error(f"Error fetching live data for {tf}: {e}")

                    # Update next_fetch_time for this TF
                    self.next_fetch_time[tf] = self.next_fetch_time[tf] + interval

            # Only yield if we received new data for any timeframe
            if updated_any:
                yield self.data_store.copy()
                logger.info(
                    f"Yielded live data snapshot at master time {current_master_time} "
                    f"(data updated)"
                )

            # Advance master clock by one interval
            current_master_time += master_step_interval

            # Calculate precise sleep timing to wake up 0.25s after next candlestick closes
            current_time = datetime.now()
            smallest_tf = sorted_timeframes[0]
            smallest_interval = self.__get_timeframe_interval(smallest_tf)

            # Round current time down to the nearest interval
            current_time_ts = current_time.timestamp()
            interval_seconds = smallest_interval.total_seconds()
            rounded_ts = (current_time_ts // interval_seconds) * interval_seconds

            # Calculate next interval start and add 5s delay for data availability
            next_interval_start = datetime.fromtimestamp(rounded_ts + interval_seconds)
            target_wake_time = next_interval_start + timedelta(seconds=5)

            # Calculate sleep duration to reach target wake time
            sleep_duration = (target_wake_time - current_time).total_seconds()
            sleep_duration = max(0.01, sleep_duration)  # Ensure positive sleep duration

            logger.debug(
                f"Next candle closes at {next_interval_start}, "
                f"targeting wake at {target_wake_time} "
                f"(sleeping {sleep_duration:.3f}s)"
            )
            await asyncio.sleep(sleep_duration)

    async def __fetch_data_from_api(
        self, symbol: str, timeframes: List[str], counts: Dict[str, int]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches initial batch of historical OHLC (candle) data for all specified timeframes
        from the Deriv API using the 'candles' call.

        Args:
            symbol (str): The trading symbol (e.g., 'R_100').
            timeframes (List[str]): A list of timeframe strings (e.g., ['1m', '5m', '1h']).
            counts (Dict[str, int]): A dictionary mapping timeframe to the number of candles to fetch.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are timeframes and values are pandas DataFrames
                                     containing 'Open', 'High', 'Low', 'Close' data.
                                     'Volume' is not provided by Deriv's candles API and will be omitted.
        """
        # logger.info(f"API: Fetching initial OHLC data for {symbol} on {timeframes}...")

        # Ensure any previous subscriptions are cleared
        await self.deriv_api.send({"forget_all": "candles"})
        self.deriv_api

        all_timeframe_data = {}
        for tf in timeframes:
            num_records = counts.get(tf, 100)  # Default to 100 if not specified
            granularity = self.__get_timeframe_interval(tf).seconds
            request = {
                "ticks_history": symbol,
                "count": num_records,
                "granularity": granularity,
                "end": "latest",
                "style": "candles",
            }

            try:
                response = await asyncio.wait_for(
                    self.deriv_api.send(request), timeout=10
                )
                if (
                    response
                    and response.get("msg_type") == "candles"
                    and response.get("candles")
                ):
                    ohlc_data = response["candles"]
                    # Create DataFrame
                    df = pd.DataFrame(ohlc_data)
                    df["Date"] = pd.to_datetime(df["epoch"], unit="s")
                    df = df.set_index("Date")
                    df = df[["open", "high", "low", "close"]].astype(float)
                    df.columns = [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                    ]  # Rename columns for consistency
                    df.index.name = "Date"

                    # self.validate_data(df, tf) # Assuming validate_data exists and handles OHLC structure
                    all_timeframe_data[tf] = df
                    # logger.debug(
                    #     f"API: Fetched {len(df)} OHLC records for {tf} ending at {df.index.max()}"
                    # )
                elif response and response.get("error"):
                    logger.error(
                        f"Error fetching candles for {symbol}/{tf}: {response['error']['message']}"
                    )
                else:
                    logger.warning(
                        f"No OHLC data found for {symbol}/{tf} or unexpected response."
                    )
            except Exception as e:
                logger.error(f"An error occurred while fetching {symbol}/{tf}: {e}")
        return all_timeframe_data

    async def __fetch_data_from_api_for_timeframe(
        self, symbol: str, timeframe: str, count: int
    ) -> pd.DataFrame:
        """
        Fetches the latest 'count' OHLC (candle) data for a specific timeframe
        from the Deriv API using the 'candles' call.

        Args:
            symbol (str): The trading symbol (e.g., 'R_100').
            timeframe (str): The timeframe string (e.g., '1m', '1h').
            count (int): The number of candles to fetch.

        Returns:
            pd.DataFrame: A pandas DataFrame containing 'Open', 'High', 'Low', 'Close' data.
                          'Volume' is not provided by Deriv's candles API and will be omitted.
        """
        # logger.debug(
        #     f"API: Fetching {count} latest OHLC candles for {symbol}/{timeframe}..."
        # )

        granularity = self.__get_timeframe_interval(timeframe).seconds

        request = {
            "ticks_history": symbol,
            "count": count,
            "granularity": granularity,
            "end": "latest",
            "style": "candles",
        }

        try:
            response = await asyncio.wait_for(self.deriv_api.send(request), timeout=10)
            if (
                response
                and response.get("msg_type") == "candles"
                and response.get("candles")
            ):
                ohlc_data = response["candles"]
                df = pd.DataFrame(ohlc_data)
                df["Date"] = pd.to_datetime(df["epoch"], unit="s")
                df = df.set_index("Date")
                df = df[["open", "high", "low", "close"]].astype(float)
                df.columns = ["Open", "High", "Low", "Close"]
                df.index.name = "Date"

                # self.validate_data(df, timeframe) # Assuming validate_data exists and handles OHLC structure
                # logger.debug(
                #     f"API: Fetched {len(df)} OHLC records for {timeframe} ending at {df.index.max()}"
                # )
                return df
            elif response and response.get("error"):
                logger.error(
                    f"Error fetching candles for {symbol}/{timeframe}: {response['error']['message']}"
                )
                return pd.DataFrame()  # Return empty DataFrame on error
            else:
                logger.warning(
                    f"No OHLC data found for {symbol}/{timeframe} or unexpected response."
                )
                return (
                    pd.DataFrame()
                )  # Return empty DataFrame if no data or unexpected response
        except Exception as e:
            logger.error(f"An error occurred while fetching {symbol}/{timeframe}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on exception
