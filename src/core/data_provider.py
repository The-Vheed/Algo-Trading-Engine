from src.utils.logger import Logger

logger = Logger(__name__)

import os
import random
import time
import pandas as pd
from pandas import DataFrame
from typing import List, Dict, Union, Generator, Tuple
from pandas import DataFrame, Series, concat
from datetime import datetime, timedelta


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
        app_id: str = "",
        base_path: str = "",
        base_url: str = "https://api.deriv.com",
        timeout: int = 30,
        retries: int = 3,
        initial_fetch_timeout: int = 60,  # New: Max time to wait for initial live data
        live_polling_interval: int = 5,  # New: How often to poll for new live data (seconds)
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
            app_id (str): Application ID for live data access.
            base_path (str): Base path for CSV files (required for 'csv' source_type).
            base_url (str): Base URL for the live API.
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
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.initial_fetch_timeout = initial_fetch_timeout
        self.live_polling_interval = live_polling_interval

        self.data_store: Dict[str, DataFrame] = (
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
            self.__load_initial_csv_data()
        elif self.source_type == "live":
            # For live, initial data will be fetched by the stream_data method
            logger.info(
                "Live data provider initialized. Data will be fetched on stream_data() call."
            )
            # Initialize next_fetch_time to allow immediate fetch
            for tf in self.timeframes:
                self.next_fetch_time[tf] = datetime.min  # Allow immediate fetch

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

    def stream_data(self) -> Generator[Dict[str, pd.DataFrame], None, None]:
        """
        Yields the latest data for each timeframe.
        In CSV mode, it simulates a stream. In live mode, it queries the API.
        """
        if self.source_type == "csv":
            yield from self.__stream_csv_data()
        elif self.source_type == "live":
            yield from self.__stream_live_data()

    def __stream_csv_data(self) -> Generator[Dict[str, pd.DataFrame], None, None]:
        """
        Streams historical data from loaded CSVs.
        Yields dataframes for each timeframe, with lengths equal to their respective counts.
        """
        if not self.data_store:
            logger.error("No CSV data available to stream.")
            return

        # Ensure timeframes are sorted from lowest to highest for consistent streaming
        # Assuming H1 < H4 < D1 etc. based on integer prefixes or standard durations
        sorted_timeframes = sorted(
            self.timeframes, key=lambda tf: self.__get_timeframe_interval(tf)
        )

        # Initialize pointers for each timeframe
        # Store index positions for each timeframe to track what has been yielded
        timeframe_indices = {tf: 0 for tf in sorted_timeframes}

        # Find the earliest start time among all loaded dataframes
        earliest_start_time = min(df.index.min() for df in self.data_store.values())
        current_global_time = earliest_start_time

        # To ensure we yield based on the lowest common time,
        # we advance each timeframe's pointer until its data is at or past current_global_time
        for tf in sorted_timeframes:
            df = self.data_store[tf]
            idx_pos = df.index.searchsorted(current_global_time, side="left")
            timeframe_indices[tf] = idx_pos
            if idx_pos >= len(df):
                logger.warning(
                    f"Timeframe {tf} has no data at or after global start time {current_global_time}. It might be skipped."
                )

        while True:
            yielded_data_this_round = {}
            all_timeframes_exhausted = True

            # Find the next global timestamp to process based on the smallest next timestamp
            # available across all timeframes that still have data.
            next_available_times = []
            for tf in sorted_timeframes:
                df = self.data_store[tf]
                current_idx = timeframe_indices[tf]
                if current_idx < len(df):
                    next_available_times.append(df.index[current_idx])

            if not next_available_times:
                logger.info("All CSV data exhausted across all timeframes.")
                break  # All data has been streamed

            current_global_time = min(next_available_times)
            logger.debug(f"CSV Stream Clock: {current_global_time}")

            # For each timeframe, get the data ending at current_global_time and of 'count' length
            for tf in sorted_timeframes:
                df = self.data_store[tf]
                current_idx = timeframe_indices[tf]

                # Check if this timeframe has data at or past the current_global_time
                # and if we haven't exhausted this timeframe's data
                if (
                    current_idx < len(df)
                    and df.index[current_idx] <= current_global_time
                ):
                    end_idx = df.index.searchsorted(current_global_time, side="right")
                    if end_idx > 0:
                        start_idx = max(0, end_idx - self.counts[tf])
                        # Slice the DataFrame
                        chunk = df.iloc[start_idx:end_idx].copy()

                        if not chunk.empty:
                            yielded_data_this_round[tf] = chunk
                            logger.debug(
                                f"Prepared {len(chunk)} records for {tf} ending at {chunk.index.max()}"
                            )
                            all_timeframes_exhausted = False
                        else:
                            logger.debug(
                                f"No new data for {tf} ending at {current_global_time}. Current index: {current_idx}"
                            )
                    else:
                        logger.debug(
                            f"No data points <= {current_global_time} for {tf}."
                        )

                    # Advance the pointer for this timeframe
                    # We advance by one step to move to the next candle in this timeframe
                    timeframe_indices[tf] = end_idx

            if yielded_data_this_round:
                yield yielded_data_this_round
                logger.info(
                    f"Yielded data for {current_global_time}. Next fetch for CSV at {current_global_time + timedelta(seconds=1)}"
                )  # Increment clock slightly
            else:
                logger.debug(
                    f"No new data to yield for any timeframe at {current_global_time}. Advancing stream time."
                )
                # If no data was yielded, we still need to advance the overall stream time
                # This could happen if some timeframes are exhausted or have gaps.
                # The next iteration will find the next available time.

            if all_timeframes_exhausted:
                logger.info("All timeframes exhausted in CSV stream.")
                break

    def __stream_live_data(self) -> Generator[Dict[str, pd.DataFrame], None, None]:
        """
        Streams live data by querying the API.
        Yields dataframes for each timeframe, with lengths equal to their respective counts.
        It intelligently polls for new data based on timeframe frequency.
        """
        logger.info("Starting live data stream...")

        # Perform initial fetch for all timeframes
        initial_fetch_successful = False
        start_time = time.time()
        while time.time() - start_time < self.initial_fetch_timeout:
            try:
                initial_data = self.__fetch_data_from_api(
                    self.symbol, self.timeframes, self.counts
                )
                for tf, df in initial_data.items():
                    if not df.empty:
                        self.data_store[tf] = df
                        self.last_fetched_time[tf] = df.index.max()
                        self.next_fetch_time[
                            tf
                        ] = datetime.now() + self.__get_timeframe_interval(
                            tf
                        )  # Set next fetch based on TF
                    else:
                        logger.warning(
                            f"Initial fetch for {tf} returned empty DataFrame."
                        )
                initial_fetch_successful = True
                logger.info("Initial live data fetch complete.")
                break
            except Exception as e:
                logger.warning(
                    f"Initial live data fetch failed: {e}. Retrying in {self.live_polling_interval}s..."
                )
                time.sleep(self.live_polling_interval)

        if not initial_fetch_successful:
            raise RuntimeError(
                "Failed to fetch initial live data after multiple retries."
            )

        yield self.data_store.copy()  # Yield initial data

        while True:
            yielded_data_this_round = {}
            current_time = datetime.now()

            # Iterate through timeframes, prioritizing lower ones
            sorted_timeframes = sorted(
                self.timeframes, key=lambda tf: self.__get_timeframe_interval(tf)
            )

            for tf in sorted_timeframes:
                interval = self.__get_timeframe_interval(tf)

                # Check if it's time to fetch data for this timeframe
                if current_time >= self.next_fetch_time[tf]:
                    logger.debug(
                        f"Attempting to fetch new data for {tf} at {current_time}. Last fetched: {self.last_fetched_time.get(tf)}"
                    )
                    try:
                        # Fetch the latest data for this specific timeframe
                        # The API call should ideally support getting 'N' latest candles
                        new_data = self.__fetch_data_from_api_for_timeframe(
                            self.symbol, tf, self.counts[tf]
                        )

                        if not new_data.empty:
                            last_fetched_dt = self.last_fetched_time.get(tf)
                            if (
                                last_fetched_dt is None
                                or new_data.index.max() > last_fetched_dt
                            ):
                                # New data found: append or update
                                logger.info(
                                    f"New data found for {tf}. Last candle: {new_data.index.max()}"
                                )

                                # Only append if we have existing data and indices align
                                if (
                                    tf in self.data_store
                                    and not self.data_store[tf].empty
                                    and new_data.index.min()
                                    > self.data_store[tf].index.max()
                                ):
                                    self.data_store[tf] = pd.concat(
                                        [self.data_store[tf], new_data],
                                        ignore_index=False,
                                    )
                                    # Trim to maintain 'count' length if needed, or manage rolling window
                                    self.data_store[tf] = self.data_store[tf].iloc[
                                        -self.counts[tf] :
                                    ]
                                elif (
                                    tf in self.data_store
                                    and not self.data_store[tf].empty
                                    and new_data.index.min()
                                    <= self.data_store[tf].index.max()
                                ):
                                    # If new data overlaps or is older, we might need to merge or simply replace
                                    # For simplicity, if overlap, we'll assume the API returns the most up-to-date 'count'
                                    # For a robust solution, you'd need a more sophisticated merge/update logic here
                                    # based on how your API provides historical data.
                                    logger.warning(
                                        f"Fetched data for {tf} overlaps with existing data. Replacing with new fetch for simplicity."
                                    )
                                    self.data_store[tf] = new_data
                                else:
                                    self.data_store[tf] = (
                                        new_data  # First data or full replacement
                                    )

                                self.validate_data(self.data_store[tf], tf)
                                self.last_fetched_time[tf] = self.data_store[
                                    tf
                                ].index.max()
                                yielded_data_this_round[tf] = self.data_store[
                                    tf
                                ].copy()  # Yield the current view

                            else:
                                logger.debug(
                                    f"No newer data for {tf}. Last candle: {new_data.index.max()}. Existing max: {last_fetched_dt}"
                                )
                        else:
                            logger.debug(
                                f"API returned empty DataFrame for {tf}. No new data."
                            )

                    except Exception as e:
                        logger.error(f"Error fetching live data for {tf}: {e}")

                    # Always update next fetch time after attempting to fetch,
                    # whether new data was found or not, to prevent continuous hammering.
                    self.next_fetch_time[tf] = current_time + interval

            if yielded_data_this_round:
                logger.info(
                    f"Yielded live data for {len(yielded_data_this_round)} timeframes."
                )
                yield self.data_store.copy()  # Yield a copy of the entire current data_store
            else:
                logger.debug("No new live data to yield this round.")

            # Sleep for the shortest interval or a polling interval to prevent busy-waiting
            # Sleep until the earliest next fetch time among all timeframes, or minimum polling interval
            sleep_duration = self.live_polling_interval
            if self.next_fetch_time:
                earliest_next_fetch = min(self.next_fetch_time.values())
                time_until_next_fetch = (
                    earliest_next_fetch - datetime.now()
                ).total_seconds()
                if time_until_next_fetch > 0:
                    sleep_duration = min(sleep_duration, time_until_next_fetch)

            if sleep_duration > 0:
                logger.debug(f"Sleeping for {sleep_duration:.2f} seconds...")
                time.sleep(sleep_duration)

    # --- Placeholder for actual API integration ---
    def __fetch_data_from_api(
        self, symbol: str, timeframes: List[str], counts: Dict[str, int]
    ) -> Dict[str, pd.DataFrame]:
        """
        Placeholder: Fetches initial batch of live data for all specified timeframes.
        This would typically involve multiple API calls, one per timeframe.
        """
        logger.info(f"API: Fetching initial data for {symbol} on {timeframes}...")
        # Simulate API call delay
        # time.sleep(1)

        # This is a mock API response
        mock_data = {}
        for tf in timeframes:
            # Generate dummy data for the count requested
            num_records = counts.get(tf, 100)

            # Adjust interval for dummy data generation
            interval = self.__get_timeframe_interval(tf)

            end_date = datetime.now().replace(microsecond=0)  # Round to nearest second
            start_date = end_date - (num_records * interval)

            dates = pd.date_range(start=start_date, end=end_date, freq=interval)

            # Ensure we get exactly num_records or slightly more/less depending on freq
            if len(dates) < num_records:
                # If the freq doesn't give exact count, just take the last 'num_records'
                # or create more dates
                dates = pd.date_range(end=end_date, periods=num_records, freq=interval)
            elif len(dates) > num_records:
                dates = dates[-num_records:]

            data = {
                "Open": [random.uniform(1.0, 1.1) for _ in range(len(dates))],
                "High": [random.uniform(1.1, 1.2) for _ in range(len(dates))],
                "Low": [random.uniform(0.9, 1.0) for _ in range(len(dates))],
                "Close": [random.uniform(1.0, 1.1) for _ in range(len(dates))],
                "Volume": [random.randint(100, 1000) for _ in range(len(dates))],
            }
            df = pd.DataFrame(data, index=dates)
            df.index.name = "Date"
            self.validate_data(df, tf)
            mock_data[tf] = df
            logger.debug(
                f"API: Generated {len(df)} records for {tf} ending at {df.index.max()}"
            )
        return mock_data

    def __fetch_data_from_api_for_timeframe(
        self, symbol: str, timeframe: str, count: int
    ) -> pd.DataFrame:
        """
        Placeholder: Fetches the latest 'count' candles for a specific timeframe from the API.
        This would be the actual API call for updates.
        """
        logger.debug(
            f"API: Fetching {count} latest candles for {symbol}/{timeframe}..."
        )
        # Simulate API call delay
        # time.sleep(0.5)

        # Generate dummy data for the count requested
        interval = self.__get_timeframe_interval(timeframe)
        end_date = datetime.now().replace(microsecond=0)
        start_date = end_date - (count * interval)

        dates = pd.date_range(start=start_date, end=end_date, freq=interval)
        if len(dates) < count:
            dates = pd.date_range(end=end_date, periods=count, freq=interval)
        elif len(dates) > count:
            dates = dates[-count:]

        data = {
            "Open": [random.uniform(1.0, 1.1) for _ in range(len(dates))],
            "High": [random.uniform(1.1, 1.2) for _ in range(len(dates))],
            "Low": [random.uniform(0.9, 1.0) for _ in range(len(dates))],
            "Close": [random.uniform(1.0, 1.1) for _ in range(len(dates))],
            "Volume": [random.randint(100, 1000) for _ in range(len(dates))],
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = "Date"
        self.validate_data(df, timeframe)
        logger.debug(
            f"API: Generated {len(df)} records for {timeframe} ending at {df.index.max()}"
        )
        return df


# class DataStore:
#     """In-memory storage for loaded data."""

#     def __init__(self):
#         self.data = {}

#     def add(self, timeframe: str, new_data: Union[DataFrame, Series]):
#         """
#         Appends new data (either a DataFrame or a Series) to the DataFrame for the given timeframe.
#         If no DataFrame exists for the timeframe, it initializes it with the new data.
#         """
#         if timeframe not in self.data:
#             # If no DataFrame exists, initialize it.
#             # If new_data is a Series, convert it to a DataFrame (single row)
#             self.data[timeframe] = (
#                 new_data.to_frame().T if isinstance(new_data, Series) else new_data
#             )
#         else:
#             # If a DataFrame already exists
#             if isinstance(new_data, Series):
#                 # If new_data is a Series, convert it to a DataFrame (single row)
#                 # Ensure the Series has a name for its index if it's meant to be a new row with a meaningful index.
#                 # If the Series needs to align by column names, ensure the Series index matches the DataFrame's columns.
#                 new_data_df = new_data.to_frame().T
#                 self.data[timeframe] = concat(
#                     [self.data[timeframe], new_data_df], ignore_index=True
#                 )
#             elif isinstance(new_data, DataFrame):
#                 # If new_data is a DataFrame, concatenate it
#                 self.data[timeframe] = concat(
#                     [self.data[timeframe], new_data], ignore_index=True
#                 )
#             else:
#                 raise TypeError("new_data must be a pandas DataFrame or Series")

#     def get(self, timeframe: str) -> DataFrame:
#         """Retrieves the DataFrame for the given timeframe."""
#         return self.data.get(timeframe)

#     def summary(self):
#         """Prints a summary (head, tail, describe) for each stored DataFrame."""
#         for tf, df in self.data.items():
#             logger.debug(f"\nTimeframe: {tf}")
#             if not df.empty:
#                 logger.debug(df.head(3))
#                 logger.debug(df.tail(3))
#                 logger.debug(df.describe())
#             else:
#                 logger.debug("DataFrame is empty.")
