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
                logger.info(
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
            at_least_one_tf_updated = False

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
                    at_least_one_tf_updated = True
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

    def __stream_live_data(self) -> Generator[Dict[str, pd.DataFrame], None, None]:
        """
        Streams live data by querying the API, optimized with a master clock
        and intelligent polling based on timeframe frequencies.
        Yields dataframes for each timeframe, with lengths equal to their respective counts,
        ensuring the end-timestamps are synchronized as much as possible, driven by
        the master clock.
        """
        logger.info("Starting live data stream with master clock...")

        # --- Initial Data Fetch Phase ---
        initial_fetch_successful = False
        fetch_start_time = time.time()

        while time.time() - fetch_start_time < self.initial_fetch_timeout:
            try:
                initial_data = self.__fetch_data_from_api(
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
                        # Set next fetch time based on when the bar *should* close
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

        # Initialize master clock to just after the latest fetched data point's "close time"
        # or the next expected bar close for the fastest timeframe, aligned to its interval.
        # We round current time down to the nearest master_step_interval boundary, then add one interval.
        current_system_time_aligned = datetime.now().replace(microsecond=0)
        if master_step_interval == timedelta(minutes=1):
            current_system_time_aligned = current_system_time_aligned.replace(second=0)
        elif master_step_interval == timedelta(minutes=5):
            current_system_time_aligned = current_system_time_aligned.replace(second=0)
            current_system_time_aligned -= timedelta(
                minutes=current_system_time_aligned.minute % 5
            )
        else:  # master_step_interval == timedelta(hours=1):
            current_system_time_aligned = current_system_time_aligned.replace(
                minute=0, second=0
            )

        # Ensure master_clock starts at the next expected tick *after* all initial data.
        latest_initial_data_end = max(self.last_fetched_time.values())
        current_master_time = (
            max(current_system_time_aligned, latest_initial_data_end)
            + master_step_interval
        )

        logger.info(f"Master clock initialized to: {current_master_time}.")

        # --- Main Streaming Loop (Live Data) ---
        while True:
            current_system_time = datetime.now()  # Actual system time for polling logic

            # Advance master clock if the real system time has passed the current master clock's expected tick
            if current_system_time >= current_master_time:
                # Move master clock to the next full interval
                current_master_time = current_master_time + master_step_interval
                logger.debug(f"Master clock advanced to {current_master_time}.")

            yielded_data_this_round = {}

            for tf in sorted_timeframes:
                interval = self.__get_timeframe_interval(tf)
                required_count = self.counts[tf]

                # Only attempt to fetch if the current master clock has reached or passed
                # the next expected fetch time for this specific timeframe.
                if current_master_time >= self.next_fetch_time[tf]:
                    logger.debug(
                        f"Fetching for {tf} as master clock {current_master_time} "
                        f"is >= next_fetch_time {self.next_fetch_time[tf]}."
                    )
                    try:
                        # Fetch the latest 'count' bars for this timeframe.
                        # API should ideally provide *closed* bars.
                        new_data = self.__fetch_data_from_api_for_timeframe(
                            self.symbol, tf, required_count
                        )

                        if not new_data.empty:
                            new_data_max_time = new_data.index.max()
                            last_fetched_dt = self.last_fetched_time.get(tf)

                            # If new data is genuinely newer or it's the first data
                            if (
                                last_fetched_dt is None
                                or new_data_max_time > last_fetched_dt
                            ):
                                logger.info(
                                    f"New data for {tf}. Latest bar ends at: {new_data_max_time}"
                                )

                                # Concatenate new data, handle potential overlaps, and trim to count
                                if (
                                    tf in self.data_store
                                    and not self.data_store[tf].empty
                                ):
                                    combined_df = pd.concat(
                                        [self.data_store[tf], new_data]
                                    )
                                    # Drop duplicates based on index (timestamps), keeping the latest
                                    combined_df = combined_df[
                                        ~combined_df.index.duplicated(keep="last")
                                    ]
                                    self.data_store[tf] = combined_df.iloc[
                                        -required_count:
                                    ]
                                else:
                                    # No existing data or empty, just store the new data (trimmed)
                                    self.data_store[tf] = new_data.iloc[
                                        -required_count:
                                    ]

                                self.validate_data(self.data_store[tf], tf)
                                self.last_fetched_time[tf] = self.data_store[
                                    tf
                                ].index.max()
                                yielded_data_this_round[tf] = self.data_store[
                                    tf
                                ].copy()  # Mark for yielding

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

                    # IMPORTANT: Always update next_fetch_time for this TF *after* the attempt,
                    # to prevent immediate re-fetching. It's set for the next time its interval passes
                    # the current master clock.
                    self.next_fetch_time[tf] = current_master_time + interval
                else:
                    logger.debug(
                        f"Not yet time to fetch for {tf}. Master clock {current_master_time} "
                        f"is < next_fetch_time {self.next_fetch_time[tf]}."
                    )

            if yielded_data_this_round:
                logger.info(
                    f"Yielded live data snapshot (containing updates for {len(yielded_data_this_round)} TFs) "
                    f"at master time {current_master_time}."
                )
                yield self.data_store.copy()  # Yield a copy of the entire current data_store
            else:
                logger.debug(
                    f"No new data ready to yield at master time {current_master_time}."
                )

            # Calculate optimal sleep duration until the next event (master clock tick or TF fetch time)
            next_event_times = list(self.next_fetch_time.values()) + [
                current_master_time + master_step_interval
            ]
            time_until_next_event_s = (
                min(next_event_times) - datetime.now()
            ).total_seconds()

            # Ensure sleep duration is non-negative and respects minimum polling interval
            sleep_duration = max(self.live_polling_interval, time_until_next_event_s)
            sleep_duration = max(
                0.01, sleep_duration
            )  # Minimum sleep to prevent busy-waiting

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
