import logging, os


class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(
            logging.DEBUG
        )  # Set logger level to DEBUG to capture all logs

        # Prevent propagation to root logger to avoid duplicate logs
        self.logger.propagate = False

        # Clear any existing handlers (important when logger is reused)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create handlers
        stream_handler = logging.StreamHandler()
        os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
        file_handler = logging.FileHandler(f"logs/{name}.log")

        # Set levels - INFO for console (no debug logs), DEBUG for file (all logs)
        stream_handler.setLevel(logging.INFO)  # Console shows INFO and above only
        file_handler.setLevel(logging.DEBUG)  # File shows DEBUG and above

        # Create formatters
        stream_formatter = logging.Formatter("%(message)s")  # Console: Only the message
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # File: Full details
        )

        # Apply formatters to handlers
        stream_handler.setFormatter(stream_formatter)
        file_handler.setFormatter(file_formatter)

        # Add handlers to logger
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)
