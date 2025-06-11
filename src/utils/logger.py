import logging, os


class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        stream_handler = logging.StreamHandler()
        os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
        file_handler = logging.FileHandler(f"logs/{name}.log")

        self.logger.setLevel(logging.DEBUG)  # Set the default logging level
        file_handler.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

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
