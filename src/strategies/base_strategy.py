class BaseStrategy:
    def __init__(self, config: dict):
        self.config = config

    def initialize(self):
        pass

    def generate_signals(self, data):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def execute(self):
        raise NotImplementedError("This method should be overridden by subclasses.")