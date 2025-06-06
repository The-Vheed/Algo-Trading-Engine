class IndicatorEngine:
    def __init__(self):
        self.indicators = {}

    def register_indicator(self, name: str, calculator):
        self.indicators[name] = calculator

    def calculate_indicator(self, name: str, data, **params):
        if name in self.indicators:
            return self.indicators[name].calculate(data, **params)
        else:
            raise ValueError(f"Indicator '{name}' not registered.")

    def get_available_indicators(self):
        return list(self.indicators.keys())

    def calculate_all_required(self, data, config):
        results = {}
        for indicator_name in config.get('indicators', []):
            results[indicator_name] = self.calculate_indicator(indicator_name, data, **config[indicator_name])
        return results