class RegimeDetector:
    def detect_regime(self, data: DataFrame, config: dict) -> str:
        # Logic to determine the market regime based on the provided data and configuration
        pass

    def confirm_regime_change(self, current: str, previous: str, bars: int) -> bool:
        # Logic to confirm if a regime change has occurred based on the number of bars
        pass

    def get_regime_strength(self, data: DataFrame) -> float:
        # Logic to calculate the strength of the current regime based on the data
        pass