class MovingAverage:
    def __init__(self, period: int):
        self.period = period
        self.prices = []

    def add_price(self, price: float):
        self.prices.append(price)
        if len(self.prices) > self.period:
            self.prices.pop(0)

    def calculate(self) -> float:
        if len(self.prices) < self.period:
            return float('nan')
        return sum(self.prices) / self.period


class ExponentialMovingAverage:
    def __init__(self, period: int):
        self.period = period
        self.multiplier = 2 / (period + 1)
        self.ema = None

    def add_price(self, price: float):
        if self.ema is None:
            self.ema = price
        else:
            self.ema = (price - self.ema) * self.multiplier + self.ema

    def calculate(self) -> float:
        return self.ema


class RelativeStrengthIndex:
    def __init__(self, period: int):
        self.period = period
        self.gains = []
        self.losses = []

    def add_price(self, current_price: float, previous_price: float):
        change = current_price - previous_price
        if change > 0:
            self.gains.append(change)
            self.losses.append(0)
        else:
            self.gains.append(0)
            self.losses.append(-change)

        if len(self.gains) > self.period:
            self.gains.pop(0)
            self.losses.pop(0)

    def calculate(self) -> float:
        if len(self.gains) < self.period or len(self.losses) < self.period:
            return float('nan')

        avg_gain = sum(self.gains) / self.period
        avg_loss = sum(self.losses) / self.period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


class BollingerBands:
    def __init__(self, period: int, num_std_dev: float):
        self.period = period
        self.num_std_dev = num_std_dev
        self.prices = []

    def add_price(self, price: float):
        self.prices.append(price)
        if len(self.prices) > self.period:
            self.prices.pop(0)

    def calculate(self):
        if len(self.prices) < self.period:
            return float('nan'), float('nan')

        mean = sum(self.prices) / self.period
        variance = sum((x - mean) ** 2 for x in self.prices) / self.period
        std_dev = variance ** 0.5

        upper_band = mean + (self.num_std_dev * std_dev)
        lower_band = mean - (self.num_std_dev * std_dev)

        return upper_band, lower_band