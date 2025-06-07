class PerformanceMetrics:
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.net_profit = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0

    def update_metrics(self, trade_result: float):
        self.total_trades += 1
        if trade_result > 0:
            self.winning_trades += 1
            self.gross_profit += trade_result
        else:
            self.losing_trades += 1
            self.gross_loss += abs(trade_result)
        
        self.net_profit += trade_result

    def calculate_drawdown(self, peak: float, current: float):
        drawdown = (peak - current) / peak
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def calculate_sharpe_ratio(self, risk_free_rate: float, returns: list):
        excess_returns = [r - risk_free_rate for r in returns]
        self.sharpe_ratio = (sum(excess_returns) / len(excess_returns)) / (self.std_dev(returns) if self.std_dev(returns) != 0 else 1)

    def std_dev(self, returns: list):
        mean = sum(returns) / len(returns)
        return (sum((x - mean) ** 2 for x in returns) / len(returns)) ** 0.5

    def get_summary(self):
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "net_profit": self.net_profit,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
        }