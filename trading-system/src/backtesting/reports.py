class BacktestReport:
    def __init__(self, results):
        self.results = results

    def generate_summary(self):
        summary = {
            "net_profit": self.results["net_profit"],
            "total_trades": self.results["total_trades"],
            "win_rate": self.results["win_rate"],
            "max_drawdown": self.results["max_drawdown"],
            "profit_factor": self.results["profit_factor"],
        }
        return summary

    def export_to_csv(self, file_path):
        import pandas as pd

        df = pd.DataFrame(self.results["trades"])
        df.to_csv(file_path, index=False)

    def display_report(self):
        print("Backtest Report Summary:")
        for key, value in self.generate_summary().items():
            print(f"{key}: {value}")