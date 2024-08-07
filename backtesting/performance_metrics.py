import numpy as np

def calculate_performance_metrics(cumulative_returns):
    risk_free_rate = 0.01
    daily_returns = cumulative_returns.pct_change().dropna()

    # Sharpe Ratio
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    # Max Drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    metrics = {
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

    return metrics
