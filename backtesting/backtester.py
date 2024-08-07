def backtest_strategy(strategy_func, symbols_file, portfolio_size):
    portfolio = strategy_func(symbols_file, int(portfolio_size))
    return portfolio