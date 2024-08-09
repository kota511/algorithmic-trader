def backtest_strategy(strategy_func, symbols_file, portfolio_size, training_period=None, testing_period=None, stop_loss=None, take_profit=None):
    if training_period and testing_period and stop_loss is not None and take_profit is not None:
        portfolio = strategy_func(symbols_file, training_period, testing_period, portfolio_size, stop_loss, take_profit)
    else:
        portfolio = strategy_func(symbols_file, portfolio_size)
    
    return portfolio