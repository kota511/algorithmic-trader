# def backtest_strategy(strategy_func, symbols_file, portfolio_size):
#     portfolio = strategy_func(symbols_file, int(portfolio_size))
#     return portfolio

def backtest_strategy(strategy_func, symbols_file, training_period, testing_period, portfolio_size, stop_loss, take_profit):
    portfolio = strategy_func(symbols_file, training_period, testing_period, int(portfolio_size), stop_loss, take_profit)
    return portfolio