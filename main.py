# from backtesting.backtester import backtest_strategy
# from strategies.equal_weighted_portfolio import equal_weighted_portfolio
# from strategies.quant_momentum_strategy import quantitative_momentum
# from strategies.quant_value_strategy import quantitative_value

# if __name__ == "__main__":
#     symbols_file = 'data/sp500_symbols.csv'
#     portfolio_size = input("Enter the value of your portfolio: ")

#     print("Running Equal-Weighted Portfolio Strategy...")
#     eq_portfolio = backtest_strategy(equal_weighted_portfolio, symbols_file, portfolio_size)

#     print("Running Quantitative Momentum Strategy...")
#     momentum_portfolio = backtest_strategy(quantitative_momentum, symbols_file, portfolio_size)

#     print("Running Quantitative Value Strategy...")
#     value_portfolio = backtest_strategy(quantitative_value, symbols_file, portfolio_size)

from backtesting.backtester import backtest_strategy
from simulations.equal_weighted_sim import equal_weighted_portfolio
from simulations.quant_momentum_sim import quantitative_momentum
from simulations.quant_value_sim import quantitative_value

if __name__ == "__main__":
    symbols_file = 'data/sp500_symbols.csv'  # Adjust the path as necessary
    portfolio_size = 1000000  # Set your portfolio size as an integer

    training_period = ('2022-01-01', '2023-01-01')
    testing_period = ('2023-01-01', '2024-01-01')

    stop_loss = 0.1  # Set stop loss level
    take_profit = 0.2  # Set take profit level

    print("Running Equal-Weighted Portfolio Strategy...")
    eq_portfolio = backtest_strategy(equal_weighted_portfolio, symbols_file, training_period, testing_period, portfolio_size, stop_loss, take_profit)

    # print("Running Quantitative Momentum Strategy...")
    # momentum_portfolio = backtest_strategy(quantitative_momentum, symbols_file, training_period, testing_period, portfolio_size, stop_loss, take_profit)

    # print("Running Quantitative Value Strategy...")
    # value_portfolio = backtest_strategy(quantitative_value, symbols_file, training_period, testing_period, portfolio_size, stop_loss, take_profit)