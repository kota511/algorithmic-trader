from backtesting.backtester import backtest_strategy
from strategies.equal_weighted_portfolio import equal_weighted_portfolio
from strategies.quant_momentum_strategy import quantitative_momentum
from strategies.quant_value_strategy import quantitative_value

if __name__ == "__main__":
    symbols_file = 'data/sp500_symbols.csv'
    portfolio_size = input("Enter the value of your portfolio: ")

    print("Running Equal-Weighted Portfolio Strategy...")
    eq_portfolio = backtest_strategy(equal_weighted_portfolio, symbols_file, portfolio_size)

    print("Running Quantitative Momentum Strategy...")
    momentum_portfolio = backtest_strategy(quantitative_momentum, symbols_file, portfolio_size)

    print("Running Quantitative Value Strategy...")
    value_portfolio = backtest_strategy(quantitative_value, symbols_file, portfolio_size)