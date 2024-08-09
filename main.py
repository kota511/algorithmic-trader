import inquirer
from backtesting.backtester import backtest_strategy
from strategies.equal_weighted_portfolio import equal_weighted_portfolio
from strategies.quant_momentum_strategy import quantitative_momentum
from strategies.quant_value_strategy import quantitative_value
from simulations.equal_weighted_sim import equal_weighted_portfolio_sim
from simulations.quant_momentum_sim import quantitative_momentum_sim
from simulations.quant_value_sim import quantitative_value_sim

def run_strategy(choice):
    symbols_file = 'data/sp500_symbols.csv'
    while True:
        try:
            portfolio_size = int(input("Enter the value of your portfolio: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    if choice == 'Equal-Weighted Portfolio Strategy':
        print("Running Equal-Weighted Portfolio Strategy...")
        backtest_strategy(equal_weighted_portfolio, symbols_file, portfolio_size)
    elif choice == 'Quantitative Momentum Strategy':
        print("Running Quantitative Momentum Strategy...")
        backtest_strategy(quantitative_momentum, symbols_file, portfolio_size)
    elif choice == 'Quantitative Value Strategy':
        print("Running Quantitative Value Strategy...")
        backtest_strategy(quantitative_value, symbols_file, portfolio_size)

def run_simulation(choice):
    symbols_file = 'data/sp500_symbols.csv'
    while True:
        try:
            portfolio_size = int(input("Enter the value of your portfolio: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    training_period = ('2022-01-01', '2023-01-01')
    testing_period = ('2023-01-01', '2024-01-01')

    stop_loss = 0.1
    take_profit = 0.2

    if choice == 'Equal-Weighted Portfolio Simulation':
        print("Running Equal-Weighted Portfolio Simulation...")
        backtest_strategy(equal_weighted_portfolio_sim, symbols_file, training_period, testing_period, portfolio_size, stop_loss, take_profit)
    elif choice == 'Quantitative Momentum Simulation':
        print("Running Quantitative Momentum Simulation...")
        backtest_strategy(quantitative_momentum_sim, symbols_file, training_period, testing_period, portfolio_size, stop_loss, take_profit)
    elif choice == 'Quantitative Value Simulation':
        print("Running Quantitative Value Simulation...")
        backtest_strategy(quantitative_value_sim, symbols_file, training_period, testing_period, portfolio_size, stop_loss, take_profit)

if __name__ == "__main__":
    main_menu = [
        inquirer.List(
            'menu',
            message="Select the type of execution",
            choices=['Run Strategy', 'Run Simulation', 'Exit'],
        ),
    ]
    selected_menu = inquirer.prompt(main_menu)

    if selected_menu['menu'] == 'Run Strategy':
        strategy_menu = [
            inquirer.List(
                'strategy',
                message="Select the strategy to run",
                choices=['Equal-Weighted Portfolio Strategy', 'Quantitative Momentum Strategy', 'Quantitative Value Strategy'],
            ),
        ]
        selected_strategy = inquirer.prompt(strategy_menu)
        run_strategy(selected_strategy['strategy'])

    elif selected_menu['menu'] == 'Run Simulation':
        simulation_menu = [
            inquirer.List(
                'simulation',
                message="Select the simulation to run",
                choices=['Equal-Weighted Portfolio Simulation', 'Quantitative Momentum Simulation', 'Quantitative Value Simulation'],
            ),
        ]
        selected_simulation = inquirer.prompt(simulation_menu)
        run_simulation(selected_simulation['simulation'])
    
    elif selected_menu['menu'] == 'Exit':
        print("Exiting...")