import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statistics import mean
from scipy import stats
import math

def quantitative_momentum_sim(symbols_file, training_period, testing_period, portfolio_size, stop_loss, take_profit):
    symbols = pd.read_csv(symbols_file, skiprows=1, header=None)[0].tolist()

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    symbol_groups = list(chunks(symbols, 100))
    symbol_strings = [' '.join(group) for group in symbol_groups]

    hqm_columns = [
        'Ticker',
        'Price',
        'Number of Shares to Buy',
        'One-Year Price Return',
        'One-Year Return Percentile',
        'Six-Month Price Return',
        'Six-Month Return Percentile',
        'Three-Month Price Return',
        'Three-Month Return Percentile',
        'One-Month Price Return',
        'One-Month Return Percentile',
        'HQM Score'
    ]

    hqm_dataframe = pd.DataFrame(columns=hqm_columns)

    for symbol_string in symbol_strings:
        try:
            data = yf.download(symbol_string.split(), start=training_period[0], end=training_period[1])['Adj Close']
            temp_dataframe = pd.DataFrame(columns=hqm_columns)
            for symbol in data.columns:
                if data[symbol].dropna().empty:
                    print(f"No data found for {symbol}. Skipping...")
                    continue
                one_year_return = (data[symbol].iloc[-1] / data[symbol].iloc[0]) - 1
                six_month_return = (data[symbol].iloc[-1] / data[symbol].iloc[-126]) - 1
                three_month_return = (data[symbol].iloc[-1] / data[symbol].iloc[-63]) - 1
                one_month_return = (data[symbol].iloc[-1] / data[symbol].iloc[-21]) - 1
                temp_dataframe = pd.concat([temp_dataframe, pd.DataFrame([{
                    'Ticker': symbol,
                    'Price': data[symbol].iloc[-1],
                    'Number of Shares to Buy': 0,
                    'One-Year Price Return': one_year_return,
                    'One-Year Return Percentile': 0,
                    'Six-Month Price Return': six_month_return,
                    'Six-Month Return Percentile': 0,
                    'Three-Month Price Return': three_month_return,
                    'Three-Month Return Percentile': 0,
                    'One-Month Price Return': one_month_return,
                    'One-Month Return Percentile': 0,
                    'HQM Score': 0
                }])], ignore_index=True)
            if not temp_dataframe.empty:
                hqm_dataframe = pd.concat([hqm_dataframe, temp_dataframe], ignore_index=True)
        except Exception as e:
            print(f"Failed download for {symbol_string}: {e}")

    time_periods = [
        'One-Year',
        'Six-Month',
        'Three-Month',
        'One-Month'
    ]

    hqm_dataframe.dropna(subset=[f'{time_period} Price Return' for time_period in time_periods], inplace=True)

    for row in hqm_dataframe.index:
        for time_period in time_periods:
            hqm_dataframe.loc[row, f'{time_period} Return Percentile'] = stats.percentileofscore(
                hqm_dataframe[f'{time_period} Price Return'].dropna(),
                hqm_dataframe.loc[row, f'{time_period} Price Return']
            )

    print(hqm_dataframe)

    for row in hqm_dataframe.index:
        momentum_percentiles = [hqm_dataframe.loc[row, f'{time_period} Return Percentile'] for time_period in time_periods]
        hqm_dataframe.loc[row, 'HQM Score'] = mean(momentum_percentiles)

    print(hqm_dataframe)

    hqm_dataframe.sort_values(by='HQM Score', ascending=False, inplace=True)
    hqm_dataframe = hqm_dataframe[:50]
    hqm_dataframe.reset_index(drop=True, inplace=True)

    print(hqm_dataframe)

    position_size = float(portfolio_size) / len(hqm_dataframe.index)
    for i in range(0, len(hqm_dataframe['Ticker'])):
        hqm_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / hqm_dataframe['Price'][i])

    tickers = hqm_dataframe['Ticker'].tolist()
    testing_data = yf.download(tickers, start=testing_period[0], end=testing_period[1])['Adj Close']

    initial_portfolio_value = portfolio_size
    current_portfolio_value = portfolio_size
    trades = []

    for ticker in tickers:
        try:
            buy_price = hqm_dataframe.loc[hqm_dataframe['Ticker'] == ticker, 'Price'].values[0]
            num_shares = hqm_dataframe.loc[hqm_dataframe['Ticker'] == ticker, 'Number of Shares to Buy'].values[0]
            stop_loss_price = buy_price * (1 - stop_loss)
            take_profit_price = buy_price * (1 + take_profit)
            for date, price in testing_data[ticker].items():
                if price <= stop_loss_price:
                    sell_price = stop_loss_price
                    trades.append({'Ticker': ticker, 'Buy Price': buy_price, 'Sell Price': sell_price, 'Shares': num_shares, 'Sell Date': date})
                    break
                elif price >= take_profit_price:
                    sell_price = take_profit_price
                    trades.append({'Ticker': ticker, 'Buy Price': buy_price, 'Sell Price': sell_price, 'Shares': num_shares, 'Sell Date': date})
                    break
            else:
                sell_price = testing_data[ticker].iloc[-1]
                trades.append({'Ticker': ticker, 'Buy Price': buy_price, 'Sell Price': sell_price, 'Shares': num_shares, 'Sell Date': testing_data[ticker].index[-1]})
        except KeyError:
            print(f"No data available for {ticker} during testing period. Skipping...")

    for trade in trades:
        current_portfolio_value += (trade['Sell Price'] - trade['Buy Price']) * trade['Shares']

    results = pd.DataFrame(trades)
    results['PnL'] = (results['Sell Price'] - results['Buy Price']) * results['Shares']

    print(f"Initial Portfolio Value: {initial_portfolio_value}")
    print(f"Final Portfolio Value: {current_portfolio_value}")
    print(f"Total PnL: {current_portfolio_value - initial_portfolio_value}")

    results.to_excel('trade_log_momentum_strategy.xlsx', index=False)
    print("Trading results saved to trade_log_momentum_strategy.xlsx")

    return results