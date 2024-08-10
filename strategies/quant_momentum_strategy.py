import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statistics import mean
from scipy import stats
import math
import xlsxwriter

def quantitative_momentum(symbols_file, portfolio_size):
    symbols = pd.read_csv(symbols_file, skiprows=1, header=None)[0].tolist()

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    symbol_groups = list(chunks(symbols, 100))
    symbol_strings = [' '.join(group) for group in symbol_groups]

    # Define columns for the DataFrame
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
            data = yf.download(symbol_string.split(), period="1y")['Adj Close']
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

    print(hqm_dataframe)

    # Calculate the percentile scores for each time period
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

    # Select the 50 best momentum stocks
    hqm_dataframe.sort_values(by='HQM Score', ascending=False, inplace=True)
    hqm_dataframe = hqm_dataframe[:50]
    hqm_dataframe.reset_index(drop=True, inplace=True)
    print(hqm_dataframe)

    position_size = float(portfolio_size) / len(hqm_dataframe.index)
    for i in range(0, len(hqm_dataframe['Ticker'])):
        hqm_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / hqm_dataframe['Price'][i])

    hqm_dataframe = hqm_dataframe[hqm_dataframe['Number of Shares to Buy'] > 0]

    print(hqm_dataframe)

    plt.figure(figsize=(12, 8))
    plt.bar(hqm_dataframe['Ticker'], hqm_dataframe['HQM Score'], color='blue')
    plt.xlabel('Ticker')
    plt.ylabel('HQM Score')
    plt.title('HQM Score for Top 50 Momentum Stocks')
    plt.xticks(rotation=90)
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].bar(hqm_dataframe['Ticker'], hqm_dataframe['One-Year Price Return'], color='green')
    axs[0, 0].set_title('One-Year Price Return')
    axs[0, 0].tick_params(axis='x', rotation=90)

    axs[0, 1].bar(hqm_dataframe['Ticker'], hqm_dataframe['Six-Month Price Return'], color='red')
    axs[0, 1].set_title('Six-Month Price Return')
    axs[0, 1].tick_params(axis='x', rotation=90)

    axs[1, 0].bar(hqm_dataframe['Ticker'], hqm_dataframe['Three-Month Price Return'], color='purple')
    axs[1, 0].set_title('Three-Month Price Return')
    axs[1, 0].tick_params(axis='x', rotation=90)

    axs[1, 1].bar(hqm_dataframe['Ticker'], hqm_dataframe['One-Month Price Return'], color='orange')
    axs[1, 1].set_title('One-Month Price Return')
    axs[1, 1].tick_params(axis='x', rotation=90)

    for ax in axs.flat:
        ax.set(xlabel='Ticker', ylabel='Price Return')

    fig.tight_layout()
    plt.show()

    writer = pd.ExcelWriter('momentum_strategy.xlsx', engine='xlsxwriter')
    hqm_dataframe.to_excel(writer, sheet_name='Momentum Strategy', index=False)

    # Set formatting options
    string_template = writer.book.add_format(
    {'font_color': '#000000', 'bg_color': '#ffffff', 'border': 1}
    )

    dollar_template = writer.book.add_format(
        {'num_format': '$0.00', 'font_color': '#000000', 'bg_color': '#ffffff', 'border': 1}
    )

    integer_template = writer.book.add_format(
        {'num_format': '0', 'font_color': '#000000', 'bg_color': '#ffffff', 'border': 1}
    )

    percent_template = writer.book.add_format(
        {'num_format': '0.0%', 'font_color': '#000000', 'bg_color': '#ffffff', 'border': 1}
    )

    float_template = writer.book.add_format(
        {'num_format': '0.00', 'font_color': '#000000', 'bg_color': '#ffffff', 'border': 1}
    )

    column_formats = {
        'A': ['Ticker', string_template],
        'B': ['Price', dollar_template],
        'C': ['Number of Shares to Buy', integer_template],
        'D': ['One-Year Price Return', percent_template],
        'E': ['One-Year Return Percentile', percent_template],
        'F': ['Six-Month Price Return', percent_template],
        'G': ['Six-Month Return Percentile', percent_template],
        'H': ['Three-Month Price Return', percent_template],
        'I': ['Three-Month Return Percentile', percent_template],
        'J': ['One-Month Price Return', percent_template],
        'K': ['One-Month Return Percentile', percent_template],
        'L': ['HQM Score', float_template]
    }

    for column in column_formats.keys():
        writer.sheets['Momentum Strategy'].set_column(f'{column}:{column}', 20, column_formats[column][1])
        writer.sheets['Momentum Strategy'].write(f'{column}1', column_formats[column][0], string_template)

    writer.close()

    print("Trades saved to momentum_strategy.xlsx")

    return hqm_dataframe
# symbols_file = 'data/sp500_symbols.csv'
# portfolio_size = input("Enter the value of your portfolio: ")
# quantitative_momentum(symbols_file, int(portfolio_size))