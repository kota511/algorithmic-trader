import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import xlsxwriter
import math

def equal_weighted_portfolio(symbols_file, portfolio_size):
    symbols = pd.read_csv(symbols_file, skiprows=1, header=None)[0].tolist()

    all_data = {}
    # Download data for all symbols, which have data available
    for symbol in symbols:
        try:
            data = yf.download(symbol, period="1y")['Adj Close']
            if not data.empty:
                all_data[symbol] = data
        except Exception as e:
            print(f"Failed download for {symbol}: {e}")

    all_data_df = pd.DataFrame(all_data)
    all_data_df = all_data_df.dropna(axis=1, how='all')

    num_stocks = all_data_df.shape[1]
    equal_weights = 1 / num_stocks

    daily_returns = all_data_df.pct_change().dropna()

    def get_stock_info(symbol):
        stock = yf.Ticker(symbol)
        info = stock.info
        market_cap = info.get('marketCap', 'N/A')
        price = info.get('currentPrice', 'N/A')
        return market_cap, price

    # Get stock info for all valid symbols
    data_list = []
    valid_symbols = list(all_data.keys())
    for symbol in valid_symbols:
        try:
            market_cap, price = get_stock_info(symbol)
            data_list.append({'Ticker': symbol, 'Price': price, 'Market Capitalization': market_cap, 'Number Of Shares to Buy': 'N/A'})
        except Exception as e:
            print(f"Failed to get info for {symbol}: {e}")

    final_dataframe = pd.DataFrame(data_list)
    print(final_dataframe)

    weighted_returns = daily_returns * equal_weights
    portfolio_returns = weighted_returns.sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    plt.figure(figsize=(10, 6))
    cumulative_returns.plot()
    plt.title('Equal-Weighted S&P 500 Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()

    position_size = portfolio_size / len(final_dataframe.index)
    for i in range(len(final_dataframe)):
        final_dataframe.loc[i, 'Number Of Shares to Buy'] = math.floor(position_size / final_dataframe['Price'][i])
    final_dataframe = final_dataframe[final_dataframe['Number Of Shares to Buy'] > 0]

    # Save trades to Excel
    writer = pd.ExcelWriter('equal_weighted_portfolio.xlsx', engine='xlsxwriter')
    final_dataframe.to_excel(writer, sheet_name='Equal Weighted Portfolio', index=False)

    background_color = '#ffffff'
    font_color = '#000000'

    string_format = writer.book.add_format(
        {'font_color': font_color, 'bg_color': background_color, 'border': 1}
    )

    dollar_format = writer.book.add_format(
        {'num_format': '$0.00', 'font_color': font_color, 'bg_color': background_color, 'border': 1}
    )

    integer_format = writer.book.add_format(
        {'num_format': '0', 'font_color': font_color, 'bg_color': background_color, 'border': 1}
    )

    column_formats = {
        'A': ['Ticker', string_format],
        'B': ['Price', dollar_format],
        'C': ['Market Capitalization', dollar_format],
        'D': ['Number Of Shares to Buy', integer_format]
    }

    worksheet = writer.sheets['Equal Weighted Portfolio']

    for column in column_formats.keys():
        worksheet.set_column(f'{column}:{column}', 20, column_formats[column][1])
        worksheet.write(f'{column}1', column_formats[column][0], string_format)

    writer.close()

    print("Trades saved to equal_weighted_portfolio.xlsx")

    return final_dataframe