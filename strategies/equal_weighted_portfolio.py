import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math

def equal_weighted_portfolio(symbols_file, portfolio_size):
    symbols = pd.read_csv(symbols_file, skiprows=1, header=None)[0].tolist()

    all_data = {}
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

    final_dataframe.to_excel('recommended_trades.xlsx', index=False)
    print("Trades saved to recommended_trades.xlsx")

    return final_dataframe

# symbols_file = 'data/sp500_symbols.csv'
# portfolio_size = input("Enter the value of your portfolio: ")
# equal_weighted_portfolio(symbols_file, int(portfolio_size))