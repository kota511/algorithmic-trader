import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import xlsxwriter
import math

def equal_weighted_portfolio_sim(symbols_file, training_period, testing_period, initial_portfolio_size, stop_loss, take_profit):
    symbols = pd.read_csv(symbols_file, skiprows=1, header=None)[0].tolist()
    initial = initial_portfolio_size
    all_data = {}
    # Download data for all symbols, which have data available
    for symbol in symbols:
        try:
            data = yf.download(symbol, start=training_period[0], end=testing_period[1])['Adj Close']
            if not data.empty:
                all_data[symbol] = data
        except Exception as e:
            print(f"Failed download for {symbol}: {e}")

    all_data_df = pd.DataFrame(all_data)
    all_data_df = all_data_df.dropna(axis=1, how='all')

    training_data = all_data_df.loc[training_period[0]:training_period[1]]
    testing_data = all_data_df.loc[testing_period[0]:testing_period[1]]

    num_stocks = training_data.shape[1]
    equal_weights = 1 / num_stocks

    daily_returns = testing_data.pct_change().dropna()

    def get_stock_info(symbol):
        stock = yf.Ticker(symbol)
        info = stock.info
        market_cap = info.get('marketCap', 'N/A')
        price = info.get('currentPrice', 'N/A')
        return market_cap, price

    # Get stock info for all valid symbols
    data_list = []
    valid_symbols = list(training_data.columns)
    for symbol in valid_symbols:
        try:
            market_cap, price = get_stock_info(symbol)
            data_list.append({'Ticker': symbol, 'Price': round(price, 2), 'Market Capitalization': market_cap})
        except Exception as e:
            print(f"Failed to get info for {symbol}: {e}")

    final_dataframe = pd.DataFrame(data_list)

    # Calculate the initial position size and number of shares to buy
    position_size = initial_portfolio_size / len(final_dataframe.index)
    trade_log = []  # Trade log to record all trades
    number_of_shares = []

    for i in range(len(final_dataframe)):
        shares_to_buy = math.floor(position_size / final_dataframe['Price'][i])
        number_of_shares.append(shares_to_buy)

        if shares_to_buy > 0:
            total_value = round(shares_to_buy * final_dataframe['Price'][i], 2)
            trade_log.append({
                'Date': training_period[1],  # Using the end of the training period as the initial buy date
                'Ticker': final_dataframe.loc[i, 'Ticker'],
                'Action': 'Buy',
                'Shares': shares_to_buy,
                'Price': final_dataframe.loc[i, 'Price'],
                'Total Value': total_value,
                'PnL': 0
            })

    # Add the calculated number of shares to the final_dataframe
    final_dataframe['Number Of Shares to Buy'] = number_of_shares

    # Remove rows where no shares were bought
    final_dataframe = final_dataframe[final_dataframe['Number Of Shares to Buy'] > 0]
    final_dataframe.reset_index(drop=True, inplace=True)

    # Adjust initial portfolio value after buying the shares
    total_invested = round(sum(final_dataframe['Number Of Shares to Buy'] * final_dataframe['Price']), 2)
    portfolio_value = round(initial_portfolio_size - total_invested, 2)
    print(f"Total initial investment: {total_invested}")
    print(f"Remaining cash after initial investment: {portfolio_value}")

    # Implement stop loss and take profit
    stop_loss_level = final_dataframe['Price'] * (1 - stop_loss)
    take_profit_level = final_dataframe['Price'] * (1 + take_profit)

    for date in testing_data.index:
        for i in range(len(final_dataframe)):
            ticker = final_dataframe.loc[i, 'Ticker']
            if ticker in testing_data.columns:
                price = testing_data.loc[date, ticker]
                shares = final_dataframe.loc[i, 'Number Of Shares to Buy']

                if shares > 0: # Ensure there's something to sell
                    if price <= stop_loss_level[i] or price >= take_profit_level[i]:
                        cash_from_sales = round(shares * price, 2)
                        final_dataframe.loc[i, 'Number Of Shares to Buy'] = 0 # Remove shares after sale
                        pnl = round((price - final_dataframe.loc[i, 'Price']) * shares, 2)
                        portfolio_value += cash_from_sales # Update portfolio value by adding cash from sales
                        trade_log.append({
                            'Date': date,
                            'Ticker': ticker,
                            'Action': 'Sell',
                            'Shares': shares,
                            'Price': round(price, 2),
                            'Total Value': cash_from_sales,
                            'PnL': pnl
                        })
                        print(f"Selling {shares} shares of {ticker} at {round(price, 2)} on {date} due to stop loss or take profit. PnL: {pnl}")

    weighted_returns = daily_returns * equal_weights
    portfolio_returns = weighted_returns.sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    plt.figure(figsize=(10, 6))
    cumulative_returns.plot()
    plt.title('Equal-Weighted S&P 500 Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()

    risk_free_rate = 0.01
    excess_returns = portfolio_returns - (risk_free_rate / 252)
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    print(f"Sharpe Ratio: {round(sharpe_ratio, 2)}")

    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    print(f"Max Drawdown: {round(max_drawdown, 2)}")

    print(f"Initial Portfolio Value: {initial}")
    print(f"Overall PnL: {round(sum(trade['PnL'] for trade in trade_log), 2)}")
    final_stock_value = round(sum(final_dataframe['Number Of Shares to Buy'] * final_dataframe['Price']), 2)
    final_portfolio_value = round(portfolio_value + final_stock_value, 2)
    print(f"Final Portfolio Value: {final_portfolio_value}")

    # Save trade log to Excel
    writer = pd.ExcelWriter('trade_log_equal_weighted.xlsx', engine='xlsxwriter')
    trade_log_df = pd.DataFrame(trade_log)
    trade_log_df.to_excel(writer, sheet_name='Trade Log', index=False)

    # Set formatting options
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

    date_format = writer.book.add_format(
        {'num_format': 'yyyy-mm-dd', 'font_color': font_color, 'bg_color': background_color, 'border': 1}
    )

    column_formats = {
        'A': ['Date', date_format],
        'B': ['Ticker', string_format],
        'C': ['Action', string_format],
        'D': ['Shares', integer_format],
        'E': ['Price', dollar_format],
        'F': ['Total Value', dollar_format],
        'G': ['PnL', dollar_format]
    }

    worksheet = writer.sheets['Trade Log']

    for column in column_formats.keys():
        worksheet.set_column(f'{column}:{column}', 20, column_formats[column][1])
        worksheet.write(f'{column}1', column_formats[column][0], string_format)

    writer.close()

    print("Trade log saved to trade_log_equal_weighted.xlsx")

    return final_dataframe
