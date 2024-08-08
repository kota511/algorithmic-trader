import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math

def equal_weighted_portfolio(symbols_file, training_period, testing_period, initial_portfolio_size, stop_loss, take_profit):
    symbols = pd.read_csv(symbols_file, skiprows=1, header=None)[0].tolist()

    all_data = {}
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

    data_list = []
    valid_symbols = list(training_data.columns)
    for symbol in valid_symbols:
        try:
            market_cap, price = get_stock_info(symbol)
            data_list.append({'Ticker': symbol, 'Price': price, 'Market Capitalization': market_cap})
        except Exception as e:
            print(f"Failed to get info for {symbol}: {e}")

    final_dataframe = pd.DataFrame(data_list)

    # Calculate the initial position size and number of shares to buy
    position_size = initial_portfolio_size / len(final_dataframe.index)
    trade_log = []  # Trade log to record all trades
    for i in range(len(final_dataframe)):
        shares_to_buy = math.floor(position_size / final_dataframe['Price'][i])
        final_dataframe.loc[i, 'Number Of Shares to Buy'] = shares_to_buy
        trade_log.append({
            'Date': training_period[1],  # Using the end of the training period as the initial buy date
            'Ticker': final_dataframe.loc[i, 'Ticker'],
            'Action': 'Buy',
            'Shares': shares_to_buy,
            'Price': final_dataframe.loc[i, 'Price'],
            'Total Value': shares_to_buy * final_dataframe.loc[i, 'Price']
        })

    # Adjust initial portfolio value after buying the shares
    total_invested = sum(final_dataframe['Number Of Shares to Buy'] * final_dataframe['Price'])
    portfolio_value = initial_portfolio_size - total_invested
    print(f"Total initial investment: {total_invested}")
    print(f"Remaining cash after initial investment: {portfolio_value}")

    # Implement stop loss and take profit
    stop_loss_level = final_dataframe['Price'] * (1 - stop_loss)
    take_profit_level = final_dataframe['Price'] * (1 + take_profit)

    for date in testing_data.index:
        stocks_sold = False
        cash_from_sales = 0
        for i in range(len(final_dataframe)):
            ticker = final_dataframe.loc[i, 'Ticker']
            if ticker in testing_data.columns:
                price = testing_data.loc[date, ticker]
                shares = final_dataframe.loc[i, 'Number Of Shares to Buy']

                if shares > 0:  # Ensure there's something to sell
                    if price <= stop_loss_level[i] or price >= take_profit_level[i]:
                        cash_from_sales += shares * price
                        final_dataframe.loc[i, 'Number Of Shares to Buy'] = 0
                        stocks_sold = True
                        trade_log.append({
                            'Date': date,
                            'Ticker': ticker,
                            'Action': 'Sell',
                            'Shares': shares,
                            'Price': price,
                            'Total Value': shares * price
                        })
                        print(f"Selling {shares} shares of {ticker} at {price} on {date} due to stop loss or take profit.")

        if stocks_sold:
            portfolio_value += cash_from_sales
            valid_prices = testing_data.loc[date].dropna()
            valid_symbols = valid_prices.index
            new_position_size = portfolio_value / len(valid_symbols)
            for symbol in valid_symbols:
                price = testing_data.loc[date, symbol]
                if not np.isnan(price) and price > 0:  # Check to avoid NaN and ensure price is valid
                    shares_to_buy = math.floor(new_position_size / price)
                    final_dataframe.loc[final_dataframe['Ticker'] == symbol, 'Number Of Shares to Buy'] += shares_to_buy
                    portfolio_value -= shares_to_buy * price
                    stop_loss_level[final_dataframe['Ticker'] == symbol] = price * (1 - stop_loss)
                    take_profit_level[final_dataframe['Ticker'] == symbol] = price * (1 + take_profit)
                    trade_log.append({
                        'Date': date,
                        'Ticker': symbol,
                        'Action': 'Buy',
                        'Shares': shares_to_buy,
                        'Price': price,
                        'Total Value': shares_to_buy * price
                    })
            print(f"Reinvesting into valid stocks on {date}. Remaining cash: {portfolio_value}")

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
    print(f"Sharpe Ratio: {sharpe_ratio}")

    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    print(f"Max Drawdown: {max_drawdown}")

    final_stock_value = sum(final_dataframe['Number Of Shares to Buy'] * final_dataframe['Price'])
    print(f"Final Stock Value: {final_stock_value}")
    final_portfolio_value = portfolio_value + final_stock_value
    print(f"Final Portfolio Value: {final_portfolio_value}")

    # Save trade log to Excel
    trade_log_df = pd.DataFrame(trade_log)
    trade_log_df.to_excel('trade_log_equal_weighted.xlsx', index=False)
    print("Trade log saved to trade_log_equal_weighted.xlsx")

    return final_dataframe
