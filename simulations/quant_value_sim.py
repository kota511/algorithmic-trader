import numpy as np
import pandas as pd
import yfinance as yf
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

def quantitative_value_sim(symbols_file, training_period, testing_period, portfolio_size, stop_loss, take_profit):
    symbols = pd.read_csv(symbols_file)
    if 'Symbol' in symbols.columns:
        symbols.rename(columns={'Symbol': 'Ticker'}, inplace=True)

    def get_financial_data(ticker, start_date, end_date):
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            pe_ratio = stock.info.get('forwardPE')
            pb_ratio = stock.info.get('priceToBook')
            ps_ratio = stock.info.get('priceToSalesTrailing12Months')
            ev = stock.info.get('enterpriseValue')
            ebitda = stock.info.get('ebitda')
            gross_margin = stock.info.get('grossMargins')
            total_revenue = stock.info.get('totalRevenue')
            
            if gross_margin is not None and total_revenue is not None:
                gross_profit = gross_margin * total_revenue
            else:
                gross_profit = np.nan
            
            ev_ebitda = ev / ebitda if ev and ebitda else np.nan
            ev_gp = ev / gross_profit if ev and gross_profit else np.nan
            
            return {
                'Ticker': ticker,
                'Price': current_price,
                'PE Ratio': pe_ratio,
                'PB Ratio': pb_ratio,
                'PS Ratio': ps_ratio,
                'EV/EBITDA': ev_ebitda,
                'EV/GP': ev_gp
            }
        else:
            return None

    def fetch_data_concurrently(tickers, start_date, end_date):
        financial_data = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(get_financial_data, ticker, start_date, end_date): ticker for ticker in tickers}
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data:
                        financial_data.append(data)
                except Exception as e:
                    print(f"Could not get data for {ticker}: {e}", flush=True)
        return financial_data

    tickers = symbols['Ticker']
    financial_data = fetch_data_concurrently(tickers, training_period[0], training_period[1])
    financial_df = pd.DataFrame(financial_data)
    
    print(f"Total number of stocks fetched: {len(financial_df)}", flush=True)
    
    financial_df.dropna(subset=['Price', 'PE Ratio', 'PB Ratio', 'PS Ratio', 'EV/EBITDA', 'EV/GP'], how='all', inplace=True)
    
    print(f"Total number of stocks after dropping missing data: {len(financial_df)}", flush=True)

    for column in ['PE Ratio', 'PB Ratio', 'PS Ratio', 'EV/EBITDA', 'EV/GP']:
        financial_df[column] = financial_df[column].fillna(financial_df[column].mean())
    
    metrics = {
        'PE Ratio': 'PE Percentile',
        'PB Ratio': 'PB Percentile',
        'PS Ratio': 'PS Percentile',
        'EV/EBITDA': 'EV/EBITDA Percentile',
        'EV/GP': 'EV/GP Percentile'
    }

    for metric in metrics.keys():
        financial_df[metrics[metric]] = financial_df[metric].rank(pct=True)

    for index, row in financial_df.iterrows():
        value_percentiles = [
            row['PE Percentile'],
            row['PB Percentile'],
            row['PS Percentile'],
            row['EV/EBITDA Percentile'],
            row['EV/GP Percentile']
        ]
        financial_df.at[index, 'RV Score'] = mean(value_percentiles)
    
    financial_df.sort_values(by='RV Score', inplace=True)
    
    top_50_stocks = financial_df.head(50).reset_index(drop=True)
    
    print(f"Total number of stocks selected: {len(top_50_stocks)}", flush=True)

    position_size = portfolio_size / len(top_50_stocks.index)
    results = []

    for i, row in top_50_stocks.iterrows():
        ticker = row['Ticker']
        shares_to_buy = math.floor(position_size / row['Price'])
        
        # Simulate the trading over the testing period
        stock_data = yf.download(ticker, start=testing_period[0], end=testing_period[1])
        if stock_data.empty:
            continue
        
        buy_price = stock_data['Close'].iloc[0]
        stop_loss_price = buy_price * (1 - stop_loss)
        take_profit_price = buy_price * (1 + take_profit)
        
        for date, price in stock_data['Close'].items():
            if price <= stop_loss_price:
                sell_price = price
                break
            elif price >= take_profit_price:
                sell_price = price
                break
        else:
            sell_price = stock_data['Close'].iloc[-1]
        
        profit = (sell_price - buy_price) * shares_to_buy
        results.append({
            'Ticker': ticker,
            'Buy Price': buy_price,
            'Sell Price': sell_price,
            'Shares': shares_to_buy,
            'Profit': profit,
            'PE Ratio': row['PE Ratio'],
            'PB Ratio': row['PB Ratio'],
            'PS Ratio': row['PS Ratio'],
            'EV/EBITDA': row['EV/EBITDA'],
            'EV/GP': row['EV/GP'],
            'RV Score': row['RV Score']
        })

    results_df = pd.DataFrame(results)
    
    total_profit = results_df['Profit'].sum()

    print("Total Profit: ${:.2f}".format(total_profit), flush=True)

    results_df.to_excel('trade_log_value_strategy.xlsx', index=False)
    print("Trade log saved to trade_log_value_strategy.xlsx", flush=True)

    return results_df, total_profit