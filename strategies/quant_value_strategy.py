import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

def quantitative_value(symbols_file, portfolio_size):
    symbols = pd.read_csv(symbols_file)
    if 'Symbol' in symbols.columns:
        symbols.rename(columns={'Symbol': 'Ticker'}, inplace=True)

    def get_financial_data(ticker):
        stock = yf.Ticker(ticker)
        data = stock.info
        
        current_price = data.get('currentPrice')
        pe_ratio = data.get('forwardPE')
        pb_ratio = data.get('priceToBook')
        ps_ratio = data.get('priceToSalesTrailing12Months')
        ev = data.get('enterpriseValue')
        ebitda = data.get('ebitda')
        gross_margin = data.get('grossMargins')
        total_revenue = data.get('totalRevenue')
        
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

    def fetch_data_concurrently(tickers):
        financial_data = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(get_financial_data, ticker): ticker for ticker in tickers}
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    financial_data.append(data)
                except Exception as e:
                    print(f"Could not get data for {ticker}: {e}")
        return financial_data

    tickers = symbols['Ticker']
    financial_data = fetch_data_concurrently(tickers)
    financial_df = pd.DataFrame(financial_data)

    financial_df.dropna(subset=['Price', 'PE Ratio', 'PB Ratio', 'PS Ratio', 'EV/EBITDA', 'EV/GP'], how='all', inplace=True)

    print(financial_df)

    for column in ['PE Ratio', 'PB Ratio', 'PS Ratio', 'EV/EBITDA', 'EV/GP']:
        if column in financial_df.columns:
            financial_df[column] = financial_df[column].fillna(financial_df[column].mean())
            financial_df[column] = financial_df[column].infer_objects(copy=False)

    ev_gp_mean = financial_df['EV/GP'].mean()
    financial_df['EV/GP'] = financial_df['EV/GP'].fillna(ev_gp_mean)

    print(financial_df)

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
    
    print(financial_df)

    financial_df.sort_values(by='RV Score', inplace=True)
    top_50_stocks = financial_df.head(50).reset_index(drop=True)

    print(top_50_stocks)

    position_size = portfolio_size / len(top_50_stocks.index)

    for i in range(len(top_50_stocks)):
        top_50_stocks.at[i, 'Number of Shares to Buy'] = math.floor(position_size / top_50_stocks.at[i, 'Price'])

    ordered_columns = [
        'Ticker', 'Price', 'Number of Shares to Buy', 'PE Ratio', 'PE Percentile', 
        'PB Ratio', 'PB Percentile', 'PS Ratio', 'PS Percentile', 'EV/EBITDA', 
        'EV/EBITDA Percentile', 'EV/GP', 'EV/GP Percentile', 'RV Score'
    ]
    top_50_stocks = top_50_stocks[ordered_columns]

    top_50_stocks.to_excel('value_strategy.xlsx', index=False)
    print("Trades saved to value_strategy.xlsx")

    return top_50_stocks
# symbols_file = 'data/sp500_symbols.csv'
# portfolio_size = input("Enter the value of your portfolio: ")
# quantitative_value(symbols_file, int(portfolio_size))