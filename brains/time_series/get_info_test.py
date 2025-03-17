from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

def _fetch_data(ticker, chunks, interval, age_days):
    data = pd.DataFrame()
    for x in range(chunks):
        start_date = (datetime.now() - timedelta(days=8) - timedelta(days=8*x) - timedelta(days=age_days)).strftime('%Y-%m-%d')
        end_date = (datetime.now() - timedelta(days=8*x) - timedelta(days=age_days)).strftime('%Y-%m-%d')
        temp_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        data = pd.concat([data, temp_data])
    data.sort_index(inplace=True)
    data.columns = data.columns.droplevel(1)
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'Datetime'}, inplace=True)
    return data

data = _fetch_data('BTC-USD', 3, '1m', 0)
print(data.shape[0])
data = _fetch_data('BTC-USD', 8, '5m', 0)
print(data.shape[0])