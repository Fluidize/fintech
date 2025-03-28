import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

import numpy as np
import scipy.stats as stats
from scipy.differentiate import derivative
import plotly.graph_objects as go
import plotly.subplots as sp

def fetch_data(symbol, chunks, interval, age_days):
    data = pd.DataFrame()
    for x in range(chunks):
        chunksize = 1
        start_date = (datetime.now() - timedelta(days=chunksize) - timedelta(days=chunksize*x) - timedelta(days=age_days)).strftime('%Y-%m-%d')
        end_date = (datetime.now() - timedelta(days=chunksize*x) - timedelta(days=age_days)).strftime('%Y-%m-%d')
        temp_data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        data = pd.concat([data, temp_data])
        
    data.sort_index(inplace=True)
    data.columns = data.columns.droplevel(1)
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'Datetime'}, inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return pd.DataFrame(data)  

def _calculate_rsi(context, period=14):
    delta_p = context['Close'].diff()
    gain = delta_p.where(delta_p > 0, 0)
    loss = -delta_p.where(delta_p < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi
def _calculate_macd(context, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = context['Close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = context['Close'].ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

data = fetch_data('BTC-USD', 29, '5m', 10)

# Calculate returns instead of standard deviation
raw_std = data['Close'].rolling(window=20).std()
raw_std.dropna(inplace=True, axis=0)

std = (raw_std - raw_std.min())/(raw_std.max() - raw_std.min())

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(std)

chi_df, chi_loc, chi_scale = stats.chi.fit(std)
x = np.linspace(np.mean(std) - 4*np.std(std), np.mean(std) + 4*np.std(std), 1000)
pdf = stats.chi2.pdf(x, chi_df, loc=chi_loc, scale=chi_scale)
cdf = stats.chi2.cdf(x, chi_df, loc=chi_loc, scale=chi_scale)
# derivative_cdf = np.gradient(cdf, x)
print(np.sum(std < 0))

fig = sp.make_subplots(rows=2, cols=1, 
                      subplot_titles=('PDF', 'CDF'))
fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF'), row=1, col=1)
# fig.add_trace(go.Histogram(x=std, name='Histogram'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=cdf, mode='lines', name='CDF'), row=2, col=1)
# fig.add_trace(go.Scatter(x=x, y=derivative_cdf, mode='lines', name="CDF'"), row=1, col=1)

fig.show()

