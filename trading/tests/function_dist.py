import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import norm
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
returns = data['Close'].pct_change().dropna()
mu, sigma = norm.fit(returns)
print(f"Mean return: {mu:.4f}, Std of returns: {sigma:.4f}")

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
pdf = norm.pdf(x, mu, sigma)
cdf = norm.cdf(x, mu, sigma)

fig = sp.make_subplots(rows=2, cols=1, 
                      subplot_titles=('Distribution of Returns', 'Cumulative Distribution of Returns'))
fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='Normal Distribution'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=cdf, mode='lines', name='Cumulative Distribution'), row=2, col=1)

# Add histogram of actual returns
fig.add_trace(go.Histogram(x=returns, name='Actual Returns', nbinsx=50, opacity=0.5), row=1, col=1)

fig.update_layout(title_text='BTC-USD Returns Distribution Analysis')
fig.update_xaxes(title_text='Return', row=1, col=1)
fig.update_xaxes(title_text='Return', row=2, col=1)
fig.update_yaxes(title_text='Probability Density', row=1, col=1)
fig.update_yaxes(title_text='Cumulative Probability', row=2, col=1)

fig.show()

