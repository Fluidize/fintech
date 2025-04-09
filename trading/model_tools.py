from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from datetime import datetime, timedelta
import requests
import yfinance as yf
from tqdm import tqdm
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rich import print

import talib
from scipy.stats import kurtosis, skew

def fetch_data(ticker, chunks, interval, age_days, kucoin: bool = True):
    print("[green]DOWNLOADING DATA[/green]")
    if not kucoin:
        data = pd.DataFrame()
        times = []
        for x in range(chunks):
            chunksize = 1
            start_date = datetime.now() - timedelta(days=chunksize) - timedelta(days=chunksize*x) - timedelta(days=age_days)
            end_date = datetime.now() - timedelta(days=chunksize*x) - timedelta(days=age_days)
            temp_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval, progress=False)
            if data.empty:
                data = temp_data
            else:
                data = pd.concat([data, temp_data])
            times.append(start_date)
            times.append(end_date)
        
        earliest = min(times)
        latest = max(times)
        difference = latest - earliest
        print(f"\n{ticker} | {difference.days} days {difference.seconds//3600} hours {difference.seconds//60%60} minutes {difference.seconds%60} seconds")

        data.sort_index(inplace=True)
        data.columns = data.columns.droplevel(1)
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Datetime'}, inplace=True)
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
        data = pd.DataFrame(data)  
    elif kucoin:
        data = pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        times = []
        
        progress_bar = tqdm(total=chunks, desc="KUCOIN PROGRESS")
        for x in range(chunks):
            chunksize = 1440  # 1d of 1m data
            end_time = datetime.now() - timedelta(minutes=chunksize*x) - timedelta(days=age_days)
            start_time = end_time - timedelta(minutes=chunksize) - timedelta(days=age_days)
            
            params = {
                "symbol": ticker,
                "type": interval,
                "startAt": str(int(start_time.timestamp())),
                "endAt": str(int(end_time.timestamp()))
            }
            
            request = requests.get("https://api.kucoin.com/api/v1/market/candles", params=params).json()
            try:
                request_data = request["data"]  # list of lists
            except:
                raise Exception(f"Error fetching Kucoin. Check request parameters.")
            
            records = []
            for dochltv in request_data:
                records.append({
                    "Datetime": dochltv[0],
                    "Open": float(dochltv[1]),
                    "Close": float(dochltv[2]),
                    "High": float(dochltv[3]),
                    "Low": float(dochltv[4]),
                    "Volume": float(dochltv[6])
                })
            
            temp_data = pd.DataFrame(records)
            if data.empty:
                data = temp_data
            else:
                data = pd.concat([data, temp_data])
            times.append(start_time)
            times.append(end_time)

            progress_bar.update(1)
        progress_bar.close()
        
        earliest = min(times)
        latest = max(times)
        difference = latest - earliest
        print(f"{ticker} | {difference.days} days {difference.seconds//3600} hours {difference.seconds//60%60} minutes {difference.seconds%60} seconds | {data.shape[0]} bars")
        
        data["Datetime"] = pd.to_datetime(pd.to_numeric(data['Datetime']), unit='s')
        data.sort_values('Datetime', inplace=True)
        data.reset_index(drop=True, inplace=True)
        
    return data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_data(data, lagged_length=5, train_split=True, scale_y=True):
    scalers = {
        'price': MinMaxScaler(feature_range=(0, 1)),
        'volume': QuantileTransformer(output_distribution='normal'),
        'technical': StandardScaler()
    }

    df = data.copy()
    df = df.drop(columns=['MA50', 'MA20', 'MA10', 'RSI'], errors='ignore')

    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    
    lagged_features = []
    for col in ['Close', 'Volume', 'High', 'Low', 'Open']:
        for i in range(1, lagged_length):
            lagged_features.append(pd.DataFrame({
                f'Prev{i}_{col}': df[col].shift(i)
            }))
    
    if lagged_features:
        df = pd.concat([df] + lagged_features, axis=1)
    
    std = df['Close'].std()
    df['Close_ZScore'] = (df['Close'] - df['Close'].mean()) / std 
    
    df['MA10'] = df['Close'].rolling(window=10).mean() / df['Close']
    df['MA20'] = df['Close'].rolling(window=20).mean() / df['Close']
    df['MA50'] = df['Close'].rolling(window=50).mean() / df['Close']
    df['MA10_MA20_Cross'] = df['MA10'] - df['MA20']
    
    df['RSI'] = compute_rsi(df['Close'], 14)

    # Check for NaN values
    if df.isnull().any().any():
        # Fill NaN values with the mean of the column
        df = df.fillna(df.mean())
        # Alternatively, drop rows with NaN values
        # df = df.dropna()

    df.dropna(inplace=True)
    
    if train_split:
        price_features = ['Open', 'High', 'Low', 'Close']
        volume_features = ['Volume'] + [f'Prev{i}_Volume' for i in range(1, lagged_length)]
        bounded_features = ['RSI']  # Features that are already bounded (e.g., 0-100)
        normalized_features = ['MA10', 'MA20', 'MA50', 'Price_Range', 'MA10_MA20_Cross', 'Close_ZScore']
        
        technical_features = [col for col in df.columns 
                            if col not in (price_features + volume_features + bounded_features + 
                                        normalized_features + ['Datetime'])]
        if scale_y:
            df[price_features] = scalers['price'].fit_transform(df[price_features])
        else:
            df[price_features] = df[price_features]
        
        df[volume_features] = df[volume_features].replace([np.inf, -np.inf], np.nan)
        df[volume_features] = df[volume_features].fillna(df[volume_features].mean())
        df[volume_features] = scalers['volume'].fit_transform(df[volume_features])
        
        if technical_features:
            df[technical_features] = scalers['technical'].fit_transform(df[technical_features])

        if 'Datetime' in df.columns:
            X = df.drop(['Datetime'], axis=1)
        else:
            X = df
        
        y = df['Close'].shift(-1)
        
        X = X[:-1]
        y = y[:-1]
        return X, y, scalers
    
    return df, scalers

def prepare_data_classifier(data, lagged_length=5, train_split=True, pct_threshold=0.01):
    """
    Enhanced data preparation for classification with advanced feature engineering using TA-Lib.
    Uses only past information for all calculations to prevent data leakage.
    
    Parameters:
        data: DataFrame with OHLCV data
        lagged_length: Number of lagged features to create
        train_split: Whether to return X,y for training
        pct_threshold: Threshold for classifying returns (in %)
    """
    pct_threshold = pct_threshold / 100
    df = data.copy()
    
    # Keep only OHLCV columns to avoid any potential leakage
    if len(df.columns) > 5:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # ==== PRICE FEATURES ====
    price_features = pd.DataFrame({
        'CloseOverOpen': df['Close'] / df['Open'],
        'HighOverLow': df['High'] / df['Low'],
        'HighOverClose': df['High'] / df['Close'],
        'LowOverClose': df['Low'] / df['Close'],
        'PriceRange': (df['High'] - df['Low']) / df['Close'],
        'UpperShadow': (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close'],
        'LowerShadow': (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
    })
    
    # ==== RETURNS ====
    returns_features = pd.DataFrame({
        'Returns': df['Close'].pct_change(),
        'LogReturns': np.log(df['Close'] / df['Close'].shift(1))
    })
    
    # ==== VOLUME FEATURES ====
    volume_features = pd.DataFrame({
        'RelativeVolume': df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean(),
        'VolumeChange': df['Volume'].pct_change(),
        'PriceVolCorr': returns_features['Returns'] * df['Volume'].pct_change()
    })
    
    # ==== VOLATILITY INDICATORS ====
    volatility_features = pd.DataFrame()
    for window in [5, 10, 20]:
        volatility_features[f'Volatility{window}'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=window)
        if window > 5:
            volatility_features[f'VolatilityRatio{window}'] = volatility_features['Volatility5'] / (volatility_features[f'Volatility{window}'] + 1e-8)
    
    # ==== MOMENTUM INDICATORS ====
    momentum_features = pd.DataFrame()
    for period in [3, 5, 10, 20]:
        momentum_features[f'Momentum{period}'] = talib.MOM(df['Close'], timeperiod=period)
        momentum_features[f'ROC{period}'] = talib.ROC(df['Close'], timeperiod=period)
    
    # ==== MOVING AVERAGES ====
    ma_features = pd.DataFrame()
    for window in [5, 10, 20, 50]:
        ma_features[f'SMA{window}'] = talib.SMA(df['Close'], timeperiod=window)
        ma_features[f'EMA{window}'] = talib.EMA(df['Close'], timeperiod=window)
        ma_features[f'CloseOverSMA{window}'] = df['Close'] / ma_features[f'SMA{window}']
        ma_features[f'CloseOverEMA{window}'] = df['Close'] / ma_features[f'EMA{window}']
    
    # MA Crossovers
    ma_crossovers = pd.DataFrame({
        'SMACross5_10': ma_features['SMA5'] / ma_features['SMA10'],
        'SMACross10_20': ma_features['SMA10'] / ma_features['SMA20'],
        'EMACross5_10': ma_features['EMA5'] / ma_features['EMA10'],
        'EMACross10_20': ma_features['EMA10'] / ma_features['EMA20']
    })
    
    # ==== OSCILLATORS ====
    oscillator_features = pd.DataFrame({
        'RSI': talib.RSI(df['Close'], timeperiod=14),
        'StochK': talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0)[0],
        'StochD': talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0)[1],
        'WillR': talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14),
        'CCI': talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=20),
        'ADX': talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    })
    
    # ==== PATTERN RECOGNITION ====
    pattern_features = pd.DataFrame({
        'DojiPattern': talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close']),
        'Hammer': talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    })
    
    # ==== LAGGED FEATURES ====
    important_cols = ['Returns', 'Close', 'Volume', 'RSI', 'CloseOverSMA20', 'StochK', 'HighOverLow', 'Volatility10']
    lagged_features = pd.DataFrame()
    
    for col in important_cols:
        for i in range(1, lagged_length + 1):
            if col in returns_features.columns:
                lagged_features[f'Lag{i}_{col}'] = returns_features[col].shift(i)
            elif col in df.columns:
                lagged_features[f'Lag{i}_{col}'] = df[col].shift(i)
            elif col in oscillator_features.columns:
                lagged_features[f'Lag{i}_{col}'] = oscillator_features[col].shift(i)
            elif col in price_features.columns:
                lagged_features[f'Lag{i}_{col}'] = price_features[col].shift(i)
            elif col in volatility_features.columns:
                lagged_features[f'Lag{i}_{col}'] = volatility_features[col].shift(i)
    
    # ==== STATISTICAL FEATURES ====
    statistical_features = pd.DataFrame()
    for window in [5, 10, 20]:
        roll_returns = returns_features['Returns'].rolling(window=window, min_periods=max(3, window//3))
        statistical_features[f'Skew{window}'] = roll_returns.apply(lambda x: skew(x) if len(x) > 3 else 0, raw=False)
        statistical_features[f'Kurt{window}'] = roll_returns.apply(lambda x: kurtosis(x) if len(x) > 3 else 0, raw=False)
    
    # Combine all features
    df = pd.concat([
        df, price_features, returns_features, volume_features, volatility_features,
        momentum_features, ma_features, ma_crossovers, oscillator_features,
        pattern_features, lagged_features, statistical_features
    ], axis=1)
    
    # ==== HANDLE NaN VALUES ====
    df = df.ffill()
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].expanding().mean())
    df = df.fillna(0)
    
    # ==== FILTER EXTREME VALUES ====
    for col in df.columns:
        if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            if not np.isnan(q1) and not np.isnan(q99):
                df[col] = df[col].clip(q1, q99)
    
    if train_split:
        feature_columns = [col for col in df.columns if col != 'Datetime']
        future_returns = df['Close'].pct_change().shift(-1)
        
        y = pd.Series(1, index=df.index)  # Default to hold
        y[future_returns < -pct_threshold] = 0  # Sell signal
        y[future_returns > pct_threshold] = 2  # Buy signal
        
        X = df[feature_columns].copy()
        
        # Scale features using per-feature robust standardization
        window_size = min(100, len(X) // 3)
        
        for col in X.columns:
            if col in ['RSI', 'StochK', 'StochD', 'WillR', 'DojiPattern', 'Hammer'] or 'Over' in col:
                continue
                
            roll_mean = X[col].rolling(window=window_size, min_periods=5).mean()
            roll_std = X[col].rolling(window=window_size, min_periods=5).std()
            
            roll_mean = roll_mean.bfill()
            roll_std = roll_std.bfill()
            
            X[col] = (X[col] - roll_mean) / (roll_std + 1e-8)
        
        X = X.fillna(0)
        X = X[:-1]
        y = y[:-1]
        
        return X, y
    
    return df

def prediction_plot(actual, predicted):
    difference = len(actual)-len(predicted) #trimmer
    actual = actual[difference:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual['Datetime'], y=actual['Close'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=actual['Datetime'], y=predicted, mode='lines', name='Predicted'))
    fig.update_layout(title='Price Prediction', xaxis_title='Date', yaxis_title='Price')
    
    return fig

def loss_plot(loss_history):
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=list(range(len(loss_history))), y=loss_history, mode='lines', name='Loss'), row=1, col=1)
    delta_loss = np.diff(loss_history)
    fig.add_trace(go.Scatter(x=list(range(len(delta_loss))), y=delta_loss, mode='lines', name='Delta Loss'), row=2, col=1)
    fig.update_layout(title='Loss', xaxis_title='Epoch', yaxis_title='Loss')

    return fig