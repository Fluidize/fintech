import yfinance as yf
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import layers, models, optimizers, callbacks, losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

feature_scaler = MinMaxScaler() #create scaler object in public to inverse in final output.
close_scaler = MinMaxScaler() #separate scaler object to store MinMax for inversing output yhat


########################################
## PRE-PROCESSING
########################################

def fetch_btc_data():
    data = yf.download('BTC-USD', period='8d', interval='1m')  
    return data

def add_features(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    df['RSI'] = compute_rsi(df['Close'], 14)
    
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Open'] = df['Open'].shift(1)
    df['Prev_Volume'] = df['Volume'].shift(1)
    
    df.dropna(inplace=True)

    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_data(df):
    # Target is the next bar's close
    df['Target'] = df['Close'].shift(-1)

    # Features (OHLCV + technical indicators)
    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50', 'RSI', 'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Open', 'Prev_Volume']]
    X = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns) #SCALE X SEPARATELY

    y = df[['Target']]
    y = pd.DataFrame(close_scaler.fit_transform(y), columns=y.columns) #SCALE Y SEPARATELY TO INVERSE LATER
    
    # Remove the last row (no target)
    X = X[:-1]
    y = y[:-1]
    
    return X, y

data = fetch_btc_data()
scaled_data = add_features(data)
X, y = prepare_data(scaled_data)

X = X.values.reshape((X.shape[0], 1, X.shape[1]))  # Reshaping to (samples, time_steps, features)

########################################
## MODEL
########################################

model = models.Sequential()
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss',  # Monitor validation loss
                              factor=0.2,
                              patience=5,
                              min_lr=1e-6)
early_stopping = callbacks.EarlyStopping(monitor='loss', mode='auto', patience=5, restore_best_weights=True)
rnn_width = 512
dense_width = 512

inputs = layers.Input(shape=(X.shape[1], X.shape[2])) # X.shape = (num_samples, num_time_steps, num_features)

rnn = layers.GRU(units=rnn_width, return_sequences=True)(inputs)
rnn = layers.GRU(units=rnn_width, return_sequences=True)(rnn)
rnn = layers.GRU(units=rnn_width, return_sequences=True)(rnn)

attention = layers.MultiHeadAttention(num_heads=13, key_dim=32)(rnn, rnn)

dense = layers.Dense(dense_width, activation='relu')(attention)
dense = layers.Dense(dense_width, activation='relu')(dense)

outputs = layers.Dense(1)(dense)

model = models.Model(inputs=inputs, outputs=outputs)
lossfn = losses.MeanAbsoluteError()
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=lossfn, metrics=['mean_squared_error'])
model_data = model.fit(X, y, epochs=1, batch_size=64, callbacks=[early_stopping])

yhat = model.predict(X)
yhat_inverse = close_scaler.inverse_transform(yhat.squeeze().reshape(-1,1)) #squeeze and reshape bc it expects a 2d dataframe

########################################
## ANALYZE
########################################

data.index = pd.to_datetime(data.index)
data = data.iloc[:-1] # extras

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplots with 2 rows and 1 column
fig = make_subplots(rows=2, cols=1,subplot_titles=('Actual vs Predicted Values', 'Loss Function'))

# Plot Actual vs Predicted values in the first subplot (top panel)
fig.append_trace(
    go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Actual', line=dict(color='blue')),
    row=1, col=1
)

fig.append_trace(
    go.Scatter(x=data.index, y=yhat_inverse.flatten(), mode='lines', name='Predicted', line=dict(color='red')),
    row=1, col=1
)

# Training loss
training_loss = model_data.history['loss']

# Plot training loss in the second subplot (bottom panel)
fig.append_trace(
    go.Scatter(x=list(range(len(training_loss))), y=training_loss, mode='lines', name='Training Loss', line=dict(color='blue')),
    row=2, col=1
)

# Update layout (titles, axis labels)
fig.update_layout(
    title_text="Actual vs Predicted & Loss Function",
    height=800,  # adjust height to avoid overlap
    showlegend=True,
    template="plotly_dark"  # Set dark mode theme
)

# Show the plot
fig.show()
