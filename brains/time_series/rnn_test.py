import yfinance as yf
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import layers, models, optimizers, callbacks, losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timedelta
import os

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

class TimeSeriesPredictor:
    def __init__(self, epochs, rnn_width, dense_width, ticker='BTC-USD', chunks=5, interval='5m', age_days=10):
        self.epochs = epochs
        self.rnn_width = rnn_width
        self.dense_width = dense_width

        self.ticker = ticker
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.feature_scaler = MinMaxScaler()
        self.ohlcv_scaler = MinMaxScaler()  # Updated from price_scaler to ohlcv_scaler

        self.train_data = None
        self.X = None
        self.y = None
        self.model_data = None
        self.yhat = None
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    ########################################
    ## PROCESSING FUNCTIONS
    ########################################

    def _fetch_data(self):
        data = pd.DataFrame()
        temp_data = None
        for x in range(self.chunks):
            start_date = (datetime.now() - timedelta(days=8) - timedelta(days=8*x) - timedelta(days=self.age_days)).strftime('%Y-%m-%d')
            end_date = (datetime.now()- timedelta(days=8*x) - timedelta(days=self.age_days)).strftime('%Y-%m-%d')
            temp_data = yf.download(self.ticker, start=start_date, end=end_date, interval=self.interval)
            data = pd.concat([data, temp_data])
        data.sort_index(inplace=True)
        self.train_data = data

    def _add_features(self):
        df = self.train_data
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        df['RSI'] = self._compute_rsi(df['Close'], 14)
        
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_High'] = df['High'].shift(1)
        df['Prev_Low'] = df['Low'].shift(1)
        df['Prev_Open'] = df['Open'].shift(1)
        df['Prev_Volume'] = df['Volume'].shift(1)
        
        df.dropna(inplace=True)

        self.train_data = df

    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _prepare_data(self):
        df = self.train_data
        ohlcv_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        other_features = ['MA20', 'MA50', 'RSI', 'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Open', 'Prev_Volume']
        
        X_ohlcv = df[ohlcv_features]
        X_other = df[other_features]
        
        X_ohlcv_scaled = pd.DataFrame(self.ohlcv_scaler.fit_transform(X_ohlcv), columns=X_ohlcv.columns)  # Updated
        X_other_scaled = pd.DataFrame(self.feature_scaler.fit_transform(X_other), columns=X_other.columns)
        X = pd.concat([X_ohlcv_scaled, X_other_scaled], axis=1)

        y = X_ohlcv_scaled.shift(-1)  # Shift the target variable to predict the next time step
        
        X = X[:-1]
        y = y[:-1]
        
        self.X = X
        self.y = y

    ########################################
    ## TRAIN MODEL
    ########################################

    def _train_model(self):
        X = self.X.values.reshape((self.X.shape[0], 1, self.X.shape[1]))
        y = self.y

        model = models.Sequential()
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-8)
        early_stopping = callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, mode='auto', patience=3, restore_best_weights=True)
        inputs = layers.Input(shape=(X.shape[1], X.shape[2]))

        rnn = layers.Bidirectional(layers.LSTM(units=self.rnn_width, return_sequences=True))(inputs)
        rnn = layers.Bidirectional(layers.GRU(units=self.rnn_width, return_sequences=True))(rnn)
        rnn = layers.LSTM(units=self.rnn_width, return_sequences=True)(rnn)
        rnn = layers.GRU(units=self.rnn_width, return_sequences=True)(rnn)

        attention = layers.MultiHeadAttention(num_heads=13, key_dim=32)(rnn, rnn)

        dense = layers.Dense(self.dense_width, activation='relu')(attention)
        dense = layers.Dense(self.dense_width, activation='relu')(dense)
        dense = layers.Dense(self.dense_width, activation='relu')(dense)
        dense = layers.Dense(self.dense_width, activation='relu')(dense)
        dense = layers.Dense(self.dense_width, activation='relu')(dense)

        outputs = layers.Dense(5, activation='relu')(dense)  # Adjusted to output 5 features

        model = models.Model(inputs=inputs, outputs=outputs)
        lossfn = losses.Huber(delta=5.0)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=lossfn, metrics=['mean_squared_error'])
        model_data = model.fit(X, y, epochs=self.epochs, batch_size=64, callbacks=[early_stopping, reduce_lr])

        self.model_data = model_data
        return model

    ########################################
    ## INVERSE TRANSFORM PREDICTIONS
    ########################################

    def _inverse_transform_predictions(self, yhat):
        yhat = yhat.reshape((yhat.shape[0], yhat.shape[2]))  # Reshape yhat to (num_samples, num_features)
        yhat_inverse = self.ohlcv_scaler.inverse_transform(yhat)  # Inverse transform using ohlcv_scaler
        return yhat_inverse

    ########################################
    ## PREDICT
    ########################################

    def _predict(self, model):
        X = self.X.values.reshape((self.X.shape[0], 1, self.X.shape[1]))
        yhat = model.predict(X)
        self.yhat = self._inverse_transform_predictions(yhat)  # Use the new method

    ########################################
    ## ANALYZE
    ########################################

    def create_plot(self, show_graph=False):
        train_data = self.train_data
        y = self.y
        yhat = self.yhat
        model_data = self.model_data

        train_data.index = pd.to_datetime(train_data.index)
        train_data = train_data.iloc[:-1]
        train_data.reset_index(inplace=True)

        loss_history = model_data.history['loss']

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Price Prediction', f'Loss: {loss_history[-1]} | MSE: {mean_squared_error(y, yhat)}'))

        fig.add_trace(go.Scatter(x=train_data['Datetime'], y=train_data['Close'].squeeze(), mode='lines', name='True Close', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=train_data['Datetime'], y=yhat[:, 3], mode='lines', name='Predicted Close', line=dict(color='red')), row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(loss_history))), y=loss_history, mode='lines', name='Loss', line=dict(color='orange')), row=2, col=1)

        fig.update_layout(template='plotly_dark', title_text='Price Prediction and Model Loss')
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=1, row=2, col=1)
        
        if show_graph:
            fig.show()

        return fig

    def run(self, show_graph=False, save=False):
        self._fetch_data()
        self._add_features()
        self._prepare_data()
        model = self._train_model()
        print("Training complete")
        self._predict(model)  # Output is already inverse transformed

        if save: 
            model.save(f'{self.ticker}_{self.interval}_{model.count_params()}.keras', overwrite=True)
            print(f'Model Saved to {os.getcwd()}')
        return self.model_data, self.create_plot(show_graph=show_graph)

class ModelTesting(TimeSeriesPredictor):
    def __init__(self, ticker, chunks, interval, age_days):
        super().__init__(epochs=0, rnn_width=0, dense_width=0, ticker=ticker, chunks=chunks, interval=interval, age_days=age_days)
        self.model = None

    def _load_model(self, model_name):
        self.model = models.load_model(model_name)
        print(f"Using Model: {model_name}")
        self.model.summary()
        print(f"{self.model.loss}")
        return self.model

    def run(self, steps=1):
        self._fetch_data()
        self._add_features()
        self._prepare_data()
        self._predict(self.model)
        return self.create_plot()

# Example usage
model = TimeSeriesPredictor(epochs=25, rnn_width=1024, dense_width=1024, ticker='BTC-USD', chunks=1, interval='5m', age_days=10)
model.run(show_graph=True)