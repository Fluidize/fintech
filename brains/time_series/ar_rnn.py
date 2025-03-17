import yfinance as yf
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import layers, models, optimizers, callbacks, losses, regularizers, backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timedelta
import os
from rich import print

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

def check_for_nan(data, stage):
    nan_count = np.sum(np.isnan(data))
    print(f"NaNs found at {stage}: {nan_count}")
    return nan_count

class TimeSeriesPredictor:
    def __init__(self, epochs, rnn_width, dense_width, ticker='BTC-USD', chunks=5, interval='5m', age_days=10):
        self.epochs = epochs
        self.rnn_width = rnn_width
        self.dense_width = dense_width
        self.ticker = ticker
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.scalers = {
            'price': MinMaxScaler(feature_range=(0, 1)),
            'volume': QuantileTransformer(output_distribution='normal'),
            'technical': StandardScaler()
        }
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def _fetch_data(self, ticker, chunks, interval, age_days):
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

    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _prepare_data(self, data, train_split=True):
        df = data.copy()
        df = df.drop(columns=['MA50', 'MA20', 'MA10', 'RSI'], errors='ignore')

        # Price-based features
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Create sequences of previous values
        self.lagged_length = 5
        print(f"Creating features for sequence length: {self.lagged_length}")
        
        # Create all lagged features at once using pd.concat
        lagged_features = []
        for col in ['Close', 'Volume', 'High', 'Low', 'Open']:
            for i in range(1, self.lagged_length):
                lagged_features.append(pd.DataFrame({
                    f'Prev{i}_{col}': df[col].shift(i)
                }))
        
        # Concatenate all lagged features at once
        if lagged_features:
            df = pd.concat([df] + lagged_features, axis=1)
        
        std = df['Close'].std()
        df['Close_ZScore'] = (df['Close'] - df['Close'].mean()) / std #Z-score
        
        # Moving averages (normalized by current price)
        df['MA10'] = df['Close'].rolling(window=10).mean() / df['Close']
        df['MA20'] = df['Close'].rolling(window=20).mean() / df['Close']
        df['MA50'] = df['Close'].rolling(window=50).mean() / df['Close']
        df['MA10_MA20_Cross'] = df['MA10'] - df['MA20']
        
        # RSI and other technical indicators
        df['RSI'] = self._compute_rsi(df['Close'], 14)

        df.dropna(inplace=True)
        print(f"Features after creation: {df.columns.tolist()}")
        print(f"Total number of features: {len(df.columns)}")
        
        if train_split:
            # Group features by their characteristics
            price_features = ['Open', 'High', 'Low', 'Close']
            volume_features = ['Volume'] + [f'Prev{i}_Volume' for i in range(1, self.lagged_length)]
            bounded_features = ['RSI']  # Features that are already bounded (e.g., 0-100)
            normalized_features = ['MA10', 'MA20', 'MA50', 'Price_Range', 'MA10_MA20_Cross', 'Close_ZScore']
            
            # Remaining features need standardization
            technical_features = [col for col in df.columns 
                               if col not in (price_features + volume_features + bounded_features + 
                                            normalized_features + ['Datetime'])]
            
            print(f"Price features: {price_features}")
            print(f"Volume features: {volume_features}")
            print(f"Technical features: {technical_features}")
            
            # Scale absolute price values with MinMaxScaler
            df[price_features] = self.scalers['price'].fit_transform(df[price_features])
            
            # Transform volume features to normal distribution
            df[volume_features] = df[volume_features].replace([np.inf, -np.inf], np.nan)
            df[volume_features] = df[volume_features].fillna(df[volume_features].mean())
            df[volume_features] = self.scalers['volume'].fit_transform(df[volume_features])
            
            # Standardize technical features that aren't already normalized
            if technical_features:
                df[technical_features] = self.scalers['technical'].fit_transform(df[technical_features])
            
            X = df.drop(['Datetime'], axis=1)
            y = df[['Open', 'High', 'Low', 'Close', 'Volume']].shift(-1)  # Predict next OHLCV
            
            X = X[:-1]  # Remove last row since we don't have target for it
            y = y[:-1]  # Remove last row since we don't have target for it
            print(f"Final X shape: {X.shape}")
            return X, y
        
        return df

    def _train_model(self, X, y):
        self.sequence_length = 25  # Sequence length for temporal information
        # Create sequences by sliding window
        n_samples = X.shape[0] - self.sequence_length + 1
        n_features = X.shape[1]
        
        # Create sequences
        X_sequences = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)
        for i in range(n_samples):
            X_sequences[i] = X[i:i + self.sequence_length]
        
        # Adjust y to match sequence predictions
        y = y[self.sequence_length - 1:]
        
        print(f"Training shapes - X: {X_sequences.shape}, y: {y.shape}")
        
        l1_reg = 1e-4
        l2_reg = 1e-4
        self.rnn_count = 3
        self.dense_count = 5
        embedding_dim = 32  # Size of embedding space
        num_heads = X_sequences.shape[1] #feature count
        ff_dim = 64       # Feed-forward network dimension
        growth_rate = 16  # Number of features added by each dense layer

        inputs = layers.Input(shape=(X_sequences.shape[1], X_sequences.shape[2]), name='input_layer')
    
        x = layers.Dense(embedding_dim, name='embedding_projection')(inputs)
        
        transformer_outputs = [x]
        for i in range(2):
            concat_inputs = layers.Concatenate(name=f'transformer_{i}_concat')(transformer_outputs)
            self_attention = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embedding_dim//num_heads,
                name=f'transformer_{i}_self_attention'
            )(
                query=concat_inputs,
                key=concat_inputs,
                value=concat_inputs
            )
            
            if i > 0:
                cross_attention = layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embedding_dim//num_heads,
                    name=f'transformer_{i}_cross_attention'
                )(
                    query=self_attention,
                    key=transformer_outputs[0],
                    value=transformer_outputs[0]
                )
                self_attention = layers.Add(name=f'transformer_{i}_cross_merge')([self_attention, cross_attention])
            
            # Position-wise feed-forward network
            ffn = layers.Dense(ff_dim, activation="relu", name=f'transformer_{i}_ffn_1')(self_attention) #expand to higher dimension and compress back
            ffn = layers.Dense(embedding_dim, name=f'transformer_{i}_ffn_2')(ffn)
            
            transformer_outputs.append(ffn)
        x = layers.Concatenate(name='transformer_final_concat')(transformer_outputs)
        
        gru_outputs = [x]
        for i in range(self.rnn_count):
            concat_inputs = layers.Concatenate(axis=-1, name=f'gru_{i}_concat')(gru_outputs)
            
            gru_out = layers.Bidirectional(layers.GRU(
                    units=self.rnn_width,
                    return_sequences=True,
                    kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                    name=f'gru_layer_{i}'
            ), name=f'bidirectional_gru_{i}')(concat_inputs)
            
            avg_pool = layers.GlobalAveragePooling1D(name=f'avg_pool_{i}')(gru_out)
            max_pool = layers.GlobalMaxPooling1D(name=f'max_pool_{i}')(gru_out)
            
            gru_out = layers.Concatenate(axis=-1, name=f'gru_features_{i}')([
                gru_out,
                layers.RepeatVector(self.sequence_length)(avg_pool),
                layers.RepeatVector(self.sequence_length)(max_pool)
            ])
            
            gru_outputs.append(gru_out)
        final_outputs = []
        for output in gru_outputs:
            avg_features = layers.GlobalAveragePooling1D(name=f'final_avg_pool_{len(final_outputs)}')(output)
            max_features = layers.GlobalMaxPooling1D(name=f'final_max_pool_{len(final_outputs)}')(output)
            final_outputs.extend([avg_features, max_features])
        x = layers.Concatenate(axis=-1, name='gru_final_concat')(final_outputs)
        
        dense_outputs = [x]
        for i in range(self.dense_count): 
            concat_inputs = layers.Concatenate(name=f'dense_{i}_concat')(dense_outputs)
            x = layers.Dense(growth_rate, name=f'dense_layer_{i}')(concat_inputs)
            x = layers.LeakyReLU(name=f'leaky_relu_layer_{i}')(x)
            
            dense_outputs.append(x)
        x = layers.Concatenate(name='final_concat')(dense_outputs)
        
        x = layers.Dense(self.dense_width, activation='relu', name='transition_layer')(x)
        
        outputs = layers.Dense(5, name='output_layer')(x)
        
        lossfn = losses.Huber(delta=5.0)
        optimizer = optimizers.Adam(learning_rate=1e-5)
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                  restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                      factor=0.5, 
                                      patience=7,
                                      min_lr=1e-6)
        ]
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer,
                     loss=lossfn,
                     metrics=['mae'])
        
        model_data = model.fit(X_sequences, y, 
                             epochs=self.epochs,
                             batch_size=32,
                             validation_split=0.1,
                             callbacks=callbacks_list,
                             verbose=1)
        
        print("\nTraining Results:")
        print(f"Final loss: {model_data.history['loss'][-1]:.6f}")
        print(f"Final validation loss: {model_data.history['val_loss'][-1]:.6f}")
        
        return model, model_data

    def _predict(self, model, X):
        n_samples = X.shape[0] - self.sequence_length + 1
        n_features = X.shape[1]
        
        X_sequences = np.zeros((n_samples, self.sequence_length, n_features))
        for i in range(n_samples):
            X_sequences[i] = X[i:i + self.sequence_length]
        
        yhat = model.predict(X_sequences)
        yhat = yhat.squeeze()
        
        price_predictions = yhat[:, :4]
        volume_predictions = yhat[:, 4].reshape(-1, 1)
        volume_predictions_extended = np.zeros((volume_predictions.shape[0], self.lagged_length))
        volume_predictions_extended[:, 0] = volume_predictions.squeeze()

        price_pred = self.scalers['price'].inverse_transform(price_predictions)
        volume_pred = self.scalers['volume'].inverse_transform(volume_predictions_extended)[:, 0]

        final_pred = np.column_stack((price_pred, volume_pred))
        
        print(f"Prediction shape: {final_pred.shape}")
        print(f"Price prediction range: min={np.min(final_pred[:, :4]):.2f}, max={np.max(final_pred[:, :4]):.2f}")
        print(f"Volume prediction range: min={np.min(final_pred[:, 4]):.2f}, max={np.max(final_pred[:, 4]):.2f}")
        return final_pred

    def create_plot(self, data, yhat, model_data, show_graph=False, save_image=False):
        df = self._prepare_data(data.copy(), train_split=False)
        actual_prices = df['Close'].values[self.sequence_length-1:-1]
        dates = df['Datetime'].values[self.sequence_length-1:-1]
        
        mse = mean_squared_error(actual_prices, yhat[:, 3])  # Compare with Close prices
        mape = np.mean(np.abs((actual_prices - yhat[:, 3]) / actual_prices)) * 100
        
        layer_info = []
        layers_per_line = 4

        layer_names = {
            'InputLayer': 'In',
            'Dense': 'D',
            'GRU': 'GRU',
            'Bidirectional': 'Bi',
            'MultiHeadAttention': 'MHA',
            'Add': '+',
            'Concatenate': 'Cat',
            'LeakyReLU': 'LReLU',
            'BatchNormalization': 'BN'
        }
        
        # Group layers by type
        for layer, index in zip(model_data.model.layers, range(len(model_data.model.layers))):
            layer_name = layer.__class__.__name__
            
            # Add line break every few layers
            if index % layers_per_line == 0 and index > 0:
                layer_info.append("<br>")
            
            # Format layer info based on type
            if layer_name == 'Dense':
                layer_info.append(f"D{layer.units}")  # D128 instead of Dense(128)
            elif layer_name == 'GRU':
                layer_info.append(f"BiGRU{layer.units}")  # BiGRU256 instead of BiGRU(256)
            elif layer_name == 'MultiHeadAttention':
                layer_info.append(f"MHA{layer.num_heads}")  # MHA4 instead of MHA(4)
            else:
                short_name = layer_names.get(layer_name, layer_name)
                layer_info.append(short_name)
        
        # Create architecture text with sections
        architecture_text = (
            "<b>Architecture:</b><br>" +
            " → ".join(layer_info) +
            "<br><br><b>Params:</b><br>" +
            f"RNN: {self.rnn_width} | Dense: {self.dense_width}<br>" +
            f"Total: {model_data.model.count_params():,}<br><br>" +
            f"<b>Training:</b><br>" +
            f"Batch: 32 | LR: 1e-5 | Seq: {self.sequence_length}"
        )
        
        fig = make_subplots(rows=3, cols=1, 
                          shared_xaxes=True, 
                          vertical_spacing=0.05,
                          subplot_titles=('Price Prediction', 
                                        f'Percent Prediction Error (MSE: {mse:.2f}, MAPE: {mape:.2f}%)',
                                        f'Training Loss History (Final Loss: {model_data.history["loss"][-1]:.6f})'))

        # Price prediction plot
        fig.add_trace(go.Scatter(x=dates, y=actual_prices, 
                               mode='lines', name='Actual Close', 
                               line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=yhat[:, 3], 
                               mode='lines', name='Predicted Close', 
                               line=dict(color='red')), row=1, col=1)

        # Error plot
        percent_error = (np.abs(actual_prices - yhat[:, 3]) / actual_prices) * 100
        fig.add_trace(go.Scatter(x=dates, y=percent_error, 
                               mode='lines', name='Prediction Error', 
                               line=dict(color='orange')), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # Loss history plot
        fig.add_trace(go.Scatter(x=list(range(len(model_data.history['loss']))), 
                               y=model_data.history['loss'],
                               mode='lines', name='Training Loss', 
                               line=dict(color='green')), row=3, col=1)

        # Add architecture text box in the loss plot
        fig.add_annotation(
            x=1, #align better with loss plot
            y=10,
            xref='paper',
            yref='paper',
            text=architecture_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='white',
            borderwidth=1,
            align='left',
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
                         template='plotly_dark', 
                         showlegend=True, 
                         title_text='Cryptocurrency Price Prediction Analysis'
                         )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Error", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=3, col=1)
        
        if show_graph:
            fig.show()
        if save_image:
            fig.write_image(f"images/{self.ticker}_{self.interval}_{model_data.model.count_params()}.png")
        return fig

    def run(self, save=False):
        data = self._fetch_data(self.ticker, self.chunks, self.interval, self.age_days)
        X, y = self._prepare_data(data)
        model, model_data = self._train_model(X, y)
        yhat = self._predict(model, X)
        if save: 
            model.save(f'{self.ticker}_{self.interval}_{model.count_params()}.keras', overwrite=True)
            print(f'Model Saved to {os.getcwd()}')
        return data, yhat, model_data

class ModelTesting(TimeSeriesPredictor):
    def __init__(self, ticker, chunks, interval, age_days):
        super().__init__(epochs=0, rnn_width=0, dense_width=0, ticker=ticker, chunks=chunks, interval=interval, age_days=age_days)
        self.model = None #doesn't exist in TimeSeriesPredictor

    def load_model(self, model_name):
        self.model = models.load_model(model_name) #models is a keras function
        self.model.summary()
        self.model_filename = model_name #store it to get interval

    def create_test_plot(self, data, yhat, show_graph=False):
        fig = go.Figure() 

        fig.add_trace(go.Scatter(x=yhat['Datetime'], y=data['Close'].squeeze(), mode='lines', name='True Close', line=dict(color='blue'), connectgaps=True))
        fig.add_trace(go.Scatter(x=yhat['Datetime'], y=yhat['Close'], mode='lines', name='Predicted Close', line=dict(color='red'), connectgaps=True))

        fig.update_layout(template='plotly_dark', title_text='Price Prediction')
        
        if show_graph:
            fig.show()

        return fig

    def _extended_predict(self, model, data, interval, extension=10):
        # PARSE INTO A TIMEDELTA
        number = int(interval[:-1])  # Take all characters except the last one as the number
        unit = interval[-1]  # Last character will be the unit (e.g., 'm', 'h')

        if unit == 'm':
            extended_time = timedelta(minutes=number)
        elif unit == 'h':
            extended_time = timedelta(hours=number)
        elif unit == 'd':
            extended_time = timedelta(days=number)
        else:
            raise ValueError("Unsupported unit, please use 'm' for minutes or 'h' for hours.")
          
        original_data = data.copy()
        predicted_data = data

        for i in range(extension):
            # Use only the last row for prediction
            X, _ = self._prepare_data(predicted_data)
            yhat = self._predict(model, X)
            
            new_data = pd.DataFrame(yhat[-1].reshape(1, -1), columns=original_data.columns[1:6]) #only ohlcv
            new_data['Datetime'] = predicted_data['Datetime'].iloc[-1] + extended_time
            predicted_data = pd.concat([predicted_data, new_data], axis=0)
            print(predicted_data)

        return original_data, predicted_data

    def run(self, extension=10):
        model_interval = self.model_filename.split("_")[1]
        data = self._fetch_data(self.ticker, self.chunks, self.interval, self.age_days)
        data = self._extended_predict(self.model, data, model_interval, extension)
        return data

test_client = TimeSeriesPredictor(epochs=15, rnn_width=256, dense_width=128, ticker='BTC-USD', chunks=3, interval='1m', age_days=0)
data, yhat, model_data = test_client.run(save=False)
test_client.create_plot(data, yhat, model_data, show_graph=True)

# Extended prediction testing
# test_client = ModelTesting(ticker='BTC-USD', chunks=1, interval='5m', age_days=0)
# test_client.load_model(model_name="best_model.keras")
# original_data, predicted_data = test_client.run(extension=100)
# test_client.create_test_plot(original_data, predicted_data, show_graph=True)