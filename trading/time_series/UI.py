import streamlit as st
import io
import os
import re
import numpy as np
import plotly.graph_objects as go

from ar_rnn import TimeSeriesPredictor, ModelTesting  # Import from ar_rnn.py

os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.sidebar.header('Adjust Train Parameters')  # TRAINING

epochs = st.sidebar.number_input('Epochs', min_value=1, max_value=1000, value=5)
rnn_width = st.sidebar.number_input('RNN Width', min_value=1, max_value=8192, value=128)
dense_width = st.sidebar.number_input('Dense Width', min_value=1, max_value=8192, value=128)

ticker = st.sidebar.selectbox('Ticker', options=['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD'])
chunks = st.sidebar.number_input('Chunks', min_value=1, max_value=100, value=7)
interval = st.sidebar.selectbox('Interval', options=['1m', '2m', '5m', '15m', '30m', '1h', '1d'], index=2)
age_days = st.sidebar.number_input('Age Days', min_value=1, max_value=365, value=3)

save_model = st.sidebar.checkbox('Save Model as .keras file')

if st.sidebar.button('Run Training'):
    with st.spinner("Training model...", show_time=True):
        model = TimeSeriesPredictor(epochs, rnn_width, dense_width, ticker, chunks, interval, age_days)
        data, yhat, model_data = model.run(save=save_model)
        print(data)
        print(data.isna().sum(), np.sum(np.isnan(yhat)))
        st.success('Model training complete!')
        st.plotly_chart(model.create_plot(data, yhat, model_data))

st.sidebar.header('Adjust Test Parameters')  # TESTING

# Regex search for .keras files in the CWD
directory_files = os.listdir(os.getcwd())
keras_files = [f for f in directory_files if re.search(r'\.keras$', f)]
test_model = st.sidebar.selectbox('Select Model', keras_files)
extensions = st.sidebar.number_input('Intervals To Extend', min_value=1, max_value=100, value=10)


if st.sidebar.button('Test Model'):
    with st.spinner("Testing model...", show_time=True):
        test_client = ModelTesting(ticker='BTC-USD', chunks=1, interval='5m', age_days=0)
        test_client.load_model(model_name=test_model)
        test_client.run(extension=extensions)  # Ensure the run method is called correctly
        plotly_figure = test_client.create_test_plot()

        st.plotly_chart(plotly_figure)

        summary_str = io.StringIO()
        test_client.model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
        summary_str = summary_str.getvalue()

        st.markdown("<h3 style='color: #9b59b6; font-weight: bold;'>Model Summary</h3>", unsafe_allow_html=True)
        summary_lines = summary_str.split('\n')
        for line in summary_lines:
            if any(char.isalnum() for char in line.strip()):
                st.write(line)
