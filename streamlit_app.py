import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

# Streamlit page settings
st.set_page_config(page_title="Multi-Stock Price Predictor", layout="wide")
st.title("📈 Multi-Stock Price Prediction using LSTM")

# Sidebar input for multiple stocks
st.sidebar.header("Stock Configuration")

# Predefined list of popular stocks for ease of use; user can still add their own
popular_stocks = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'FB', 'NVDA', 'NFLX', 'INTC', 'AMD']
stock_symbols = st.sidebar.multiselect(
    "Select Stock Tickers (choose one or more):",
    options=popular_stocks,
    default=['AAPL']
)

# Allow custom ticker input if user wants to add more (comma-separated)
custom_tickers = st.sidebar.text_input("Or add custom tickers (comma separated):")
if custom_tickers:
    custom_list = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
    # Combine and deduplicate tickers
    stock_symbols = list(set(stock_symbols + custom_list))

start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('2024-12-31'))

# Time horizon selection
time_horizon = st.sidebar.selectbox(
    "Select Prediction Horizon:",
    options=['1 Day', '1 Week', '1 Month']
)

# Map time horizon to number of days ahead for prediction
horizon_map = {'1 Day': 1, '1 Week': 7, '1 Month': 30}

# LSTM sequence length
SEQ_LEN = 60

if st.sidebar.button("Predict"):
    if not stock_symbols:
        st.error("Please select or enter at least one stock ticker.")
    else:
        for stock_symbol in stock_symbols:
            st.header(f"🔹 {stock_symbol} Stock Analysis")

            with st.spinner(f"Fetching data for {stock_symbol}..."):
                # Fetch data, handle MultiIndex columns
                data = yf.download(stock_symbol, start=start_date, end=end_date, group_by='ticker')
                if isinstance(data.columns, pd.MultiIndex):
                    data = data[stock_symbol]

                if data.empty:
                    st.error(f"❌ No data found for {stock_symbol}. Check ticker or date range.")
                    continue

                # Show latest basic stock details
                latest = data.tail(1).squeeze()
                st.write({
                    "Open": f"${float(latest['Open']):.2f}",
                    "Close": f"${float(latest['Close']):.2f}",
                    "Volume": f"{int(latest['Volume'])}",
                    "Lower Circuit": f"${float(latest['Close']) * 0.95:.2f}",
                    "Upper Circuit": f"${float(latest['Close']) * 1.05:.2f}",
                    "Today's Low": f"${float(latest['Low']):.2f}",
                    "Today's High": f"${float(latest['High']):.2f}"
                })

                # Plot raw close price
                close_data = data[['Close']]
                st.subheader(f"📉 Raw Close Price Data for {stock_symbol}")
                st.line_chart(close_data)

                # Prepare data for LSTM
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(close_data)

                # Create sequences
                def create_sequences(dataset, seq_length):
                    x, y = [], []
                    for i in range(seq_length, len(dataset)):
                        x.append(dataset[i - seq_length:i, 0])
                        y.append(dataset[i, 0])
                    return np.array(x), np.array(y)

                x, y = create_sequences(scaled_data, SEQ_LEN)
                x = np.reshape(x, (x.shape[0], x.shape[1], 1))

                # Train-test split
                split = int(len(x) * 0.8)
                x_train, x_test = x[:split], x[split:]
                y_train, y_test = y[:split], y[split:]

                # Build LSTM model
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)))
                model.add(LSTM(50))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train model (quietly)
                model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

                # Predict on test data
                predictions = model.predict(x_test)
                predictions_actual = scaler.inverse_transform(predictions)
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Show prediction vs actual plot
                st.subheader(f"📊 Predicted vs Actual Prices for {stock_symbol}")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(y_test_actual, color='blue', label='Actual Price')
                ax.plot(predictions_actual, color='red', label='Predicted Price')
                ax.set_title(f'{stock_symbol} Stock Price Prediction')
                ax.set_xlabel('Time')
                ax.set_ylabel('Price')
                ax.legend()
                ax.grid()
                st.pyplot(fig)

                # RMSE & Accuracy
                rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
                accuracy = 100 - (rmse / np.mean(y_test_actual) * 100)
                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
                st.metric("Model Accuracy Estimate", f"{accuracy:.2f}%")

                # Predict for chosen horizon
                days_ahead = horizon_map[time_horizon]

                # To predict for multiple days ahead:
                # We'll iteratively predict next day, append it to input, then predict the next...
                last_seq = scaled_data[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
                predicted_scaled = []

                for _ in range(days_ahead):
                    pred = model.predict(last_seq)[0][0]
                    predicted_scaled.append(pred)
                    # Slide window forward with predicted value
                    last_seq = np.append(last_seq[:,1:,:], [[[pred]]], axis=1)

                predicted_prices = scaler.inverse_transform(np.array(predicted_scaled).reshape(-1, 1))

                if days_ahead == 1:
                    st.success(f"📅 Predicted {stock_symbol} price for next day ({time_horizon}): ${predicted_prices[0][0]:.2f}")
                else:
                    st.success(f"📅 Predicted {stock_symbol} prices for next {time_horizon}:")
                    pred_dates = pd.date_range(end_date + timedelta(days=1), periods=days_ahead)
                    pred_df = pd.DataFrame({
                        'Date': pred_dates,
                        'Predicted Close Price': predicted_prices.flatten()
                    }).set_index('Date')
                    st.line_chart(pred_df)
                    st.dataframe(pred_df.style.format({'Predicted Close Price': '${:.2f}'}))
