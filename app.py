import datetime
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import streamlit as st

# Set up Streamlit app title
st.title("Stock Price Prediction App")

# Set up date inputs
start_date = datetime.date(2019, 10, 8)
end_date = datetime.date.today()

# User inputs for stock symbol
stock_symbol = st.text_input("Enter Stock Symbol (e.g., STOCKNAME.NS): ")

# Days to predict into the future
days_to_predict = 365

# Function to fetch data
def fetch_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        if data.empty or len(data.dropna()) < 2:
            st.error("Not enough data available. Please adjust the date range or check the stock symbol.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to prepare and scale data
def preprocess_data(data):
    data.reset_index(inplace=True)
    data_lstm = data[['Date', 'Adj Close']].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_lstm['Adj Close'].values.reshape(-1, 1))
    return data_lstm, scaled_data, scaler

# Function to create dataset for LSTM model
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to plot predictions
def plot_predictions(data, train_predict, test_predict, time_step, scaler):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Adj Close'], label='Actual Data', color='black')

    train_predict_plot = np.empty_like(scaler.inverse_transform(train_predict))
    train_predict_plot[:, :] = np.nan
    train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

    test_predict_plot = np.empty_like(scaler.inverse_transform(test_predict))
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaler.inverse_transform(test_predict)) - 1, :] = test_predict

    plt.plot(data['Date'][:len(train_predict_plot)], train_predict_plot, label='Train Prediction', color='blue')
    plt.plot(data['Date'][len(train_predict_plot):], test_predict_plot, label='Test Prediction', color='red')

    plt.title(f"Stock Price Prediction for {stock_symbol.upper()}")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Function to make future predictions
def predict_future(model, scaled_data, scaler, days_to_predict):
    last_60_days = scaled_data[-60:]
    future_input = last_60_days.reshape(1, -1)
    future_input = future_input.reshape((1, 60, 1))

    future_predictions = []
    for _ in range(days_to_predict):
        next_pred = model.predict(future_input)
        future_predictions.append(next_pred[0][0])
        future_input = np.append(future_input[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# Main execution block
if stock_symbol:
    if start_date > end_date:
        st.error("End Date cannot be earlier than Start Date!")
    else:
        data = fetch_data(stock_symbol, start_date, end_date)
        if data is not None:
            st.success("Data Fetched Successfully!")
            data_lstm, scaled_data, scaler = preprocess_data(data)

            # Create training and testing datasets
            train_data = scaled_data[:int(0.8 * len(data_lstm))]
            test_data = scaled_data[int(0.8 * len(data_lstm)):]
            time_step = 60
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, y_test = create_dataset(test_data, time_step)

            # Reshape input to be [samples, time steps, features]
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Build and train the LSTM model
            model = build_lstm_model((time_step, 1))
            model.fit(X_train, y_train, batch_size=1, epochs=1)

            # Generate predictions
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)
            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)

            # Plot predictions
            plot_predictions(data_lstm, train_predict, test_predict, time_step, scaler)

            # Display historical data
            st.write("Historical Data:")
            st.dataframe(data)

            # Predict future stock prices for 365 days
            future_predictions = predict_future(model, scaled_data, scaler, days_to_predict)
            future_dates = pd.date_range(start=end_date, periods=days_to_predict + 1).tolist()
            forecast_table = pd.DataFrame(future_predictions, columns=['Predicted Price'])
            forecast_table['Date'] = future_dates[1:]  # Exclude the end_date

            st.write("Future Forecast for 365 Days:")
            st.dataframe(forecast_table)
