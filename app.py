import datetime
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st
import time
from ta.utils import dropna

# Start timer
start_time = time.time()

# Title and Subtitle
st.title("Stock Price Prediction App")
st.subheader("Project made by Aditya, Ahil, Aneesh, and Farhan")

# Set the date range
start_date = datetime.date(2019, 10, 8)
end_date = datetime.date.today()

# Input for stock symbol and prediction days
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL): ")
days_to_predict = st.number_input("Days to Predict into the Future:", min_value=1, max_value=365, value=30)

if stock_symbol:
    if start_date > end_date:
        st.error("End Date cannot be earlier than Start Date!")
    else:
        try:
            # Download stock data
            data = yf.download(stock_symbol, start=start_date, end=end_date)

            if data.empty or len(data.dropna()) < 2:
                st.error("Not enough data available. Please adjust the date range or check the stock symbol.")
                st.stop()

        except (ConnectionError, ValueError) as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
        else:
            st.success("Data Fetched Successfully!")

            # Prepare data by removing NaN values
            data = dropna(data)
            data.reset_index(inplace=True)

            # Check data structure
            st.write("Data preview:", data.head())

            # Prepare data for Prophet
            data_prophet = data[['Date', 'Adj Close']].rename(columns={'Date': 'ds', 'Adj Close': 'y'})

            # Drop rows with missing values in the 'y' column
            data_prophet.dropna(subset=['y'], inplace=True)

            # Verify if data is in correct format
            if data_prophet['y'].empty:
                st.error("Data for prediction is missing or invalid. Please check the stock symbol or date range.")
                st.stop()

            # Train Prophet model
            model = Prophet()
            model.fit(data_prophet)

            # Make predictions
            future_dates = model.make_future_dataframe(periods=days_to_predict)
            forecast = model.predict(future_dates)

            # Plot results
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data_prophet['ds'], data_prophet['y'], label='Actual Data', color='black')
            ax.plot(forecast['ds'], forecast['yhat'], label='Predicted Data', color='blue')
            ax.set_title(f"Stock Price Prediction for {stock_symbol.upper()}", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stock Price (USD)", fontsize=12)
            ax.legend(loc='upper left')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Display historical data
            st.write("Historical Data:")
            st.dataframe(data)

            # Display forecast data
            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_table.columns = ['Date', 'Predicted Price (USD)', 'Lower Bound (USD)', 'Upper Bound (USD)']
            forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')
            st.write("Future Forecast:")
            st.dataframe(forecast_table)

            # Fetch additional stock information
            stock_info = yf.Ticker(stock_symbol).info
            st.write("Stock Information:")
            market_cap = stock_info.get('marketCap', 'N/A')
            dividend_yield = stock_info.get('dividendYield', 'N/A')
            total_revenue = stock_info.get('totalRevenue', 'N/A')
            earnings_growth = stock_info.get('earningsGrowth', 'N/A')

            st.write(f"**Market Cap:** {market_cap}")
            st.write(f"**Dividend Yield:** {dividend_yield}")
            st.write(f"**Total Revenue:** {total_revenue}")
            st.write(f"**Earnings Growth:** {earnings_growth}")

            # Display time taken to run
            end_time = time.time()
            time_taken = end_time - start_time
            st.write(f"Time taken to load the site and perform predictions: {time_taken:.2f} seconds")
