import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import date

# Configure Streamlit page layout
st.set_page_config(page_title="Stock Market Predictor", layout="wide")

# App Title
st.header('Stock Market Predictor')

# Sidebar: User Input
st.sidebar.header("User Input Parameters")
stock = st.sidebar.text_input("Enter Stock Symbol", "GOOG")
# Default dates: start date from 2015 and end date till March 1, 2025
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-03-01"))

# Load model (using caching for faster reloads)
@st.cache_resource
def load_stock_model():
    # Use a relative path to the model file for portability
    model_path = "Stock Predictions Model.keras"
    return load_model(model_path)

model = load_stock_model()

# Data fetching with caching (to speed up repeated runs)
@st.cache_data
def get_stock_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

with st.spinner("Fetching data..."):
    data = get_stock_data(stock, start_date, end_date)

if data.empty:
    st.error("No data fetched. Check the stock symbol or date range.")
    st.stop()

st.subheader('Stock Data')
st.write(data)

# Prepare training and testing data
train_size = int(len(data) * 0.80)
data_train = pd.DataFrame(data["Close"].iloc[:train_size])
data_test = pd.DataFrame(data["Close"].iloc[train_size:])

# Scale data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit on training data
_ = scaler.fit_transform(data_train)
# Use the last 100 days of training data as a base for test scaling
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)

# Calculate moving averages for visualization
ma_50 = data["Close"].rolling(window=50).mean()
ma_100 = data["Close"].rolling(window=100).mean()
ma_200 = data["Close"].rolling(window=200).mean()

# Plot: Price vs MA50
st.subheader('Price vs MA50')
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(data.index, data["Close"], color="green", label="Close Price")
ax1.plot(ma_50.index, ma_50, color="red", label="MA50")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)

# Plot: Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(data.index, data["Close"], color="green", label="Close Price")
ax2.plot(ma_50.index, ma_50, color="red", label="MA50")
ax2.plot(ma_100.index, ma_100, color="blue", label="MA100")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# Plot: Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(data.index, data["Close"], color="green", label="Close Price")
ax3.plot(ma_100.index, ma_100, color="red", label="MA100")
ax3.plot(ma_200.index, ma_200, color="blue", label="MA200")
ax3.set_xlabel("Date")
ax3.set_ylabel("Price")
ax3.legend()
st.pyplot(fig3)

# Prepare data for prediction with a rolling window of 100 days
x_test = []
y_test = []
for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i-100:i])
    y_test.append(data_test_scale[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Generate predictions
predictions = model.predict(x_test)

# Rescale predictions and ground truth values to the original scale
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y_test = y_test * scale_factor

# Plot: Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(predictions, color="red", label="Predicted Price")
ax4.plot(y_test, color="green", label="Original Price")
ax4.set_xlabel("Time")
ax4.set_ylabel("Price")
ax4.legend()
st.pyplot(fig4)
