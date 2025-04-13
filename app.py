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

# NEW FEATURE 1: RSI (Relative Strength Index) Calculation and Visualization
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate RSI
rsi = calculate_rsi(data["Close"])

# Plot RSI - Fixed version without fill_between
st.subheader('Relative Strength Index (RSI)')
fig_rsi, ax_rsi = plt.subplots(figsize=(8, 4))
ax_rsi.plot(data.index, rsi, color="purple")
ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.5, label="Overbought (70)")
ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.5, label="Oversold (30)")
ax_rsi.set_xlabel("Date")
ax_rsi.set_ylabel("RSI")
ax_rsi.set_ylim(0, 100)
ax_rsi.legend()
st.pyplot(fig_rsi)

# RSI interpretation
st.write("""
**RSI Interpretation:**
- RSI > 70: The stock may be overbought (potential sell signal)
- RSI < 30: The stock may be oversold (potential buy signal)
- RSI = 50: Neutral market conditions
""")

# Current RSI value - FIXED
latest_rsi_value = float(rsi.iloc[-1])  # Convert Series to float
rsi_status = "Overbought" if latest_rsi_value > 70 else "Oversold" if latest_rsi_value < 30 else "Neutral"
st.metric("Current RSI", f"{latest_rsi_value:.2f}", delta=rsi_status)

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

# NEW FEATURE 2: Model Performance Metrics
def calculate_performance_metrics(y_true, y_pred):
    # Ensure arrays have the same shape
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 0.0001))) * 100
    
    # Fix for directional accuracy - handle the dimension issue
    try:
        # Ensure predictions is a 1D array
        y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        
        # Calculate directional movements
        if len(y_true_flat) > 1 and len(y_pred_flat) > 1:
            direction_true = np.sign(np.diff(y_true_flat))
            direction_pred = np.sign(np.diff(y_pred_flat))
            
            # Ensure both have the same length
            min_dir_len = min(len(direction_true), len(direction_pred))
            if min_dir_len > 0:
                matches = (direction_true[:min_dir_len] == direction_pred[:min_dir_len])
                dir_accuracy = np.mean(matches) * 100
            else:
                dir_accuracy = 0.0
        else:
            dir_accuracy = 0.0
    except Exception as e:
        st.warning(f"Error calculating directional accuracy: {e}")
        dir_accuracy = 0.0
    
    return {
        "Root Mean Squared Error (RMSE)": f"${rmse:.2f}",
        "Mean Absolute Error (MAE)": f"${mae:.2f}",
        "Mean Absolute Percentage Error (MAPE)": f"{mape:.2f}%",
        "Directional Accuracy": f"{dir_accuracy:.2f}%"
    }

# Debug information
st.write(f"Debug - y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
st.write(f"Debug - predictions shape: {predictions.shape}, dtype: {predictions.dtype}")

# Display performance metrics
st.subheader("Model Performance Metrics")
metrics = calculate_performance_metrics(y_test, predictions)

# Create a nicer display for metrics using columns
col1, col2 = st.columns(2)
with col1:
    st.metric("Root Mean Squared Error", metrics["Root Mean Squared Error (RMSE)"])
    st.metric("Mean Absolute Error", metrics["Mean Absolute Error (MAE)"])
with col2:
    st.metric("Mean Absolute Percentage Error", metrics["Mean Absolute Percentage Error (MAPE)"])
    st.metric("Directional Accuracy", metrics["Directional Accuracy"])

# Explanation of metrics
st.write("""
**Metrics Explanation:**
- **RMSE**: The standard deviation of prediction errors (lower is better)
- **MAE**: Average absolute difference between predicted and actual values (lower is better)
- **MAPE**: Percentage error, giving sense of error relative to the actual values
- **Directional Accuracy**: How often the model correctly predicts the direction of price movement (up/down)
""")

# Add an evaluation of the model's performance
avg_mape = float(metrics["Mean Absolute Percentage Error (MAPE)"].replace("%", ""))
avg_dir_acc = float(metrics["Directional Accuracy"].replace("%", ""))

if avg_dir_acc > 60:
    model_evaluation = "Good directional prediction capability"
elif avg_dir_acc > 50:
    model_evaluation = "Better than random chance for direction prediction"
else:
    model_evaluation = "Poor directional prediction capability"

if avg_mape < 5:
    price_accuracy = "Excellent price prediction accuracy"
elif avg_mape < 10:
    price_accuracy = "Good price prediction accuracy"
elif avg_mape < 20:
    price_accuracy = "Moderate price prediction accuracy"
else:
    price_accuracy = "Poor price prediction accuracy"

st.info(f"**Model Evaluation:** {model_evaluation}. {price_accuracy}.")