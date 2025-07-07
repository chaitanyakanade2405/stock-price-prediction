import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import date

# Configure Streamlit page layout
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.header('Stock Price Predictor')

# Sidebar: Market and Stock Symbol Input
st.sidebar.header("User Input Parameters")
exchange = st.sidebar.selectbox("Select Stock Exchange", ["NASDAQ/NYSE", "BSE (India)"])
stock_input = st.sidebar.text_input("Enter Stock Symbol", "GOOG")

# Format the stock symbol
if exchange == "BSE (India)":
    stock = f"{stock_input.upper()}.BO"
    st.sidebar.caption("Examples: RELIANCE, TCS, INFY, HDFCBANK")
else:
    stock = stock_input.upper()

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-03-01"))

# Load model
@st.cache_resource
def load_stock_model():
    model_path = "Stock Predictions Model.keras"
    return load_model(model_path)

model = load_stock_model()

# Fetch stock data with error handling
@st.cache_data
def get_stock_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

with st.spinner("Fetching data..."):
    data = get_stock_data(stock, start_date, end_date)

if data.empty or 'error' in str(data).lower():
    st.error("⚠️ No data fetched. This may be due to an invalid symbol, wrong date range, or Yahoo Finance rate limiting. Try again after a few minutes.")
    st.stop()

# Optional: Show company name safely
@st.cache_data
def get_stock_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info
    except Exception as e:
        st.warning(f"Could not retrieve stock info: {e}")
        return {}

info = get_stock_info(stock)
st.write(f"**Company Name:** {info.get('longName', 'N/A')}")
st.write(f"**Showing data for:** `{stock}`")

# Show data
st.subheader('Stock Data')
st.write(data)

# Preprocess data
train_size = int(len(data) * 0.80)
data_train = pd.DataFrame(data["Close"].iloc[:train_size])
data_test = pd.DataFrame(data["Close"].iloc[train_size:])

scaler = MinMaxScaler(feature_range=(0, 1))
_ = scaler.fit_transform(data_train)

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)

# Moving averages
ma_50 = data["Close"].rolling(window=50).mean()
ma_100 = data["Close"].rolling(window=100).mean()
ma_200 = data["Close"].rolling(window=200).mean()

# Price vs MA50
st.subheader('Price vs MA50')
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(data.index, data["Close"], color="green", label="Close Price")
ax1.plot(ma_50.index, ma_50, color="red", label="MA50")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)

# Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(data.index, data["Close"], color="green", label="Close Price")
ax2.plot(ma_50.index, ma_50, color="red", label="MA50")
ax2.plot(ma_100.index, ma_100, color="blue", label="MA100")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(data.index, data["Close"], color="green", label="Close Price")
ax3.plot(ma_100.index, ma_100, color="red", label="MA100")
ax3.plot(ma_200.index, ma_200, color="blue", label="MA200")
ax3.set_xlabel("Date")
ax3.set_ylabel("Price")
ax3.legend()
st.pyplot(fig3)

# RSI Calculation
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi = calculate_rsi(data["Close"])

# RSI Plot
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

st.write("""
**RSI Interpretation:**
- RSI > 70: The stock may be overbought (potential sell signal)
- RSI < 30: The stock may be oversold (potential buy signal)
- RSI = 50: Neutral market conditions
""")

# Show current RSI
latest_rsi_value = float(rsi.iloc[-1])
rsi_status = "Overbought" if latest_rsi_value > 70 else "Oversold" if latest_rsi_value < 30 else "Neutral"
st.metric("Current RSI", f"{latest_rsi_value:.2f}", delta=rsi_status)

# Prepare data for prediction
x_test = []
y_test = []
for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i-100:i])
    y_test.append(data_test_scale[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
predictions = model.predict(x_test)

# Rescale
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y_test = y_test * scale_factor

# Price prediction plot
st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(predictions, color="red", label="Predicted Price")
ax4.plot(y_test, color="green", label="Original Price")
ax4.set_xlabel("Time")
ax4.set_ylabel("Price")
ax4.legend()
st.pyplot(fig4)

# Model performance metrics
def calculate_performance_metrics(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 0.0001))) * 100
    try:
        y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        direction_true = np.sign(np.diff(y_true_flat))
        direction_pred = np.sign(np.diff(y_pred_flat))
        min_dir_len = min(len(direction_true), len(direction_pred))
        if min_dir_len > 0:
            matches = (direction_true[:min_dir_len] == direction_pred[:min_dir_len])
            dir_accuracy = np.mean(matches) * 100
        else:
            dir_accuracy = 0.0
    except:
        dir_accuracy = 0.0
    return {
        "Root Mean Squared Error (RMSE)": f"${rmse:.2f}",
        "Mean Absolute Error (MAE)": f"${mae:.2f}",
        "Mean Absolute Percentage Error (MAPE)": f"{mape:.2f}%",
        "Directional Accuracy": f"{dir_accuracy:.2f}%"
    }

# Display metrics
st.subheader("Model Performance Metrics")
metrics = calculate_performance_metrics(y_test, predictions)

col1, col2 = st.columns(2)
with col1:
    st.metric("Root Mean Squared Error", metrics["Root Mean Squared Error (RMSE)"])
    st.metric("Mean Absolute Error", metrics["Mean Absolute Error (MAE)"])
with col2:
    st.metric("Mean Absolute Percentage Error", metrics["Mean Absolute Percentage Error (MAPE)"])
    st.metric("Directional Accuracy", metrics["Directional Accuracy"])

st.write("""
**Metrics Explanation:**
- **RMSE**: The standard deviation of prediction errors (lower is better)
- **MAE**: Average absolute difference between predicted and actual values (lower is better)
- **MAPE**: Percentage error relative to actual values (lower is better)
- **Directional Accuracy**: How often the model correctly predicts the price direction (higher is better)
""")

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
