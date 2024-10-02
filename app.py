import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests

# Set page config
st.set_page_config(page_title="Advanced Stock Market Analyzer", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #1a1a1a;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        color: #ffffff;
        background-color: #2b2b2b;
    }
    .stPlotlyChart {
        background-color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# Function to fetch historical data
def fetch_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Function to calculate technical indicators
def calculate_indicators(data):
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].diff(1).fillna(0).rolling(window=14).mean() / data['Close'].diff(1).fillna(0).abs().rolling(window=14).mean()))
    return data

# Function to convert USD to INR
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_usd_to_inr_rate():
    url = "https://api.exchangerate-api.com/v4/latest/USD"
    response = requests.get(url)
    data = response.json()
    return data["rates"]["INR"]

# Function to format currency in INR
def format_inr(amount):
    return f"₹{amount:,.2f}"

# ML-based price prediction
def predict_price(data):
    if len(data) < 101:  # We need at least 101 data points for prediction
        return None
    
    df = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)
    
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)
    
    time_step = 100
    X, y = create_dataset(df_scaled, time_step)
    
    if len(X) == 0 or len(y) == 0:
        return None
    
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    model.fit(X, y, validation_split=0.1, epochs=50, batch_size=64, verbose=0)
    
    last_100_days = df_scaled[-100:]
    X_test = np.array([last_100_days])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    
    return pred_price[0][0]

# Sidebar for user input
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
analyze_button = st.sidebar.button("Analyze Stock")

# Main page
st.title(f"Advanced Stock Market Analyzer - {ticker}")

# Display real-time stock price
real_time_data = yf.Ticker(ticker).history(period="1d")
if not real_time_data.empty:
    current_price = real_time_data['Close'].iloc[-1]
    usd_to_inr_rate = get_usd_to_inr_rate()
    current_price_inr = current_price * usd_to_inr_rate
    st.metric("Current Price", format_inr(current_price_inr))

if analyze_button:
    # Fetch and process data
    data = fetch_data(ticker)
    data = calculate_indicators(data)
    
    # Overview section
    st.header("Stock Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Open", format_inr(data['Open'].iloc[-1] * usd_to_inr_rate))
    col2.metric("High", format_inr(data['High'].iloc[-1] * usd_to_inr_rate))
    col3.metric("Low", format_inr(data['Low'].iloc[-1] * usd_to_inr_rate))

    # Historical price chart
    st.subheader("Historical Price Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'] * usd_to_inr_rate, label='Close Price', color='#4CAF50')
    ax.set_facecolor('#0e1117')
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('Price (INR)', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend(facecolor='#1a1a1a', edgecolor='#ffffff')
    fig.patch.set_facecolor('#0e1117')
    st.pyplot(fig)

    # Technical indicators
    st.subheader("Technical Indicators")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(data.index, data['Close'] * usd_to_inr_rate, label='Close Price', color='#4CAF50')
    ax1.plot(data.index, data['SMA50'] * usd_to_inr_rate, label='50-day SMA', color='#FFA500')
    ax1.plot(data.index, data['SMA200'] * usd_to_inr_rate, label='200-day SMA', color='#FF69B4')
    ax1.set_ylabel('Price (INR)', color='white')
    ax1.set_facecolor('#0e1117')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.legend(facecolor='#1a1a1a', edgecolor='#ffffff')

    ax2.plot(data.index, data['RSI'], label='RSI', color='#00BFFF')
    ax2.axhline(y=70, color='#FF6347', linestyle='--')
    ax2.axhline(y=30, color='#32CD32', linestyle='--')
    ax2.set_ylabel('RSI', color='white')
    ax2.set_xlabel('Date', color='white')
    ax2.set_facecolor('#0e1117')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.legend(facecolor='#1a1a1a', edgecolor='#ffffff')

    fig.patch.set_facecolor('#0e1117')
    st.pyplot(fig)

    # Buy/sell recommendation
    st.subheader("Buy/Sell Recommendation")
    last_price = data['Close'].iloc[-1]
    sma50 = data['SMA50'].iloc[-1]
    sma200 = data['SMA200'].iloc[-1]
    rsi = data['RSI'].iloc[-1]

    if last_price > sma50 > sma200 and 30 <= rsi <= 70:
        recommendation = "Buy"
        explanation = "The stock is in an uptrend with the price above both SMAs, and RSI indicates a balanced market."
    elif last_price < sma50 < sma200 and rsi > 70:
        recommendation = "Sell"
        explanation = "The stock is in a downtrend with the price below both SMAs, and RSI indicates overbought conditions."
    else:
        recommendation = "Hold"
        explanation = "The current market conditions are mixed. Consider waiting for a clearer signal."

    st.write(f"Recommendation: {recommendation}")
    st.write(f"Explanation: {explanation}")

    # Volume analysis
    st.subheader("Volume Analysis")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(data.index, data['Volume'], color='#4CAF50')
    ax.set_ylabel('Volume', color='white')
    ax.set_xlabel('Date', color='white')
    ax.set_facecolor('#0e1117')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    fig.patch.set_facecolor('#0e1117')
    st.pyplot(fig)

    # ML-based price prediction
    st.subheader("ML-based Price Prediction")
    predicted_price = predict_price(data)
    if predicted_price is not None:
        predicted_price_inr = predicted_price * usd_to_inr_rate
        st.write(f"Predicted price for next day: {format_inr(predicted_price_inr)}")
    else:
        st.write("Not enough data for price prediction. Please try a stock with more historical data.")

# Add some space at the bottom
st.write("")
st.write("")
st.write("Data provided by Yahoo Finance. This is not financial advice. Please do your own research before making investment decisions.")