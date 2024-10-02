# Advanced Stock Market Analyzer

## Project Overview

The **Advanced Stock Market Analyzer** is a powerful tool designed to provide real-time stock analysis, technical indicator insights, and machine learning-based price predictions. It converts stock prices from USD to INR and generates buy/sell recommendations based on technical indicators such as the Simple Moving Averages (SMA) and the Relative Strength Index (RSI). The tool is built using Python and Streamlit to create an interactive and user-friendly interface.

## Features

- **Real-time Stock Data**: Fetches historical and live stock price data using Yahoo Finance.
- **Technical Indicators**:
  - **SMA (Simple Moving Averages)**: 50-day and 200-day SMAs.
  - **RSI (Relative Strength Index)**: Indicates overbought/oversold market conditions.
- **ML-based Price Prediction**: Uses an LSTM (Long Short-Term Memory) neural network to predict stock prices based on historical data.
- **Currency Conversion**: Automatically converts stock prices from USD to INR.
- **Buy/Sell Recommendations**: Provides recommendations based on SMA and RSI analysis.
- **Volume Analysis**: Visualizes stock trading volume over time.
- **Custom UI**: Dark-themed UI with interactive charts for better visualization.

## How It Works

1. **Data Fetching**: The user inputs a stock ticker (e.g., `AAPL`), and the app fetches the stock's historical data and real-time price from Yahoo Finance.
2. **Technical Analysis**: The tool calculates 50-day and 200-day SMAs, RSI, and other indicators based on the stock’s historical data.
3. **Machine Learning**: A pre-trained LSTM model predicts future stock prices based on the last 100 days of historical data.
4. **Visualization**: Data is presented in an easy-to-understand format using charts and graphs.
5. **Buy/Sell Recommendation**: The tool generates recommendations based on the relationship between current stock price, SMAs, and RSI.

## Technologies Used

- **Frontend**: Streamlit for building interactive web apps.
- **Backend**: 
  - **Yahoo Finance API (`yfinance`)**: To fetch real-time and historical stock price data.
  - **Requests**: For live currency exchange rates.
- **Data Processing**: 
  - **Pandas & Numpy**: For data manipulation and analysis.
  - **Matplotlib**: For data visualization (technical indicators, stock prices, volume charts).
- **Machine Learning**: 
  - **TensorFlow/Keras**: For LSTM-based stock price prediction.
  - **MinMaxScaler**: For data scaling.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/stock-market-analyzer.git
   cd stock-market-analyzer
