"""
Nifty 50 Stock Price Forecasting with Random Forest (and Linear Regression Baseline)
------------------------------------------------------------------------------------
This script forecasts daily closing prices for all 50 Nifty 50 stocks using Random Forest.
A simple Linear Regression serves as a baseline for comparison.

Requirements:
    - yfinance
    - numpy
    - pandas
    - scikit-learn
    - matplotlib

Install missing packages using:
    pip install yfinance numpy pandas scikit-learn matplotlib
"""

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# -----------------------------
# 1. Define Nifty 50 Stock List
# -----------------------------
NIFTY50_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'SBIN.NS',
    'BHARTIARTL.NS', 'AXISBANK.NS', 'BAJAJFINSV.NS', 'HCLTECH.NS', 'MARUTI.NS',
    'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'BAJFINANCE.NS', 'WIPRO.NS',
    'ASIANPAINT.NS', 'POWERGRID.NS', 'TITAN.NS', 'DIVISLAB.NS', 'HDFCLIFE.NS',
    'ONGC.NS', 'TECHM.NS', 'CIPLA.NS', 'ADANIGREEN.NS', 'BAJAJ-AUTO.NS',
    'GRASIM.NS', 'HDFCAMC.NS', 'TATASTEEL.NS', 'COALINDIA.NS', 'BRITANNIA.NS',
    'EICHERMOT.NS', 'JSWSTEEL.NS', 'ADANIPORTS.NS', 'M&M.NS', 'SHREECEM.NS',
    'HEROMOTOCO.NS', 'BPCL.NS', 'INDUSINDBK.NS', 'DRREDDY.NS', 'SBILIFE.NS',
    'HINDALCO.NS', 'ICICIPRULI.NS', 'GAIL.NS', 'NTPC.NS', 'UPL.NS'
]

TOP5_TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']

# ----------------------------------
# 2. Download Data from Yahoo Finance
# ----------------------------------
def download_data(ticker, period="5y"):
    """
    Downloads historical daily stock data for a given ticker
    """
    try:
        df = yf.download(ticker, period=period, interval='1d')
        # Keep only relevant columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return None

# ----------------------------------
# 3. Prepare Data for Time Series Models
# ----------------------------------
def prepare_data(df, seq_len=60):
    """
    Prepares time-series data for machine learning models:
    - Handles missing values
    - Normalizes prices
    - Creates input sequences
    """
    # Fill missing values (forward fill, then backward fill)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # We'll forecast 'Close' price
    data = df[['Close']].values
    
    # Normalize 'Close' prices between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        # Use past `seq_len` days to predict the next day
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    # For ML models, flatten input: [samples, features]
    X = np.reshape(X, (X.shape[0], X.shape[1]))
    return X, y, scaler

# ----------------------------------
# 4. Random Forest Model
# ----------------------------------
def build_rf_model(trainX, trainY, testX):
    """
    Trains a Random Forest Regressor and predicts test set
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(trainX, trainY)
    y_pred_rf = rf.predict(testX)
    return y_pred_rf

# ----------------------------------
# 5. Linear Regression Baseline
# ----------------------------------
def build_lr_baseline(trainX, trainY, testX):
    """
    Trains a Linear Regression model as a baseline
    """
    lr_model = LinearRegression()
    lr_model.fit(trainX, trainY)
    y_pred_lr = lr_model.predict(testX)
    return y_pred_lr

# ----------------------------------
# 6. Train/Evaluate Models for All Stocks
# ----------------------------------
results = []
PLOTS = {}

for ticker in NIFTY50_TICKERS:
    print(f"\nProcessing {ticker} ...")
    df = download_data(ticker)
    if df is None or len(df) < 100:
        print(f"Insufficient data for {ticker}, skipping.")
        continue
    
    # Prepare the data
    X, y, scaler = prepare_data(df)
    
    # Train/Test Split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    trainX, testX = X[:split_idx], X[split_idx:]
    trainY, testY = y[:split_idx], y[split_idx:]
    
    # Random Forest model
    y_pred_rf = build_rf_model(trainX, trainY, testX)
    y_pred_rf_actual = scaler.inverse_transform(y_pred_rf.reshape(-1, 1)).flatten()
    testY_actual = scaler.inverse_transform(testY.reshape(-1, 1)).flatten()
    rmse_rf = np.sqrt(mean_squared_error(testY_actual, y_pred_rf_actual))
    
    # Baseline: Linear Regression
    y_pred_lr = build_lr_baseline(trainX, trainY, testX)
    y_pred_lr_actual = scaler.inverse_transform(y_pred_lr.reshape(-1, 1)).flatten()
    rmse_lr = np.sqrt(mean_squared_error(testY_actual, y_pred_lr_actual))
    
    # Save results
    results.append({
        'Ticker': ticker,
        'RMSE_RF': rmse_rf,
        'RMSE_LR': rmse_lr,
        'RF_Better': rmse_rf < rmse_lr
    })
    
    # Save data for plotting (only for top 5 stocks)
    if ticker in TOP5_TICKERS:
        PLOTS[ticker] = {
            'Dates': df.index[-len(testY_actual):],
            'Actual': testY_actual,
            'RF_Pred': y_pred_rf_actual,
            'LR_Pred': y_pred_lr_actual
        }
    print(f"Done: RF RMSE={rmse_rf:.2f}, LR RMSE={rmse_lr:.2f}")

# ----------------------------------
# 7. Visualization: Actual vs Predicted
# ----------------------------------
for ticker in TOP5_TICKERS:
    if ticker in PLOTS:
        plt.figure(figsize=(10,5))
        plt.plot(PLOTS[ticker]['Dates'], PLOTS[ticker]['Actual'], label='Actual Price')
        plt.plot(PLOTS[ticker]['Dates'], PLOTS[ticker]['RF_Pred'], label='Random Forest Predicted')
        plt.plot(PLOTS[ticker]['Dates'], PLOTS[ticker]['LR_Pred'], label='Linear Regression Predicted', linestyle='--')
        plt.title(f"{ticker} - Actual vs Predicted Closing Price")
        plt.xlabel('Date')
        plt.ylabel('Price (INR)')
        plt.legend()
        plt.tight_layout()
        plt.show()

# ----------------------------------
# 8. Print Summary Table
# ----------------------------------
results_df = pd.DataFrame(results)
print("\n==== Model Performance Summary (RMSE) ====")
print(results_df[['Ticker', 'RMSE_RF', 'RMSE_LR', 'RF_Better']])

num_better = results_df['RF_Better'].sum()
print(f"\nRandom Forest performed better than Linear Regression on {num_better} out of {len(results_df)} stocks.")

print("""
Takeaway:
---------
- Random Forest models can capture non-linear relationships and complex patterns in stock price data better than simple linear models in many cases.
- Results vary by stock and data quality.
- This project shows how AI/ML can be applied to financial forecasting using free data and open-source libraries.
""")