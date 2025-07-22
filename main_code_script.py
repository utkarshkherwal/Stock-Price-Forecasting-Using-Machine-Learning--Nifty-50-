import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

# Nifty 50 tickers latest data available till date
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

def download_data(ticker):
    try:
        df = yf.download(ticker, start="2019-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'), interval='1d')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return None

def prepare_data(df, seq_len=60):
    df = df.fillna(method='ffill').fillna(method='bfill')
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1]))
    return X, y, scaler

def build_xgb_model(trainX, trainY, testX):
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb_model.fit(trainX, trainY)
    return xgb_model.predict(testX)

def build_rf_model(trainX, trainY, testX):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(trainX, trainY)
    return rf.predict(testX)

def build_lr_baseline(trainX, trainY, testX):
    lr_model = LinearRegression()
    lr_model.fit(trainX, trainY)
    return lr_model.predict(testX)

results = []
PLOTS = {}

for ticker in NIFTY50_TICKERS:
    print(f"\nProcessing {ticker} ...")
    df = download_data(ticker)
    if df is None or len(df) < 100:
        print(f"Insufficient data for {ticker}, skipping.")
        continue

    X, y, scaler = prepare_data(df)
    split_idx = int(len(X) * 0.8)
    trainX, testX = X[:split_idx], X[split_idx:]
    trainY, testY = y[:split_idx], y[split_idx:]

    # XGBoost
    y_pred_xgb = build_xgb_model(trainX, trainY, testX)
    y_pred_xgb_actual = scaler.inverse_transform(y_pred_xgb.reshape(-1, 1)).flatten()
    testY_actual = scaler.inverse_transform(testY.reshape(-1, 1)).flatten()
    rmse_xgb = np.sqrt(mean_squared_error(testY_actual, y_pred_xgb_actual))

    # Random Forest
    y_pred_rf = build_rf_model(trainX, trainY, testX)
    y_pred_rf_actual = scaler.inverse_transform(y_pred_rf.reshape(-1, 1)).flatten()
    rmse_rf = np.sqrt(mean_squared_error(testY_actual, y_pred_rf_actual))

    # Linear Regression
    y_pred_lr = build_lr_baseline(trainX, trainY, testX)
    y_pred_lr_actual = scaler.inverse_transform(y_pred_lr.reshape(-1, 1)).flatten()
    rmse_lr = np.sqrt(mean_squared_error(testY_actual, y_pred_lr_actual))

    # Saving results
    results.append({
        'Ticker': ticker,
        'RMSE_XGB': rmse_xgb,
        'RMSE_RF': rmse_rf,
        'RMSE_LR': rmse_lr,
        'Best_Model': min(
            ('XGBoost', rmse_xgb),
            ('RandomForest', rmse_rf),
            ('LinearRegression', rmse_lr),
            key=lambda x: x[1]
        )[0]
    })

    # Save data for plotting (for all stocks)
    PLOTS[ticker] = {
        'Dates': df.index[-len(testY_actual):],
        'Actual': testY_actual,
        'XGB_Pred': y_pred_xgb_actual,
        'RF_Pred': y_pred_rf_actual,
        'LR_Pred': y_pred_lr_actual
    }
    print(f"Done: XGB RMSE={rmse_xgb:.2f}, RF RMSE={rmse_rf:.2f}, LR RMSE={rmse_lr:.2f}")

#VISUALIZATION OF ALL STOCKS


#Visualization for RELIANCE.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['RELIANCE.NS']['Dates'], PLOTS['RELIANCE.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['RELIANCE.NS']['Dates'], PLOTS['RELIANCE.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['RELIANCE.NS']['Dates'], PLOTS['RELIANCE.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['RELIANCE.NS']['Dates'], PLOTS['RELIANCE.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("RELIANCE.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for HINDUNILVR.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['HINDUNILVR.NS']['Dates'], PLOTS['HINDUNILVR.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['HINDUNILVR.NS']['Dates'], PLOTS['HINDUNILVR.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['HINDUNILVR.NS']['Dates'], PLOTS['HINDUNILVR.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['HINDUNILVR.NS']['Dates'], PLOTS['HINDUNILVR.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("HINDUNILVR.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for ITC.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['ITC.NS']['Dates'], PLOTS['ITC.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['ITC.NS']['Dates'], PLOTS['ITC.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['ITC.NS']['Dates'], PLOTS['ITC.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['ITC.NS']['Dates'], PLOTS['ITC.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("ITC.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for KOTAKBANK.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['KOTAKBANK.NS']['Dates'], PLOTS['KOTAKBANK.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['KOTAKBANK.NS']['Dates'], PLOTS['KOTAKBANK.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['KOTAKBANK.NS']['Dates'], PLOTS['KOTAKBANK.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['KOTAKBANK.NS']['Dates'], PLOTS['KOTAKBANK.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("KOTAKBANK.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for LT.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['LT.NS']['Dates'], PLOTS['LT.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['LT.NS']['Dates'], PLOTS['LT.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['LT.NS']['Dates'], PLOTS['LT.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['LT.NS']['Dates'], PLOTS['LT.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("LT.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for SBIN.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['SBIN.NS']['Dates'], PLOTS['SBIN.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['SBIN.NS']['Dates'], PLOTS['SBIN.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['SBIN.NS']['Dates'], PLOTS['SBIN.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['SBIN.NS']['Dates'], PLOTS['SBIN.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("SBINT.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for BHARTIARTL.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['BHARTIARTL.NS']['Dates'], PLOTS['BHARTIARTL.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['BHARTIARTL.NS']['Dates'], PLOTS['BHARTIARTL.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['BHARTIARTL.NS']['Dates'], PLOTS['BHARTIARTL.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['BHARTIARTL.NS']['Dates'], PLOTS['BHARTIARTL.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("BHARTIARTL.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for AXISBANK.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['AXISBANK.NS']['Dates'], PLOTS['AXISBANK.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['AXISBANK.NS']['Dates'], PLOTS['AXISBANK.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['AXISBANK.NS']['Dates'], PLOTS['AXISBANK.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['AXISBANK.NS']['Dates'], PLOTS['AXISBANK.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("AXISBANK.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for BAJAJFINSV.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['BAJAJFINSV.NS']['Dates'], PLOTS['BAJAJFINSV.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['BAJAJFINSV.NS']['Dates'], PLOTS['BAJAJFINSV.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['BAJAJFINSV.NS']['Dates'], PLOTS['BAJAJFINSV.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['BAJAJFINSV.NS']['Dates'], PLOTS['BAJAJFINSV.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("BAJAJFINSV.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for HCLTECH.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['HCLTECH.NS']['Dates'], PLOTS['HCLTECH.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['HCLTECH.NS']['Dates'], PLOTS['HCLTECH.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['HCLTECH.NS']['Dates'], PLOTS['HCLTECH.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['HCLTECH.NS']['Dates'], PLOTS['HCLTECH.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("HCLTECH.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for MARUTI.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['MARUTI.NS']['Dates'], PLOTS['MARUTI.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['MARUTI.NS']['Dates'], PLOTS['MARUTI.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['MARUTI.NS']['Dates'], PLOTS['MARUTI.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['MARUTI.NS']['Dates'], PLOTS['MARUTI.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("MARUTI.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for SUNPHARMA.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['SUNPHARMA.NS']['Dates'], PLOTS['SUNPHARMA.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['SUNPHARMA.NS']['Dates'], PLOTS['SUNPHARMA.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['SUNPHARMA.NS']['Dates'], PLOTS['SUNPHARMA.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['SUNPHARMA.NS']['Dates'], PLOTS['SUNPHARMA.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("SUNPHARMA.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for ULTRACEMCO.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['ULTRACEMCO.NS']['Dates'], PLOTS['ULTRACEMCO.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['ULTRACEMCO.NS']['Dates'], PLOTS['ULTRACEMCO.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['ULTRACEMCO.NS']['Dates'], PLOTS['ULTRACEMCO.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['ULTRACEMCO.NS']['Dates'], PLOTS['ULTRACEMCO.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("ULTRACEMCO.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for NESTLEIND.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['NESTLEIND.NS']['Dates'], PLOTS['NESTLEIND.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['NESTLEIND.NS']['Dates'], PLOTS['NESTLEIND.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['NESTLEIND.NS']['Dates'], PLOTS['NESTLEIND.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['NESTLEIND.NS']['Dates'], PLOTS['NESTLEIND.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("NESTLEIND.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for BAJFINANCE.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['BAJFINANCE.NS']['Dates'], PLOTS['BAJFINANCE.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['BAJFINANCE.NS']['Dates'], PLOTS['BAJFINANCE.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['BAJFINANCE.NS']['Dates'], PLOTS['BAJFINANCE.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['BAJFINANCE.NS']['Dates'], PLOTS['BAJFINANCE.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("BAJFINANCE.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()



#Visualization for WIPRO.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['WIPRO.NS']['Dates'], PLOTS['WIPRO.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['WIPRO.NS']['Dates'], PLOTS['WIPRO.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['WIPRO.NS']['Dates'], PLOTS['WIPRO.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['WIPRO.NS']['Dates'], PLOTS['WIPRO.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("WIPRO.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()
#Visualization for ASIANPAINT.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['ASIANPAINT.NS']['Dates'], PLOTS['ASIANPAINT.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['ASIANPAINT.NS']['Dates'], PLOTS['ASIANPAINT.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['ASIANPAINT.NS']['Dates'], PLOTS['ASIANPAINT.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['ASIANPAINT.NS']['Dates'], PLOTS['ASIANPAINT.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("ASIANPAINT.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for POWERGRID.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['POWERGRID.NS']['Dates'], PLOTS['POWERGRID.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['POWERGRID.NS']['Dates'], PLOTS['POWERGRID.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['POWERGRID.NS']['Dates'], PLOTS['POWERGRID.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['POWERGRID.NS']['Dates'], PLOTS['POWERGRID.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("POWERGRID.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for TITAN.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['TITAN.NS']['Dates'], PLOTS['TITAN.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['TITAN.NS']['Dates'], PLOTS['TITAN.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['TITAN.NS']['Dates'], PLOTS['TITAN.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['TITAN.NS']['Dates'], PLOTS['TITAN.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("TITAN.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for DIVISLAB.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['DIVISLAB.NS']['Dates'], PLOTS['DIVISLAB.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['DIVISLAB.NS']['Dates'], PLOTS['DIVISLAB.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['DIVISLAB.NS']['Dates'], PLOTS['DIVISLAB.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['DIVISLAB.NS']['Dates'], PLOTS['DIVISLAB.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("DIVISLAB.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for HDFCLIFE.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['HDFCLIFE.NS']['Dates'], PLOTS['HDFCLIFE.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['HDFCLIFE.NS']['Dates'], PLOTS['HDFCLIFE.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['HDFCLIFE.NS']['Dates'], PLOTS['HDFCLIFE.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['HDFCLIFE.NS']['Dates'], PLOTS['HDFCLIFE.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("HDFCLIFE.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for ONGC.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['ONGC.NS']['Dates'], PLOTS['ONGC.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['ONGC.NS']['Dates'], PLOTS['ONGC.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['ONGC.NS']['Dates'], PLOTS['ONGC.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['ONGC.NS']['Dates'], PLOTS['ONGC.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("ONGC.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for TECHM.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['TECHM.NS']['Dates'], PLOTS['TECHM.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['TECHM.NS']['Dates'], PLOTS['TECHM.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['TECHM.NS']['Dates'], PLOTS['TECHM.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['TECHM.NS']['Dates'], PLOTS['TECHM.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("TECHM.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for CIPLA.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['CIPLA.NS']['Dates'], PLOTS['CIPLA.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['CIPLA.NS']['Dates'], PLOTS['CIPLA.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['CIPLA.NS']['Dates'], PLOTS['CIPLA.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['CIPLA.NS']['Dates'], PLOTS['CIPLA.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("CIPLA.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for ADANIGREEN.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['ADANIGREEN.NS']['Dates'], PLOTS['ADANIGREEN.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['ADANIGREEN.NS']['Dates'], PLOTS['ADANIGREEN.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['ADANIGREEN.NS']['Dates'], PLOTS['ADANIGREEN.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['ADANIGREEN.NS']['Dates'], PLOTS['ADANIGREEN.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("ADANIGREEN.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for BAJAJ-AUTO.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['BAJAJ-AUTO.NS']['Dates'], PLOTS['BAJAJ-AUTO.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['BAJAJ-AUTO.NS']['Dates'], PLOTS['BAJAJ-AUTO.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['BAJAJ-AUTO.NS']['Dates'], PLOTS['BAJAJ-AUTO.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['BAJAJ-AUTO.NS']['Dates'], PLOTS['BAJAJ-AUTO.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("BAJAJ-AUTO.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for GRASIM.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['GRASIM.NS']['Dates'], PLOTS['GRASIM.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['GRASIM.NS']['Dates'], PLOTS['GRASIM.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['GRASIM.NS']['Dates'], PLOTS['GRASIM.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['GRASIM.NS']['Dates'], PLOTS['GRASIM.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("GRASIM.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for HDFCAMC.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['HDFCAMC.NS']['Dates'], PLOTS['HDFCAMC.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['HDFCAMC.NS']['Dates'], PLOTS['HDFCAMC.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['HDFCAMC.NS']['Dates'], PLOTS['HDFCAMC.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['HDFCAMC.NS']['Dates'], PLOTS['HDFCAMC.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("HDFCAMC.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for TATASTEEL.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['TATASTEEL.NS']['Dates'], PLOTS['TATASTEEL.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['TATASTEEL.NS']['Dates'], PLOTS['TATASTEEL.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['TATASTEEL.NS']['Dates'], PLOTS['TATASTEEL.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['TATASTEEL.NS']['Dates'], PLOTS['TATASTEEL.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("TATASTEEL.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for COALINDIA.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['COALINDIA.NS']['Dates'], PLOTS['COALINDIA.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['COALINDIA.NS']['Dates'], PLOTS['COALINDIA.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['COALINDIA.NS']['Dates'], PLOTS['COALINDIA.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['COALINDIA.NS']['Dates'], PLOTS['COALINDIA.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("COALINDIA.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for BRITANNIA.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['BRITANNIA.NS']['Dates'], PLOTS['BRITANNIA.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['BRITANNIA.NS']['Dates'], PLOTS['BRITANNIA.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['BRITANNIA.NS']['Dates'], PLOTS['BRITANNIA.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['BRITANNIA.NS']['Dates'], PLOTS['BRITANNIA.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("BRITANNIA.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for EICHERMOT.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['EICHERMOT.NS']['Dates'], PLOTS['EICHERMOT.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['EICHERMOT.NS']['Dates'], PLOTS['EICHERMOT.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['EICHERMOT.NS']['Dates'], PLOTS['EICHERMOT.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['EICHERMOT.NS']['Dates'], PLOTS['EICHERMOT.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("EICHERMOT.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for JSWSTEEL.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['JSWSTEEL.NS']['Dates'], PLOTS['JSWSTEEL.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['JSWSTEEL.NS']['Dates'], PLOTS['JSWSTEEL.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['JSWSTEEL.NS']['Dates'], PLOTS['JSWSTEEL.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['JSWSTEEL.NS']['Dates'], PLOTS['JSWSTEEL.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("JSWSTEEL.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for ADANIPORTS.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['ADANIPORTS.NS']['Dates'], PLOTS['ADANIPORTS.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['ADANIPORTS.NS']['Dates'], PLOTS['ADANIPORTS.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['ADANIPORTS.NS']['Dates'], PLOTS['ADANIPORTS.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['ADANIPORTS.NS']['Dates'], PLOTS['ADANIPORTS.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("ADANIPORTS.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for M&M.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['M&M.NS']['Dates'], PLOTS['M&M.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['M&M.NS']['Dates'], PLOTS['M&M.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['M&M.NS']['Dates'], PLOTS['M&M.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['M&M.NS']['Dates'], PLOTS['M&M.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("M&M.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()



#Visualization for SHREECEM.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['SHREECEM.NS']['Dates'], PLOTS['SHREECEM.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['SHREECEM.NS']['Dates'], PLOTS['SHREECEM.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['SHREECEM.NS']['Dates'], PLOTS['SHREECEM.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['SHREECEM.NS']['Dates'], PLOTS['SHREECEM.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("SHREECEM.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()



#Visualization for HEROMOTOCO.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['HEROMOTOCO.NS']['Dates'], PLOTS['HEROMOTOCO.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['HEROMOTOCO.NS']['Dates'], PLOTS['HEROMOTOCO.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['HEROMOTOCO.NS']['Dates'], PLOTS['HEROMOTOCO.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['HEROMOTOCO.NS']['Dates'], PLOTS['HEROMOTOCO.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("HEROMOTOCO.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for BPCL.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['BPCL.NS']['Dates'], PLOTS['BPCL.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['BPCL.NS']['Dates'], PLOTS['BPCL.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['BPCL.NS']['Dates'], PLOTS['BPCL.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['BPCL.NS']['Dates'], PLOTS['BPCL.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("BPCL.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for INDUSINDBK.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['INDUSINDBK.NS']['Dates'], PLOTS['INDUSINDBK.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['INDUSINDBK.NS']['Dates'], PLOTS['INDUSINDBK.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['INDUSINDBK.NS']['Dates'], PLOTS['INDUSINDBK.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['INDUSINDBK.NS']['Dates'], PLOTS['INDUSINDBK.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("INDUSINDBK.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for DRREDDY.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['DRREDDY.NS']['Dates'], PLOTS['DRREDDY.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['DRREDDY.NS']['Dates'], PLOTS['DRREDDY.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['DRREDDY.NS']['Dates'], PLOTS['DRREDDY.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['DRREDDY.NS']['Dates'], PLOTS['DRREDDY.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("DRREDDY.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for SBILIFE.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['SBILIFE.NS']['Dates'], PLOTS['SBILIFE.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['SBILIFE.NS']['Dates'], PLOTS['SBILIFE.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['SBILIFE.NS']['Dates'], PLOTS['SBILIFE.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['SBILIFE.NS']['Dates'], PLOTS['SBILIFE.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("SBILIFE.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for HINDALCO.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['HINDALCO.NS']['Dates'], PLOTS['HINDALCO.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['HINDALCO.NS']['Dates'], PLOTS['HINDALCO.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['HINDALCO.NS']['Dates'], PLOTS['HINDALCO.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['HINDALCO.NS']['Dates'], PLOTS['HINDALCO.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("HINDALCO.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for ICICIPRULI.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['ICICIPRULI.NS']['Dates'], PLOTS['ICICIPRULI.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['ICICIPRULI.NS']['Dates'], PLOTS['ICICIPRULI.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['ICICIPRULI.NS']['Dates'], PLOTS['ICICIPRULI.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['ICICIPRULI.NS']['Dates'], PLOTS['ICICIPRULI.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("ICICIPRULI.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()


#Visualization for GAIL.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['GAIL.NS']['Dates'], PLOTS['GAIL.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['GAIL.NS']['Dates'], PLOTS['GAIL.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['GAIL.NS']['Dates'], PLOTS['GAIL.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['GAIL.NS']['Dates'], PLOTS['GAIL.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("GAIL.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for NTPC.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['NTPC.NS']['Dates'], PLOTS['NTPC.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['NTPC.NS']['Dates'], PLOTS['NTPC.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['NTPC.NS']['Dates'], PLOTS['NTPC.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['NTPC.NS']['Dates'], PLOTS['NTPC.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("NTPC.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Visualization for UPL.NS
plt.figure(figsize=(10,5))
plt.plot(PLOTS['UPL.NS']['Dates'], PLOTS['UPL.NS']['Actual'], label='Actual Price')
plt.plot(PLOTS['UPL.NS']['Dates'], PLOTS['UPL.NS']['XGB_Pred'], label='XGBoost')
plt.plot(PLOTS['UPL.NS']['Dates'], PLOTS['UPL.NS']['RF_Pred'], label='Random Forest')
plt.plot(PLOTS['UPL.NS']['Dates'], PLOTS['UPL.NS']['LR_Pred'], label='Linear Regression', linestyle='--')
plt.title("UPL.NS - Actual vs Predicted Closing Price")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()
# Print RMSE summary
results_df = pd.DataFrame(results)
print("\n==== Model Performance Summary (RMSE) ====")
print(results_df[['Ticker', 'RMSE_XGB', 'RMSE_RF', 'RMSE_LR', 'Best_Model']])
print(f"\nCount of best model per stock:\n{results_df['Best_Model'].value_counts()}")

print("""
Takeaway:
---------
- XGBoost and tree-based models are often more accurate for stock price forecasting with this setup.
- This script uses the latest available data for all stocks and visualizes predictions for individual Nifty 50 stocks.
- To plot any other stock, just copy one of the blocks above and change the ticker symbol!
""")


# ----------------------------------End of main_code_script.py----------------------------------
