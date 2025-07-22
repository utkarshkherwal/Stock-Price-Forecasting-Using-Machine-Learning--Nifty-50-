## Stock Price Forecasting Using Machine Learning (Nifty 50)

Predict daily closing prices for Nifty 50 stocks with interactive experiments!  
This project compares Linear Regression, Random Forest, and XGBoost models for short-term stock forecasting on real Indian market data.

---

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkarshkherwal/Stock-Price-Forecasting-Using-Machine-Learning--Nifty-50-/blob/main/Stock_Price_Forecasting_Nifty50.ipynb)

---

### 📦 Requirements

Install all dependencies with:

```bash
pip install yfinance numpy pandas scikit-learn matplotlib xgboost
```

---

### 🚀 Quick Start

1. **Clone this repository:**
   ```bash
   git clone https://github.com/utkarshkherwal/Stock-Price-Forecasting-Using-Machine-Learning--Nifty-50-.git
   cd Stock-Price-Forecasting-Using-Machine-Learning--Nifty-50-
   ```

2. **Run the notebook:**
   - Open [Stock_Price_Forecasting_Nifty50.ipynb](Final.ipynb) in Jupyter or Colab.

---

### ⚡ Interactive Usage

- **Choose your stock:**  
  Change the `ticker` in the notebook (e.g., `RELIANCE.NS`, `TCS.NS`, etc.)

- **Adjust time window:**  
  Update the `window_size` variable to experiment with how much history to use for predictions.

- **Pick a model:**  
  Run model cells for Linear Regression, Random Forest, or XGBoost and compare results.

- **Visualize results:**  
  The notebook will generate plots and RMSE scores for each model.

---

### 📊 What You'll Learn

- How to fetch and preprocess real Nifty 50 data with `yfinance`
- How to build, train, and test three classic machine learning models on time series data
- Why sometimes the simplest models (like Linear Regression) outperform more complex ones in financial forecasting
- How to interpret RMSE and compare model accuracy visually

---

### 📚 Project Summary

This project aims to make stock price prediction approachable and practical for data scientists and retail investors alike.  
Surprisingly, for Nifty 50 stocks from 2019 onward, **Linear Regression** consistently produced the lowest RMSE, outperforming even advanced models like Random Forest and XGBoost.  
This demonstrates a key lesson: always start simple and validate with real data before reaching for complex solutions.

---

### 🙋‍♂️ Get Involved

- **Fork the repo** and try it on your favorite stock!
- **Open an issue** if you have ideas, questions, or improvements.
- **Share your findings** in the Discussions tab.

---

**Empowering data-driven financial insights with approachable machine learning. Try it out, explore, and have fun!**
