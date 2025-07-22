### Stock Price Forecasting Using Machine Learning (Nifty 50)

This project focuses on predicting the daily closing prices of Nifty 50 stocks using historical data and three machine learning models: Linear Regression, Random Forest, and XGBoost. It aims to identify which model performs best in terms of prediction accuracy and to visualize how closely each model can follow real stock market behavior. By analyzing and comparing the models' performance across all 50 companies, the project provides valuable insight into the reliability of different forecasting approaches in the context of Indian financial markets.

The central objective of the project is to make short-term stock forecasting accessible, interpretable, and effective for data scientists, analysts, and retail investors. With the increasing use of machine learning in financial markets, this project demonstrates how AI tools can analyze past price patterns to estimate future prices. This is particularly relevant for short-term traders, investors building strategies, or anyone interested in exploring data-driven decision-making in finance.

### Requirements

Install all dependencies with:

```bash
pip install yfinance numpy pandas scikit-learn matplotlib xgboost
```

To collect stock data, the project uses the Python library `yfinance`, which enables seamless access to historical price information from Yahoo Finance. This library fetches key features like Open, High, Low, Close, and Volume for each stock from January 2019 to the current date. Using this real-world data ensures that the analysis reflects actual market behavior, and it eliminates the need for manual downloading or expensive APIs.

Each stock’s historical price data is processed using a time-window approach, where the past 60 days are used to predict the next day’s closing price. This forms the basis for training all three machine learning models. The dataset is split into training and testing sets, and models are evaluated using RMSE (Root Mean Squared Error), which measures how far the predicted values are from the actual prices. Lower RMSE means better performance.

Surprisingly, the final results show that **Linear Regression** performed the best for all 50 stocks in the Nifty 50 index, giving the lowest RMSE in every single case. While more advanced models like Random Forest and XGBoost are often expected to outperform simple models, in this setup, the linear model was consistently more accurate, possibly due to the stability of short-term trends or the smooth nature of the data used.

To understand the models in layman’s terms, let's start with **Linear Regression**. This model works like drawing a straight line through past stock prices and assuming that the future price will follow that line. It’s simple, fast, and often surprisingly effective when prices move in a steady trend. It doesn’t handle sudden spikes or drops well, but if the market is calm, it can work very well, as seen in this project.

**Random Forest**, on the other hand, is like a group of smart trees—each tree gives its own prediction, and then the model averages them out. This model is very good at handling messy or unpredictable data. However, it can sometimes be too complex, especially if the data doesn’t have sharp patterns to learn from. In this project, it was outperformed by Linear Regression, possibly because the patterns in the data weren’t complex enough to need such heavy modeling.

**XGBoost** (Extreme Gradient Boosting) is another tree-based model, but it builds its trees one by one, where each new tree tries to fix the errors made by the previous ones. This makes XGBoost extremely powerful in many real-world cases, like Kaggle competitions or financial risk modeling. But it’s also sensitive to data quality and tuning. Here, it didn’t perform as well as expected, which tells us that sometimes, simpler models like Linear Regression can outperform even the most sophisticated algorithms when the data fits certain conditions.

This project highlights an important truth in machine learning: **the most complex model is not always the best**. While tools like XGBoost and Random Forest are powerful, in this case, the simplest model—Linear Regression—gave the best results for all 50 companies. That makes this project an important reminder that before jumping into complex AI, we should always test the basics first.

In conclusion, this project not only forecasts stock prices but also educates users about model selection, data preparation, and performance evaluation using real market data. It encourages the use of open-source tools and simple techniques to solve problems in a high-stakes domain like the stock market. This project has real-world impact by empowering data-driven investment decisions, promoting transparency, and showing that machine learning can be both powerful and approachable.
