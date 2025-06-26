Stock Prediction Tool
Overview
The Stock Prediction Tool is a Python-based desktop application built using Tkinter for visualizing and predicting stock prices. It fetches historical stock data, trains machine learning models, and provides technical analysis and comparison features for single or multiple stocks.
Features

Fetch Stock Data: Retrieve historical stock data for specified symbols using the yfinance library.
Technical Analysis: Display candlestick, OHLC, and line charts with technical indicators like RSI, MACD, VWAP, and Bollinger Bands.
Train Models: Train RandomForestRegressor models to predict stock prices based on historical data and technical indicators.
Multiple Stock Training: Train models for multiple stock symbols simultaneously.
Future Predictions: Forecast future stock prices with confidence intervals for trained models.
Stock Comparison: Compare multiple stocks' price trends, normalized performance, and trading volumes.

Requirements
To run the application, you need Python 3.x and the following libraries:

tkinter (usually included with Python)
pandas
numpy
matplotlib
mplfinance
yfinance
joblib
scikit-learn

Install the dependencies using pip:
pip install pandas numpy matplotlib mplfinance yfinance joblib scikit-learn

Installation

Clone or download the repository containing the script.
Ensure you have Python 3.x installed.
Install the required libraries as listed above.
Run the script using:python stock_predictor.py



Usage

Launch the Application: Run the script to open the Tkinter GUI.
Enter Stock Symbols: Input one or more stock symbols (e.g., AAPL, MSFT) in the "Stock Symbol(s)" field, separated by commas for multiple stocks.
Select Period: Choose a historical data period (1 month, 3 months, 6 months, 1 year, or 2 years) from the dropdown.
Fetch Data: Click "Fetch Data" to retrieve and cache stock data.
View Technical Analysis: Use the candlestick chart button (currently commented out) to visualize charts with technical indicators.
Train Models:
Click "Train Model" to train a model for a single stock.
Click "Train Multiple" to train models for multiple stocks entered.


Predict Future Prices: Select a trained model from the dropdown and click "Predict Future" to forecast prices with confidence intervals.
Compare Stocks: Enter multiple stock symbols and click "Compare Stocks" to view price, performance, and volume comparisons.

File Structure

stock_predictor.py: Main application script containing the StockPredictor class.
stock_model_[symbol].pkl: Saved RandomForest models for each trained stock symbol.
[symbol]_[period]_stock_data.csv: Cached stock data for each symbol and period.
temp_[symbol]_[chart_type].png: Temporary files for chart images (automatically cleaned up).

How It Works

Data Fetching: Uses yfinance to fetch historical stock data, with caching to reduce API calls. Data is saved as CSV files for reuse within 24 hours.
Technical Indicators:
RSI (Relative Strength Index)
MACD (Moving Average Convergence Divergence)
VWAP (Volume Weighted Average Price)
SMA (Simple Moving Averages) and Bollinger Bands


Model Training: Trains a RandomForestRegressor using features like lagged close prices, SMAs, RSI, MACD, and VWAP.
Prediction: Generates future price predictions with confidence intervals using bootstrap resampling.
Visualization: Uses matplotlib and mplfinance for charts, embedded in Tkinter via FigureCanvasTkAgg.

Notes

Rate Limiting: The application handles yfinance rate limits with retries and delays.
Data Validation: Ensures valid stock symbols using regex and checks for sufficient data before training.
Threading: Uses threading to prevent GUI freezing during long operations like data fetching or model training.
Error Handling: Displays user-friendly error messages for invalid inputs, API failures, or insufficient data.
Commented Code: The "Candlestick Chart" button is currently commented out but can be enabled for chart visualization.

Limitations

Requires an internet connection for fetching stock data via yfinance.
Model accuracy depends on the quality and quantity of historical data.
Predictions are based on historical patterns and may not account for sudden market events.
The application may generate temporary chart files, which are cleaned up automatically.

Future Improvements

Enable the candlestick chart button for direct chart access.
Add more technical indicators or machine learning models.
Implement real-time data streaming.
Enhance UI with customizable chart options and themes.

License
This project is for educational purposes and is not licensed for commercial use. Ensure compliance with yfinance terms of service when using the application.
