import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import re
import os
import time
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Prediction Tool")
        self.root.geometry("800x600")
        self.root.configure(bg="#1E1E2F")
        self.models = {}  # Dictionary to store models by stock symbol
        self.df = None
        self.setup_gui()

    def setup_gui(self):
        # Stock Symbol Entry
        tk.Label(self.root, text="Stock Symbol(s):", bg="#1E1E2F", fg="white").place(x=50, y=20)
        self.stock_entry = tk.Entry(self.root, width=20)
        self.stock_entry.place(x=150, y=20)
        self.stock_entry.insert(0, "AAPL")

        # Period Selection
        tk.Label(self.root, text="Period:", bg="#1E1E2F", fg="white").place(x=300, y=20)
        self.period_var = tk.StringVar(value="1y")
        period_menu = ttk.Combobox(self.root, textvariable=self.period_var, values=["1mo", "3mo", "6mo", "1y", "2y"], state="readonly", width=10)
        period_menu.place(x=350, y=20)

        # Trained Models Dropdown
        tk.Label(self.root, text="Trained Models:", bg="#1E1E2F", fg="white").place(x=50, y=60)
        self.model_var = tk.StringVar()
        self.model_menu = ttk.Combobox(self.root, textvariable=self.model_var, state="readonly", width=20)
        self.model_menu.place(x=150, y=60)

        # Buttons
        button_style = {"bg": "#00CED1", "fg": "black", "font": ("Arial", 10, "bold"), "width": 15}
        tk.Button(self.root, text="Fetch Data", command=self.fetch_data_action, **button_style).place(x=50, y=100)
        # tk.Button(self.root, text="Candlestick Chart", command=self.plot_candlestick_action, **button_style).place(x=200, y=100)
        tk.Button(self.root, text="Train Model", command=self.train_model_action, **button_style).place(x=50, y=140)
        tk.Button(self.root, text="Train Multiple", command=self.train_multiple_action, **button_style).place(x=200, y=140)
        tk.Button(self.root, text="Predict Future", command=self.predict_future_action, **button_style).place(x=50, y=180)
        tk.Button(self.root, text="Compare Stocks", command=self.compare_stocks_action, **button_style).place(x=200, y=180)

        # Status and Progress
        self.status_label = tk.Label(self.root, text="", bg="#1E1E2F", fg="white")
        self.status_label.place(x=50, y=220)
        self.progress_bar = ttk.Progressbar(self.root, mode="indeterminate", length=300)
        self.progress_bar.place(x=50, y=250)

        # Results Text
        tk.Label(self.root, text="Model Results:", bg="#1E1E2F", fg="white").place(x=50, y=280)
        self.result_text = tk.Text(self.root, height=10, width=80, bg="#2E2E4F", fg="white")
        self.result_text.place(x=50, y=300)

    def is_valid_symbol(self, symbol):
        return bool(re.match(r'^[A-Z.-]+$', symbol.strip().upper()))

    def fetch_stock_data(self, stock_symbol, period="1y", retries=3, delay=5, max_cache_age=86400):  # 86400 seconds = 1 day
        if not stock_symbol or not any(self.is_valid_symbol(s.strip()) for s in stock_symbol.split(',')):
            messagebox.showerror("Error", "Please enter at least one valid stock symbol (e.g., AAPL, MSFT).")
            return None
        symbols = [s.strip() for s in stock_symbol.split(',') if self.is_valid_symbol(s.strip())]
        if not symbols:
            messagebox.showerror("Error", "No valid stock symbols provided.")
            return None
        symbol = symbols[0]
        if len(symbols) > 1:
            messagebox.showwarning("Warning", f"Using only the first symbol: {symbol}. Use 'Train Multiple' for multiple symbols.")

        csv_file = f"{symbol}_{period}_stock_data.csv"

        # Check if CSV file exists and is recent
        if os.path.exists(csv_file):
            file_age = time.time() - os.path.getmtime(csv_file)
            if file_age < max_cache_age:
                try:
                    self.status_label.config(text=f"Loading data for {symbol} from CSV...")
                    self.progress_bar.start()
                    self.root.update()
                    
                    # Use explicit dtype to handle CSV format issues
                    try:
                        data = pd.read_csv(csv_file, parse_dates=['Date'],
                                          dtype={'Open': float, 'High': float, 'Low': float, 
                                                'Close': float, 'Volume': float})
                        self.df = data
                        self.status_label.config(text=f"Data for {symbol} loaded from CSV successfully!")
                        return data
                    except Exception as csv_error:
                        # If CSV reading fails, consider the file corrupted and fetch fresh data
                        self.status_label.config(text=f"CSV format issue: {csv_error}. Fetching fresh data...")
                        # Continue to fetch data section below
                        pass
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load data from CSV: {str(e)}")
                    return None
                finally:
                    self.progress_bar.stop()
                    self.root.update()

        # Fetch data if CSV doesn't exist or is too old or corrupted
        for attempt in range(retries):
            try:
                self.status_label.config(text=f"Fetching data for {symbol}...")
                self.progress_bar.start()
                self.root.update()
                time.sleep(2)  # Initial delay to avoid rate limit
                stock = yf.Ticker(symbol)
                data = stock.history(period=period)
                if data.empty:
                    messagebox.showerror("Error", f"No data found for {symbol}. Check the symbol or try again.")
                    return None
                data.reset_index(inplace=True)
                self.df = data
                data.to_csv(csv_file, index=False)
                self.status_label.config(text=f"Data fetched for {symbol} and saved to CSV successfully!")
                return data
            except Exception as e:
                if "Too Many Requests" in str(e):
                    wait_time = delay * (2 ** attempt)
                    messagebox.showwarning("Warning", f"Rate limit hit. Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    messagebox.showerror("Error", f"Failed to fetch data: {str(e)}")
                    return None
            finally:
                self.progress_bar.stop()
                self.root.update()
        messagebox.showerror("Error", "Max retries reached. Please try again later.")
        return None

    def calculate_rsi(self, data, periods=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data

    def calculate_macd(self, data):
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        return data

    def calculate_vwap(self, data):
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        data['VWAP'] = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return data

    def prepare_features(self, data, stock_symbol):
        if len(data) < 15:
            messagebox.showerror("Error", f"Not enough data for {stock_symbol}.")
            return None
        data['Close_Lag1'] = data['Close'].shift(1)
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data = self.calculate_rsi(data)
        data = self.calculate_macd(data)
        data = self.calculate_vwap(data)
        data.dropna(inplace=True)
        return data

    def plot_candlestick_chart(self, stock_symbol):
        data = self.fetch_stock_data(stock_symbol, self.period_var.get())
        if data is None or data.empty:
            return
            
        # Create a new toplevel window for the charts
        chart_window = tk.Toplevel(self.root)
        chart_window.title(f"Technical Analysis: {stock_symbol}")
        chart_window.geometry("1200x800")
        
        # Create notebook for different chart types
        notebook = ttk.Notebook(chart_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Prepare data
        data.set_index("Date", inplace=True)
        data = self.calculate_rsi(data)
        data = self.calculate_macd(data)
        data = self.calculate_vwap(data)
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['Upper_BB'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
        data['Lower_BB'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)
        
        # Tab 1: Traditional Candlestick Chart
        candle_frame = ttk.Frame(notebook)
        notebook.add(candle_frame, text="Candlestick")
        
        # Create a figure for the candlestick chart
        fig_candle = plt.Figure(figsize=(12, 8))
        
        # Use mpf to create the candlestick chart
        apds = [
            mpf.make_addplot(data['SMA_20'], color='blue', label='SMA 20'),
            mpf.make_addplot(data['SMA_50'], color='red', label='SMA 50'),
            mpf.make_addplot(data['Upper_BB'], color='green', linestyle='dashed', label='Upper BB'),
            mpf.make_addplot(data['Lower_BB'], color='green', linestyle='dashed', label='Lower BB'),
            mpf.make_addplot(data['VWAP'], color='purple', label='VWAP')
        ]
        
        # Save the candlestick chart to a temporary file
        temp_file = f"temp_{stock_symbol}_candle.png"
        mpf.plot(data, type='candle', style='yahoo', title=f"Candlestick Chart: {stock_symbol}",
                 volume=True, mav=(5, 10), addplot=apds, savefig=temp_file)
        
        # Display the saved image in the tkinter window
        img = tk.PhotoImage(file=temp_file)
        label = tk.Label(candle_frame, image=img)
        label.image = img  # Keep a reference to prevent garbage collection
        label.pack(fill='both', expand=True)
        
        # Tab 2: OHLC Chart
        ohlc_frame = ttk.Frame(notebook)
        notebook.add(ohlc_frame, text="OHLC")
        
        # Save the OHLC chart to a temporary file
        temp_file_ohlc = f"temp_{stock_symbol}_ohlc.png"
        mpf.plot(data, type='ohlc', style='yahoo', title=f"OHLC Chart: {stock_symbol}",
                 volume=True, mav=(5, 10), savefig=temp_file_ohlc)
        
        # Display the saved image in the tkinter window
        img_ohlc = tk.PhotoImage(file=temp_file_ohlc)
        label_ohlc = ttk.Label(ohlc_frame, image=img_ohlc)
        label_ohlc.image = img_ohlc
        label_ohlc.pack(fill='both', expand=True)
        
        # Tab 3: Line Chart with Indicators
        line_frame = ttk.Frame(notebook)
        notebook.add(line_frame, text="Line Chart")
        
        fig_line, ax_line = plt.subplots(figsize=(12, 8))
        ax_line.plot(data.index, data['Close'], label='Close Price')
        ax_line.plot(data.index, data['SMA_20'], label='SMA 20')
        ax_line.plot(data.index, data['SMA_50'], label='SMA 50')
        ax_line.plot(data.index, data['Upper_BB'], label='Upper BB', linestyle='--')
        ax_line.plot(data.index, data['Lower_BB'], label='Lower BB', linestyle='--')
        ax_line.set_title(f"Line Chart: {stock_symbol}")
        ax_line.set_xlabel("Date")
        ax_line.set_ylabel("Price")
        ax_line.legend()
        ax_line.grid(True)
        plt.tight_layout()
        
        canvas_line = FigureCanvasTkAgg(fig_line, master=line_frame)
        canvas_line.draw()
        canvas_line.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 4: Technical Indicators
        tech_frame = ttk.Frame(notebook)
        notebook.add(tech_frame, text="Technical Indicators")
        
        fig_tech, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # RSI
        axes[0].plot(data.index, data['RSI'], color='orange')
        axes[0].axhline(y=70, color='r', linestyle='-', alpha=0.3)
        axes[0].axhline(y=30, color='g', linestyle='-', alpha=0.3)
        axes[0].set_title("RSI")
        axes[0].set_ylabel("RSI Value")
        axes[0].grid(True)
        
        # MACD
        axes[1].plot(data.index, data['MACD'], color='blue', label='MACD')
        axes[1].plot(data.index, data['Signal'], color='red', label='Signal')
        axes[1].bar(data.index, data['MACD'] - data['Signal'], color=np.where(data['MACD'] >= data['Signal'], 'g', 'r'), alpha=0.3)
        axes[1].set_title("MACD")
        axes[1].set_ylabel("MACD Value")
        axes[1].legend()
        axes[1].grid(True)
        
        # Volume
        axes[2].bar(data.index, data['Volume'], color='blue', alpha=0.6)
        axes[2].set_title("Volume")
        axes[2].set_xlabel("Date")
        axes[2].set_ylabel("Volume")
        axes[2].grid(True)
        
        plt.tight_layout()
        
        canvas_tech = FigureCanvasTkAgg(fig_tech, master=tech_frame)
        canvas_tech.draw()
        canvas_tech.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Clean up temporary files after a delay
        def cleanup_temp_files():
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                if os.path.exists(temp_file_ohlc):
                    os.remove(temp_file_ohlc)
            except Exception as e:
                print(f"Error cleaning up temporary files: {e}")
                
        chart_window.after(5000, cleanup_temp_files)

    def train_model(self, stock_symbol):
        data = self.fetch_stock_data(stock_symbol, self.period_var.get())
        if data is None or data.empty:
            return
        data = self.prepare_features(data, stock_symbol)
        if data is None:
            return
        X = data[['Close_Lag1', 'SMA_5', 'SMA_10', 'RSI', 'MACD', 'VWAP']]
        y = data['Close']
        if len(X) < 2:
            messagebox.showerror("Error", "Not enough data to train the model.")
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.models[stock_symbol] = model
        joblib.dump(model, f"stock_model_{stock_symbol}.pkl")
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        self.result_text.insert(tk.END, f"{stock_symbol} - MAE: {mae:.2f}, R²: {r2:.2f}\n")
        messagebox.showinfo("Model Training", f"Model for {stock_symbol} trained!\nMAE: {mae:.2f}, R²: {r2:.2f}")
        self.update_trained_models()

    def train_multiple_models(self):
        symbols = self.stock_entry.get().replace(" ", "").split(",")
        valid_symbols = [s for s in symbols if self.is_valid_symbol(s)]
        if not valid_symbols:
            messagebox.showerror("Error", "No valid stock symbols provided.")
            return
            
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Training multiple models...\n")
        self.root.update()
        
        success_count = 0
        for i, symbol in enumerate(valid_symbols):
            try:
                self.status_label.config(text=f"Training model for {symbol} ({i+1}/{len(valid_symbols)})...")
                self.progress_bar.start()
                self.root.update()
                
                # Fetch data
                data = self.fetch_stock_data(symbol, self.period_var.get())
                if data is None or data.empty:
                    self.result_text.insert(tk.END, f"Failed to fetch data for {symbol}. Skipping.\n")
                    continue
                    
                # Prepare features
                data = self.prepare_features(data, symbol)
                if data is None:
                    self.result_text.insert(tk.END, f"Failed to prepare features for {symbol}. Skipping.\n")
                    continue
                    
                # Train model
                X = data[['Close_Lag1', 'SMA_5', 'SMA_10', 'RSI', 'MACD', 'VWAP']]
                y = data['Close']
                
                if len(X) < 2:
                    self.result_text.insert(tk.END, f"Not enough data to train model for {symbol}. Skipping.\n")
                    continue
                    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Save model
                self.models[symbol] = model
                joblib.dump(model, f"stock_model_{symbol}.pkl")
                
                # Evaluate model
                predictions = model.predict(X_test)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                self.result_text.insert(tk.END, f"{symbol} - MAE: {mae:.2f}, R²: {r2:.2f}\n")
                success_count += 1
                
            except Exception as e:
                self.result_text.insert(tk.END, f"Error training model for {symbol}: {str(e)}\n")
            finally:
                self.progress_bar.stop()
                self.root.update()
                
                # Add a delay to avoid rate limiting
                if i < len(valid_symbols) - 1:
                    time.sleep(2)
        
        self.update_trained_models()
        self.status_label.config(text=f"Training complete. {success_count}/{len(valid_symbols)} models trained successfully.")
        
        if success_count > 0:
            messagebox.showinfo("Training Complete", f"{success_count} out of {len(valid_symbols)} models trained successfully!")
        else:
            messagebox.showerror("Training Failed", "Failed to train any models. Check the logs for details.")

    def predict_future(self, stock_symbol, days_ahead=5):
        if stock_symbol not in self.models:
            messagebox.showerror("Error", f"No model trained for {stock_symbol}. Select from trained models.")
            return
        model = self.models[stock_symbol]
        data = self.fetch_stock_data(stock_symbol, self.period_var.get())
        if data is None or data.empty:
            return
        data = self.prepare_features(data, stock_symbol)
        if data is None:
            return
        X = data[['Close_Lag1', 'SMA_5', 'SMA_10', 'RSI', 'MACD', 'VWAP']]
        predictions = model.predict(X)
        confidence_intervals = []
        for _ in range(100):
            bootstrap_model = RandomForestRegressor(n_estimators=10, random_state=np.random.randint(0, 1000))
            bootstrap_model.fit(X, data['Close'])
            confidence_intervals.append(bootstrap_model.predict(X))
        ci_lower = np.percentile(confidence_intervals, 5, axis=0)
        ci_upper = np.percentile(confidence_intervals, 95, axis=0)

        last_features = X.iloc[-1:].copy()
        future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=days_ahead + 1, freq='B')[1:]
        future_predictions = []
        for _ in range(days_ahead):
            pred = model.predict(last_features)[0]
            future_predictions.append(pred)
            last_features['Close_Lag1'] = pred
            last_features['SMA_5'] = np.mean([pred] + last_features['Close_Lag1'].tolist()[-4:])
            last_features['SMA_10'] = np.mean([pred] + last_features['Close_Lag1'].tolist()[-9:])
            last_features['RSI'] = last_features['RSI'].iloc[0]
            last_features['MACD'] = last_features['MACD'].iloc[0]
            last_features['VWAP'] = last_features['VWAP'].iloc[0]

        # Create a new toplevel window for the prediction graph
        pred_window = tk.Toplevel(self.root)
        pred_window.title(f"Price Prediction: {stock_symbol}")
        pred_window.geometry("1000x600")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data['Date'], data['Close'], label="Actual Price", color='blue')
        ax.plot(data['Date'], predictions, label="Predicted Price", color='orange', linestyle='dashed')
        ax.fill_between(data['Date'], ci_lower, ci_upper, color='gray', alpha=0.3, label="90% Confidence Interval")
        ax.plot(future_dates, future_predictions, label="Future Prediction", color='red', linestyle='dashed')
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.set_title(f"Price Prediction: {stock_symbol}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Embed the matplotlib figure in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=pred_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add prediction details in text format
        details_frame = tk.Frame(pred_window, bg="#2E2E4F")
        details_frame.pack(fill=tk.X, padx=10, pady=10)
        
        last_price = data['Close'].iloc[-1]
        pred_text = f"Last Known Price: ${last_price:.2f}\n\nPredicted Prices:\n"
        for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
            change = ((price - last_price) / last_price) * 100
            pred_text += f"{date.strftime('%Y-%m-%d')}: ${price:.2f} ({change:+.2f}%)\n"
        
        details_label = tk.Label(details_frame, text=pred_text, bg="#2E2E4F", fg="white", 
                                justify=tk.LEFT, font=("Courier", 10))
        details_label.pack(padx=10, pady=10)
        # Remove the plt.show() call here

    def update_trained_models(self):
        self.model_menu['values'] = list(self.models.keys())
        if self.models:
            self.model_var.set(list(self.models.keys())[0])

    def fetch_data_action(self):
        threading.Thread(target=self.fetch_stock_data, args=(self.stock_entry.get(), self.period_var.get())).start()

    def plot_candlestick_action(self):
        threading.Thread(target=self.plot_candlestick_chart, args=(self.stock_entry.get(),)).start()

    def train_model_action(self):
        threading.Thread(target=self.train_model, args=(self.stock_entry.get(),)).start()

    def train_multiple_action(self):
        threading.Thread(target=self.train_multiple_models).start()

    def predict_future_action(self):
        threading.Thread(target=self.predict_future, args=(self.model_var.get(),)).start()

    def compare_stocks_action(self):
        symbols = self.stock_entry.get().replace(" ", "").split(",")
        valid_symbols = [s for s in symbols if self.is_valid_symbol(s)]
        if len(valid_symbols) < 2:
            messagebox.showerror("Error", "Please enter at least two valid stock symbols separated by commas.")
            return
            
        # Create a new window for comparison
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Stock Comparison")
        compare_window.geometry("1000x800")
        
        # Create figure for comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Get data for each symbol and normalize
        period = self.period_var.get()
        start_date = None
        all_data = {}
        volumes = {}
        
        for symbol in valid_symbols:
            try:
                self.status_label.config(text=f"Loading data for {symbol}...")
                self.progress_bar.start()
                self.root.update()
                
                # Use a consistent date format when reading CSV
                csv_file = f"{symbol}_{period}_stock_data.csv"
                if os.path.exists(csv_file):
                    try:
                        # Use explicit dtype and parse_dates to handle CSV format issues
                        data = pd.read_csv(csv_file, parse_dates=['Date'], 
                                          dtype={'Open': float, 'High': float, 'Low': float, 
                                                'Close': float, 'Volume': float})
                    except Exception as e:
                        # If CSV reading fails, fetch fresh data
                        self.status_label.config(text=f"CSV format issue for {symbol}, fetching fresh data...")
                        stock = yf.Ticker(symbol)
                        data = stock.history(period=period)
                        if data.empty:
                            messagebox.showwarning("Warning", f"No data found for {symbol}. Skipping.")
                            continue
                        data.reset_index(inplace=True)
                        # Overwrite problematic CSV with correct format
                        data.to_csv(csv_file, index=False)
                else:
                    stock = yf.Ticker(symbol)
                    data = stock.history(period=period)
                    if data.empty:
                        messagebox.showwarning("Warning", f"No data found for {symbol}. Skipping.")
                        continue
                    data.reset_index(inplace=True)
                    data.to_csv(csv_file, index=False)
                
                # Ensure data has the expected columns
                if 'Date' not in data.columns or 'Close' not in data.columns:
                    messagebox.showwarning("Warning", f"Invalid data format for {symbol}. Skipping.")
                    continue
                    
                # Set start date to the latest common date if not set
                if start_date is None:
                    start_date = data['Date'].min()
                else:
                    start_date = max(start_date, data['Date'].min())
                
                all_data[symbol] = data
                volumes[symbol] = data['Volume'] if 'Volume' in data.columns else None
                
            except Exception as e:
                messagebox.showwarning("Warning", f"Error processing {symbol}: {str(e)}")
                continue
            finally:
                self.progress_bar.stop()
                self.root.update()
        
        if not all_data:
            messagebox.showerror("Error", "No valid data found for any symbols.")
            return
            
        # Plot normalized prices
        for symbol, data in all_data.items():
            # Filter data to common date range
            filtered_data = data[data['Date'] >= start_date]
            if filtered_data.empty:
                continue
                
            # Normalize to percentage change from first day
            first_price = filtered_data['Close'].iloc[0]
            normalized_prices = (filtered_data['Close'] / first_price - 1) * 100
            
            # Plot on the first axis
            ax1.plot(filtered_data['Date'], normalized_prices, label=f"{symbol}")
        
        ax1.set_title("Percentage Change Comparison")
        ax1.set_ylabel("% Change")
        ax1.legend()
        ax1.grid(True)
        
        # Plot volumes on the second axis (if available)
        for symbol, volume in volumes.items():
            if volume is not None:
                filtered_data = all_data[symbol][all_data[symbol]['Date'] >= start_date]
                if not filtered_data.empty and 'Volume' in filtered_data.columns:
                    ax2.plot(filtered_data['Date'], filtered_data['Volume'], label=f"{symbol} Volume", alpha=0.7)
        
        ax2.set_title("Volume Comparison")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volume")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Embed the plot in the window
        canvas = FigureCanvasTkAgg(fig, master=compare_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add correlation information
        corr_frame = tk.Frame(compare_window, bg="#2E2E4F")
        corr_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Calculate correlations between stocks
        if len(all_data) >= 2:
            corr_text = "Price Correlations:\n"
            symbols_list = list(all_data.keys())
            
            for i in range(len(symbols_list)):
                for j in range(i+1, len(symbols_list)):
                    sym1, sym2 = symbols_list[i], symbols_list[j]
                    
                    # Merge data on date
                    df1 = all_data[sym1][['Date', 'Close']].rename(columns={'Close': sym1})
                    df2 = all_data[sym2][['Date', 'Close']].rename(columns={'Close': sym2})
                    merged = pd.merge(df1, df2, on='Date', how='inner')
                    
                    if not merged.empty and len(merged) > 1:
                        correlation = merged[sym1].corr(merged[sym2])
                        corr_text += f"{sym1} vs {sym2}: {correlation:.4f}\n"
                    else:
                        corr_text += f"{sym1} vs {sym2}: Insufficient data\n"
            
            corr_label = tk.Label(corr_frame, text=corr_text, bg="#2E2E4F", fg="white", 
                                justify=tk.LEFT, font=("Courier", 10))
            corr_label.pack(padx=10, pady=10)
        
        threading.Thread(target=self.compare_stocks, args=(valid_symbols,)).start()
        
    def compare_stocks(self, symbols, period=None):
        if period is None:
            period = self.period_var.get()
            
        self.status_label.config(text=f"Comparing stocks: {', '.join(symbols)}...")
        self.progress_bar.start()
        
        # Create a new toplevel window for the comparison
        compare_window = tk.Toplevel(self.root)
        compare_window.title(f"Stock Comparison: {', '.join(symbols)}")
        compare_window.geometry("1000x800")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(compare_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Price comparison tab
        price_frame = ttk.Frame(notebook)
        notebook.add(price_frame, text="Price Comparison")
        
        # Performance comparison tab
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="Performance Comparison")
        
        # Volume comparison tab
        vol_frame = ttk.Frame(notebook)
        notebook.add(vol_frame, text="Volume Comparison")
        
        try:
            # Fetch data for all symbols
            all_data = {}
            for symbol in symbols:
                data = self.fetch_stock_data(symbol, period)
                if data is None or data.empty:
                    continue
                all_data[symbol] = data
            
            if not all_data:
                messagebox.showerror("Error", "Could not fetch data for any of the provided symbols.")
                compare_window.destroy()
                return
                
            # 1. Price Comparison Chart
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            for symbol, data in all_data.items():
                ax1.plot(data['Date'], data['Close'], label=symbol)
            
            ax1.set_title("Stock Price Comparison")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Price")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            canvas1 = FigureCanvasTkAgg(fig1, master=price_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 2. Normalized Performance Comparison (percentage change)
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            for symbol, data in all_data.items():
                normalized = data['Close'] / data['Close'].iloc[0] * 100
                ax2.plot(data['Date'], normalized, label=symbol)
            
            ax2.set_title("Normalized Performance Comparison (Base=100)")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Performance (%)")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            canvas2 = FigureCanvasTkAgg(fig2, master=perf_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 3. Volume Comparison
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            for symbol, data in all_data.items():
                ax3.bar(data['Date'], data['Volume'], alpha=0.3, label=symbol)
            
            ax3.set_title("Trading Volume Comparison")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Volume")
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            canvas3 = FigureCanvasTkAgg(fig3, master=vol_frame)
            canvas3.draw()
            canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.status_label.config(text=f"Stock comparison completed for {', '.join(symbols)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during comparison: {str(e)}")
        finally:
            self.progress_bar.stop()
if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictor(root)
    root.mainloop()
