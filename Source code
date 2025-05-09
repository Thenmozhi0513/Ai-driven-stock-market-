import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Step 1: Load stock data from Yahoo Finance or CSV
def load_stock_data(ticker="AAPL", start_date="2024-01-01", end_date="2024-12-31", use_csv=True, csv_path='stock_data.csv'):
    if use_csv and csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df.sort_values("Date", inplace=True)
        data = df['Close'].values.reshape(-1, 1)
        print(f"Loaded data from CSV: {csv_path}")
    else:
        df = yf.download(ticker, start=start_date, end=end_date)
        data = df['Close'].values.reshape(-1, 1)
        print(f"Downloaded data for {ticker} from Yahoo Finance.")
    return data, df

# Step 2: Preprocess data
def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # reshape for LSTM
    return X, y, scaler

# Step 3: Build LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Train the model
def train_model(model, X_train, y_train, epochs=20, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Step 5: Predict and plot results
def predict_and_plot(model, data, scaler, sequence_length=60):
    test_inputs = data[-sequence_length:]
    scaled_inputs = scaler.transform(test_inputs)

    X_test = [scaled_inputs.reshape(-1)]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    plt.figure(figsize=(10, 5))
    plt.title('Predicted Next Day Closing Price')
    plt.plot(data, label='Historical Price')
    plt.plot(len(data), predicted_price[0], 'ro', label='Predicted Price')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid()
    plt.show()

    print(f"Predicted next closing price: ${predicted_price[0][0]:.2f}")

# Main execution
if __name__ == "__main__":
    # Settings
    use_csv = True  # Set to False to use Yahoo Finance
    csv_path = "your_stock_data.csv"  # Replace with your actual file path
    ticker = "AAPL"  # Used only if use_csv = False
    sequence_length = 60

    raw_data, df = load_stock_data(ticker, use_csv=use_csv, csv_path=csv_path)
    X, y, scaler = preprocess_data(raw_data, sequence_length)

    model = build_model((X.shape[1], 1))
    train_model(model, X, y, epochs=20)

    predict_and_plot(model, raw_data, scaler, sequence_length)
