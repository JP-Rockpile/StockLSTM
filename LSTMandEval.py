import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# companies = ['BBY', 'LOW', 'WBA', 'KR', 'WMT', 'AMZN', 'TGT', 'HD', 'COST', 'AAPL']
companies = ['NEE', 'VWDRY', 'FSLR', 'ENPH', 'ORA', 'BEP', 'SPWR', 'PLUG', 'BLDP', 'TSLA']
# companies = ['XOM', 'CVX', 'BP', 'SHEL', 'COP', 'TTE', 'OXY', 'PSX', 'MPC', 'VLO']

# Process each company's data
def process_company(company, start_date='2017-07-01', end_date='2017-09-30'):
    # Download stock price data
    data = yf.download(company, start=start_date, end=end_date)

    #Calculate Moving Average For Additional Data Point
    data['Moving_Avg'] = data['Close'].rolling(window=10).mean()
    data.dropna(inplace=True)

    # Double Check Data
    if len(data) < 10:
        print(f"Not enough data for {company} to create sequences.")
        return None, None, None, None

    # Normalization
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'Moving_Avg']])

    # Sequence Creation
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, 0])
        return np.array(X), np.array(y)

    seq_length = 10  # Adjusted sequence length to fit within the available data
    X, y = create_sequences(scaled_data, seq_length)

    return X, y, scaler, data.index

# Process and store data for each company
company_data = []
for company in companies:
    X, y, scaler, dates = process_company(company)
    if X is not None:
        print(f"{company} - X shape: {X.shape}, y shape: {y.shape}")
        company_data.append((X, y, scaler, dates))

# Split into training and testing sets
split_idx = int(0.8 * len(company_data))
train_data = company_data[:split_idx]
test_data = company_data[split_idx:]

# Combine training data
X_train = np.concatenate([data[0] for data in train_data])
y_train = np.concatenate([data[1] for data in train_data])

# Combine test data
X_test = np.concatenate([data[0] for data in test_data])
y_test = np.concatenate([data[1] for data in test_data])
test_dates = np.concatenate([data[3][10:] for data in test_data])

# Build and train the LSTM model
def build_and_train_model(X_train, y_train):
    print(f"Building model - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.2)

    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test, scalers):
    predictions = model.predict(X_test)

    # Inverse transform the predictions and y_test
    y_test_actual = []
    predictions_actual = []
    start_idx = 0

    for scaler, (X, y, _, _) in zip(scalers, test_data):
        end_idx = start_idx + len(y)

        # Create a temporary array to match the scaler's expected shape
        temp_array = np.zeros((len(y), 2))
        temp_array[:, 0] = predictions[start_idx:end_idx].flatten()

        # Inverse transform the predictions and y_test
        predictions_actual.extend(scaler.inverse_transform(temp_array)[:, 0])
        temp_array[:, 0] = y.reshape(-1)
        y_test_actual.extend(scaler.inverse_transform(temp_array)[:, 0])

        start_idx = end_idx

    predictions_actual = np.array(predictions_actual)
    y_test_actual = np.array(y_test_actual)

    mae = mean_absolute_error(y_test_actual, predictions_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    r2 = r2_score(y_test_actual, predictions_actual)

    return mae, rmse, r2, y_test_actual, predictions_actual

# Train and evaluate the model
scalers = [data[2] for data in test_data]
model = build_and_train_model(X_train, y_train)
mae, rmse, r2, y_test_actual, predictions_actual = evaluate_model(model, X_test, y_test, scalers)

# Visualize the results for the test companies
plt.figure(figsize=(12,6))
for i, (X, y, scaler, dates) in enumerate(test_data):
    actual_prices = scaler.inverse_transform(np.hstack((y.reshape(-1, 1), np.zeros((len(y), 1)))))[:, 0]
    plt.plot(dates[10:], actual_prices, label=f'{companies[split_idx + i]} Actual')

plt.plot(test_dates, predictions_actual, color='red', label='Predicted Stock Price')

plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)

plt.savefig('BASEstock_price_prediction.png', dpi=300, bbox_inches='tight')

plt.show()
