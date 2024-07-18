import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.ffill(inplace=True)
    data['SettlementDate'] = pd.to_datetime(data['SettlementDate'])
    data['StartTime'] = pd.to_datetime(data['StartTime'])
    data.set_index('StartTime', inplace=True)
    data = data.sort_index()
    return data

train_files = [
    'ML/ImbalancePricing/data/train/train_1.csv',
]

all_train_data = pd.concat([load_and_preprocess_data(file) for file in train_files])
test_file_path = 'ML/ImbalancePricing/data/test/test_1.csv'
test_data = load_and_preprocess_data(test_file_path)
train_price_data = all_train_data[['SystemSellPrice']].values
test_price_data = test_data[['SystemSellPrice']].values

train_series = pd.Series(train_price_data.flatten(), index=all_train_data.index)
test_series = pd.Series(test_price_data.flatten(), index=test_data.index)

combined_price_data = np.concatenate((train_price_data, test_price_data), axis=0)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(combined_price_data)

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, y = create_dataset(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

lstm_model = Sequential([
    Input(shape=(time_step, 1)),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=100),
    Dropout(0.2),
    Dense(units=50),
    Dense(units=1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X, y, batch_size=32, epochs=50)

test_scaled_data = scaler.transform(test_price_data)
X_test, y_test = create_dataset(test_scaled_data, time_step)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

test_predictions = lstm_model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)

test_series_adjusted = test_series[time_step + 1:]

mse = mean_squared_error(test_series_adjusted, test_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_series_adjusted, test_predictions)
r2 = r2_score(test_series_adjusted, test_predictions)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R2 Score: {r2}')

test_predictions_df = pd.DataFrame(test_predictions, index=test_series_adjusted.index, columns=['Predicted Price'])

plt.figure(figsize=(8, 8))
plt.scatter(test_series_adjusted, test_predictions, color='purple')
plt.plot([test_series_adjusted.min(), test_series_adjusted.max()], [test_series_adjusted.min(), test_series_adjusted.max()], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices Scatter Plot')
plt.show()

plt.figure(figsize=(14, 8))
plt.plot(test_series.index, test_series, label='Actual Test Data', color='blue')
plt.plot(test_predictions_df.index, test_predictions_df['Predicted Price'], label='Predicted Test Data', color='green')
plt.xlabel('Date')
plt.ylabel('System Sell Price')
plt.legend()
plt.show()

lstm_model.save('ML/ImbalancePricing/src/models/LSTM_price.keras')