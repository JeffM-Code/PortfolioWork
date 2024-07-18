import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.ffill(inplace=True)
    data['SettlementDate'] = pd.to_datetime(data['SettlementDate'])
    data['StartTime'] = pd.to_datetime(data['StartTime'])
    data.set_index('StartTime', inplace=True)
    data = data.sort_index()
    return data

test_file_path = 'ML/ImbalancePricing/data/test/test_1.csv'
test_data = load_and_preprocess_data(test_file_path)
test_price_data = test_data[['SystemSellPrice']].values

model_file_name = 'ML/ImbalancePricing/src/models/RF_price.pkl'
scaler_file_name = 'ML/ImbalancePricing/src/models/scaler.pkl'

with open(model_file_name, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_file_name, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

test_scaled_data = scaler.transform(test_price_data)

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X_test, y_test = create_dataset(test_scaled_data, time_step)

test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()

mse = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R2 Score: {r2}')

plt.figure(figsize=(14, 8))
test_series_adjusted = pd.Series(test_price_data.flatten()[time_step + 1:], index=test_data.index[time_step + 1:])
plt.plot(test_series_adjusted.index, test_series_adjusted, label='Actual Test Data', color='blue')
plt.plot(test_series_adjusted.index, test_predictions[:len(test_series_adjusted)], label='Predicted Test Data', color='green')
plt.xlabel('Date')
plt.ylabel('System Sell Price')
plt.legend()
plt.title('Actual vs Predicted Prices')
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(test_series_adjusted, test_predictions[:len(test_series_adjusted)], color='purple')
plt.plot([test_series_adjusted.min(), test_series_adjusted.max()], [test_series_adjusted.min(), test_series_adjusted.max()], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices Scatter Plot')
plt.show()