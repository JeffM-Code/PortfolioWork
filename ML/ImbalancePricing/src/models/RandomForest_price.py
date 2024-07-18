import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.ffill(inplace=True)
    data['SettlementDate'] = pd.to_datetime(data['SettlementDate'])
    data['StartTime'] = pd.to_datetime(data['StartTime'])
    data.set_index('StartTime', inplace=True)
    data = data.sort_index()
    return data

train_files = ['ML/ImbalancePricing/data/train/train_1.csv']
all_train_data = pd.concat([load_and_preprocess_data(file) for file in train_files])
test_file_path = 'ML/ImbalancePricing/data/test/test_1.csv'
test_data = load_and_preprocess_data(test_file_path)

train_price_data = all_train_data[['SystemSellPrice']].values
test_price_data = test_data[['SystemSellPrice']].values
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

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

with open('ML/ImbalancePricing/src/models/RF_price.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('ML/ImbalancePricing/src/models/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)