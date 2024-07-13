import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import seaborn as sns

file_paths = [
    'ML\\GasTurbineConsumption\\data\\train\\ex_9.csv',
    'ML\\GasTurbineConsumption\\data\\train\\ex_20.csv',
    'ML\\GasTurbineConsumption\\data\\train\\ex_21.csv',
    'ML\\GasTurbineConsumption\\data\\train\\ex_1.csv',
    'ML\\GasTurbineConsumption\\data\\train\\ex_23.csv',
    'ML\\GasTurbineConsumption\\data\\train\\ex_24.csv'
]

dfs = [pd.read_csv(file) for file in file_paths]

training_data = pd.concat(dfs, ignore_index=True)
X_train = training_data[['time', 'input_voltage']]
y_train = training_data['el_power']

model = LinearRegression()

model.fit(X_train, y_train)

test_file_paths = [
    'ML\\GasTurbineConsumption\\data\\test\\ex_4.csv'
]

test_dfs = [pd.read_csv(file) for file in test_file_paths]

test_data = pd.concat(test_dfs, ignore_index=True)

X_test = test_data[['time', 'input_voltage']]
y_test = test_data['el_power']

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Data:", mse)

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', 
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = -train_scores.mean(axis=1)
val_scores_mean = -val_scores.mean(axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training error')
plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation error')
plt.xlabel('Training Size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.show()

sns.histplot(y_pred - y_test, kde=True, bins=30)
plt.xlabel('Prediction Error')
plt.title('Error Distribution')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(test_data['time'], y_test, label='Ground Truth', color='gray', alpha=0.6)
plt.plot(test_data['time'], y_pred, label='Test Prediction', color='green', linestyle='--')
plt.xlabel('Time (sec)')
plt.ylabel('Electrical Power (W)')
plt.title('Electrical Power Predictions over Time')
plt.legend()
plt.show()