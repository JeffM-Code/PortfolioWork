import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the datasets for training
file_paths = [
    'GasTurbineConsumption\\data\\train\\ex_9.csv',
    'GasTurbineConsumption\\data\\train\\ex_20.csv',
    'GasTurbineConsumption\\data\\train\\ex_21.csv',
    'GasTurbineConsumption\\data\\train\\ex_1.csv',
    'GasTurbineConsumption\\data\\train\\ex_23.csv',
    'GasTurbineConsumption\\data\\train\\ex_24.csv'
]

# Load all datasets into a list
dfs = [pd.read_csv(file) for file in file_paths]

# Concatenate all datasets into one DataFrame
training_data = pd.concat(dfs, ignore_index=True)

# Define features and target for training
X_train = training_data[['time', 'input_voltage']]
y_train = training_data['el_power']

# Create the regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

"""
Regression - Test

"""
# Load the test datasets
test_file_paths = [
    'GasTurbineConsumption\\data\\test\\ex_4.csv'
]

# Load all test datasets into a list
test_dfs = [pd.read_csv(file) for file in test_file_paths]

# Concatenate all test datasets into one DataFrame
test_data = pd.concat(test_dfs, ignore_index=True)

# Define features and target for testing
X_test = test_data[['time', 'input_voltage']]
y_test = test_data['el_power']

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error on the test data
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Data:", mse)

# Plot the ground truth and predictions
plt.figure(figsize=(10, 6))
plt.plot(test_data['time'], y_test, label='Ground Truth', color='gray', alpha=0.6)
plt.plot(test_data['time'], y_pred, label='Test Prediction', color='green', linestyle='--')
plt.xlabel('Time (sec)')
plt.ylabel('Electrical Power (W)')
plt.title('Electrical Power Predictions over Time')
plt.legend()
plt.show()