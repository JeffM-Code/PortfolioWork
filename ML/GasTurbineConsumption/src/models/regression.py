import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import seaborn as sns

# Load the datasets for training
file_paths = [
    'ML\\GasTurbineConsumption\\data\\train\\ex_9.csv',
    'ML\\GasTurbineConsumption\\data\\train\\ex_20.csv',
    'ML\\GasTurbineConsumption\\data\\train\\ex_21.csv',
    'ML\\GasTurbineConsumption\\data\\train\\ex_1.csv',
    'ML\\GasTurbineConsumption\\data\\train\\ex_23.csv',
    'ML\\GasTurbineConsumption\\data\\train\\ex_24.csv'
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
    'ML\\GasTurbineConsumption\\data\\test\\ex_4.csv'
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

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', 
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = -train_scores.mean(axis=1)
val_scores_mean = -val_scores.mean(axis=1)

# Learning curves
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training error')
plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation error')
plt.xlabel('Training Size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.show()

# Error distribution plot 
sns.histplot(y_pred - y_test, kde=True, bins=30)
plt.xlabel('Prediction Error')
plt.title('Error Distribution')
plt.show()

# Plot the ground truth and predictions
plt.figure(figsize=(10, 6))
plt.plot(test_data['time'], y_test, label='Ground Truth', color='gray', alpha=0.6)
plt.plot(test_data['time'], y_pred, label='Test Prediction', color='green', linestyle='--')
plt.xlabel('Time (sec)')
plt.ylabel('Electrical Power (W)')
plt.title('Electrical Power Predictions over Time')
plt.legend()
plt.show()