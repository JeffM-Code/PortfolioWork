import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def plot_feature_importance(importances, feature_names, title='Feature Importance'):
    plt.figure(figsize=(10, 5))
    feature_importance = pd.Series(importances, index=feature_names)
    feature_importance.nlargest(10).plot(kind='barh', title=title)
    plt.xlabel('Importance')

def plot_predictions_vs_actual(y_test, y_pred, title='Actual vs Predicted'):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)

# Fetch dataset
energy_efficiency = fetch_ucirepo(id=242)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

# Create a mapping dictionary for feature names
feature_mapping = {
    'X1': 'Relative Compactness',
    'X2': 'Surface Area',
    'X3': 'Wall Area',
    'X4': 'Roof Area',
    'X5': 'Overall Height',
    'X6': 'Orientation',
    'X7': 'Glazing Area',
    'X8': 'Glazing Area Distribution'
}

# Apply the feature renaming
X = X.rename(columns=feature_mapping)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Separate targets for Heating Load and Cooling Load
y_train_heating = y_train['Y1']
y_test_heating = y_test['Y1']
y_train_cooling = y_train['Y2']
y_test_cooling = y_test['Y2']

# Train a linear regression model for Heating Load
model_heating = LinearRegression()
model_heating.fit(X_train, y_train_heating)

# Train a linear regression model for Cooling Load
model_cooling = LinearRegression()
model_cooling.fit(X_train, y_train_cooling)

# Predict and evaluate for Heating Load
y_pred_heating = model_heating.predict(X_test)
mse_heating = mean_squared_error(y_test_heating, y_pred_heating)
r2_heating = r2_score(y_test_heating, y_pred_heating)

# Predict and evaluate for Cooling Load
y_pred_cooling = model_cooling.predict(X_test)
mse_cooling = mean_squared_error(y_test_cooling, y_pred_cooling)
r2_cooling = r2_score(y_test_cooling, y_pred_cooling)

# Print evaluation metrics
print(f'Heating Load - Mean Squared Error: {mse_heating}, R2 Score: {r2_heating}')
print(f'Cooling Load - Mean Squared Error: {mse_cooling}, R2 Score: {r2_cooling}')

# Plot feature importance for Heating Load
plot_feature_importance(model_heating.coef_, X.columns, title='Feature Importance for Heating Load')

# Plot feature importance for Cooling Load
plot_feature_importance(model_cooling.coef_, X.columns, title='Feature Importance for Cooling Load')

# Plot predictions vs actual values for Heating Load
plot_predictions_vs_actual(y_test_heating, y_pred_heating, title='Actual vs Predicted Heating Load')

# Plot predictions vs actual values for Cooling Load
plot_predictions_vs_actual(y_test_cooling, y_pred_cooling, title='Actual vs Predicted Cooling Load')

plt.show()