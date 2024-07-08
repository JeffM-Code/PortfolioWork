from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

"""
Metadata

"""
# fetch dataset
energy_efficiency = fetch_ucirepo(id=242)

# data (as pandas dataframes)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

# metadata
print(energy_efficiency.metadata)

# variable information
print(energy_efficiency.variables)

"""
Data format alternative

"""
# Combine features and targets into a single DataFrame
data = pd.concat([X, y], axis=1)

# Export to CSV
data.to_csv('energy_efficiency.csv', index=False)

"""
Classification

"""
energy_efficiency = fetch_ucirepo(id=242)

# Data (as pandas dataframes)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

# Combine features and targets into a single DataFrame
data = pd.concat([X, y], axis=1)

# Define ranges for classification (example: low, medium, high)
def create_classes(value, bins, labels):
    return pd.cut(value, bins=bins, labels=labels)

# Define bins and labels for the targets
bins = [0, 10, 20, 40, 100]  # Example bin edges
labels = ['low', 'medium', 'high', 'very high']

# Apply the classification
data['Heating Load Class'] = create_classes(data['Y1'], bins, labels)
data['Cooling Load Class'] = create_classes(data['Y2'], bins, labels)

# Drop the original continuous targets
data = data.drop(['Y1', 'Y2'], axis=1)

# Features and target
X = data.drop(['Heating Load Class', 'Cooling Load Class'], axis=1)
y_heating = data['Heating Load Class']
y_cooling = data['Cooling Load Class']

# Split the data
X_train, X_test, y_heating_train, y_heating_test = train_test_split(X, y_heating, test_size=0.3, random_state=42)
X_train, X_test, y_cooling_train, y_cooling_test = train_test_split(X, y_cooling, test_size=0.3, random_state=42)

# Fit a classifier for Heating Load
clf_heating = RandomForestClassifier()
clf_heating.fit(X_train, y_heating_train)

# Fit a classifier for Cooling Load
clf_cooling = RandomForestClassifier()
clf_cooling.fit(X_train, y_cooling_train)

# Predict for Heating Load
y_heating_pred = clf_heating.predict(X_test)
heating_cm = confusion_matrix(y_heating_test, y_heating_pred)

# Predict for Cooling Load
y_cooling_pred = clf_cooling.predict(X_test)
cooling_cm = confusion_matrix(y_cooling_test, y_cooling_pred)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot confusion matrices
plot_confusion_matrix(heating_cm, labels, title='Heating Load Classification')
plot_confusion_matrix(cooling_cm, labels, title='Cooling Load Classification')

# Feature importance for Heating Load
heating_feature_importance = pd.Series(clf_heating.feature_importances_, index=X.columns)
heating_feature_importance.nlargest(10).plot(kind='barh', title='Feature Importance for Heating Load')

# Feature importance for Cooling Load
cooling_feature_importance = pd.Series(clf_cooling.feature_importances_, index=X.columns)
cooling_feature_importance.nlargest(10).plot(kind='barh', title='Feature Importance for Cooling Load')


"""
Regression

"""
# Data (as pandas dataframes)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

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

# Plot predictions vs actual values for Heating Load
plt.figure(figsize=(10, 5))
plt.scatter(y_test_heating, y_pred_heating, color='blue')
plt.plot([y_test_heating.min(), y_test_heating.max()], [y_test_heating.min(), y_test_heating.max()], color='red', linewidth=2)
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')
plt.title('Actual vs Predicted Heating Load')

# Plot predictions vs actual values for Cooling Load
plt.figure(figsize=(10, 5))
plt.scatter(y_test_cooling, y_pred_cooling, color='blue')
plt.plot([y_test_cooling.min(), y_test_cooling.max()], [y_test_cooling.min(), y_test_cooling.max()], color='red', linewidth=2)
plt.xlabel('Actual Cooling Load')
plt.ylabel('Predicted Cooling Load')
plt.title('Actual vs Predicted Cooling Load')
plt.show()