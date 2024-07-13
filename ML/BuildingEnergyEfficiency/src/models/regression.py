from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate

train_file_path = 'ML\\BuildingEnergyEfficiency\\data\\train\\train.csv'
test_file_path = 'ML\\BuildingEnergyEfficiency\\data\\test\\test.csv' 

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

X_train = train_data.drop(['Y1', 'Y2'], axis=1)
y_train = train_data[['Y1', 'Y2']]
X_test = test_data.drop(['Y1', 'Y2'], axis=1)
y_test = test_data[['Y1', 'Y2']]

y_train_heating = y_train['Y1']
y_test_heating = y_test['Y1']
y_train_cooling = y_train['Y2']
y_test_cooling = y_test['Y2']

model_heating = LinearRegression()
model_heating.fit(X_train, y_train_heating)

model_cooling = LinearRegression()
model_cooling.fit(X_train, y_train_cooling)

y_pred_heating = model_heating.predict(X_test)
mse_heating = mean_squared_error(y_test_heating, y_pred_heating)
r2_heating = r2_score(y_test_heating, y_pred_heating)

y_pred_cooling = model_cooling.predict(X_test)
mse_cooling = mean_squared_error(y_test_cooling, y_pred_cooling)
r2_cooling = r2_score(y_test_cooling, y_pred_cooling)

plt.figure(figsize=(10, 5))
plt.scatter(y_test_heating, y_pred_heating, color='blue')
plt.plot([y_test_heating.min(), y_test_heating.max()], [y_test_heating.min(), y_test_heating.max()], color='red', linewidth=2)
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')
plt.title('Actual vs Predicted Heating Load')
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(y_test_cooling, y_pred_cooling, color='blue')
plt.plot([y_test_cooling.min(), y_test_cooling.max()], [y_test_cooling.min(), y_test_cooling.max()], color='red', linewidth=2)
plt.xlabel('Actual Cooling Load')
plt.ylabel('Predicted Cooling Load')
plt.title('Actual vs Predicted Cooling Load')
plt.show()

print(f'Heating Load - Mean Squared Error: {mse_heating}, R2 Score: {r2_heating}')
print(f'Cooling Load - Mean Squared Error: {mse_cooling}, R2 Score: {r2_cooling}')

predictions1 = pd.DataFrame({
    'Actual Heating Load': y_test_heating,
    'Predicted Heating Load': y_pred_heating,
})

predictions2 = pd.DataFrame({
    'Actual Cooling Load': y_test_cooling,
    'Predicted Cooling Load': y_pred_cooling
})

print("\n\nHeating Predictions:\n")
print(predictions1)

print("\n\nCooling Predictions:\n")
print(predictions2)