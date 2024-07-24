import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

location1_path = 'ML\\WindTurbineGeneration\\data\\train\\Location1.csv'
location2_path = 'ML\\WindTurbineGeneration\\data\\train\\Location2.csv'
location1_data = pd.read_csv(location1_path)
location2_data = pd.read_csv(location2_path)

location1_data['Time'] = pd.to_datetime(location1_data['Time'])
location2_data['Time'] = pd.to_datetime(location2_data['Time'])

merged_data = pd.concat([location1_data, location2_data], ignore_index=True)

plt.figure(figsize=(12, 6))
plt.plot(merged_data['Time'], merged_data['Power'], label='Power')
plt.xlabel('Time')
plt.ylabel('Power Generated')
plt.title('Power Generated Over Time')
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = merged_data.drop(['Time'], axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

features = merged_data.drop(['Time', 'Power'], axis=1).columns
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(merged_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()