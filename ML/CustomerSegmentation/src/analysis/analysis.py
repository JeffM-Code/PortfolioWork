import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

file_path = 'ML\\CustomerSegmentation\\data\\train\\train.csv'

data = pd.read_csv(file_path, delimiter=',')
data.replace('?', np.nan, inplace=True)

data['Date'] = data['Date'].astype(str)
data['Time'] = data['Time'].astype(str)
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], dayfirst=True, errors='coerce')
data = data.drop(columns=['Date', 'Time'])
data = data.dropna(subset=['Datetime'])

for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

data = data.fillna(data.mean(numeric_only=True))

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(columns=['Datetime']))
scaled_data = pd.DataFrame(scaled_data, columns=data.columns[:-1])

sns.pairplot(scaled_data)
plt.suptitle("Pair Plot", y=1.02)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(scaled_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.show()