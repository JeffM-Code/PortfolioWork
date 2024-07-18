import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import pickle

file_path = 'ML/CustomerSegmentation/data/train/train_1.csv'

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

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

silhouette_avg = silhouette_score(scaled_data, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')
db_index = davies_bouldin_score(scaled_data, data['Cluster'])
print(f'Davies-Bouldin Index: {db_index}')

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Cluster'], cmap='viridis', marker='.')
plt.title('K-means Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

model_file_path = 'ML/CustomerSegmentation/src/models/kmeans_model.pkl'
scaler_file_path = 'ML/CustomerSegmentation/src/models/scaler.pkl'

with open(model_file_path, 'wb') as model_file:
    pickle.dump(kmeans, model_file)

with open(scaler_file_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)