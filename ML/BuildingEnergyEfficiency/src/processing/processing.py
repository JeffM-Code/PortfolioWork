from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split

energy_efficiency = fetch_ucirepo(id=242)

X = energy_efficiency.data.features
y = energy_efficiency.data.targets

print(energy_efficiency.metadata)
print(energy_efficiency.variables)

file_path = 'ML\\BuildingEnergyEfficiency\\data\\energy_efficiency.csv'
data = pd.read_csv(file_path)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_file_path = 'ML\\BuildingEnergyEfficiency\\data\\train\\train.csv'
test_file_path = 'ML\\BuildingEnergyEfficiency\\data\\test\\test.csv' 

train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)