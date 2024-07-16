import pandas as pd

file_path = 'ML\\CustomerSegmentation\\data\\train\\train.csv'

data = pd.read_csv(file_path)

print(data.head())