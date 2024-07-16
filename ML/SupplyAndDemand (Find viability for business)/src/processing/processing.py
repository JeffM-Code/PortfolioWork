import pandas as pd

file_paths = [
    'ML\\GasTurbineConsumption\\data\\train\\train_1.csv',
    'ML\\GasTurbineConsumption\\data\\train\\train_2.csv',
    'ML\\GasTurbineConsumption\\data\\train\\train_3.csv',
    'ML\\GasTurbineConsumption\\data\\train\\train_4.csv',
    'ML\\GasTurbineConsumption\\data\\train\\train_5.csv',
    'ML\\GasTurbineConsumption\\data\\train\\train_6.csv',
    'ML\\GasTurbineConsumption\\data\\test\\test_1.csv'
]

train_dfs = [pd.read_csv(file) for file in file_paths[:6]]
train_df = pd.concat(train_dfs, ignore_index=True)

test_df = pd.read_csv(file_paths[6])

print(train_df.head())