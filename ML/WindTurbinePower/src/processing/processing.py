import pandas as pd

location1_path = 'ML\\WindTurbineGeneration\\data\\train\\Location1.csv'
location2_path = 'ML\\WindTurbineGeneration\\data\\train\\Location2.csv'
location3_path = 'ML\\WindTurbineGeneration\\data\\test\\Location3.csv'
location4_path = 'ML\\WindTurbineGeneration\\data\\test\\Location4.csv'
location1_data = pd.read_csv(location1_path)
location2_data = pd.read_csv(location2_path)
location3_data = pd.read_csv(location3_path)
location4_data = pd.read_csv(location4_path)

location1_data['Time'] = pd.to_datetime(location1_data['Time'])
location2_data['Time'] = pd.to_datetime(location2_data['Time'])
location3_data['Time'] = pd.to_datetime(location3_data['Time'])
location4_data['Time'] = pd.to_datetime(location4_data['Time'])

X_test_location3 = location3_data.drop(['Time', 'Power'], axis=1)
y_test_location3 = location3_data['Power']

X_test_location4 = location4_data.drop(['Time', 'Power'], axis=1)
y_test_location4 = location4_data['Power']

merged_data = pd.concat([location1_data, location2_data], ignore_index=True)

print(merged_data.head())

print(location3_data.head())
print(location4_data.head())