import pandas as pd

train_test_files = [
    'ML\\ImbalancePricing\\data\\train\\train_1.csv',
    'ML\\ImbalancePricing\\data\\test\\test_1.csv'
]

test_1 = train_test_files[1]
test_1_df = pd.read_csv(test_1)

print(test_1_df.head())