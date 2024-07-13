import pandas as pd

train_test_files = [
    'ML\\ImbalancePricing\\data\\train\\train_1.csv',
    'ML\\ImbalancePricing\\data\\train\\train_2.csv',
    'ML\\ImbalancePricing\\data\\train\\train_3.csv',
    'ML\\ImbalancePricing\\data\\train\\train_4.csv',
    'ML\\ImbalancePricing\\data\\train\\train_5.csv',
    'ML\\ImbalancePricing\\data\\test\\test_1.csv'
]

test_1 = train_test_files[5]
test_1_df = pd.read_csv(test_1)

print(test_1_df.head())