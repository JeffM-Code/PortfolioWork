import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'ML\\BuildingEnergyEfficiency\\data\\energy_efficiency.csv'
data = pd.read_csv(file_path)

data_renamed = data.rename(columns={
    'X1': 'Relative Compactness',
    'X2': 'Surface Area',
    'X3': 'Wall Area',
    'X4': 'Roof Area',
    'X5': 'Overall Height',
    'X6': 'Orientation',
    'X7': 'Glazing Area',
    'X8': 'Glazing Area Distribution',
    'Y1': 'Heating Load',
    'Y2': 'Cooling Load'
})

plt.figure(figsize=(10, 8))
correlation_matrix = data_renamed.corr()
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(12, 8))
boxplot = sns.boxplot(data=data_renamed)
plt.title("Boxplot of all variables")
plt.xticks(rotation=90)
plt.show()