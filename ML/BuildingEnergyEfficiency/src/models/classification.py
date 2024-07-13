import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def create_classes(value, bins, labels):
    return pd.cut(value, bins=bins, labels=labels)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_feature_importance(importances, feature_names, title='Feature Importance'):
    plt.figure(figsize=(10, 5))
    feature_importance = pd.Series(importances, index=feature_names)
    feature_importance.nlargest(10).plot(kind='barh', title=title)
    plt.xlabel('Importance')

energy_efficiency = fetch_ucirepo(id=242)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

data = pd.concat([X, y], axis=1)

feature_mapping = {
    'X1': 'Relative Compactness',
    'X2': 'Surface Area',
    'X3': 'Wall Area',
    'X4': 'Roof Area',
    'X5': 'Overall Height',
    'X6': 'Orientation',
    'X7': 'Glazing Area',
    'X8': 'Glazing Area Distribution'
}

data = data.rename(columns=feature_mapping)

bins = [0, 10, 20, 40, 100]
labels = ['low', 'medium', 'high', 'very high']

data['Heating Load Class'] = create_classes(data['Y1'], bins, labels)
data['Cooling Load Class'] = create_classes(data['Y2'], bins, labels)

data = data.drop(['Y1', 'Y2'], axis=1)

X = data.drop(['Heating Load Class', 'Cooling Load Class'], axis=1)
y_heating = data['Heating Load Class']
y_cooling = data['Cooling Load Class']

X_train, X_test, y_heating_train, y_heating_test = train_test_split(X, y_heating, test_size=0.3, random_state=42)
X_train, X_test, y_cooling_train, y_cooling_test = train_test_split(X, y_cooling, test_size=0.3, random_state=42)

clf_heating = RandomForestClassifier()
clf_heating.fit(X_train, y_heating_train)

clf_cooling = RandomForestClassifier()
clf_cooling.fit(X_train, y_cooling_train)

y_heating_pred = clf_heating.predict(X_test)
heating_cm = confusion_matrix(y_heating_test, y_heating_pred)

y_cooling_pred = clf_cooling.predict(X_test)
cooling_cm = confusion_matrix(y_cooling_test, y_cooling_pred)

plot_feature_importance(clf_heating.feature_importances_, X.columns, title='Feature Importance for Heating Load')

plot_feature_importance(clf_cooling.feature_importances_, X.columns, title='Feature Importance for Cooling Load')

plot_confusion_matrix(heating_cm, labels, title='Heating Load Classification')
plot_confusion_matrix(cooling_cm, labels, title='Cooling Load Classification')

plt.show()