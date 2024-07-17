import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
import joblib

train_file_path = 'ML\\BuildingEnergyEfficiency\\data\\train\\train.csv'
test_file_path = 'ML\\BuildingEnergyEfficiency\\data\\test\\test.csv' 

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

def create_classes(value, bins, labels):
    return pd.cut(value, bins=bins, labels=labels)

bins = [0, 10, 20, 40, 100]
labels = ['low', 'medium', 'high', 'very high']

train_data['Heating Load Class'] = create_classes(train_data['Y1'], bins, labels)
train_data['Cooling Load Class'] = create_classes(train_data['Y2'], bins, labels)

train_data = train_data.drop(['Y1', 'Y2'], axis=1)

X_train = train_data.drop(['Heating Load Class', 'Cooling Load Class'], axis=1)
y_heating_train = train_data['Heating Load Class']
y_cooling_train = train_data['Cooling Load Class']

test_data['Heating Load Class'] = create_classes(test_data['Y1'], bins, labels)
test_data['Cooling Load Class'] = create_classes(test_data['Y2'], bins, labels)

test_data = test_data.drop(['Y1', 'Y2'], axis=1)

X_test = test_data.drop(['Heating Load Class', 'Cooling Load Class'], axis=1)
y_heating_test = test_data['Heating Load Class']
y_cooling_test = test_data['Cooling Load Class']

clf_heating = RandomForestClassifier(random_state=42)
clf_heating.fit(X_train, y_heating_train)

clf_cooling = RandomForestClassifier(random_state=42)
clf_cooling.fit(X_train, y_cooling_train)

y_heating_pred = clf_heating.predict(X_test)
heating_cm = confusion_matrix(y_heating_test, y_heating_pred)
heating_accuracy = accuracy_score(y_heating_test, y_heating_pred)
heating_report = classification_report(y_heating_test, y_heating_pred, output_dict=True)

y_cooling_pred = clf_cooling.predict(X_test)
cooling_cm = confusion_matrix(y_cooling_test, y_cooling_pred)
cooling_accuracy = accuracy_score(y_cooling_test, y_cooling_pred)
cooling_report = classification_report(y_cooling_test, y_cooling_pred, output_dict=True)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(heating_cm, clf_heating.classes_, title='Heating Load Classification')
plot_confusion_matrix(cooling_cm, clf_cooling.classes_, title='Cooling Load Classification')

heating_report_df = pd.DataFrame(heating_report).transpose()
cooling_report_df = pd.DataFrame(cooling_report).transpose()

print("Heating Load Classification Accuracy: {:.2f}%\n".format(heating_accuracy * 100))
print(tabulate(heating_report_df, headers='keys', tablefmt='psql'))
print("\nCooling Load Classification Accuracy: {:.2f}%\n".format(cooling_accuracy * 100))
print(tabulate(cooling_report_df, headers='keys', tablefmt='psql'))

test_data['Heating Load Prediction'] = y_heating_pred
test_data['Cooling Load Prediction'] = y_cooling_pred

print("Predictions for Heating Load:\n")
print(test_data[['Heating Load Class', 'Heating Load Prediction']])

print("\n\nPredictions for Cooling Load:\n")
print(test_data[['Cooling Load Class', 'Cooling Load Prediction']])

joblib.dump(clf_heating, 'ML\\BuildingEnergyEfficiency\\src\\models\\heating_load_classifier.bin')
joblib.dump(clf_cooling, 'ML\\BuildingEnergyEfficiency\\src\\models\\cooling_load_classifier.bin')