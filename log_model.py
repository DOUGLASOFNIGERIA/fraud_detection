import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv')
x = dataset.iloc[:, 1:-1]
x['Dependents'] = x['Dependents'].astype(str)
x['Dependents'] = x['Dependents'].apply(lambda x: x.replace('+', ''))
x['Dependents'] = x['Dependents'].astype(float)
x = x.values
y = dataset.iloc[:, -1].values

# Change missing value with imputer
from sklearn.impute import SimpleImputer
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:, 5:9] = imputer_mean.fit_transform(x[:, 5:9])

imputer_frequency = SimpleImputer(strategy='most_frequent')
categorical_index = [0, 1, 2, 4, 9]
for i in categorical_index:
    x[:, i] = imputer_frequency.fit_transform(x[:, i].reshape(-1, 1)).flatten()

# Label Encoder
from sklearn.preprocessing import LabelEncoder
index = [0, 1, 3, 4, 10]
le = LabelEncoder()
y = le.fit_transform(y)

for i in index:
    le = LabelEncoder()
    x[:, i] = le.fit_transform(x[:, i]) 

# Split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(random_state = 1)
log_model.fit(x_train, y_train)

y_pred = log_model.predict(x_test)
df_pred = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis = 1)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

# Plot bar charts for each column
fig, axes = plt.subplots(5, 2, figsize=(10, 20))

for i, ax in enumerate(axes.flat):
    column_values = x[:, i]
    unique_values = np.unique(column_values)
    mean_target_values = [np.mean(y[column_values == val]) for val in unique_values]
    ax.bar(unique_values, mean_target_values)
    ax.set_xlabel(f'Column {i+1} Unique Values')
    ax.set_ylabel('Mean Target Value')
    ax.set_title(f'Column {i+1} vs Target')

plt.tight_layout()
plt.show()


# KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2)
knn.fit(x_train, y_train)

knn_predictions = knn.predict(x_test)
knn_result = np.concatenate((knn_predictions.reshape(len(knn_predictions), 1), y_test.reshape(len(knn_predictions),1)), axis = 1)

knn_cm = confusion_matrix(y_test, knn_predictions)
accuracy_score(y_test, knn_predictions)


# SVM model
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', random_state = 1)
svc.fit(x_train, y_train)

svc_predictions = svc.predict(x_test)
svc_result = np.concatenate((svc_predictions.reshape(len(svc_predictions), 1), y_test.reshape(len(svc_predictions),1)), axis = 1)

svc_cm = confusion_matrix(y_test, svc_predictions)
accuracy_score(y_test, svc_predictions)

# Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

hidden_layers = 5
ann = Sequential()
ann.add(Dense(hidden_layers, activation = 'relu'))
ann.add(Dense(units = 1, activation = 'sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(x_train, y_train, batch_size = 1, epochs = 12, verbose = 2)





