import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('data/train.csv', index_col='id')
test = pd.read_csv('data/test.csv', index_col='id')

# check if any columns have missing values
print(train.isnull().any())
print(test.isnull().any())

print(train.columns)

y_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
y = train[y_columns]
X = train.drop(y_columns, axis=1)

# print the first 5 rows of the dataframe
print(X.head())
print(y.head())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(test)

for y_col in y_columns:
    print('Training model for:', y_col)
    roc_scores = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y[y_col], test_size=0.2)
        model = KNeighborsClassifier(n_neighbors=15, weights='distance')
        model.fit(X_train, y_train)
        y_score = model.predict(X_test)
        roc_scores.append(roc_auc_score(y_test, y_score))
    print('Mean ROC AUC Score:', np.mean(roc_scores))

