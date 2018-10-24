import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

import imp
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

data = pd.read_csv('Data Set.csv')


X = data.drop(['Id', 'Class'], axis=1)
y = data['Class']


k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))

logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)

joblib_file = "filename.joblib"  
joblib.dump(knn, joblib_file)

knn.predict([[0.0200153, -0.3014402, -0.3008801, -0.2884384,-0.2709138,-0.4670434,-0.5239125,-0.5501519,-0.5606456,-0.1176529,-0.07766194,-0.05077618,-0.04127074,-0.3825685,-0.4515708,-0.4349683,-0.4293025,-0.2254354,-0.2188429,-0.2053359,-0.227809]])
