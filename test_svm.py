import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('sample_data.csv')

X = df[df.columns[1:6]]
y = df['Tekanan Tulisan']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random)

import matplotlib.pyplot as plt 

plt.scatter(y, X[:,1], c=y, s=50, cmap='spring'); 
plt.show()

clf = SVC(C=1000000.0, cache_size=200, class_weight=None,
            coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
            max_iter=-1, probability=False, random_state=None,
            shrinking=True, tol=0.001, verbose=False)

clf.fit(X_train, y_train)
clf.predict(X_test)

print(clf.score(X_test, y_test))