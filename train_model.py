import os
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pressure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

X_data = []
y_data = []

if os.path.isfile("label_list"):
    print("Info: label_list found.")
    # =================================================================
    with open("label_list", "r") as labels:
        for line in labels:
            content = line.split()

            X_data.append(float(content[0]))
            y_data.append(content[1])

    # ===============================================================

    X = []
    for a, b in zip(X_data, y_data):
        X.append([a, b])
        
    X = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=.3)

    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)
    print("Classifier 1 accuracy: ", accuracy_score(clf.predict(X_test), y_test))

    # ===================================================================================================

    h = .02
    X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 1, stop = X[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X[:, 1].min() - 1, stop = X[:, 1].max() + 1, step = 0.01))

    plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_train)):
        plt.scatter(X[:, 0], X[:, 1], c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

    plt.title('SVM (Training set)')
    plt.xlabel('Pressure')
    plt.ylabel('Pressure Key')
    plt.xticks(())
    plt.yticks(())
    plt.show()
   
    # ===================================================================================================

    while True:
        file_name = input("Enter file name to predict or z to exit: ")
        if file_name == 'z':
            break

        raw_features = pressure.start(file_name)

        raw_pen_pressure = raw_features[0]
        pen_pressure, comment = pressure.determine_pen_pressure(
            raw_pen_pressure)
        print("Pen Pressure: "+comment)

        print("Personality: ", clf.predict(
            [[raw_pen_pressure, pen_pressure]]))
        print("---------------------------------------------------")

else:
    print("Error: label_list file not found.")
