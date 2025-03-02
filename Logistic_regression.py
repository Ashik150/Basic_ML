from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
#print(iris['DESCR'])
x = iris["data"][:, 3:] # petal width
y = (iris["target"] == 2).astype(np.int64)

#Train a Logistic Regression classifier
clf = LogisticRegression()
clf.fit(x, y)
prediction = clf.predict(([[1.6]]))
print(prediction)

#Using matplotlib to plot the visualization
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = clf.predict_proba(X_new)
plt.plot(X_new, y_prob[:, 1], "g-", label="Iris-Virginica")
plt.show()