from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
#print(iris.DESCR)
features = iris.data
labels = iris.target

#print (features[20],labels[20])

#Training Classifier

clf = KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[5.4, 3.4, 1.7, 0.2]])

if preds == 0:
    ans = "Iris-Setosa"
elif preds == 1:
    ans = "Iris-Versicolor"
elif preds == 2:
    ans = "Iris-Virginica"
print(ans)