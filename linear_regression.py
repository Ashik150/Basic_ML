import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
#(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

dia_x = np.array([[1],[2],[3]])

train_x = dia_x
test_x = dia_x

dia_y = np.array([3,2,4])

train_y = dia_y
test_y = dia_y

#Creating the model
model = linear_model.LinearRegression()
#Training the model
model.fit(train_x,train_y)
#Making Prediction
diabetes_predict =model.predict(test_x)

print("Mean Squared Error is: ",mean_squared_error(test_y,diabetes_predict))

print("Weights: ",model.coef_)
print("Intercept", model.intercept_)

plt.scatter(test_x,test_y)
plt.plot(test_x,diabetes_predict)
plt.show()

# Mean Squared Error is:  3035.060115291269
# Weights:  [941.43097333]
# Intercept 153.39713623331644


