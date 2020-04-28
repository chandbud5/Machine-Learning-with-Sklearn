import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('X_Y_Sinusoid_Data.csv')

x = df['x']
y = df['y']

x = np.asanyarray(x)
y = np.asanyarray(y)

plt.scatter(x,y)

x = x.reshape(-1,1)
y = y.reshape(-1,1)

train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.33)

poly = preprocessing.PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)
test_x_poly = poly.fit_transform(test_x)

model = linear_model.LinearRegression()
model.fit(train_x_poly, train_y)

a = x
eq_b = model.intercept_[0] + model.coef_[0][1]*a + model.coef_[0][2]*(a**2) + model.coef_[0][3]*(a**3)
plt.plot(a, eq_b)
plt.show()

prediction = model.predict(test_x_poly)

print("The r2_score of model is ",r2_score(test_y,prediction))
print("The Mean square error of model is ",mean_squared_error(test_y,prediction))