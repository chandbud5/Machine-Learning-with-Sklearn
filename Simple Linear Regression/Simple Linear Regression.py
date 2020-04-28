import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,r2_score,mean_squared_error

# Loading dataset
df = pd.read_csv("sat_score.csv")

# Selecting X & Y from dataframe
X = df['SAT']
Y = df['GPA']

# Converting to numpy arrays
X = np.asanyarray(X)
Y = np.asanyarray(Y)

# Splitting dataset
train_x , test_x , train_y , test_y = train_test_split(X, Y, test_size=0.33, random_state=42)

# Creating Model
model = LinearRegression()

train_x = train_x.reshape(-1,1)
train_y = train_y.reshape(-1,1)
test_x = test_x.reshape(-1,1)
test_y = test_y.reshape(-1,1)

# Fitting Model
model.fit(train_x, train_y)

# predicting
pred = model.predict(test_x)
print(model.intercept_,model.coef_)

plt.scatter(X,Y)
x = train_x
y = model.intercept_[0] + model.coef_[0][0]*train_x
plt.plot(x,y,color='black')
plt.show()

print(r2_score(test_y,pred))