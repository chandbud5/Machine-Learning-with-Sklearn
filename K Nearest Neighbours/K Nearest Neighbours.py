from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# READ DATA
df = pd.read_csv("teleCust1000t.csv")

# SPLIT INTO LABELS AND FEATURES
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
Y = df[['custcat']].values

# FEATURE NORMALISATION
X = StandardScaler().fit(X).transform(X.astype(float))
Y = Y.reshape(1000)

# DATA SPLITTING
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.2,random_state=4)

# CREATING MODEL
model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_x,train_y)

prediction = model.predict(test_x)

print("Accuracy of a model is ", accuracy_score(test_y,prediction))