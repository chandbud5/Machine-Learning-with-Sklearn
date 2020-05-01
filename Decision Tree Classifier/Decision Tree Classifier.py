from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd

# IMPORTING DATASET
df = pd.read_csv("drug200.csv")

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
Y = df["Drug"]


# ASSIGNING LABELS FOR MAKING TREE
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=3)

model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
model.fit(train_x,train_y)

prediction = model.predict(test_x)

ac = metrics.accuracy_score(test_y, prediction)

print(ac)