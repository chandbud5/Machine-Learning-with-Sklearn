import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ChurnData.csv")

df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]

X = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless']]
Y = df[['churn']].astype(int)

X = StandardScaler().fit(X).transform(X)

train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.2)
model = LogisticRegression(C=0.01, solver='liblinear')
model.fit(train_x,train_y)

prediction = model.predict(test_x)

ac = metrics.accuracy_score(test_y, prediction)
jac = metrics.jaccard_score(test_y, prediction)
log = metrics.log_loss(test_y, prediction)
recall = metrics.recall_score(test_y, prediction)
conf_mat = metrics.confusion_matrix(test_y, prediction, labels=[1, 0])

cmap = plt.cm.Blues
plt.imshow(conf_mat,cmap=cmap)
plt.title("Confusion matrix")
plt.text(0,0,conf_mat[0][0])
plt.text(0,1,conf_mat[0][1])
plt.text(1,0,conf_mat[1][0])
plt.text(1,1,conf_mat[1][1])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("Confusion matrix plotted")
print("Accuracy Score is ",ac)
print("Jaccard index is ",jac)
print("Recall is",recall)
print("Log loss is ",log)