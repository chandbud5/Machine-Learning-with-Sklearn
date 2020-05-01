from sklearn import metrics
from sklearn import svm
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cell_samples.csv")

# To replace all null values
df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]

X = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']].values.astype(int)
Y = df[['Class']].values.astype(int)


ax = df[df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant')
df[df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax)
plt.show()

train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size=0.3)

model = svm.SVC(kernel='rbf')

model.fit(train_x,train_y)

prediction = model.predict(test_x)

ac = metrics.accuracy_score(test_y, prediction)
jac = metrics.jaccard_score(test_y, prediction, pos_label=4)
recall = metrics.recall_score(test_y, prediction, pos_label=4)
f1 = metrics.f1_score(test_y,prediction, pos_label=4)
conf_mat = metrics.confusion_matrix(test_y, prediction, labels=[2, 4])
print("Accuracy score is",ac)
print("Jaccard score", jac)
print("Recall is",recall)
print("F1 score is",f1)
print("Confusion matrix is plotted")

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