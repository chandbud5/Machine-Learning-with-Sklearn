from sklearn.cluster import DBSCAN
from sklearn.datasets import _samples_generator
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X,Y = _samples_generator.make_blobs(n_samples=1500, centers=[[4,3], [2,-1], [-1,4]], cluster_std=0.7)

plt.scatter(X[:,0], X[:,1])
plt.savefig("dataset.png",dpi=500)
plt.show()

X = StandardScaler().fit(X).transform(X)

model = DBSCAN(eps=0.3,min_samples=7)
model.fit(X)

labels = model.labels_
sl = set(labels)
plt.figure(dpi=350)
for i in sl:
    if i==-1:
        plt.scatter(X[:,0], X[:,1], labels==-1,color='yellow',marker='^')
    else:
        plt.scatter(X[:,0], X[:,1], labels==i)
plt.axes().set_facecolor("black")
plt.savefig("Clusters.png", dpi=1080)
plt.show()