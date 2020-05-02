import numpy as np
from scipy import cluster
from scipy.spatial import distance
from sklearn.datasets import _samples_generator
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

X, Y = _samples_generator.make_blobs(n_samples=50, centers=[[4,4], [-2,- 1], [1,1], [10,4]], cluster_std=0.9)

plt.scatter(X[:,0], X[:,1], marker='o')
plt.savefig("dataset.png",dpi=500)
plt.show()

model = AgglomerativeClustering(n_clusters=4, linkage='complete')
model.fit(X,Y)

X = preprocessing.MinMaxScaler().fit_transform(X)
plt.figure(figsize=(5,4),dpi=350)
plt.scatter(X[:,0], X[:,1], model.labels_==0,color='red')
plt.scatter(X[:,0], X[:,1], model.labels_==1,color='blue')
plt.scatter(X[:,0], X[:,1], model.labels_==2,color='yellow')
plt.scatter(X[:,0], X[:,1], model.labels_==3,color='green')
plt.axes().set_facecolor('black')
plt.savefig("cluster.png",dpi=500)
plt.show()

l = X.shape[0]

dist_mat = np.zeros([l,l])

for i in range(l):
    for j in range(l):
        dist_mat[i,j] = distance.euclidean(X[i], X[j])


Z = cluster.hierarchy.linkage(dist_mat, 'complete')
dendro = cluster.hierarchy.dendrogram(Z)
plt.savefig("Dendrogram.png",dpi=500)
plt.show()