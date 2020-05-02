import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import _samples_generator
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

X,Y = _samples_generator.make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]],cluster_std=0.9)

plt.scatter(X[:,0], X[:,1], marker='.')
plt.savefig("dataset.png", dpi=500)
plt.show()

X = MinMaxScaler().fit_transform(X)

model = KMeans(n_clusters=4, init='k-means++', n_init=12)
model.fit(X)

label = model.labels_
centers = model.cluster_centers_
print(label)
plt.figure(dpi=350)
plt.scatter(X[:,0], X[:,1], label==0)
plt.scatter(centers[0,0], centers[0,1], marker='^',color='black')
plt.scatter(X[:,0], X[:,1], label==1)
plt.scatter(centers[1,0], centers[1,1], marker='^',color='black')
plt.scatter(X[:,0], X[:,1], label==2)
plt.scatter(centers[2,0], centers[2,1], marker='^',color='black')
plt.scatter(X[:,0], X[:,1], label==3)
plt.scatter(centers[3,0], centers[3,1], marker='^',color='black')
plt.savefig("Clusters.png",dpi=500)
plt.show()