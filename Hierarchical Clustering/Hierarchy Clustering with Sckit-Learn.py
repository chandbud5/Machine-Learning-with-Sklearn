import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from sklearn import datasets, preprocessing, manifold
from sklearn.cluster import AgglomerativeClustering

X, Y = datasets._samples_generator.make_blobs(n_samples=70, centers=[[4,4], [-2,- 1], [1,1], [10,4]], cluster_std=0.9)
# larger the cluster_std more apart the clusters will be choose it b/w 0.5-1.5

plt.scatter(X[:,0], X[:,1], marker='o')
plt.savefig("dataset.png",dpi=600)
plt.show()

scale = preprocessing.MinMaxScaler()
X = scale.fit_transform(X)

model = AgglomerativeClustering(n_clusters=4,linkage='complete')
model.fit(X,Y)

# plotting data with their clusters
plt.figure(figsize=(5,4), dpi=300)
plt.scatter(X[:,0], X[:,1],Y==0 ,color='red')
plt.scatter(X[:,0], X[:,1],Y==1 ,color='green')
plt.scatter(X[:,0], X[:,1],Y==2 ,color='yellow')
plt.scatter(X[:,0], X[:,1],Y==3 ,color='blue')
ax = plt.axes()
ax.set_facecolor("black")
plt.savefig("clusters.png",dpi=300)
plt.show()

# Finding Dendrogram
dist_mat = distance_matrix(X,X)
z = hierarchy.linkage(dist_mat,'complete')
dendro = hierarchy.dendrogram(z)
plt.savefig("dendrogram.png",dpi=300)
plt.show()