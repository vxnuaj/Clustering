import jax.numpy as jp
from KMeans import KMeans

X = jp.array([
[1.0, 2.0],
[1.5, 1.8],
[5.0, 8.0],
[8.0, 8.0],
[1.0, 0.6],
[9.0, 11.0],
[8.0, 2.0],
[10.0, 2.0],
[9.0, 3.0]
])

verbose = True
seed = 1
n_clusters = 3
max_iter = 10

model = KMeans(verbose = verbose, seed = seed)
model.cluster(X = X, n_clusters = n_clusters, max_iter = max_iter)