from scipy.spatial.distance import pdist, squareform
import numpy as np

def rbf(X, sigma=0.5)
	pairwise_dists = squareform(pdist(X, 'euclidean'))
	K = scipy.exp(-pairwise_dists ** 2 / sigma ** 2)
	return K
