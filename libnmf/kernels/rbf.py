from scipy.spatial.distance import pdist, squareform
import scipy

def rbf(X, sigma=0.5):
	pairwise_dists = squareform(pdist(X, 'euclidean'))
	A = scipy.exp(-pairwise_dists ** 2 / sigma ** 2)
	return A
