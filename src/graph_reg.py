import numpy as np
import pandas as pd
from numpy import random
import numpy.linalg as LA
import scipy.sparse as sp
import itertools



def check_non_negativity(A):
	if A.min()<0:
		return 1
	else:
		return -1

def graph_reg(X, A,  n_components=None, max_iter=200):

	if check_non_negativity(X):
		print "Invalid Input"
		return -1

    W = random.rand(A.shape[0], n_components)
    H = random.rand(n_components, A.shape[1])


    list_reconstruction_err_ = []
    reconst_err_ = LA.norm(A - np.dot(W, H))
    list_reconstruction_err_.append(reconstruction_err_)

    eps = np.spacing(1) 

    for n_iter in range(1, max_iter + 1):
        h1 = lambd*np.dot(H, Lm)+np.dot(W.T, (X+eps)/(np.dot(W, H)+eps))
        h2 = lambd*np.dot(H, Lp)+np.dot(W.T, np.ones(X.shape))
        H = np.multiply(H, (h1+eps)/(h2+eps))
        H[H <= 0] = eps
        H[np.isnan(H)] = eps

        w1 = np.dot((X+eps)/(np.dot(W, H)+eps), H.T)
        w2 = np.dot(np.ones(X.shape), H.T)
        W = np.multiply(W, (w1+eps)/(w2+eps))
        W[W <= 0] = eps
        W[np.isnan(W)] = eps

        if reconstruction_err_ > LA.norm(X - np.dot(W, H)):
            H = (1-eps)*H + eps*np.random.normal(
                0, 1, (n_components, n_features))**2
            W = (1-eps)*W + eps*np.random.normal(
                0, 1, (n_samples, n_components))**2
        reconstruction_err_ = LA.norm(X - np.dot(W, H))
        list_reconstruction_err_.append(reconstruction_err_)


    print "Reconstruction Error: " + str(list_reconstruction_err_[-1])

    return  ( np.squeeze(np.asarray(W)),np.squeeze(np.asarray(H)),
            list_reconstruction_err_[-1])
