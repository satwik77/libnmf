import numpy as np
import pandas as pd
from numpy import random
import numpy.linalg as LA
import scipy.sparse as sp
import itertools
from kernels.rbf import rbf


def check_non_negativity(X):
	if X.min()<0:
		return 1
	else:
		return -1

def knmf(X, param=0, n_components=None, max_iter=200):

	if check_non_negativity(X):
		print "Invalid Input"
		return -1

    A = rbf(X, param)
    W = random.rand(A.shape[0], n_components)
    H = random.rand(A.shape[1], n_components)


    list_reconstruction_err_ = []
    reconst_err_ = LA.norm(A - np.dot(W, H))
    list_reconstruction_err_.append(reconstruction_err_)

    eps = np.spacing(1) 

    for n_iter in range(1, max_iter + 1):

        AtW = A.T.dot(W)
        HWtW = H.dot(W.T.dot(W)) + eps
        H = H * AtW
        H = H / HWtW

        AH = A.dot(H)
        WHtH = W.dot(H.T.dot(H)) +eps
        W = W * AH
        W = W / WHtH

        reconstruction_err_ = LA.norm(A - np.dot(W, H.T))
        list_reconstruction_err_.append(reconstruction_err_)


    print "Reconstruction Error: " + str(list_reconstruction_err_[-1])

    return  ( np.squeeze(np.asarray(W)),np.squeeze(np.asarray(H.T)),
            list_reconstruction_err_[-1])