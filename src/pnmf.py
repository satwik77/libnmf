import numpy as np
import pandas as pd
from numpy import random
import numpy.linalg as LA
import scipy.sparse as sp
import itertools



def check_non_negativity(X):
	if X.min()<0:
		return 1
	else:
		return -1

def pnmf(X, alpha=0.3, beta=0.3, n_components=None, max_iter=200):

	if check_non_negativity(X):
		print "Invalid Input"
		return -1

    W = random.rand(X.shape[0], n_components)
    H = random.rand(n_components, X.shape[1])


    list_reconstruction_err_ = []
    reconst_err_ = LA.norm(X - np.dot(W, H))
    list_reconstruction_err_.append(reconstruction_err_)

    eps = np.spacing(1) 

    for n_iter in range(1, max_iter + 1):

        XtW = (W.T).dot(X)
        HWtW = ((W.T.dot(W)).dot(H) + beta*H) 
        H = H * XtW
        H = H / HWtW

        XH = X.dot(H.T)
        WHtH = (W.dot(H.T.dot(H)) + alpha*W)
        W = W * XH
        W = W / WHtH

        reconstruction_err_ = LA.norm(X - np.dot(W, H))
        list_reconstruction_err_.append(reconstruction_err_)


    print "Reconstruction Error: " + str(list_reconstruction_err_[-1])

    return  ( np.squeeze(np.asarray(W)),np.squeeze(np.asarray(H)),
            list_reconstruction_err_[-1])