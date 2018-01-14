#Author: Satwik Bhattamishra

"""
Classical NMF:

[1] Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. 
    In Advances in neural information processing systems (pp. 556-562).

"""

import numpy as np
import pandas as pd
from numpy import random
import numpy.linalg as LA
import scipy.sparse as sp
import itertools
from nmfbase import NMFBase



class NMF(NMFBase):
    """

    Attributes
    ----------
    W : matrix of basis vectors
    H : matrix of coefficients
    frob_error : frobenius norm

    """
       
    def update_h(self):

        XtW = np.dot(self.W.T, self.X)
        HWtW = np.dot(self.W.T.dot(self.W), self.H ) + 2**-8
        self.H *= XtW
        self.H /= HWtW



    def update_w(self):

        XH = self.X.dot(self.H.T)
        WHtH = self.W.dot(self.H.dot(self.H.T)) + 2**-8
        self.W *= XH
        self.W /= WHtH


