#Author: Satwik Bhattamishra

"""
Probabilistic NMF:

[4] Bayar, B., Bouaynaya, N., & Shterenberg, R. (2014). Probabilistic 
    non-negative matrix factorization: theory and application to microarray data analysis. 
    Journal of bioinformatics and computational biology, 12(01), 1450001.

"""


import numpy as np
from numpy import random
import numpy.linalg as LA
from nmfbase import NMFBase
from sys import exit


class PNMF(NMFBase):

    """

    Attributes
    ----------
    W : matrix of basis vectors
    H : matrix of coefficients
    frob_error : frobenius norm

    """
       
    def compute_factors(self, max_iter=100, alpha= 0.2, beta= 0.2):
    
        if self.check_non_negativity():
            pass
        else:
            print "The given matrix contains negative values"
            exit()

        if not hasattr(self,'W'):
            self.initialize_w()
               
        if not hasattr(self,'H'):
            self.initialize_h()

        self.frob_error = np.zeros(max_iter)

        for i in xrange(max_iter):

            self.update_h(alpha)  
            self.update_w(beta)                                      
         
            self.frob_error[i] = self.frobenius_norm()   


    def update_h(self, beta):

        XtW = np.dot(self.W.T, self.X)
        HWtW = np.dot(self.W.T.dot(self.W), self.H ) + beta+ 2**-8
        self.H *= XtW
        self.H /= HWtW



    def update_w(self, alpha):

        XH = self.X.dot(self.H.T)
        WHtH = self.W.dot(self.H.dot(self.H.T)) + alpha+ 2**-8
        self.W *= XH
        self.W /= WHtH


