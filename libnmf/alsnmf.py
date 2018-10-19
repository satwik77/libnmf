#Author: Satwik Bhattamishra

"""
Alternating Least Squares NMF:

[2] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative
    Matrix Factorization, Nature 401(6755), 788-799.

for more details refer: http://cvxopt.org/examples/tutorial/qp.html
"""

from cvxopt import matrix, solvers
import numpy as np
from numpy import random
import numpy.linalg as LA
from .nmfbase import NMFBase
import cvxopt


class ALSNMF(NMFBase):
    """

    Attributes
    ----------
    W : matrix of basis vectors
    H : matrix of coefficients
    frob_error : frobenius norm

    """

    def update_h(self):

        WtW= np.float64(np.dot(self.W.T, self.W))       #Float64 for cvxopt
        Q = matrix(WtW)
        G = matrix(-np.eye(self._rank))
        h = matrix(0.0, (self._rank, 1))
        samples = self.X.T
        cvxopt.solvers.options['show_progress'] = False
        for i in range(self._samples):
            p = matrix(np.float64(np.dot(-self.W.T, samples[i])))

            sol = solvers.qp(Q, p, G, h )
            self.H[:,i] = np.array(sol['x']).reshape((1,-1))



    def update_w(self):

        HHt= np.float64(np.dot(self.H, self.H.T))       #Float64 for cvxopt
        Q = matrix(HHt)
        G = matrix(-np.eye(self._rank))
        h = matrix(0.0, (self._rank, 1))
        samples = self.X
        cvxopt.solvers.options['show_progress'] = False
        for i in range(self._samples):
            p = matrix(np.float64(np.dot(-self.H, samples[i].T)))

            sol = solvers.qp(Q, p, G, h)
            self.W[i,:] = np.array(sol['x']).reshape((1,-1))




