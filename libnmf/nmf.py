#Author: Satwik Bhattamishra

"""
Classical NMF (Multiplicative Update Rule):

[1] Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. 
	In Advances in neural information processing systems (pp. 556-562).

"""

import numpy as np
from numpy import random
import numpy.linalg as LA
import scipy.sparse as sp
from nmfbase import NMFBase



class NMF(NMFBase):
	"""

	Attributes
	----------
	W : matrix of basis vectors
	H : matrix of coefficients
	frob_error : frobenius norm

	Usage:
	------

	>>> from libnmf.nmf import NMF
	>>> import numpy as np

	>>> X = np.random.random((10,10))
	>>> nmf = NMF(X, rank=4)
	>>> nmf.compute_factors(50)

	>>> nmf.W
	>>> nmf.H
	>>> nmf.frob_error

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


