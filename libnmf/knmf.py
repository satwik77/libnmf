#Author: Satwik Bhattamishra

"""
Kernel NMF:

[5] Zhang, D., Zhou, Z. H., & Chen, S. (2006). Non-negative matrix 
	factorization on kernels. PRICAI 2006: Trends in Artificial Intelligence, 404-412.

"""


import numpy as np
from numpy import random
import numpy.linalg as LA
import scipy.sparse as sp
from nmfbase import NMFBase
from libnmf.kernels.rbf import rbf
from sys import exit

class KNMF(NMFBase):

	"""

	Attributes
	----------
	W : matrix of basis vectors
	H : matrix of coefficients
	frob_error : frobenius norm 

	"""

	def compute_kernel_mapping(self, kernel_type='rbf', param=0.3):
		
		if kernel_type == 'rbf':
			A = rbf(self.X, param)
			return A
		
		else:
			return self.X
		

	def compute_factors(self, max_iter=100, kernel_type='rbf', param=None):
	
		if self.check_non_negativity():
			pass
		else:
			print "The given matrix contains negative values"
			exit()

		if not hasattr(self,'W'):
			self.initialize_w()
			   
		if not hasattr(self,'H'):
			self.initialize_h()              

		A = self.compute_kernel_mapping(kernel_type, param)

		self.frob_error = np.zeros(max_iter)

		for i in xrange(max_iter):

			self.update_w(A)

			self.update_h(A)                                        
		 
			self.frob_error[i] = self.frobenius_norm()   
	   
	def update_h(self, A):
		
		AtW = np.dot(self.W.T, A)
		HWtW = np.dot(self.W.T.dot(self.W), self.H)
		self.H *= AtW
		self.H /= HWtW


	def update_w(self, A):

		AH = A.dot(self.H.T)
		WHtH = self.W.dot(self.H.dot(self.H.T))
		self.W *= AH
		self.W /= WHtH
