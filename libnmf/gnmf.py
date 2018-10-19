#Author: Satwik Bhattamishra

"""
Graph Regularized NMF:

[3] Cai, D., He, X., Han, J., & Huang, T. S. (2011). Graph regularized
	nonnegative matrix factorization for data representation. IEEE Transactions
	on Pattern Analysis and Machine Intelligence, 33(8), 1548-1560.

"""



import numpy as np
from numpy import random
import numpy.linalg as LA
import scipy.sparse as sp
from sys import exit
from .nmfbase import NMFBase

class GNMF(NMFBase):

	"""

	Attributes
	----------
	W : matrix of basis vectors
	H : matrix of coefficients
	frob_error : frobenius norm

	"""

	def compute_graph(self, weight_type='heat-kernel', param=0.3):
		if weight_type == 'heat-kernel':
			samples = np.matrix(self.X.T)
			sigma= param
			A= np.zeros((samples.shape[0], samples.shape[0]))

			for i in range(A.shape[0]):
				for j in range(A.shape[1]):
					A[i][j]= np.exp(-(LA.norm(samples[i] - samples[j] ))/sigma )

			return A
		elif weight_type == 'dot-weighting':
			samples = np.matrix(self.X.T)
			A= np.zeros((samples.shape[0], samples.shape[0]))

			for i in range(A.shape[0]):
				for j in range(A.shape[1]):
					A[i][j]= np.dot(samples[i],samples[j])

			return A


	def compute_factors(self, max_iter=100, lmd=0, weight_type='heat-kernel', param=None):

		if self.check_non_negativity():
			pass
		else:
			print("The given matrix contains negative values")
			exit()

		if not hasattr(self,'W'):
			self.initialize_w()

		if not hasattr(self,'H'):
			self.initialize_h()

		A = self.compute_graph(weight_type, param)

		D = np.matrix(np.diag(np.asarray(A).sum(axis=0)))

		self.frob_error = np.zeros(max_iter)

		for i in range(max_iter):

			self.update_w(lmd, A, D)

			self.update_h(lmd, A, D)

			self.frob_error[i] = self.frobenius_norm()


	def update_h(self, lmd, A, D):

		eps = 2**-8
		h_num = lmd*np.dot(A, self.H.T)+np.dot(self.X.T, self.W )
		h_den = lmd*np.dot(D, self.H.T)+np.dot(self.H.T, np.dot(self.W.T, self.W))


		self.H = np.multiply(self.H.T, (h_num+eps)/(h_den+eps))
		self.H = self.H.T
		self.H[self.H <= 0] = eps
		self.H[np.isnan(self.H)] = eps


	def update_w(self, lmd, A, D):

		XH = self.X.dot(self.H.T)
		WHtH = self.W.dot(self.H.dot(self.H.T)) + 2**-8
		self.W *= XH
		self.W /= WHtH


