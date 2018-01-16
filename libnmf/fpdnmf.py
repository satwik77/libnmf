#Author: Satwik Bhattamishra

"""
Chambolle-Pock based First-order Primal-dual Algo for NMF:
(Minimizes KL Divergence)

[6] Yanez, Felipe, and Francis Bach. "Primal-dual algorithms for
	non-negative matrix factorization with the Kullback-Leibler divergence." 
	In Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International 
	Conference on, pp. 2257-2261. IEEE, 2017.

"""


import numpy as np
from numpy import random
import numpy.linalg as LA
from nmfbase import NMFBase
from sys import exit


class FPDNMF(NMFBase):

	"""

	Attributes
	----------
	W : matrix of basis vectors
	H : matrix of coefficients
	frob_error : Frobenius norm
	div_error  : KL Divergence Error

	"""
	def initialize_vars(self):
		
		self.Wbar= self.W
		self.Wold= self.W

		self.Hbar= self.H
		self.Hold= self.H

		self.chi = -self.X / np.dot(self.W, self.H)
		self.chi1 = np.max( (np.dot(self.W.T, self.chi).T * (1.0/np.sum(self.W, 0))).T  , 0)
		self.chi = self.chi* (1.0/self.chi1)
	   
	def compute_factors(self, max_iter=100, nditer=5):
	
		if self.check_non_negativity():
			pass
		else:
			print "The given matrix contains negative values"
			exit()

		if not hasattr(self,'W'):
			self.initialize_w()
			   
		if not hasattr(self,'H'):
			self.initialize_h()

		self.initialize_vars()

		m= np.float(self.X.shape[1])
		n= np.float(self.X.shape[0])
		r= self._rank

		sigma=0
		tau=0

		self.frob_error = np.zeros(max_iter)
		self.div_error = np.zeros(max_iter)

		for i in xrange(max_iter):

			sigma= (np.sum(self.W) / np.sum(self.X, 0)) / LA.norm(self.W)
			sigma *= np.power(n/r, 0.5)

			tau= ( np.sum(self.X, 0) / np.sum(self.W) ) / LA.norm(self.W)
			tau *= np.power(r/n, 0.5)

			for j in xrange(nditer):
				self.update_h(sigma, tau)



			sigma= (np.sum(self.H) / np.sum(self.X, 1)) / LA.norm(self.H)
			sigma *= np.power(m/r, 0.5)

			tau= ( np.sum(self.X, 1) / np.sum(self.H) ) / LA.norm(self.H)
			tau *= np.power(r/m, 0.5)

			for j in xrange(nditer):      
				self.update_w(sigma, tau)                                      
		 
			self.frob_error[i] = self.frobenius_norm() 
			self.div_error[i] = self.kl_divergence()  


	def update_h(self, sigma, tau):

		self.chi += (np.dot(self.W, self.Hbar) * sigma)
		self.chi = (self.chi - np.sqrt( np.square(self.chi) + (self.X * (4*sigma) ) ))/2
		self.H = self.H - np.dot(self.W.T, (self.chi+1)) * tau
		self.H[self.H<0] = 0.0

		self.Hbar= 2*self.H- self.Hold
		self.Hold= self.H




	def update_w(self, sigma, tau):

		self.chi += (np.dot(self.Wbar, self.H) * sigma)
		self.chi = (self.chi - np.sqrt( np.square(self.chi) + (self.X * (4*sigma) ) ))/2
		self.W = self.W - (np.dot((self.chi+1), self.H.T).T * tau).T
		self.W[self.W<0] = 0.0

		self.Wbar= 2*self.W- self.Wold
		self.Wold= self.W

