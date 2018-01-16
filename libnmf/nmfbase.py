# Authors: Satwik Bhattamishra
# License: MIT

"""
Base class used all the methods

"""
import numpy as np
import scipy.sparse
import numpy.linalg as LA
from scipy.stats import entropy
from sys import exit

class NMFBase():
	
	def __init__(self, X, rank=10, **kwargs):
		
		self.X = X       
		self._rank = rank             
	  
		
		self.X_dim, self._samples = self.X.shape
		
	def frobenius_norm(self):
		""" Euclidean error between X and W*H """

		if hasattr(self,'H') and hasattr(self,'W'):
			error = LA.norm(self.X - np.dot(self.W, self.H))            
		else:
			error = None

		return error

	def kl_divergence(self):
		""" KL Divergence between X and W*H """

		if hasattr(self,'H') and hasattr(self,'W'):
			V = np.dot(self.W, self.H)
			error = entropy(self.X, V).sum()        
		else:
			error = None

		return error
		
	def initialize_w(self):
		""" Initalize W to random values [0,1]."""

		self.W = np.random.random((self.X_dim, self._rank)) 
		
	def initialize_h(self):
		""" Initalize H to random values [0,1]."""

		self.H = np.random.random((self._rank, self._samples)) 
		
	def update_h(self):
		"Override in subclasses"
		pass

	def update_w(self):
		"Override in subclasses"
		pass


	def check_non_negativity(self):

		if self.X.min()<0:
			return 0
		else:
			return 1

	def compute_factors(self, max_iter=100):
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

			self.update_w()

			self.update_h()                                        
		 
			self.frob_error[i] = self.frobenius_norm()                
		   
