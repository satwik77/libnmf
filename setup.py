import os
from setuptools import setup, find_packages

setup(name='libNMF',
      version='0.1',
      description='Optimization and Regularization variants of NMF',
      author='Satwik Bhattamishra',
      author_email='satwik55@gmail.com',
      url='https://github.com/satwik77/libnmf/',
      packages = find_packages(),    
      license = 'MIT',
      install_requires=['numpy', 'scipy', 'cvxopt'],
      long_description=open('README.md').read(),
      )     
