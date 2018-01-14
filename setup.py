import os
from setuptools import setup

setup(name='libNMF',
      version='0.1',
      description='Optimization and Regularization variants of NMF',
      author='Satwik Bhattamishra',
      author_email='satwik55@gmail.com',
      url='https://github.com/satwik77/libnmf/',
      packages = ['libnmf'],    
      license = 'MIT',
      install_requires=['numpy', 'scipy'],
      long_description=open('README.md').read(),
      )     
