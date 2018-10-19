import os
from setuptools import setup, find_packages

setup(name='libNMF',
      version='0.1.0',
      description='Optimization and Regularization variants of NMF',
      author='Satwik Bhattamishra',
      author_email='satwik55@gmail.com',
      url='https://github.com/satwik77/libnmf/',
      packages = find_packages(),
      license = 'MIT',
      install_requires=['numpy>=1.1.0', 'scipy>=1.1.0', 'cvxopt>=1.2.0'],
      long_description=open('README.md').read(),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
      )
