# libNMF

The library contains implementations of different optimization and regularization variants of non-negative matrix factorization.

List of algorithms implemented:
1. Multiplicative Update Rule (nmf.py)
2. Alternating Least Squares NMF (alsnmf.py)
3. Graph Regularized NMF (gnmf.py)
4. Probabilistc NMF (pnmf.py)
5. Kernel NMF (knmf.py)
6. Chambolle-Pock based first-order primal dual algo (fpdnmf.py)






## Setup:

To get the project's source code, clone the github repository:

```shell
$ git clone https://github.com/satwik77/libnmf.git
```

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv venv
$ source venv/bin/activate
```

Install all the required packages:

```shell
$ pip install -r requirements.txt
```

Install the library by running the following command from the root directory of the repository:

```shell
$ python setup.py install	
```


## Usage:

```python
>>> import numpy as np

>>> # For Graph Regularized NMF
>>> from libnmf.gnmf import GNMF
>>> X = np.random.random((10,10))
>>> gnmf= GNMF(X, rank=4)
>>> gnmf.compute_factors(max_iter= 20, lmd= 0.3, weight_type='heat-kernel', param= 0.4)

>>> # For first-order primal-dual algo
>>> from libnmf.fpdnmf import FPDNMF
>>> fpdnmf= FPDNMF(X, rank=4)
>>> fpdnmf.compute_factors(max_iter=30, nditer=5)
>>> #print fpdnmf.W, fpdnmf.H, fpdnmf.div_error
```


Refer to examples/Simple-Usage.ipynb for more on usage.

## References

* [1] Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In Advances in neural information processing systems (pp. 556-562). [Paper](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf)

* [2] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative Matrix Factorization, Nature 401(6755), 788-799. [Paper](http://lsa.colorado.edu/LexicalSemantics/seung-nonneg-matrix.pdf)

* [3] Cai, D., He, X., Han, J., & Huang, T. S. (2011). Graph regularized nonnegative matrix factorization for data representation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(8), 1548-1560. [Paper](http://www.cad.zju.edu.cn/home/dengcai/Publication/Journal/TPAMI-GNMF.pdf)

* [4] Bayar, B., Bouaynaya, N., & Shterenberg, R. (2014). Probabilistic non-negative matrix factorization: theory and application to microarray data analysis. Journal of bioinformatics and computational biology, 12(01), 1450001. [Paper](https://pdfs.semanticscholar.org/18c2/302cbf1fe01a8338a186999b69abc5701c2e.pdf)

* [5] Zhang, D., Zhou, Z. H., & Chen, S. (2006). Non-negative matrix factorization on kernels. PRICAI 2006: Trends in Artificial Intelligence, 404-412. [Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/pricai06a.pdf)

* [6] Yanez, Felipe, and Francis Bach. "Primal-dual algorithms for non-negative matrix factorization with the Kullback-Leibler divergence." In Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on, pp. 2257-2261. IEEE, 2017. [Paper](https://arxiv.org/pdf/1412.1788.pdf)