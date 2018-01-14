# libNMF

The project contains implementations of different optimization and regularization variants of non-negative matrix factorization.

## Setup:

To get the project's source code, clone the github repository:

    $ git clone https://github.com/satwik77/libnmf.git

Install VirtualEnv using the following (optional):

    $ [sudo] pip install virtualenv

Create and activate your virtual environment (optional):

    $ virtualenv venv
    $ source venv/bin/activate

Install all the required packages:

	$ pip install -r requirements.txt

Install the library by running the following command from the root directory of the repository:

	$ python setup.py install


## Usage:

	>>> import numpy as np
	>>> from libnmf.gnmf import GNMF
	
	>>> X = np.random.random((10,10))
	>>> gnmf= GNMF(X, rank=4)
	>>> gnmf.compute_factors(20, 0.3, 'heat-kernel', 0.4)
	>>> print gnmf.W
	>>> print gnmf.frob_error

## References

* [1] Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In Advances in neural information processing systems (pp. 556-562). [Paper](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf)

* [2] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative Matrix Factorization, Nature 401(6755), 788-799. [Paper](http://lsa.colorado.edu/LexicalSemantics/seung-nonneg-matrix.pdf)

* [3] Cai, D., He, X., Han, J., & Huang, T. S. (2011). Graph regularized nonnegative matrix factorization for data representation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(8), 1548-1560. [Paper](http://www.cad.zju.edu.cn/home/dengcai/Publication/Journal/TPAMI-GNMF.pdf)

* [4] Bayar, B., Bouaynaya, N., & Shterenberg, R. (2014). Probabilistic non-negative matrix factorization: theory and application to microarray data analysis. Journal of bioinformatics and computational biology, 12(01), 1450001. [Paper](https://pdfs.semanticscholar.org/18c2/302cbf1fe01a8338a186999b69abc5701c2e.pdf)

* [5] Zhang, D., Zhou, Z. H., & Chen, S. (2006). Non-negative matrix factorization on kernels. PRICAI 2006: Trends in Artificial Intelligence, 404-412. [Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/pricai06a.pdf)