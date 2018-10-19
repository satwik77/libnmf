#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `libnmf` package."""

import unittest

class Testlibnmf(unittest.TestCase):

    def setUp(self):
        pass

    def test_libnmf_import(self):
        try:
            from libnmf.gnmf import GNMF
            from libnmf.fpdnmf import FPDNMF
            from libnmf.knmf import KNMF
            from libnmf.alsnmf import ALSNMF
            from libnmf.pnmf import PNMF
            from libnmf.nmf import NMF
        except:
            self.fail("Import Error")

    def test_NMF_instances(self):
        try:
            import numpy as np
            X = np.random.random((10,10))
            from libnmf.gnmf import GNMF
            from libnmf.fpdnmf import FPDNMF
            from libnmf.knmf import KNMF
            from libnmf.alsnmf import ALSNMF
            from libnmf.pnmf import PNMF
            from libnmf.nmf import NMF

            gnmf= GNMF(X, rank=4)
            fpdnmf= FPDNMF(X, rank=4)
            knmf= KNMF(X, rank=4)
            alsnmf= ALSNMF(X, rank=4)
            pnmf= PNMF(X, rank=4)
            nmf= NMF(X, rank=4)


        except:
            self.fail("Cannot create NMF instance")



if __name__ == '__main__':
    unittest.main()
