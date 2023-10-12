#!/usr/bin/env python3

import unittest
import testdata as td
import numpy as np


class TestTools(unittest.TestCase):
    def test_get_spl(self):
        # float as argument
        p = 1.
        spl = td.get_spl(p, 'power')
        self.assertAlmostEqual(spl, 93.9794, places=5)
        # list as argument
        p = [1., 2., 3.]
        spl = td.get_spl(p, 'power')
        self.assertAlmostEqual(spl[0], 93.9794, places=5)
        self.assertAlmostEqual(spl[1], 93.9794+3.01030, places=5)
        self.assertAlmostEqual(spl[2], 93.9794+4.77121, places=5)
        # xydata as argument
        f = [100, 200, 300.]
        p = [1., 2., 3.]
        s = td.Spectrum(f, p)
        spl = td.get_spl(s, 'power')
        self.assertAlmostEqual(spl.y[0], 93.9794, places=5)
        self.assertAlmostEqual(spl.y[1], 93.9794+3.01030, places=5)
        self.assertAlmostEqual(spl.y[2], 93.9794+4.77121, places=5)
        self.assertTrue(isinstance(spl, td.Spectrum))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
