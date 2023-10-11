#!/usr/bin/env python3

import numpy as np
import copy
import pickle
import unittest
import testdata as td


class TestXYData(unittest.TestCase):

    def test_XYData(self):
        x = np.linspace(0, 100, 1001)
        y = np.random.random(len(x))
        info = {'infomation': 'no info'}
        fd = td.XYData(x, y, info)
        fd2 = pickle.loads(pickle.dumps(fd))
        self.assertTrue(np.allclose(x, fd2.x))
        self.assertTrue(np.allclose(y, fd2.y))

    def test_Spectrum(self):
        x = np.linspace(0, 100, 1001)
        y = np.random.random(len(x))
        fd = td.Spectrum(x, y)
        self.assertTrue(np.allclose(fd.pxx, y))

    def test_derive(self):
        x = np.linspace(0, 100, 1001)
        y = np.random.random(len(x))
        fd = td.Spectrum(x, y)
        fd2 = fd.derive(td.XYData)
        fd3 = fd2.derive(td.Spectrum)
        self.assertTrue(np.allclose(fd.pxx, fd3.pxx))

    def test_linrange(self):
        x = np.arange(10000)
        data = td.as_linrange(x)
        self.assertTrue(np.allclose(list(data), x))
        self.assertEqual(x[0], 0.)
        self.assertEqual(x[9999], 9999.)
        self.assertEqual(x[-1], 9999.)
        try:
            x[10000]
        except IndexError:
            pass
        else:
            self.assertTrue(False, 'Exception not raised')

        x = [1, 2, 3, 4, 5, 6, 6.5]
        try:
            # Should raise error here
            data = td.as_linrange(x)
        except ValueError:
            pass
        else:
            self.assertTrue(False, 'Exception not raised')

    def test_pickle_LinRange(self):
        a = td.LinRange(20, 1)
        b = pickle.loads(pickle.dumps(a))
        self.assertTrue(np.allclose(list(a), list(b)))
        self.assertEqual(a.step, b.step)

    def test_copy_LinRange(self):
        a = td.LinRange(20, 1, 1)
        b = copy.copy(a)
        c = copy.deepcopy(a)
        self.assertTrue(np.allclose(list(a), list(b)))
        self.assertEqual(a.step, b.step)
        self.assertTrue(np.allclose(list(a), list(c)))
        self.assertEqual(a.step, c.step)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
