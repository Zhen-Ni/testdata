#!/usr/bin/env python3

import numpy as np
import copy
import pickle
import unittest
import testdata as td


class TestXYData(unittest.TestCase):

    def test_as_storage(self):
        x1 = [1., 2., 3., 4., 5.]
        x = td.as_storage(x1)
        self.assertTrue(isinstance(x, td.LinRange))
        self.assertTrue(np.allclose(x, x1))
        x2 = 2 ** np.array(x1)
        x = td.as_storage(x2)
        self.assertTrue(isinstance(x, td.LogRange))
        self.assertTrue(np.allclose(x, x2))
        x3 = [1, 3, 2, 5]
        x = td.as_storage(x3)
        self.assertTrue(isinstance(x, td.Array))
        self.assertTrue(np.allclose(x, x3))

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

    def test_linrange_getitem(self):
        a = td.LinRange(10, 1, 0)[::-1][::-1]
        self.assertEqual(a[:], a)
        self.assertEqual(a[4:], td.LinRange(6, 1, 4))
        self.assertEqual(a[:4], td.LinRange(4, 1, 0))
        self.assertEqual(a[1:4], td.LinRange(3, 1, 1))
        self.assertEqual(a[4:1], td.LinRange(0, 1, 4))
        self.assertEqual(a[-6:], td.LinRange(6, 1, 4))
        self.assertEqual(a[:-6], td.LinRange(4, 1, 0))
        self.assertEqual(a[-16:], a)
        self.assertEqual(a[:-16], td.LinRange(0, 1, 0))
        self.assertEqual(a[::2], td.LinRange(5, 2, 0))
        self.assertEqual(a[::-2], td.LinRange(5, -2, 9))
        self.assertEqual(a[1:9:2], td.LinRange(4, 2, 1))
        self.assertEqual(a[8::-2], td.LinRange(5, -2, 8))
        self.assertEqual(a[9::-2], td.LinRange(5, -2, 9))
        self.assertEqual(a[:8:-2], td.LinRange(1, -2, 9))
        self.assertEqual(a[:9:-2], td.LinRange(0, -2, 9))
        self.assertEqual(a[4:][:4], td.LinRange(4, 1, 4))

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

    def test_array(self):
        a = td.LinRange(10, 1, 0)
        b = td.Array(a)
        self.assertEqual(len(b), 10)
        self.assertEqual(a, b)
        self.assertEqual(a[:5], b[:5])
        self.assertEqual(b[9:2:-1][::2], a[9:2:-1][::2])
        self.assertTrue(np.allclose(np.array(b[9:2:-1][::2]),
                                    np.array(b)[9:2:-1][::2]))
        self.assertEqual(pickle.loads(pickle.dumps(b)), b)
        self.assertEqual(pickle.loads(pickle.dumps(b[:5])), b[:5])
        self.assertEqual(len(pickle.dumps([b, b[:2]])),
                         len(pickle.dumps([b, b[:9][::2]])))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
