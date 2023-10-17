#!/usr/bin/env python3


import unittest
import numpy as np
import testdata as td


class TestSegmentation(unittest.TestCase):

    def test_segment(self):
        x = np.linspace(0, 100, 1001)
        y = np.linspace(100, 200, 1001)
        xydata = td.XYData(x, y)
        segments = td.segment(xydata, 100, 50)
        self.assertEqual(len(segments), 19)
        self.assertTrue(np.allclose(segments[10].x,
                                    np.linspace(50, 60, 101)[:100]))
        self.assertTrue(np.allclose(segments[10].y,
                                    np.linspace(150, 160, 101)[:100]))
        segments = td.segment(y, 100, 50)
        self.assertEqual(len(segments), 19)
        self.assertTrue(np.allclose(segments[10],
                                    np.linspace(150, 160, 101)[:100]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
