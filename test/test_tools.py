#!/usr/bin/env python3

import unittest
import testdata as td
import numpy as np


class TestTools(unittest.TestCase):
    def test_get_spl(self):
        p = 1.
        spl = td.get_spl(p, 'power')
        self.assertAlmostEqual(spl, 93.9794, places=5)

    def test_get_octave_frequency(self):
        ob = td.OctaveBand(3, 'base10')  # one third octave
        index = [-2, -1, 0, 1, 2]
        fm = ob.midband_frequency(index)
        f1, f2 = ob.bandedge_frequency(index)
        self.assertTrue(np.allclose(f1[1:], f2[:-1]))
        self.assertTrue(np.allclose(fm,
                                    [630.96, 794.33, 1000, 1258.9, 1584.9],
                                    atol=0.02))
        self.assertTrue(np.allclose(index, ob.index(f1 + 0.01)))
        self.assertTrue(np.allclose(index, ob.index(f2 - 0.01)))

    def test_get_octave_index(self):
        ob = td.OctaveBand(3, 'base10')  # one third octave
        index = [-2, -1, 0, 1, 2]
        fm = ob.midband_frequency(index)
        f1, f2 = ob.bandedge_frequency(index)
        self.assertTrue(np.allclose(index, ob.index(f1 + 0.01)))
        self.assertTrue(np.allclose(index, ob.index(f2 - 0.01)))

    def test_get_octave_power(self):
        frequency_resolution = 1.
        t = np.linspace(0, 10, 100001)  # fs = 10000Hz
        x = np.sin(2 * np.pi * 1000 * t)
        x += 2 * np.sin(2 * np.pi * 1100 * t)
        x += 4 * np.sin(2 * np.pi * 1200 * t)
        data = td.XYData(t, x)
        data = td.Spectrum.from_time_data(data, df=frequency_resolution,
                                          window='boxcar',scaling='density',
                                          normalization='power')
        ob = td.OctaveBand()
        res = ob.power(data, kind='linear')
        fm = np.asarray(res.x)
        power = np.asarray(res.y)
        self.assertAlmostEqual(power[ob.index(fm)==-1][0], 0)
        self.assertAlmostEqual(power[ob.index(fm)==0][0], 2.5)
        self.assertAlmostEqual(power[ob.index(fm)==1][0], 8)
    
        
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
