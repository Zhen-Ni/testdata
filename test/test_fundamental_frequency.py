#!/usr/bin/env python3


import unittest
import numpy as np
import testdata as td


class TestFundamentalFrequency(unittest.TestCase):
    def test_alias(self):
        self.assertTrue(td.find_fundamental is td.find_fundamental_hps)

    def test_hps(self):
        sec = td.import_wav("./test/fundamental.wav")
        sec[0].derive(td.SpectrumChannel)
        spec = sec[0].get_spectrum()
        f = td.find_fundamental_hps(spec, 5)
        self.assertEqual(f, 1165.0)

        # 50 Hz overtone with 10k Hz sampling frequency
        t = np.arange(0, 10, 1 / 10000)
        x = [np.sin(2 * np.pi * 50 * i * t)
             for i in range(1, 100) if i != 1]
        x = np.array(x).sum(axis=0)
        x += 0.02 * np.sin(2 * np.pi * 50 * 1 * t)
        xydata = td.XYData(t, x)
        spec = td.get_spectrum(xydata)
        f = td.utility.fundamental_frequency.find_fundamental_hps(
            spec, 5, threshold=None)
        self.assertNotAlmostEqual(f,  50, msg='Not a useful test case')
        self.assertAlmostEqual(f / 50, round(f / 50),
                               msg='Result should be multiples of 50'
                               ' (factor > 1)')
        f = td.utility.fundamental_frequency.find_fundamental_hps(
            spec, 5, threshold=0.5)
        self.assertAlmostEqual(f, 50.)

    def test_brute(self):
        sec = td.import_wav("./test/fundamental.wav")
        sec[0].derive(td.SpectrumChannel)
        spec = sec[0].get_spectrum()
        start = 500
        stop = 2000
        step = 1.0
        f = td.utility.fundamental_frequency.find_fundamental_brute(
            spec, start, stop, step)
        self.assertEqual(f, 1165.0)

    def test_ceps(self):
        # 50 Hz overtone with 10k Hz sampling frequency
        t = np.arange(0, 10, 1 / 10000)
        x = [1 / i * np.sin(2 * np.pi * 50 * i * t)
             for i in range(1, 100)]
        x = np.array(x).sum(axis=0)
        xydata = td.XYData(t, x)
        spec = td.get_spectrum(xydata)
        start = 40
        stop = 80
        f = td.utility.fundamental_frequency.find_fundamental_cepstrum(
            spec, start, stop)
        self.assertAlmostEqual(f, 50.)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
