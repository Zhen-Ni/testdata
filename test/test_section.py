#!/usr/bin/env python3


import numpy as np
import copy
import unittest
import testdata as td


class TestChannel(unittest.TestCase):

    def test_Section(self):
        section = td.import_wav('./test/320-spoiler.wav', 'test wav')
        section2 = copy.deepcopy(section)
        [c.derive(td.SpectrumChannel).update_spectrum()
         for c in section.channels]
        self.assertAlmostEqual(section.find_channels('0')[0].df, 1.0)
        try:
            section2[0].df
        except AttributeError:
            pass
        else:
            self.assertTrue(False, 'section2 should have no attribute "df"')
        self.assertTrue(section2[0].section is section2)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
