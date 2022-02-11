#!/usr/bin/env python3

import sys
if '..' not in sys.path:
    sys.path.append('..')

import numpy as np
import copy
import unittest
import testdata as td


class TestImporterAndExporter(unittest.TestCase):

    def test_wav(self):
        section = td.import_wav('./320-spoiler.wav', 'test wav')
        td.export_wav(section, 'generated.wav')
        section2 = td.import_wav('generated.wav')
        self.assertEqual(section[0].source_data.y,
                         section2[0].source_data.y)
        self.assertEqual(section[1].source_data.x,
                         section2[1].source_data.x)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)



