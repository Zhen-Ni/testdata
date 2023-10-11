#!/usr/bin/env python3

import numpy as np
import copy
import os
import unittest
import testdata as td


class TestImporterAndExporter(unittest.TestCase):

    def test_wav(self):
        section = td.import_wav('./test/320-spoiler.wav', 'test wav')
        td.export_wav(section, './test/generated.wav')
        section2 = td.import_wav('./test/generated.wav')
        os.remove('./test/generated.wav')
        self.assertEqual(section[0].source_data.y,
                         section2[0].source_data.y)
        self.assertEqual(section[1].source_data.x,
                         section2[1].source_data.x)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)



