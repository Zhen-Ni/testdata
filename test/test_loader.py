#!/usr/bin/env python3

import numpy as np
import copy
import unittest
import testdata as td


class TestChannel(unittest.TestCase):

    def test_importerloader(self):
        info = td.LoaderInfo('importerloader',
                             './test/320-spoiler.wav',
                             {'import_function': td.import_wav,
                              'channel_index': 1})
        data = td.load_data(info)
        section = td.import_wav('./test/320-spoiler.wav')
        self.assertTrue(section[1].source_data.x == data.x)
        self.assertTrue(section[1].source_data.y == data.y)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)



