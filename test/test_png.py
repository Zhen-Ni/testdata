#!/usr/bin/env python3

import numpy as np
import copy
import unittest
import testdata as td
import xml.etree.ElementTree as ET


class TestPng(unittest.TestCase):
    def test_dump_load(self):
        section = td.import_wav('./test/320-spoiler.wav', 'test wav')
        ch = section.channels[0]
        data = ch.get_source_data()
        td.png.dump(data, 'test.png')
        data2 = td.png.load('test.png')
        self.assertEqual(data, data2)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
