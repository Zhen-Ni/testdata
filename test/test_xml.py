#!/usr/bin/env python3

import sys
if '..' not in sys.path:
    sys.path.append('..')

import numpy as np
import copy
import unittest
import testdata as td
import xml.etree.ElementTree as ET


class TestXml(unittest.TestCase):
    def test_basic_types(self):
        data = {'a': [1, 2.3, 4.5+567j, (1, 2, 'text', b'\11\22\33'), None],
                'b': ('test', set([1, 2, 3, 4, 4])),
                'c': True
                }
        element = td.xml.dump_element(data)
        data2 = td.xml.load_element(element)
        self.assertEqual(data, data2)

    def test_storage(self):
        section = td.import_wav('./320-spoiler.wav', 'test wav')
        xydata = section[0].source_data
        xml_x = td.xml.dump_element(xydata.x)
        x = td.xml.load_element(xml_x)
        self.assertEqual(x, xydata.x)
        xml_y = td.xml.dump_element(xydata.y)
        y = td.xml.load_element(xml_y)
        self.assertEqual(y, xydata.y)

    def test_xydata(self):
        section = td.import_wav('./320-spoiler.wav', 'test wav')
        xydata = section[0].source_data
        xydata.info['testinfo'] = (1, 2, 3)
        xml = td.xml.dump_element(xydata)
        data2 = td.xml.load_element(xml)
        self.assertEqual(xydata, data2)

    def test_section(self):
        section = td.import_wav('./320-spoiler.wav', 'test wav')
        xml = td.xml.dump_element(section)
        section2 = td.xml.load_element(xml)
        self.assertEqual(section.name, section2.name)
        self.assertEqual(section.records, section2.records)
        self.assertEqual(section[0].source_data, section2[0].source_data)
        
    def test_dumps_loads(self):
        section = td.import_wav('./320-spoiler.wav', 'test wav')
        xml = td.xml.dumps(section)
        section2 = td.xml.loads(xml)
        self.assertEqual(section.name, section2.name)
        self.assertEqual(section.records, section2.records)
        self.assertEqual(section[0].source_data, section2[0].source_data)

    def test_dump_load(self):
        section = td.import_wav('./320-spoiler.wav', 'test wav')
        [channel.derive(td.SpectrumChannel).update_spectrum() for
         channel in section.channels]
        td.xml.dump(section, 'test.xml')
        section2 = td.xml.load('test.xml')
        self.assertEqual(section.name, section2.name)
        self.assertEqual(section.records, section2.records)
        self.assertEqual(section[0].source_data, section2[0].source_data)
        
        
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)



