#!/usr/bin/env python3

import unittest

import testdata as td


class Test_database(unittest.TestCase):

    def test_DataBase_context(self):
        section = td.import_wav('./test/320-spoiler.wav', name='test')
        section2 = td.import_wav('./test/320-spoiler.wav', name='test2')
        # Open database with uri mode 'memory', which creates a pure
        # in-memory database that never interacts with disk.
        with td.SectionDB('test#.db', 'memory') as db:
            if section.name in db.list():
                db.update(section)
            else:
                db.insert(section)
            if section2.name in db.list():
                db.delete('test2')
            db.insert(section2)
            res, res2 = db.findall('test')
        self.assertTrue(section[0].source_data.x ==
                        res[0].source_data.x)
        self.assertTrue((section2[1].source_data.y ==
                         res2[1].source_data.y).all())


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
