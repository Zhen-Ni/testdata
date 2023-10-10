#!/usr/bin/env python3

import numpy as np
import pickle
import unittest
import testdata as td


class TestChannel(unittest.TestCase):

    def test_Channel_pickle(self):
        data = td.XYData(np.random.random(100), np.random.random(100))
        channel = td.Channel('test channel',
                             source_data=data)
        c2 = pickle.loads(pickle.dumps(channel))
        self.assertEqual(c2.name, 'test channel')
        self.assertFalse(c2.source_data is None)

    def test_Channel_loader(self):
        loader_info = td.LoaderInfo('textloader', './test/channel0.txt')
        data0 = td.load_data(loader_info)
        channel = td.SpectrumChannel('test channel',
                                     loader_info=loader_info)
        channel.update_source_data()
        self.assertTrue(np.allclose(data0.x, channel.t))
        self.assertTrue(np.allclose(data0.y, channel.y))

    def test_method_from_channel(self):
        data = td.XYData(np.random.random(100), np.random.random(100))
        channel = td.Channel('test channel',
                             source_data=data)
        channel2 = td.SpectrumChannel.from_channel(channel)
        self.assertTrue(np.allclose(channel.source_data.y, channel2.y))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
