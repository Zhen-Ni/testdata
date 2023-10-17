#!/usr/bin/env python3

"""Import Section from other files.

All functions defined in this file used as interface should begin with
`import_`. It should return a Section object.
"""


from typing import Optional
import functools
import copy

from scipy.io import wavfile, loadmat

from .. import core

from .loader import LoaderInfo

__all__ = 'import_wav', 'import_testlab_mat'


def import_wav(filename: str,
               name: Optional[str] = None,) -> core.Section:
    """Import a section from wave file."""
    name = filename if name is None else name
    fs, data = wavfile.read(filename)
    if len(data.shape) == 1:
        data.reshape(-1, 1)
    section = core.Section(name, records={'importer_filename': filename})
    x = core.LinRange(len(data), 1/fs, 0.)
    for i in range(data.shape[1]):
        y = data[:, i]
        xydata = core.XYData(x, y)
        channel = core.Channel(f'channel-{i}', {}, xydata,
                               LoaderInfo('wavloader', filename,
                                          {'index': i}),
                               True)
        section.append_channel(channel)
    return section


_IMPORT_TESTLAB_MAT_CACHE_SIZE = 1

@functools.lru_cache(_IMPORT_TESTLAB_MAT_CACHE_SIZE)
# This function is tested by real LMS exported data. Do not easily
# modify this function unless testing environment is ready or you are
# sure no bugs will be written.
def _import_testlab_mat_helper(filename: str) -> core.Section:
    mat = loadmat(filename)
    idx = 0
    section = core.Section(filename, records={'importer_filename': filename})
    for key, item in mat.items():
        if not key.startswith('Signal'):
            continue
        start = item[0, 0]['x_values'][0, 0]['start_value'][0, 0]
        step = item[0, 0]['x_values'][0, 0]['increment'][0, 0]
        size = item[0, 0]['x_values'][0, 0]['number_of_values'][0, 0]
        x = core.LinRange(size, step, start)
        ys = item[0, 0]['y_values'][0, 0]['values']
        xlabel = item[0, 0]['x_values'][0, 0]['quantity'][0, 0]['label'][0]
        ylabel = item[0, 0]['y_values'][0, 0]['quantity'][0, 0]['label'][0]
        name = item[0, 0]['function_record'][0, 0]['name']
        if len(name.shape) == 1:
            name = [''.join(name)]
        else:
            name = [''.join(i) for i in name[0]]
        for i, namei in enumerate(name):
            data = core.XYData(x, ys[:, i],
                               info={'xlabel': xlabel, 'ylabel': ylabel})
            loader = LoaderInfo('importerloader', filename,
                                {'import_function': import_testlab_mat,
                                 'channel_index': idx})
            records = {'channel_name': namei}
            channel = core.Channel(namei, records,
                                   source_data=data, loader_info=loader)
            section.append_channel(channel)
            idx += 1
    return section


def import_testlab_mat(filename: str,
                       name: Optional[str] = None
) -> core.Section:
    """Import a section from matlab file exported by testlab."""
    name = filename if name is None else name
    section = copy.deepcopy(_import_testlab_mat_helper(filename))
    section.name = name
    return section
