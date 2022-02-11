#!/usr/bin/env python3

"""Export Section to other file formats.

All functions defined in this file used as interface should begin with
`export_`, its first input argument should be `section`, and second argument
is optional `filename`

"""


from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile

from .. import core
from ..core.misc import as_int


__all__ = 'export_wav',


def export_wav(section: core.Section,
               filename: Optional[str] = None,
) -> None:
    """Export a section to wave file."""
    filename = section.name if filename is None else filename
    fs = None
    data = []
    for channel in section.channels:
        xydata = channel.get_source_data()
        if not isinstance(xydata.x, core.LinRange):
            raise AttributeError('all channels should have source data'
                                 ' with linranged x')
        dt = xydata.x.step
        if fs is None:
            fs = as_int(1 / dt)
        else:
            if fs != as_int(1 / dt):
                raise AttributeError('all channels should have the '
                                     'same sampling frequency')
        data.append(xydata.y)
    wavfile.write(filename, fs, np.array(data).T)
