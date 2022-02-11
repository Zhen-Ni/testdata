#!/usr/bin/env python3

"""Loaders for loading xydata using given informatoin."""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING, Callable
from dataclasses import dataclass, field

import scipy.io.wavfile as wavfile

from ..core.xydata import XYData, LinRange
if TYPE_CHECKING:
    from ..core.section import Section


__all__ = 'LoaderInfo', 'load_data'


LoaderNameType = Literal['textloader', 'wavloader', 'importerloader']


@dataclass
class LoaderInfo:
    """Information for loading xydata.

    It contains three values. `loader_name` specifies which loader
    function to use to load the data. `source_file` specifies the
    source filename.  `kwargs` is a dict which stores other key
    information for loading data, which can be different for different
    types of loader functions.
    """
    loader_name: str
    source_file: str
    kwargs: dict = field(default_factory=lambda: {})


def load_data(info: LoaderInfo) -> XYData:
    """Load source data from given loader info.

    Implemented loader_names are: textloader, wavloader.
    """
    if info.loader_name == 'importerloader':
        return importerloader(info.source_file, **info.kwargs)
    elif info.loader_name == 'textloader':
        return textloader(info.source_file)
    elif info.loader_name == 'wavloader':
        return wavloader(info.source_file, **info.kwargs)
    else:
        raise NotImplementedError('loader "{}" not implemented'
                                  .format(info.loader_name))


def importerloader(source_file: str,
                   import_function: Callable[..., Section],
                   channel_index: int, **kwargs) -> XYData:
    """Load data using importer."""
    # from . import importer
    # import_function = getattr(importer, import_function_name)
    section = import_function(source_file, **kwargs)
    channel = section.channels[channel_index]
    if channel.source_data is None:
        raise AttributeError('source data not loaded by import function')
    return channel.source_data


def textloader(source_file: str) -> XYData:
    """Load data from given file.

    The file should contain two columns seperated by whitespace.
    """
    xs = []
    ys = []
    with open(source_file, 'rb') as f:
        for line in f:
            try:
                xi, yi = [float(i) for i in line.split()]
                xs.append(xi)
                ys.append(yi)
            except Exception:
                pass
    return XYData(xs, ys, {})


def wavloader(source_file: str,
              index: int) -> XYData:
    """Load data from the `index` channel from giben wave file."""
    fs, data = wavfile.read(source_file)
    x = LinRange(len(data), 1/fs, 0.)
    y = data[:, index]
    xydata = XYData(x, y)
    return xydata
