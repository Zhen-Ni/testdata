#!/usr/bin/env python3

"""Interact with png images storing XYData.

XYData are plotted in the figure, and the binary data can also be
dumped into the image, and thus be recovered. This makes png image
works much like vector graphics."""

from .. import core
import base64
import pickle
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=10)
plt.rc('mathtext', fontset='stix')


__all__ = 'dump', 'load'


FIGSIZE = 3, 2.25
DPI = 300


def dump(data: core.XYData,
         filename: str,
         meta_data: bool = True,
         figsize: tuple[float, float] | None = None,
         dpi: int | None = None,
         ) -> None:
    if not filename.endswith('.png'):
        filename += '.png'
    figsize_ = FIGSIZE if figsize is None else figsize
    dpi_ = DPI if dpi is None else dpi
    fig = plt.figure(figsize=figsize_)
    ax = fig.add_subplot(111)
    ax.plot(data.x, data.y)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.tight_layout(pad=0)
    fig.savefig(filename, dpi=dpi_)
    if meta_data:
        write_pnginfo(data, filename)


def load(filename: str) -> core.XYData | None:
    return read_pnginfo(filename)


def write_pnginfo(data: core.XYData,
                  filename: str):
    img = Image.open(filename)
    info = PngInfo()
    info.add_text('data',
                  base64.b85encode(pickle.dumps(data)).decode())
    img.save(filename, pnginfo=info)
    img.close()


def read_pnginfo(filename: str) -> core.XYData | None:
    img = Image.open(filename)
    data = img.info.get('data')
    img.close()
    if data is None:
        return
    data = base64.b85decode(data)
    return pickle.loads(data)
