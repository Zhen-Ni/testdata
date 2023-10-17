#!/usr/bin/env python3

from typing import overload, Sequence
import numpy as np
import numpy.typing as npt
from ..core import XYData, LinRange


__all__ = ('segment',)


@overload
def segment(xydata: XYData,
            length: int,
            hop_length: int,
            window: npt.NDArray | None = ...,
            ) -> list[XYData]: ...


@overload
def segment(xydata: Sequence,
            length: int,
            hop_length: int,
            window: npt.NDArray | None = ...,
            ) -> list[npt.NDArray]: ...


def segment(data, length, hop_length, window=None):
    """Divide data into several segments with given length."""
    if isinstance(data, XYData):
        if not isinstance(data.x, LinRange):
            raise ValueError('xydata input should have linranged x')
    else:
        data = np.asarray(data)
    data_length = len(data)
    datas = []
    n_segments = (data_length - length) // hop_length + 1
    for i in range(n_segments):
        start = hop_length * i
        stop = start + length
        datai = data[start: stop]
        if window is not None:
            if isinstance(data, XYData):
                x = datai.x
                y = datai.y * window
                datai = XYData(x, y)
            else:
                datai = datai * window
        datas.append(datai)
    return datas
