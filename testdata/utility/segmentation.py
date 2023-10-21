#!/usr/bin/env python3

"""Divide data into overlapping segments."""


from typing import overload
from ..core import ArrayLike, XYData, LinRange, Storage, as_storage


__all__ = ('segment',)


@overload
def segment(data: XYData,
            length: int,
            hop_length: int,
            window: ArrayLike | None = ...,
            ) -> list[XYData]: ...


@overload
def segment(data: ArrayLike,
            length: int,
            hop_length: int,
            window: ArrayLike | None = ...,
            ) -> Storage: ...


def segment(data, length, hop_length, window=None):
    """Divide data into several segments with given length."""
    if isinstance(data, XYData):
        if not isinstance(data.x, LinRange):
            raise ValueError('xydata input should have linranged x')
    else:
        data = as_storage(data)
    if window is not None:
        window = as_storage(window)
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
